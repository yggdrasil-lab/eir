"""Biometrics router for reading and managing health time-series."""

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import BiometricMeasurement, HealthFeedback, HealthSample
from app.schemas import (
    BiometricCreate,
    BiometricRead,
    DailyBiometricSummary,
    DetectedWorkoutRead,
    HealthSampleRead,
)
from app.services.biometrics import deduplicate_step_samples

router = APIRouter(prefix="/biometrics", tags=["biometrics"])


def detect_workouts_for_day(db: Session, target_date: date) -> list[DetectedWorkoutRead]:
    """Infers sustained physical exertion blocks from granular active energy and heart rate samples."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=settings.tz)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=settings.tz)

    samples = list(
        db.scalars(
            select(HealthSample)
            .where(
                HealthSample.deleted_at.is_(None),
                HealthSample.sample_type.in_(["active_energy", "heart_rate", "step_count"]),
                HealthSample.measured_at >= start_dt,
                HealthSample.measured_at <= end_dt,
            )
            .order_by(HealthSample.measured_at.asc())
        )
    )

    if not samples:
        return []

    # Partition 24 hours into 288 5-minute slots for exertion thresholding
    slots = [{"energy": Decimal(0), "hr_vals": []} for _ in range(288)]
    for s in samples:
        dt = s.measured_at.astimezone(settings.tz) if s.measured_at.tzinfo else s.measured_at.replace(tzinfo=settings.tz)
        offset = (dt - start_dt).total_seconds()
        idx = int(offset // 300)
        if 0 <= idx < 288:
            if s.sample_type in ("active_energy", "active_energy_kcal"):
                slots[idx]["energy"] += s.value
            elif s.sample_type in ("heart_rate", "resting_hr"):
                slots[idx]["hr_vals"].append(float(s.value))

    active_indices = []
    for i, slot in enumerate(slots):
        avg_hr = (sum(slot["hr_vals"]) / len(slot["hr_vals"])) if slot["hr_vals"] else None
        # Exertion threshold: >= 20 kcal in 5m (~4 kcal/min) OR average HR in 5m >= 120 bpm
        is_active = (slot["energy"] >= Decimal(20)) or (avg_hr is not None and avg_hr >= 120.0)
        if is_active:
            active_indices.append(i)

    if not active_indices:
        return []

    # Merge clusters with max gap of 3 slots (15 minutes) to bridge rest/water intervals
    clusters: list[tuple[int, int]] = []
    curr = [active_indices[0]]
    for idx in active_indices[1:]:
        if idx - curr[-1] <= 3:
            curr.append(idx)
        else:
            clusters.append((curr[0], curr[-1]))
            curr = [idx]
    if curr:
        clusters.append((curr[0], curr[-1]))

    detected: list[DetectedWorkoutRead] = []
    for start_idx, end_idx in clusters:
        duration_mins = (end_idx - start_idx + 1) * 5
        block_energy = sum((slots[i]["energy"] for i in range(start_idx, end_idx + 1)), Decimal(0))
        all_hrs: list[float] = []
        for i in range(start_idx, end_idx + 1):
            all_hrs.extend(slots[i]["hr_vals"])

        avg_hr = int(round(sum(all_hrs) / len(all_hrs))) if all_hrs else None
        peak_hr = int(round(max(all_hrs))) if all_hrs else None

        # Filter noise: minimum 30 minutes AND (active energy >= 250 kcal OR avg HR >= 120)
        if duration_mins >= 30 and (block_energy >= Decimal(250) or (avg_hr is not None and avg_hr >= 120)):
            s_time = start_dt + timedelta(minutes=start_idx * 5)
            e_time = start_dt + timedelta(minutes=(end_idx + 1) * 5)
            # Centralized multi-device step deduplication for the workout window
            cluster_steps = [
                s for s in samples
                if s.sample_type in ("step_count", "steps")
                and s_time <= (s.measured_at.astimezone(settings.tz) if s.measured_at.tzinfo else s.measured_at.replace(tzinfo=settings.tz)) < e_time
            ]
            block_steps = deduplicate_step_samples(cluster_steps)

            conf = "high" if (avg_hr and avg_hr >= 130 and block_energy >= Decimal(350)) else "medium"
            detected.append(
                DetectedWorkoutRead(
                    started_at=s_time,
                    ended_at=e_time,
                    duration_minutes=duration_mins,
                    active_energy_kcal=int(round(block_energy)),
                    avg_hr_bpm=avg_hr,
                    peak_hr_bpm=peak_hr,
                    step_count=int(round(block_steps)) if block_steps > 0 else None,
                    confidence=conf,
                )
            )

    return detected


@router.get("/granular", response_model=list[HealthSampleRead])
def get_granular_samples(
    target_date: date = Query(description="Calendar date (YYYY-MM-DD)"),
    sample_type: str = Query(description="Sample type, e.g. 'heart_rate', 'sleep', 'active_energy'"),
    limit: int = Query(default=2000, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> list[HealthSample]:
    """Retrieves high-resolution time-series samples for a specific date and metric."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=settings.tz)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=settings.tz)

    stmt = (
        select(HealthSample)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type == sample_type,
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.asc())
        .limit(limit)
    )
    return list(db.scalars(stmt))


@router.get("/latest", response_model=BiometricRead | None)
def get_latest_biometrics(
    source: str | None = Query(default=None, description="Optional source filter (e.g. 'withings', 'apple_health')"),
    db: Session = Depends(get_db),
) -> BiometricMeasurement | None:
    """Returns the most recent biometric record."""
    stmt = select(BiometricMeasurement).where(BiometricMeasurement.deleted_at.is_(None))
    if source:
        stmt = stmt.where(BiometricMeasurement.source == source)
    stmt = stmt.order_by(BiometricMeasurement.measured_at.desc()).limit(1)
    return db.scalar(stmt)


@router.get("/daily/{target_date}", response_model=DailyBiometricSummary)
def get_daily_summary(
    target_date: date,
    db: Session = Depends(get_db),
) -> DailyBiometricSummary:
    """Aggregates all biometric readings for a single calendar day."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=settings.tz)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=settings.tz)

    stmt = (
        select(BiometricMeasurement)
        .where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.measured_at >= start_dt,
            BiometricMeasurement.measured_at <= end_dt,
        )
        .order_by(BiometricMeasurement.measured_at.asc())
    )
    records = list(db.scalars(stmt))

    summary = DailyBiometricSummary(date=target_date)
    readiness_notes = []

    for r in records:
        if r.weight_kg is not None:
            summary.latest_weight_kg = r.weight_kg
        if r.body_fat_pct is not None:
            summary.latest_body_fat_pct = r.body_fat_pct
        if r.resting_hr_bpm is not None:
            summary.resting_hr_bpm = r.resting_hr_bpm
        if r.hrv_ms is not None:
            summary.hrv_ms = r.hrv_ms
        if r.sleep_duration_s is not None:
            summary.sleep_duration_hours = round(r.sleep_duration_s / 3600.0, 2)
        if r.sleep_deep_s is not None:
            summary.sleep_deep_hours = round(r.sleep_deep_s / 3600.0, 2)
        if r.sleep_score is not None:
            summary.sleep_score = r.sleep_score
        if r.step_count is not None:
            summary.step_count = (summary.step_count or 0) + r.step_count
        if r.active_energy_kcal is not None:
            summary.active_energy_kcal = (summary.active_energy_kcal or 0) + r.active_energy_kcal
        if r.notes:
            readiness_notes.append(r.notes)

    # Granular samples take precedence if recorded for this day
    granular_active_energy = db.scalar(
        select(func.sum(HealthSample.value)).where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type == "active_energy",
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
    )
    if granular_active_energy is not None:
        summary.active_energy_kcal = int(round(granular_active_energy))

    # Step count comes from BiometricMeasurement rollup or deduplicated granular samples
    if summary.step_count is None:
        raw_steps = list(
            db.scalars(
                select(HealthSample).where(
                    HealthSample.deleted_at.is_(None),
                    HealthSample.sample_type.in_(["step_count", "steps"]),
                    HealthSample.measured_at >= start_dt,
                    HealthSample.measured_at <= end_dt,
                )
            )
        )
        if raw_steps:
            summary.step_count = int(round(deduplicate_step_samples(raw_steps)))

    # Active feedbacks for this day
    feedback_stmt = select(HealthFeedback).where(
        HealthFeedback.deleted_at.is_(None),
        HealthFeedback.date == target_date,
    )
    for fb in db.scalars(feedback_stmt):
        if fb.feedback_type == "fasting":
            summary.fasting = fb
        elif fb.feedback_type == "workout":
            summary.workout = fb
        elif fb.feedback_type == "wellness":
            summary.wellness = fb

    summary.readiness_notes = readiness_notes
    summary.detected_workouts = detect_workouts_for_day(db, target_date)
    return summary


@router.get("/detected-workouts", response_model=list[DetectedWorkoutRead])
def get_detected_workouts(
    target_date: date = Query(description="Calendar date (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
) -> list[DetectedWorkoutRead]:
    """Scans high-resolution biometrics to detect unlogged physical exertion blocks."""
    return detect_workouts_for_day(db, target_date)


@router.get("", response_model=list[BiometricRead])
def list_biometrics(
    source: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[BiometricMeasurement]:
    """Lists historical biometric measurements."""
    stmt = (
        select(BiometricMeasurement)
        .where(BiometricMeasurement.deleted_at.is_(None))
        .order_by(BiometricMeasurement.measured_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if source:
        stmt = stmt.where(BiometricMeasurement.source == source)
    return list(db.scalars(stmt))


@router.post("", response_model=BiometricRead, status_code=status.HTTP_201_CREATED)
def create_biometric_entry(
    payload: BiometricCreate,
    db: Session = Depends(get_db),
) -> BiometricMeasurement:
    """Manually records a biometric measurement."""
    measurement = BiometricMeasurement(
        id=uuid.uuid4(),
        measured_at=payload.measured_at,
        source=payload.source,
        weight_kg=payload.weight_kg,
        body_fat_pct=payload.body_fat_pct,
        muscle_mass_kg=payload.muscle_mass_kg,
        water_pct=payload.water_pct,
        bone_mass_kg=payload.bone_mass_kg,
        resting_hr_bpm=payload.resting_hr_bpm,
        hrv_ms=payload.hrv_ms,
        sleep_duration_s=payload.sleep_duration_s,
        sleep_deep_s=payload.sleep_deep_s,
        sleep_rem_s=payload.sleep_rem_s,
        sleep_score=payload.sleep_score,
        step_count=payload.step_count,
        active_energy_kcal=payload.active_energy_kcal,
        notes=payload.notes,
        extra=payload.extra,
    )
    db.add(measurement)
    db.commit()
    db.refresh(measurement)
    return measurement


@router.get("/{measurement_id}", response_model=BiometricRead)
def get_biometric_entry(
    measurement_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> BiometricMeasurement:
    """Fetches a single biometric measurement by ID."""
    measurement = db.get(BiometricMeasurement, measurement_id)
    if not measurement or measurement.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Biometric measurement not found")
    return measurement
