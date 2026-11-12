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
from app.services.biometrics import (
    classify_activity_pattern,
    compute_workout_exertion_metrics,
    deduplicate_step_samples,
)

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
    slots = [{"energy": Decimal(0), "hr_vals": [], "step_samples": []} for _ in range(288)]
    for s in samples:
        dt = s.measured_at.astimezone(settings.tz) if s.measured_at.tzinfo else s.measured_at.replace(tzinfo=settings.tz)
        offset = (dt - start_dt).total_seconds()
        idx = int(offset // 300)
        if 0 <= idx < 288:
            if s.sample_type in ("active_energy", "active_energy_kcal"):
                slots[idx]["energy"] += s.value
            elif s.sample_type in ("heart_rate", "resting_hr"):
                slots[idx]["hr_vals"].append(float(s.value))
            elif s.sample_type in ("step_count", "steps"):
                slots[idx]["step_samples"].append(s)

    active_indices = []
    for i, slot in enumerate(slots):
        avg_hr = (sum(slot["hr_vals"]) / len(slot["hr_vals"])) if slot["hr_vals"] else None
        slot_step_val = deduplicate_step_samples(slot["step_samples"]) if slot["step_samples"] else Decimal(0)
        # Exertion threshold: >= 20 kcal in 5m (~4 kcal/min), avg HR >= 120 bpm, cadence >= 250 steps/5m with avg HR >= 95 bpm, or gym exertion (energy >= 15 kcal with HR >= 100 bpm)
        is_active = (
            (slot["energy"] >= Decimal(20))
            or (avg_hr is not None and avg_hr >= 120.0)
            or (slot_step_val >= Decimal(250) and avg_hr is not None and avg_hr >= 95.0)
            or (slot["energy"] >= Decimal(15) and avg_hr is not None and avg_hr >= 100.0)
        )
        if is_active:
            active_indices.append(i)

    if not active_indices:
        return []

    # Merge clusters with max gap of 3 slots (15 minutes) to bridge short rest/water intervals.
    # Rest Gap Splitting: Ensure stationary intervals (>= 10 min with < 50 steps/5min)
    # split sessions into separate blocks instead of bridging across rest gaps.
    # Transit Walk Splitting: Ensure transit walking intervals (>= 250 steps for low-cadence
    # activities like swimming, or >= 500 steps fleet-wide) split sessions rather than merging.
    clusters: list[tuple[int, int]] = []
    curr = [active_indices[0]]
    for idx in active_indices[1:]:
        gap_has_stationary_rest = False
        consecutive_stationary = 0
        gap_has_transit_walk = False

        curr_total_steps = sum(
            deduplicate_step_samples(slots[c]["step_samples"]) if slots[c]["step_samples"] else Decimal(0)
            for c in curr
        )
        curr_cadence = float(curr_total_steps) / (len(curr) * 5) if curr else 0.0

        for g_idx in range(curr[-1] + 1, idx):
            g_steps = (
                deduplicate_step_samples(slots[g_idx]["step_samples"])
                if slots[g_idx]["step_samples"]
                else Decimal(0)
            )
            if g_steps < Decimal(50):
                consecutive_stationary += 1
                if consecutive_stationary >= 2:  # 2 slots * 5 min = 10 min
                    gap_has_stationary_rest = True
                    break
            else:
                consecutive_stationary = 0

            # Transit walk detection: if non-walking activity (cadence < 25) and gap steps >= 250,
            # or any activity and gap steps >= 500 (e.g. outdoor transit walk between sessions)
            if (curr_cadence < 25.0 and g_steps >= Decimal(250)) or (g_steps >= Decimal(500)):
                gap_has_transit_walk = True
                break

        if (idx - curr[-1] <= 3) and not gap_has_stationary_rest and not gap_has_transit_walk:
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
        slot_hrs: list[float | None] = []
        slot_steps: list[Decimal] = []
        slot_energies: list[Decimal] = []
        for i in range(start_idx, end_idx + 1):
            all_hrs.extend(slots[i]["hr_vals"])
            s_hr = (
                (sum(slots[i]["hr_vals"]) / len(slots[i]["hr_vals"]))
                if slots[i]["hr_vals"]
                else None
            )
            slot_hrs.append(s_hr)
            s_steps = (
                deduplicate_step_samples(slots[i]["step_samples"])
                if slots[i]["step_samples"]
                else Decimal(0)
            )
            slot_steps.append(s_steps)
            slot_energies.append(slots[i]["energy"])

        avg_hr = int(round(sum(all_hrs) / len(all_hrs))) if all_hrs else None
        peak_hr = int(round(max(all_hrs))) if all_hrs else None

        s_time = start_dt + timedelta(minutes=start_idx * 5)
        e_time = start_dt + timedelta(minutes=(end_idx + 1) * 5)
        # Centralized multi-device step deduplication for the workout window
        cluster_steps = [
            s for s in samples
            if s.sample_type in ("step_count", "steps")
            and s_time <= (s.measured_at.astimezone(settings.tz) if s.measured_at.tzinfo else s.measured_at.replace(tzinfo=settings.tz)) < e_time
        ]
        block_steps = deduplicate_step_samples(cluster_steps)

        # Filter noise:
        # 1. High-intensity workout: >= 30 mins AND (energy >= 250 kcal OR avg HR >= 120)
        # 2. Sustained walking: >= 25 mins AND block_steps >= 1500 AND avg HR >= 90
        # 3. Gym / Resistance: >= 30 mins AND block_energy >= 120 kcal AND (hr_spread >= 20 OR peak_hr >= 115)
        # 4. Swimming / Aquatic: >= 20 mins AND block_energy >= 100 kcal AND (avg_hr is not None and avg_hr >= 110) AND cadence < 25
        hr_spread = (max(all_hrs) - min(all_hrs)) if all_hrs else 0
        cadence = float(block_steps) / duration_mins if duration_mins > 0 else 0.0

        is_high_intensity = duration_mins >= 30 and (block_energy >= Decimal(250) or (avg_hr is not None and avg_hr >= 120))
        is_sustained_walk = duration_mins >= 25 and (block_steps >= Decimal(1500)) and (avg_hr is None or avg_hr >= 90)
        is_resistance = duration_mins >= 30 and block_energy >= Decimal(120) and (hr_spread >= 20 or (peak_hr is not None and peak_hr >= 115))
        is_swim = duration_mins >= 20 and block_energy >= Decimal(100) and (avg_hr is not None and avg_hr >= 110) and cadence < 25.0

        if is_high_intensity or is_sustained_walk or is_resistance or is_swim:
            conf = "high" if (avg_hr and avg_hr >= 130 and block_energy >= Decimal(350)) else "medium"
            pattern = classify_activity_pattern(
                duration_mins=duration_mins,
                block_energy=block_energy,
                avg_hr=avg_hr,
                peak_hr=peak_hr,
                all_hrs=all_hrs,
                block_steps=block_steps,
                slot_steps=slot_steps,
                slot_hrs=slot_hrs,
                slot_energies=slot_energies,
            )
            exertion = compute_workout_exertion_metrics(
                all_hrs=all_hrs,
                duration_mins=duration_mins,
                block_energy=block_energy,
            )

            # Post-workout heart rate recovery (drop 60-90s post-exercise)
            hr_recovery_60s = None
            if all_hrs:
                post_e_time = e_time + timedelta(seconds=90)
                post_hrs = [
                    float(s.value)
                    for s in samples
                    if s.sample_type in ("heart_rate", "resting_hr")
                    and e_time
                    <= (
                        s.measured_at.astimezone(settings.tz)
                        if s.measured_at.tzinfo
                        else s.measured_at.replace(tzinfo=settings.tz)
                    )
                    <= post_e_time
                ]
                if post_hrs:
                    drop = all_hrs[-1] - min(post_hrs)
                    if drop > 0:
                        hr_recovery_60s = int(round(drop))

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
                    classification=pattern["classification"],
                    activity_type=pattern["activity_type"],
                    routine_name=pattern["routine_name"],
                    effort_rating=exertion["effort_rating"],
                    rpe=exertion["rpe"],
                    trimp=exertion["trimp"],
                    hr_zones=exertion["hr_zones"],
                    hr_recovery_60s=hr_recovery_60s,
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
        if r.sleep_rem_s is not None:
            summary.sleep_rem_hours = round(r.sleep_rem_s / 3600.0, 2)
        if r.sleep_score is not None:
            summary.sleep_score = r.sleep_score
        if r.step_count is not None:
            summary.step_count = (summary.step_count or 0) + r.step_count
        if r.active_energy_kcal is not None:
            summary.active_energy_kcal = (summary.active_energy_kcal or 0) + r.active_energy_kcal
        if r.extra:
            if "wrist_temperature_c" in r.extra:
                summary.wrist_temperature_c = r.extra["wrist_temperature_c"]
            if "respiratory_rate_bpm" in r.extra:
                summary.respiratory_rate_bpm = r.extra["respiratory_rate_bpm"]
            if "oxygen_saturation_pct" in r.extra:
                summary.oxygen_saturation_pct = r.extra["oxygen_saturation_pct"]
            if "walking_asymmetry_pct" in r.extra:
                summary.walking_asymmetry_pct = r.extra["walking_asymmetry_pct"]
            if "vo2_max" in r.extra:
                summary.vo2_max = r.extra["vo2_max"]
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

    # Granular sample aggregations for new health metrics if missing
    if summary.wrist_temperature_c is None:
        wt = db.scalar(
            select(func.avg(HealthSample.value)).where(
                HealthSample.deleted_at.is_(None),
                HealthSample.sample_type.in_(["wrist_temperature", "wrist_temp", "wrist_temperature_c"]),
                HealthSample.measured_at >= start_dt,
                HealthSample.measured_at <= end_dt,
            )
        )
        if wt is not None:
            summary.wrist_temperature_c = round(float(wt), 2)

    if summary.respiratory_rate_bpm is None:
        rr = db.scalar(
            select(func.avg(HealthSample.value)).where(
                HealthSample.deleted_at.is_(None),
                HealthSample.sample_type.in_(["respiratory_rate", "sleeping_respiratory_rate", "respiratory_rate_bpm"]),
                HealthSample.measured_at >= start_dt,
                HealthSample.measured_at <= end_dt,
            )
        )
        if rr is not None:
            summary.respiratory_rate_bpm = round(float(rr), 1)

    if summary.oxygen_saturation_pct is None:
        ox = db.scalar(
            select(func.avg(HealthSample.value)).where(
                HealthSample.deleted_at.is_(None),
                HealthSample.sample_type.in_(["oxygen_saturation", "spo2", "blood_oxygen"]),
                HealthSample.measured_at >= start_dt,
                HealthSample.measured_at <= end_dt,
            )
        )
        if ox is not None:
            summary.oxygen_saturation_pct = round(float(ox), 1)

    if summary.walking_asymmetry_pct is None:
        wa = db.scalar(
            select(func.avg(HealthSample.value)).where(
                HealthSample.deleted_at.is_(None),
                HealthSample.sample_type.in_([
                    "walking_asymmetry",
                    "walking_ asymmetry",
                    "walking asymmetry",
                    "walking_asymmetry_percentage",
                ]),
                HealthSample.measured_at >= start_dt,
                HealthSample.measured_at <= end_dt,
            )
        )
        if wa is not None:
            summary.walking_asymmetry_pct = round(float(wa), 2)

    if summary.vo2_max is None:
        vm = db.scalar(
            select(HealthSample.value)
            .where(
                HealthSample.deleted_at.is_(None),
                HealthSample.sample_type.in_(["vo2_max", "cardio_fitness", "vo2max"]),
                HealthSample.measured_at >= start_dt,
                HealthSample.measured_at <= end_dt,
            )
            .order_by(HealthSample.measured_at.desc())
            .limit(1)
        )
        if vm is not None:
            summary.vo2_max = round(float(vm), 1)

    if summary.sleep_rem_hours is None:
        sleep_samples_for_day = list(
            db.scalars(
                select(HealthSample).where(
                    HealthSample.deleted_at.is_(None),
                    HealthSample.sample_type.in_(["sleep", "sleep_duration", "sleep_duration_s"]),
                    HealthSample.measured_at >= start_dt,
                    HealthSample.measured_at <= end_dt,
                )
            )
        )
        rem_total = sum(
            (float(s.value) for s in sleep_samples_for_day if str((s.extra or {}).get("stage") or "").lower() == "rem"),
            0.0,
        )
        if rem_total > 0:
            summary.sleep_rem_hours = round(rem_total / 3600.0, 2)

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
