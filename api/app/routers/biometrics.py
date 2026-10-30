"""Biometrics router for reading and managing health time-series."""

from datetime import date, datetime, time, timezone
from decimal import Decimal
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import BiometricMeasurement, HealthSample
from app.schemas import BiometricCreate, BiometricRead, DailyBiometricSummary, HealthSampleRead

router = APIRouter(prefix="/biometrics", tags=["biometrics"])


@router.get("/granular", response_model=list[HealthSampleRead])
def get_granular_samples(
    target_date: date = Query(description="Calendar date (YYYY-MM-DD)"),
    sample_type: str = Query(description="Sample type, e.g. 'heart_rate', 'sleep', 'active_energy'"),
    limit: int = Query(default=2000, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> list[HealthSample]:
    """Retrieves high-resolution time-series samples for a specific date and metric."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=timezone.utc)

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
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=timezone.utc)

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

    granular_steps = db.scalar(
        select(func.sum(HealthSample.value)).where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type == "step_count",
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
    )
    if granular_steps is not None:
        summary.step_count = int(round(granular_steps))

    summary.readiness_notes = readiness_notes
    return summary


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
