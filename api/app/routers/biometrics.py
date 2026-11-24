"""Biometrics router for reading and managing health time-series."""

from datetime import date, datetime, time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.exertion import ExertionAnalyzer, InvalidWindowError
from app.models import BiometricMeasurement, HealthFeedback, HealthSample
from app.schemas import (
    BiometricCreate,
    BiometricRead,
    DailyBiometricSummary,
    DetectedWorkoutRead,
    HealthSampleRead,
    WorkoutTimelineRead,
)
from app.services.biometrics import deduplicate_step_samples

router = APIRouter(prefix="/biometrics", tags=["biometrics"])


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
        # Fall back to latest known VO2 max measurement on or before target date
        vm = db.scalar(
            select(HealthSample.value)
            .where(
                HealthSample.deleted_at.is_(None),
                HealthSample.sample_type.in_(["vo2_max", "cardio_fitness", "vo2max"]),
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
    summary.detected_workouts = ExertionAnalyzer(db).detect_day(target_date)
    if summary.detected_workouts:
        detected_energy_total = sum(w.active_energy_kcal for w in summary.detected_workouts)
        if summary.active_energy_kcal is None or detected_energy_total > summary.active_energy_kcal:
            summary.active_energy_kcal = detected_energy_total
    return summary


@router.get("/detected-workouts", response_model=list[DetectedWorkoutRead])
def get_detected_workouts(
    target_date: date = Query(description="Calendar date (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
) -> list[DetectedWorkoutRead]:
    """Scans high-resolution biometrics to detect unlogged physical exertion blocks."""
    return ExertionAnalyzer(db).detect_day(target_date)


@router.get("/workout-metrics", response_model=DetectedWorkoutRead)
def get_workout_metrics(
    started_at: datetime = Query(description="Window start ISO-8601 timestamp"),
    ended_at: datetime = Query(description="Window end ISO-8601 timestamp"),
    classification: str | None = Query(default=None, description="Optional activity classification override"),
    activity_type: str | None = Query(default=None, description="Optional activity_type override"),
    routine_name: str | None = Query(default=None, description="Optional routine_name override"),
    db: Session = Depends(get_db),
) -> DetectedWorkoutRead:
    """Computes full exertion metrics (HR zones, TRIMP, RPE, steps, recovery) for an arbitrary window."""
    try:
        return ExertionAnalyzer(db).analyze_window(
            started_at,
            ended_at,
            classification=classification,
            activity_type=activity_type,
            routine_name=routine_name,
        )
    except InvalidWindowError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/workout-timeline", response_model=WorkoutTimelineRead)
def get_workout_timeline(
    started_at: datetime = Query(description="Window start ISO-8601 timestamp"),
    ended_at: datetime = Query(description="Window end ISO-8601 timestamp"),
    db: Session = Depends(get_db),
) -> WorkoutTimelineRead:
    """Returns 5-minute biometric bins and Watch mode transition split suggestions for a window."""
    try:
        return ExertionAnalyzer(db).build_timeline(started_at, ended_at)
    except InvalidWindowError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


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
