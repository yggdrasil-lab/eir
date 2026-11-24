"""Biometrics router for reading and managing health time-series."""

from datetime import date, datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.daily_biometrics import BiometricsInputError, DailyBiometrics
from app.db import get_db
from app.exertion import ExertionAnalyzer, InvalidWindowError
from app.models import BiometricMeasurement, HealthSample
from app.schemas import (
    BiometricCreate,
    BiometricRead,
    DailyBiometricSummary,
    DetectedWorkoutRead,
    HealthSampleRead,
    WorkoutTimelineRead,
)

router = APIRouter(prefix="/biometrics", tags=["biometrics"])


@router.get("/granular", response_model=list[HealthSampleRead])
def get_granular_samples(
    target_date: date = Query(description="Calendar date (YYYY-MM-DD)"),
    sample_type: str = Query(description="Sample type, e.g. 'heart_rate', 'sleep', 'active_energy'"),
    limit: int = Query(default=2000, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> list[HealthSample]:
    """Retrieves high-resolution time-series samples for a specific date and metric."""
    try:
        return DailyBiometrics(db).granular_samples(target_date, sample_type, limit=limit)
    except BiometricsInputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/latest", response_model=BiometricRead | None)
def get_latest_biometrics(
    source: str | None = Query(default=None, description="Optional source filter (e.g. 'withings', 'apple_health')"),
    db: Session = Depends(get_db),
) -> BiometricMeasurement | None:
    """Returns the most recent biometric record."""
    return DailyBiometrics(db).latest_measurement(source=source)


@router.get("/daily/{target_date}", response_model=DailyBiometricSummary)
def get_daily_summary(
    target_date: date,
    db: Session = Depends(get_db),
) -> DailyBiometricSummary:
    """Aggregates all biometric readings for a single calendar day."""
    try:
        return DailyBiometrics(db).daily_summary(target_date)
    except BiometricsInputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


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
    try:
        measurement = DailyBiometrics(db).record_measurement(payload)
    except BiometricsInputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
