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
    WorkoutTimelineRead,
)
from app.services.biometrics import (
    SampleSlot,
    bucket_samples_into_5min_slots,
    build_workout_timeline,
    classify_activity_pattern,
    compute_workout_exertion_metrics,
    deduplicate_step_samples,
    estimate_workout_active_calories,
    sample_local_dt,
)

router = APIRouter(prefix="/biometrics", tags=["biometrics"])


def _sample_local_dt(s: HealthSample) -> datetime:
    """Normalizes a HealthSample timestamp to settings.tz."""
    return sample_local_dt(s, settings.tz)


def _normalize_window_bounds(started_at: datetime, ended_at: datetime) -> tuple[datetime, datetime]:
    """Normalizes window timestamps to settings.tz and validates chronological ordering."""
    s_time = started_at.astimezone(settings.tz) if started_at.tzinfo else started_at.replace(tzinfo=settings.tz)
    e_time = ended_at.astimezone(settings.tz) if ended_at.tzinfo else ended_at.replace(tzinfo=settings.tz)
    if e_time <= s_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ended_at must be after started_at",
        )
    return s_time, e_time


def _fetch_exertion_samples(db: Session, start_dt: datetime, end_dt: datetime) -> list[HealthSample]:
    """Fetches active energy, heart rate, and step samples inside [start_dt, end_dt]."""
    return list(
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


def _build_detected_workout(
    db: Session,
    *,
    s_time: datetime,
    e_time: datetime,
    end_dt: datetime,
    duration_mins: int,
    block_energy: Decimal,
    all_hrs: list[float],
    block_steps: Decimal,
    samples: list[HealthSample],
    pattern: dict[str, str],
    confidence: str,
) -> DetectedWorkoutRead:
    """Builds a DetectedWorkoutRead with calorie fallback, exertion metrics, and HR recovery."""
    avg_hr = int(round(sum(all_hrs) / len(all_hrs))) if all_hrs else None
    peak_hr = int(round(max(all_hrs))) if all_hrs else None
    final_energy_kcal = int(round(block_energy))
    energy_rate = float(block_energy) / duration_mins if duration_mins > 0 else 0.0

    # Calorie undercounting fallback: when avg_hr indicates sustained exertion (>= 115 bpm)
    # but raw energy burn rate is suspiciously low (< 3.5 kcal/min), Apple Watch was in
    # passive background mode rather than active workout mode. Fall back to physiological MET estimate.
    if (avg_hr is not None and avg_hr >= 115) and energy_rate < 3.5:
        latest_weight = (
            db.scalar(
                select(HealthSample.value)
                .where(
                    HealthSample.deleted_at.is_(None),
                    HealthSample.sample_type.in_(["weight", "weight_kg"]),
                    HealthSample.measured_at <= end_dt,
                )
                .order_by(HealthSample.measured_at.desc())
                .limit(1)
            )
            or 85.0
        )
        estimated_energy = estimate_workout_active_calories(
            duration_mins=duration_mins,
            avg_hr=avg_hr,
            classification=pattern["classification"],
            weight_kg=float(latest_weight),
        )
        if estimated_energy > final_energy_kcal:
            final_energy_kcal = estimated_energy

    exertion = compute_workout_exertion_metrics(
        all_hrs=all_hrs,
        duration_mins=duration_mins,
        block_energy=Decimal(final_energy_kcal),
    )

    # Post-workout heart rate recovery (drop 60-90s post-exercise)
    hr_recovery_60s = None
    if all_hrs:
        post_e_time = e_time + timedelta(seconds=90)
        post_hrs = [
            float(s.value)
            for s in samples
            if s.sample_type in ("heart_rate", "resting_hr")
            and e_time <= _sample_local_dt(s) <= post_e_time
        ]
        if post_hrs:
            drop = all_hrs[-1] - min(post_hrs)
            if drop > 0:
                hr_recovery_60s = int(round(drop))

    return DetectedWorkoutRead(
        started_at=s_time,
        ended_at=e_time,
        duration_minutes=duration_mins,
        active_energy_kcal=final_energy_kcal,
        avg_hr_bpm=avg_hr,
        peak_hr_bpm=peak_hr,
        step_count=int(round(block_steps)) if block_steps > 0 else None,
        confidence=confidence,
        classification=pattern["classification"],
        activity_type=pattern["activity_type"],
        routine_name=pattern["routine_name"],
        effort_rating=exertion["effort_rating"],
        rpe=exertion["rpe"],
        trimp=exertion["trimp"],
        hr_zones=exertion["hr_zones"],
        hr_recovery_60s=hr_recovery_60s,
    )


def _resolve_window_pattern(
    slots: tuple[SampleSlot, ...],
    duration_mins: int,
    block_energy: Decimal,
    avg_hr: int | None,
    peak_hr: int | None,
    all_hrs: list[float],
    block_steps: Decimal,
    classification: str | None,
    activity_type: str | None,
    routine_name: str | None,
) -> dict[str, str]:
    """Determines workout classification from explicit overrides or 5-minute slot patterns."""
    if classification or activity_type or routine_name:
        cls_val = classification or activity_type or "gym"
        return {
            "classification": cls_val,
            "activity_type": activity_type or cls_val,
            "routine_name": routine_name or "Workout",
        }
    return classify_activity_pattern(
        duration_mins=duration_mins,
        block_energy=block_energy,
        avg_hr=avg_hr,
        peak_hr=peak_hr,
        all_hrs=all_hrs,
        block_steps=block_steps,
        slot_steps=[slot.steps for slot in slots],
        slot_hrs=[(sum(slot.hr_values) / len(slot.hr_values)) if slot.hr_values else None for slot in slots],
        slot_energies=[slot.energy for slot in slots],
    )


def compute_workout_metrics_for_window(
    db: Session,
    started_at: datetime,
    ended_at: datetime,
    classification: str | None = None,
    activity_type: str | None = None,
    routine_name: str | None = None,
) -> DetectedWorkoutRead:
    """Computes full Eir workout exertion metrics (HR zones, TRIMP, RPE, steps, recovery) for any window."""
    s_time, e_time = _normalize_window_bounds(started_at, ended_at)
    samples = _fetch_exertion_samples(db, s_time, e_time + timedelta(seconds=90))
    bucketed = bucket_samples_into_5min_slots(samples, s_time, e_time, settings.tz)

    duration_mins = max(1, int(round((e_time - s_time).total_seconds() / 60)))
    block_energy = sum((slot.energy for slot in bucketed.slots), Decimal(0))
    block_steps = sum((slot.steps for slot in bucketed.slots), Decimal(0))
    all_hrs = [
        float(s.value) for s in samples
        if s.sample_type in ("heart_rate", "resting_hr") and s_time <= _sample_local_dt(s) <= e_time
    ]
    avg_hr = int(round(sum(all_hrs) / len(all_hrs))) if all_hrs else None
    peak_hr = int(round(max(all_hrs))) if all_hrs else None
    pattern = _resolve_window_pattern(
        slots=bucketed.slots, duration_mins=duration_mins, block_energy=block_energy,
        avg_hr=avg_hr, peak_hr=peak_hr, all_hrs=all_hrs, block_steps=block_steps,
        classification=classification, activity_type=activity_type, routine_name=routine_name,
    )
    return _build_detected_workout(
        db, s_time=s_time, e_time=e_time, end_dt=e_time, duration_mins=duration_mins,
        block_energy=block_energy, all_hrs=all_hrs, block_steps=block_steps,
        samples=samples, pattern=pattern, confidence="high",
    )


def compute_workout_timeline(db: Session, started_at: datetime, ended_at: datetime) -> WorkoutTimelineRead:
    """Builds 5-minute biometric bins and passive<->active Watch split hints for a window."""
    s_time, e_time = _normalize_window_bounds(started_at, ended_at)
    samples = _fetch_exertion_samples(db, s_time, e_time)
    bucketed = bucket_samples_into_5min_slots(samples, s_time, e_time, settings.tz)
    return build_workout_timeline(bucketed.slots, s_time, e_time)


def _is_active_slot(slot: SampleSlot) -> bool:
    avg_hr = (sum(slot.hr_values) / len(slot.hr_values)) if slot.hr_values else None
    return (
        (slot.energy >= Decimal(20))
        or (avg_hr is not None and avg_hr >= 120.0)
        or (slot.steps >= Decimal(250) and avg_hr is not None and avg_hr >= 95.0)
        or (slot.energy >= Decimal(15) and avg_hr is not None and avg_hr >= 100.0)
    )


def _gap_splits_cluster(slots: tuple[SampleSlot, ...], curr: list[int], next_idx: int) -> bool:
    """Returns True when the inter-slot gap contains >= 10m stationary rest or transit walking."""
    curr_total_steps = sum((slots[c].steps for c in curr), Decimal(0))
    curr_cadence = float(curr_total_steps) / (len(curr) * 5) if curr else 0.0
    consecutive_stationary = 0
    for g_idx in range(curr[-1] + 1, next_idx):
        g_steps = slots[g_idx].steps
        if g_steps < Decimal(50):
            consecutive_stationary += 1
            if consecutive_stationary >= 2:
                return True
        else:
            consecutive_stationary = 0
        if (curr_cadence < 25.0 and g_steps >= Decimal(250)) or (g_steps >= Decimal(500)):
            return True
    return False


def _cluster_active_slots(slots: tuple[SampleSlot, ...]) -> list[tuple[int, int]]:
    active_indices = [i for i, slot in enumerate(slots) if _is_active_slot(slot)]
    if not active_indices:
        return []
    clusters: list[tuple[int, int]] = []
    curr = [active_indices[0]]
    for idx in active_indices[1:]:
        if (idx - curr[-1] <= 3) and not _gap_splits_cluster(slots, curr, idx):
            curr.append(idx)
            continue
        clusters.append((curr[0], curr[-1]))
        curr = [idx]
    clusters.append((curr[0], curr[-1]))
    return clusters


def _passes_cluster_noise_filter(
    duration_mins: int,
    block_energy: Decimal,
    block_steps: Decimal,
    all_hrs: list[float],
    avg_hr: int | None,
    peak_hr: int | None,
) -> bool:
    hr_spread = (max(all_hrs) - min(all_hrs)) if all_hrs else 0.0
    cadence = float(block_steps) / duration_mins if duration_mins > 0 else 0.0
    is_high_intensity = duration_mins >= 30 and (block_energy >= Decimal(250) or (avg_hr is not None and avg_hr >= 120))
    is_sustained_walk = duration_mins >= 25 and block_steps >= Decimal(1500) and (avg_hr is None or avg_hr >= 90)
    is_resistance = duration_mins >= 30 and block_energy >= Decimal(120) and (hr_spread >= 20 or (peak_hr is not None and peak_hr >= 115))
    is_swim = duration_mins >= 20 and block_energy >= Decimal(100) and (avg_hr is not None and avg_hr >= 110) and cadence < 25.0
    return is_high_intensity or is_sustained_walk or is_resistance or is_swim


def _build_cluster_workout(
    db: Session,
    *,
    start_dt: datetime,
    end_dt: datetime,
    samples: list[HealthSample],
    slots: tuple[SampleSlot, ...],
    start_idx: int,
    end_idx: int,
) -> DetectedWorkoutRead | None:
    cluster_slots = slots[start_idx : end_idx + 1]
    duration_mins = (end_idx - start_idx + 1) * 5
    block_energy = sum((slot.energy for slot in cluster_slots), Decimal(0))
    block_steps = sum((slot.steps for slot in cluster_slots), Decimal(0))
    all_hrs = [hr for slot in cluster_slots for hr in slot.hr_values]
    avg_hr = int(round(sum(all_hrs) / len(all_hrs))) if all_hrs else None
    peak_hr = int(round(max(all_hrs))) if all_hrs else None
    if not _passes_cluster_noise_filter(duration_mins, block_energy, block_steps, all_hrs, avg_hr, peak_hr):
        return None

    pattern = classify_activity_pattern(
        duration_mins=duration_mins, block_energy=block_energy, avg_hr=avg_hr, peak_hr=peak_hr,
        all_hrs=all_hrs, block_steps=block_steps,
        slot_steps=[slot.steps for slot in cluster_slots],
        slot_hrs=[(sum(slot.hr_values) / len(slot.hr_values)) if slot.hr_values else None for slot in cluster_slots],
        slot_energies=[slot.energy for slot in cluster_slots],
    )
    conf = "high" if (avg_hr and avg_hr >= 130 and block_energy >= Decimal(350)) else "medium"
    return _build_detected_workout(
        db,
        s_time=start_dt + timedelta(minutes=start_idx * 5),
        e_time=start_dt + timedelta(minutes=(end_idx + 1) * 5),
        end_dt=end_dt, duration_mins=duration_mins, block_energy=block_energy,
        all_hrs=all_hrs, block_steps=block_steps, samples=samples, pattern=pattern, confidence=conf,
    )


def detect_workouts_for_day(db: Session, target_date: date) -> list[DetectedWorkoutRead]:
    """Infers sustained physical exertion blocks from granular active energy and heart rate samples."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=settings.tz)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=settings.tz)
    samples = _fetch_exertion_samples(db, start_dt, end_dt)
    if not samples:
        return []

    bucketed = bucket_samples_into_5min_slots(samples, start_dt, end_dt, settings.tz, fixed_slot_count=288)
    detected: list[DetectedWorkoutRead] = []
    for start_idx, end_idx in _cluster_active_slots(bucketed.slots):
        workout = _build_cluster_workout(
            db, start_dt=start_dt, end_dt=end_dt, samples=samples,
            slots=bucketed.slots, start_idx=start_idx, end_idx=end_idx,
        )
        if workout is not None:
            detected.append(workout)
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
    summary.detected_workouts = detect_workouts_for_day(db, target_date)
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
    return detect_workouts_for_day(db, target_date)


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
    return compute_workout_metrics_for_window(
        db,
        started_at=started_at,
        ended_at=ended_at,
        classification=classification,
        activity_type=activity_type,
        routine_name=routine_name,
    )


@router.get("/workout-timeline", response_model=WorkoutTimelineRead)
def get_workout_timeline(
    started_at: datetime = Query(description="Window start ISO-8601 timestamp"),
    ended_at: datetime = Query(description="Window end ISO-8601 timestamp"),
    db: Session = Depends(get_db),
) -> WorkoutTimelineRead:
    """Returns 5-minute biometric bins and Watch mode transition split suggestions for a window."""
    return compute_workout_timeline(db, started_at=started_at, ended_at=ended_at)


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
