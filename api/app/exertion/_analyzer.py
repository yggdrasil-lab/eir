"""Unified ExertionAnalyzer entrypoint coordinating workout detection, scoring, and timelines."""

from datetime import date, datetime, time, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session

from app.exertion._classification import classify_activity_pattern
from app.exertion._clustering import cluster_active_slots, passes_cluster_noise_filter
from app.exertion._queries import fetch_exertion_samples, fetch_latest_weight_kg
from app.exertion._scoring import (
    compute_hr_recovery_60s,
    compute_workout_exertion_metrics,
    estimate_workout_active_calories,
)
from app.exertion._slots import (
    bucket_samples_into_5min_slots,
    normalize_window_bounds,
    sample_local_dt,
)
from app.exertion._timeline import build_workout_timeline
from app.exertion._types import ActivityPattern, ExertionProfile, SampleSlot
from app.models import HealthSample
from app.schemas import DetectedWorkoutRead, WorkoutTimelineRead


class ExertionAnalyzer:
    """Cohesive seam for daily workout detection, window exertion metrics, and timelines."""

    def __init__(self, db: Session, profile: ExertionProfile | None = None) -> None:
        self._db = db
        self._profile = profile if profile is not None else ExertionProfile.from_settings()

    def detect_day(self, target_date: date) -> list[DetectedWorkoutRead]:
        """Infers sustained physical exertion blocks for a single calendar date."""
        start_dt = datetime.combine(target_date, time.min).replace(tzinfo=self._profile.tz)
        end_dt = datetime.combine(target_date, time.max).replace(tzinfo=self._profile.tz)
        samples = fetch_exertion_samples(self._db, start_dt, end_dt)
        if not samples:
            return []

        bucketed = bucket_samples_into_5min_slots(
            samples, start_dt, end_dt, self._profile.tz, fixed_slot_count=288
        )
        detected: list[DetectedWorkoutRead] = []
        for start_idx, end_idx in cluster_active_slots(bucketed.slots):
            workout = self._build_cluster_workout(
                start_dt=start_dt,
                end_dt=end_dt,
                samples=samples,
                slots=bucketed.slots,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            if workout is not None:
                detected.append(workout)
        return detected

    def analyze_window(
        self,
        started_at: datetime,
        ended_at: datetime,
        *,
        classification: str | None = None,
        activity_type: str | None = None,
        routine_name: str | None = None,
    ) -> DetectedWorkoutRead:
        """Computes full workout exertion metrics for an arbitrary [started_at, ended_at] window."""
        s_time, e_time = normalize_window_bounds(started_at, ended_at, self._profile.tz)
        samples = fetch_exertion_samples(self._db, s_time, e_time + timedelta(seconds=90))
        bucketed = bucket_samples_into_5min_slots(samples, s_time, e_time, self._profile.tz)

        duration_mins = max(1, int(round((e_time - s_time).total_seconds() / 60)))
        block_energy = sum((slot.energy for slot in bucketed.slots), Decimal(0))
        block_steps = sum((slot.steps for slot in bucketed.slots), Decimal(0))
        all_hrs = [
            float(s.value)
            for s in samples
            if s.sample_type in ("heart_rate", "resting_hr")
            and s_time <= sample_local_dt(s, self._profile.tz) <= e_time
        ]
        avg_hr = int(round(sum(all_hrs) / len(all_hrs))) if all_hrs else None
        peak_hr = int(round(max(all_hrs))) if all_hrs else None
        pattern = self._resolve_window_pattern(
            slots=bucketed.slots,
            duration_mins=duration_mins,
            block_energy=block_energy,
            avg_hr=avg_hr,
            peak_hr=peak_hr,
            all_hrs=all_hrs,
            block_steps=block_steps,
            classification=classification,
            activity_type=activity_type,
            routine_name=routine_name,
        )
        return self._build_detected_workout(
            s_time=s_time,
            e_time=e_time,
            end_dt=e_time,
            duration_mins=duration_mins,
            block_energy=block_energy,
            all_hrs=all_hrs,
            block_steps=block_steps,
            samples=samples,
            pattern=pattern,
            confidence="high",
        )

    def build_timeline(self, started_at: datetime, ended_at: datetime) -> WorkoutTimelineRead:
        """Builds 5-minute biometric bins and passive<->active Watch split hints for a window."""
        s_time, e_time = normalize_window_bounds(started_at, ended_at, self._profile.tz)
        samples = fetch_exertion_samples(self._db, s_time, e_time)
        bucketed = bucket_samples_into_5min_slots(samples, s_time, e_time, self._profile.tz)
        return build_workout_timeline(bucketed.slots, s_time, e_time)

    def _resolve_window_pattern(
        self,
        *,
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
    ) -> ActivityPattern:
        if classification or activity_type or routine_name:
            cls_val = classification or activity_type or "gym"
            return ActivityPattern(
                classification=cls_val,
                activity_type=activity_type or cls_val,
                routine_name=routine_name or "Workout",
            )
        return classify_activity_pattern(
            slots=slots,
            duration_mins=duration_mins,
            block_energy=block_energy,
            avg_hr=avg_hr,
            peak_hr=peak_hr,
            all_hrs=all_hrs,
            block_steps=block_steps,
        )

    def _build_cluster_workout(
        self,
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
        if not passes_cluster_noise_filter(
            duration_mins, block_energy, block_steps, all_hrs, avg_hr, peak_hr
        ):
            return None

        pattern = classify_activity_pattern(
            slots=cluster_slots,
            duration_mins=duration_mins,
            block_energy=block_energy,
            avg_hr=avg_hr,
            peak_hr=peak_hr,
            all_hrs=all_hrs,
            block_steps=block_steps,
        )
        conf = "high" if (avg_hr and avg_hr >= 130 and block_energy >= Decimal(350)) else "medium"
        return self._build_detected_workout(
            s_time=start_dt + timedelta(minutes=start_idx * 5),
            e_time=start_dt + timedelta(minutes=(end_idx + 1) * 5),
            end_dt=end_dt,
            duration_mins=duration_mins,
            block_energy=block_energy,
            all_hrs=all_hrs,
            block_steps=block_steps,
            samples=samples,
            pattern=pattern,
            confidence=conf,
        )

    def _build_detected_workout(
        self,
        *,
        s_time: datetime,
        e_time: datetime,
        end_dt: datetime,
        duration_mins: int,
        block_energy: Decimal,
        all_hrs: list[float],
        block_steps: Decimal,
        samples: list[HealthSample],
        pattern: ActivityPattern,
        confidence: str,
    ) -> DetectedWorkoutRead:
        avg_hr = int(round(sum(all_hrs) / len(all_hrs))) if all_hrs else None
        peak_hr = int(round(max(all_hrs))) if all_hrs else None
        final_energy_kcal = self._resolve_active_energy(
            duration_mins=duration_mins,
            avg_hr=avg_hr,
            block_energy=block_energy,
            end_dt=end_dt,
            classification=pattern.classification,
        )
        exertion = compute_workout_exertion_metrics(
            all_hrs=all_hrs,
            duration_mins=duration_mins,
            max_hr=self._profile.max_hr,
        )
        hr_recovery_60s = compute_hr_recovery_60s(
            samples=samples,
            e_time=e_time,
            all_hrs=all_hrs,
            tz=self._profile.tz,
        )
        return DetectedWorkoutRead(
            started_at=s_time,
            ended_at=e_time,
            duration_minutes=duration_mins,
            active_energy_kcal=final_energy_kcal,
            avg_hr_bpm=avg_hr,
            peak_hr_bpm=peak_hr,
            step_count=int(round(block_steps)) if block_steps > 0 else None,
            confidence=confidence,
            classification=pattern.classification,
            activity_type=pattern.activity_type,
            routine_name=pattern.routine_name,
            effort_rating=exertion.effort_rating,
            rpe=exertion.rpe,
            trimp=exertion.trimp,
            hr_zones=exertion.hr_zones,
            hr_recovery_60s=hr_recovery_60s,
        )

    def _resolve_active_energy(
        self,
        *,
        duration_mins: int,
        avg_hr: int | None,
        block_energy: Decimal,
        end_dt: datetime,
        classification: str,
    ) -> int:
        final_energy_kcal = int(round(block_energy))
        energy_rate = float(block_energy) / duration_mins if duration_mins > 0 else 0.0
        if (avg_hr is not None and avg_hr >= 115) and energy_rate < 3.5:
            latest_weight = fetch_latest_weight_kg(self._db, end_dt)
            estimated = estimate_workout_active_calories(
                duration_mins=duration_mins,
                avg_hr=avg_hr,
                classification=classification,
                weight_kg=latest_weight,
            )
            if estimated > final_energy_kcal:
                return estimated
        return final_energy_kcal
