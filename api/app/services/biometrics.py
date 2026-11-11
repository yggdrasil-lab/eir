"""Biometric processing, multi-device deduplication, and exertion inference engines."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.models import HealthSample


def _normalize_dt(dt_val: datetime) -> datetime:
    """Ensures datetime comparisons are consistently timezone-aware."""
    return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)


def deduplicate_step_samples(samples: Sequence[HealthSample]) -> Decimal:
    """Calculates true total steps across multi-device Apple Health step samples.

    When both Apple Watch and iPhone record steps during the same interval, Apple Watch
    takes priority. Non-overlapping iPhone intervals (e.g. phone carried while watch is charging)
    are retained. Third-party/unlabeled pedometer samples are added cleanly.
    """
    if not samples:
        return Decimal(0)

    watch_step_samples: list[HealthSample] = []
    phone_step_samples: list[HealthSample] = []
    other_step_samples: list[HealthSample] = []

    for s in samples:
        dev = str((s.extra or {}).get("device") or "").lower()
        if "watch" in dev:
            watch_step_samples.append(s)
        elif "phone" in dev:
            phone_step_samples.append(s)
        else:
            other_step_samples.append(s)

    if watch_step_samples and phone_step_samples:
        total_steps = sum((s.value for s in watch_step_samples), Decimal(0))
        for p in phone_step_samples:
            p_start = _normalize_dt(p.measured_at)
            p_end = (
                _normalize_dt(p.ended_at)
                if (p.ended_at and p.ended_at > p.measured_at)
                else p_start + timedelta(seconds=300)
            )
            overlaps = any(
                max(p_start, _normalize_dt(w.measured_at))
                <= min(
                    p_end,
                    _normalize_dt(w.ended_at)
                    if (w.ended_at and w.ended_at > w.measured_at)
                    else _normalize_dt(w.measured_at) + timedelta(seconds=300),
                )
                for w in watch_step_samples
            )
            if not overlaps:
                total_steps += p.value
        total_steps += sum((s.value for s in other_step_samples), Decimal(0))
        return total_steps
    elif watch_step_samples:
        return sum((s.value for s in watch_step_samples), Decimal(0)) + sum(
            (s.value for s in other_step_samples), Decimal(0)
        )
    else:
        return sum((s.value for s in samples), Decimal(0))


def classify_activity_pattern(
    duration_mins: int,
    block_energy: Decimal,
    avg_hr: int | None,
    peak_hr: int | None,
    all_hrs: Sequence[float],
    block_steps: Decimal,
    slot_steps: Sequence[Decimal],
    slot_hrs: Sequence[float | None],
) -> dict[str, str]:
    """Classifies an exertion window into a sport/workout activity pattern.

    Patterns:
    - Walk: continuous cadence (>50 steps/min), flat Zone 1/2 HR (95-115 bpm), steady burn.
    - Badminton / Court Sports: elevated HR (Zone 3-5), bursty steps, rest break intervals (>=10 min stationary gap).
    - Gym / Resistance: cyclic HR spikes alternating with rest intervals, lower step count, optional warmup jog segment.
    """
    cadence = float(block_steps) / duration_mins if duration_mins > 0 else 0.0

    # Check stationary gaps (slots with < 50 steps, meaning < 10 steps/min)
    consecutive_stationary = 0
    max_stationary_mins = 0
    for s in slot_steps:
        if s < Decimal(50):
            consecutive_stationary += 5
            if consecutive_stationary > max_stationary_mins:
                max_stationary_mins = consecutive_stationary
        else:
            consecutive_stationary = 0
    has_rest_break_interval = max_stationary_mins >= 10

    hr_range = 0.0
    if all_hrs and len(all_hrs) > 1:
        hr_range = float(max(all_hrs) - min(all_hrs))
    elif peak_hr is not None and avg_hr is not None:
        hr_range = float(peak_hr - avg_hr)

    # Check warmup jog segment: first 1 or 2 slots (5-10 min) have elevated cadence (>= 50 steps/min),
    # followed by lower cadence (< 35 steps/min) for the remainder.
    has_warmup_jog = False
    if len(slot_steps) >= 3:
        warmup_slots = slot_steps[:2]
        post_warmup_slots = slot_steps[2:]
        warmup_cadence = (
            float(sum(warmup_slots)) / (len(warmup_slots) * 5) if warmup_slots else 0.0
        )
        post_cadence = (
            float(sum(post_warmup_slots)) / (len(post_warmup_slots) * 5) if post_warmup_slots else 0.0
        )
        if warmup_cadence >= 50.0 and post_cadence < 35.0:
            has_warmup_jog = True

    # 1. Walk:
    # - continuous cadence (>50 steps/min)
    # - flat Zone 1/2 HR (95-115 bpm, allow up to 120 bpm)
    # - steady burn (no long rest breaks)
    is_walk = (
        cadence > 50.0
        and not has_warmup_jog
        and (avg_hr is None or avg_hr <= 120)
        and (peak_hr is None or peak_hr <= 135)
        and not has_rest_break_interval
    )

    # 2. Badminton / Court Sports:
    # - elevated HR (Zone 3-5, avg_hr >= 125 or peak_hr >= 140)
    # - bursty steps
    # - rest break intervals (>=10 min stationary gap) OR active court sport intensity
    is_badminton = (
        (avg_hr is not None and avg_hr >= 125) or (peak_hr is not None and peak_hr >= 140)
    ) and (cadence >= 30.0 or has_rest_break_interval)

    # 3. Gym / Resistance:
    # - cyclic HR spikes alternating with rest intervals
    # - lower step count (< 50 steps/min)
    # - optional warmup jog segment
    is_gym = (cadence < 50.0 or has_warmup_jog) and (
        hr_range >= 20.0 or has_warmup_jog or cadence < 30.0
    )

    if is_walk:
        return {
            "classification": "walk",
            "activity_type": "walk",
            "routine_name": "Walk / Cardio",
        }
    elif is_badminton:
        return {
            "classification": "badminton",
            "activity_type": "badminton",
            "routine_name": "Badminton Match Play",
        }
    elif is_gym:
        return {
            "classification": "gym",
            "activity_type": "gym",
            "routine_name": "Resistance Training",
        }
    else:
        if cadence > 50.0:
            return {
                "classification": "walk",
                "activity_type": "walk",
                "routine_name": "Walk / Cardio",
            }
        return {
            "classification": "gym",
            "activity_type": "gym",
            "routine_name": "Resistance Training",
        }
