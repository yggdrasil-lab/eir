"""Biometric processing, multi-device deduplication, and exertion inference engines."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

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
    slot_energies: Sequence[Decimal] | None = None,
) -> dict[str, str]:
    """Classifies an exertion window into a sport/workout activity pattern.

    Patterns:
    - Swim: continuous aquatic propulsion, near-zero cadence (<20 steps/min), sustained Zone 2+ HR (>=115 bpm), and steady high burn rate (>=3.0 kcal/min).
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
    # followed by lower cadence (< 35 steps/min) for the remainder WITHOUT repeated high-cadence bursts.
    has_warmup_jog = False
    if len(slot_steps) >= 3:
        has_post_warmup_bursts_2 = any(s >= Decimal(250) for s in slot_steps[2:])
        has_post_warmup_bursts_1 = any(s >= Decimal(250) for s in slot_steps[1:])

        # 2-slot warmup (10 mins)
        warmup_slots_2 = slot_steps[:2]
        post_warmup_slots_2 = slot_steps[2:]
        warmup_cadence_2 = (
            float(sum(warmup_slots_2)) / (len(warmup_slots_2) * 5) if warmup_slots_2 else 0.0
        )
        post_cadence_2 = (
            float(sum(post_warmup_slots_2)) / (len(post_warmup_slots_2) * 5) if post_warmup_slots_2 else 0.0
        )
        # 1-slot warmup (5 mins)
        warmup_slots_1 = slot_steps[:1]
        post_warmup_slots_1 = slot_steps[1:]
        warmup_cadence_1 = (
            float(sum(warmup_slots_1)) / (len(warmup_slots_1) * 5) if warmup_slots_1 else 0.0
        )
        post_cadence_1 = (
            float(sum(post_warmup_slots_1)) / (len(post_warmup_slots_1) * 5) if post_warmup_slots_1 else 0.0
        )
        if (warmup_cadence_2 >= 50.0 and post_cadence_2 < 35.0 and not has_post_warmup_bursts_2) or (
            warmup_cadence_1 >= 50.0 and post_cadence_1 < 35.0 and not has_post_warmup_bursts_1
        ):
            has_warmup_jog = True

    # Count court burst slots (slots with >= 250 steps, meaning >= 50 steps/min of active movement)
    burst_slots = sum(1 for s in slot_steps if s >= Decimal(250))

    # 1. Swimming / Aquatic:
    # - very low overall cadence (< 20 steps/min) and no warmup jog
    # - sustained aerobic heart rate (avg_hr >= 115 bpm or peak_hr >= 130 bpm)
    # - sustained non-stepping exertion: >= 50% of duration spent in minimal-step (< 25 steps/5m)
    #   exertion slots with at least 10 consecutive minutes (>= 2 slots)
    # - steady high burn rate (>= 3.0 kcal/min)
    burn_rate = float(block_energy) / duration_mins if duration_mins > 0 else 0.0
    minimal_step_exertion_slots = 0
    consecutive_minimal_step = 0
    max_consecutive_minimal_step = 0
    for i, s_step in enumerate(slot_steps):
        s_energy = slot_energies[i] if (slot_energies and i < len(slot_energies)) else Decimal(0)
        s_hr = slot_hrs[i] if i < len(slot_hrs) else None
        is_slot_exertion = (s_energy >= Decimal(15)) or (s_hr is not None and s_hr >= 110.0)
        if s_step < Decimal(25) and is_slot_exertion:
            minimal_step_exertion_slots += 1
            consecutive_minimal_step += 1
            if consecutive_minimal_step > max_consecutive_minimal_step:
                max_consecutive_minimal_step = consecutive_minimal_step
        else:
            consecutive_minimal_step = 0

    minimal_step_ratio = (minimal_step_exertion_slots / len(slot_steps)) if slot_steps else 0.0
    is_swim = (
        not has_warmup_jog
        and cadence < 20.0
        and (avg_hr is not None and avg_hr >= 115)
        and max_consecutive_minimal_step >= 2
        and minimal_step_ratio >= 0.50
        and burn_rate >= 3.0
    )

    # 2. Walk:
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

    # 3. Badminton / Court Sports:
    # - elevated HR (Zone 3-5, avg_hr >= 125 or peak_hr >= 140)
    # - sustained active court movement:
    #   * cadence >= 40 steps/min, OR
    #   * cadence >= 28 steps/min with burst slots (>= 2 slots with >= 250 steps), OR
    #   * cadence >= 32 steps/min with rest break intervals (game-rest-game pattern)
    # - excludes gym workouts with warmup jog and swimming
    is_badminton = (
        not has_warmup_jog
        and not is_swim
        and (
            cadence >= 40.0
            or (cadence >= 28.0 and burst_slots >= 2)
            or (cadence >= 32.0 and has_rest_break_interval)
        )
        and ((avg_hr is not None and avg_hr >= 125) or (peak_hr is not None and peak_hr >= 140))
    )

    # 4. Gym / Resistance:
    # - cyclic HR spikes alternating with rest intervals
    # - lower step count (< 50 steps/min) or warmup jog segment followed by resistance sets
    # - excludes court sports (badminton) and swimming
    is_gym = (
        not is_swim
        and not is_badminton
        and (cadence < 50.0 or has_warmup_jog)
        and (hr_range >= 20.0 or has_warmup_jog or cadence < 30.0)
    )

    if is_swim:
        return {
            "classification": "swim",
            "activity_type": "swim",
            "routine_name": "Swimming",
        }
    elif is_walk:
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


def estimate_workout_active_calories(
    duration_mins: int,
    avg_hr: int | None,
    classification: str | None,
    weight_kg: float = 85.0,
) -> int:
    """Estimates realistic net active calories for a workout when raw samples are depressed.

    Triggers when Apple Watch background passive monitoring mode causes severe
    undercounting (< 3.5 kcal/min despite elevated heart rate >= 115 bpm).
    """
    if not avg_hr or duration_mins <= 0:
        return 0

    c = (classification or "").lower()
    if c in ("badminton", "sport"):
        if avg_hr >= 160:
            met = 9.0
        elif avg_hr >= 145:
            met = 8.0
        else:
            met = 7.0
    elif c in ("swim", "swimming"):
        met = 7.5 if avg_hr >= 130 else 6.0
    elif c in ("gym", "resistance"):
        met = 5.5 if avg_hr >= 125 else 4.5
    elif c in ("walk", "walking"):
        met = 4.2 if avg_hr >= 105 else 3.5
    else:
        if avg_hr >= 150:
            met = 8.0
        elif avg_hr >= 130:
            met = 6.5
        else:
            met = 5.0

    # Net active kcal = (MET - 1.0) * weight_kg * (duration_mins / 60)
    net_active = (met - 1.0) * weight_kg * (duration_mins / 60.0)
    return int(round(net_active))


def compute_workout_exertion_metrics(
    all_hrs: Sequence[float],
    duration_mins: int,
    block_energy: Decimal,
    max_hr: float = 186.0,
    rest_hr: float = 63.0,
) -> dict[str, Any]:
    """Computes heart rate zones, Edwards TRIMP score, and inferred RPE/effort rating.

    HR Zones (% of max HR):
    - Zone 1 (< 60% max HR, < 112 bpm): Active Recovery
    - Zone 2 (60% - 69% max HR, 112 - 129 bpm): Aerobic Base
    - Zone 3 (70% - 79% max HR, 130 - 148 bpm): Tempo / Aerobic Endurance
    - Zone 4 (80% - 89% max HR, 149 - 166 bpm): Anaerobic Threshold
    - Zone 5 (>= 90% max HR, >= 167 bpm): Anaerobic Capacity / Neuromuscular
    """
    if not all_hrs:
        return {
            "effort_rating": "optimal",
            "rpe": 6.0,
            "trimp": 0.0,
            "hr_zones": None,
        }

    total_pts = len(all_hrs)
    avg_hr = sum(all_hrs) / total_pts
    peak_hr = max(all_hrs)

    z1_count = sum(1 for h in all_hrs if h < 112.0)
    z2_count = sum(1 for h in all_hrs if 112.0 <= h < 130.0)
    z3_count = sum(1 for h in all_hrs if 130.0 <= h < 149.0)
    z4_count = sum(1 for h in all_hrs if 149.0 <= h < 167.0)
    z5_count = sum(1 for h in all_hrs if h >= 167.0)

    pct_z1 = round((z1_count / total_pts) * 100.0, 1)
    pct_z2 = round((z2_count / total_pts) * 100.0, 1)
    pct_z3 = round((z3_count / total_pts) * 100.0, 1)
    pct_z4 = round((z4_count / total_pts) * 100.0, 1)
    pct_z5 = round((z5_count / total_pts) * 100.0, 1)

    mins_z1 = round((z1_count / total_pts) * duration_mins, 1)
    mins_z2 = round((z2_count / total_pts) * duration_mins, 1)
    mins_z3 = round((z3_count / total_pts) * duration_mins, 1)
    mins_z4 = round((z4_count / total_pts) * duration_mins, 1)
    mins_z5 = round((z5_count / total_pts) * duration_mins, 1)

    trimp = round(
        mins_z1 * 1.0
        + mins_z2 * 2.0
        + mins_z3 * 3.0
        + mins_z4 * 4.0
        + mins_z5 * 5.0,
        1,
    )

    high_intensity_pct = pct_z4 + pct_z5
    if (pct_z5 >= 50.0 and avg_hr >= 160.0) or (
        pct_z5 >= 40.0 and duration_mins >= 50 and avg_hr >= 160.0
    ):
        effort_rating = "too_hard"
        rpe = round(min(10.0, 9.0 + (avg_hr - 160.0) * 0.1), 1)
    elif pct_z5 >= 35.0 or high_intensity_pct >= 60.0 or avg_hr >= 150.0:
        effort_rating = "hard"
        rpe = round(min(8.9, 8.0 + (pct_z5 / 100.0) * 1.5), 1)
    elif high_intensity_pct >= 15.0 or pct_z3 >= 30.0 or avg_hr >= 120.0:
        effort_rating = "optimal"
        rpe = round(min(7.9, 5.0 + (pct_z3 + pct_z4) / 100.0 * 2.5), 1)
    else:
        effort_rating = "too_easy"
        rpe = round(max(1.0, 2.0 + (avg_hr - 60.0) * 0.05), 1)

    hr_zones = {
        "z1_recovery_pct": pct_z1,
        "z2_aerobic_pct": pct_z2,
        "z3_tempo_pct": pct_z3,
        "z4_threshold_pct": pct_z4,
        "z5_anaerobic_pct": pct_z5,
        "z1_mins": mins_z1,
        "z2_mins": mins_z2,
        "z3_mins": mins_z3,
        "z4_mins": mins_z4,
        "z5_mins": mins_z5,
    }

    return {
        "effort_rating": effort_rating,
        "rpe": rpe,
        "trimp": trimp,
        "hr_zones": hr_zones,
    }
