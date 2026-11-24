"""Sport and workout activity pattern classification."""

from collections.abc import Sequence
from decimal import Decimal

from app.exertion._types import ActivityPattern, SampleSlot

_SWIM_PATTERN = ActivityPattern(classification="swim", activity_type="swim", routine_name="Swimming")
_WALK_PATTERN = ActivityPattern(classification="walk", activity_type="walk", routine_name="Walk / Cardio")
_BADMINTON_PATTERN = ActivityPattern(
    classification="badminton",
    activity_type="badminton",
    routine_name="Badminton Match Play",
)
_GYM_PATTERN = ActivityPattern(
    classification="gym",
    activity_type="gym",
    routine_name="Resistance Training",
)


def _max_consecutive_stationary_mins(slots: Sequence[SampleSlot]) -> int:
    consecutive = 0
    max_mins = 0
    for slot in slots:
        if slot.steps < Decimal(50):
            consecutive += 5
            if consecutive > max_mins:
                max_mins = consecutive
        else:
            consecutive = 0
    return max_mins


def _compute_hr_range(
    all_hrs: Sequence[float],
    avg_hr: int | None,
    peak_hr: int | None,
) -> float:
    if all_hrs and len(all_hrs) > 1:
        return float(max(all_hrs) - min(all_hrs))
    if peak_hr is not None and avg_hr is not None:
        return float(peak_hr - avg_hr)
    return 0.0


def _matches_warmup_split(slots: Sequence[SampleSlot], split_idx: int) -> bool:
    warmup = slots[:split_idx]
    post = slots[split_idx:]
    if not warmup or not post:
        return False
    if any(slot.steps >= Decimal(250) for slot in post):
        return False
    warmup_cadence = float(sum(slot.steps for slot in warmup)) / (len(warmup) * 5)
    post_cadence = float(sum(slot.steps for slot in post)) / (len(post) * 5)
    return warmup_cadence >= 50.0 and post_cadence < 35.0


def _has_warmup_jog(slots: Sequence[SampleSlot]) -> bool:
    if len(slots) < 3:
        return False
    return _matches_warmup_split(slots, 2) or _matches_warmup_split(slots, 1)


def _is_swim_pattern(
    slots: Sequence[SampleSlot],
    duration_mins: int,
    block_energy: Decimal,
    avg_hr: int | None,
    cadence: float,
    has_warmup_jog: bool,
) -> bool:
    if has_warmup_jog or cadence >= 20.0 or avg_hr is None or avg_hr < 115:
        return False
    burn_rate = float(block_energy) / duration_mins if duration_mins > 0 else 0.0
    if burn_rate < 3.0 or not slots:
        return False

    minimal_slots = 0
    consecutive = 0
    max_consecutive = 0
    for slot in slots:
        slot_hr = (sum(slot.hr_values) / len(slot.hr_values)) if slot.hr_values else None
        is_exertion = (slot.energy >= Decimal(15)) or (slot_hr is not None and slot_hr >= 110.0)
        if slot.steps < Decimal(25) and is_exertion:
            minimal_slots += 1
            consecutive += 1
            if consecutive > max_consecutive:
                max_consecutive = consecutive
        else:
            consecutive = 0
    return max_consecutive >= 2 and (minimal_slots / len(slots)) >= 0.50


def _has_declining_or_steady_hr(slots: Sequence[SampleSlot]) -> bool:
    slot_avgs = [
        sum(slot.hr_values) / len(slot.hr_values)
        for slot in slots
        if slot.hr_values
    ]
    if len(slot_avgs) < 2:
        return True
    tail_avg = sum(slot_avgs[1:]) / (len(slot_avgs) - 1)
    return tail_avg <= slot_avgs[0] + 5.0


def _is_walk_pattern(
    slots: Sequence[SampleSlot],
    cadence: float,
    has_warmup_jog: bool,
    avg_hr: int | None,
    peak_hr: int | None,
    has_rest_break: bool,
) -> bool:
    if has_warmup_jog or has_rest_break:
        return False
    is_low_hr_walk = (
        cadence > 50.0
        and (avg_hr is None or avg_hr <= 120)
        and (peak_hr is None or peak_hr <= 135)
    )
    is_elevated_steady_walk = (
        cadence >= 58.0
        and (avg_hr is None or avg_hr <= 138)
        and (
            peak_hr is None
            or peak_hr <= 153
            or (peak_hr <= 156 and _has_declining_or_steady_hr(slots))
        )
    )
    return is_low_hr_walk or is_elevated_steady_walk


def _is_badminton_pattern(
    cadence: float,
    burst_slots: int,
    has_warmup_jog: bool,
    has_rest_break: bool,
    avg_hr: int | None,
    peak_hr: int | None,
) -> bool:
    if has_warmup_jog or avg_hr is None:
        return False
    if avg_hr >= 140 and (peak_hr is None or peak_hr >= 154) and cadence >= 24.5:
        return True
    if avg_hr < 132:
        return False
    return (
        cadence >= 40.0
        or (cadence >= 28.0 and burst_slots >= 2)
        or (cadence >= 22.0 and burst_slots >= 3)
        or (cadence >= 32.0 and has_rest_break)
    )


def classify_activity_pattern(
    slots: Sequence[SampleSlot],
    duration_mins: int,
    block_energy: Decimal,
    avg_hr: int | None,
    peak_hr: int | None,
    all_hrs: Sequence[float],
    block_steps: Decimal,
) -> ActivityPattern:
    """Classifies an exertion window into swim, walk, badminton, or resistance training."""
    cadence = float(block_steps) / duration_mins if duration_mins > 0 else 0.0
    has_rest_break = _max_consecutive_stationary_mins(slots) >= 10
    hr_range = _compute_hr_range(all_hrs, avg_hr, peak_hr)
    has_warmup_jog = _has_warmup_jog(slots)
    burst_slots = sum(1 for slot in slots if slot.steps >= Decimal(250))

    if _is_swim_pattern(slots, duration_mins, block_energy, avg_hr, cadence, has_warmup_jog):
        return _SWIM_PATTERN
    if _is_walk_pattern(slots, cadence, has_warmup_jog, avg_hr, peak_hr, has_rest_break):
        return _WALK_PATTERN
    if _is_badminton_pattern(cadence, burst_slots, has_warmup_jog, has_rest_break, avg_hr, peak_hr):
        return _BADMINTON_PATTERN
    if (cadence < 58.0 or has_warmup_jog) and (hr_range >= 20.0 or has_warmup_jog or cadence < 30.0):
        return _GYM_PATTERN
    return _WALK_PATTERN if cadence > 50.0 else _GYM_PATTERN
