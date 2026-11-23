"""Biometric processing, multi-device deduplication, and exertion inference engines."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from typing import Any

from app.config import settings
from app.models import HealthSample
from app.schemas import SuggestedSplitRead, TimelineBinRead, WorkoutTimelineRead

ACTIVE_WATCH_HR_SAMPLES_MIN = 15
SLOT_SECONDS = 300


@dataclass(frozen=True)
class SampleSlot:
    """Immutable 5-minute biometric bucket for workout detection and timeline analysis."""

    started_at: datetime
    ended_at: datetime
    energy: Decimal
    hr_values: tuple[float, ...]
    steps: Decimal


@dataclass(frozen=True)
class BucketedWindow:
    """Container of 5-minute biometric slots and deduplicated step samples."""

    slots: tuple[SampleSlot, ...]
    deduped_step_samples: tuple[HealthSample, ...]


def sample_local_dt(sample: HealthSample, tz: tzinfo) -> datetime:
    """Normalizes a HealthSample measured_at timestamp to the target timezone."""
    if sample.measured_at.tzinfo:
        return sample.measured_at.astimezone(tz)
    return sample.measured_at.replace(tzinfo=tz)


def _accumulate_slot_samples(
    samples: Sequence[HealthSample],
    deduped_steps: Sequence[HealthSample],
    start_dt: datetime,
    end_dt: datetime,
    tz: tzinfo,
    num_slots: int,
    inclusive_end: bool,
) -> tuple[list[Decimal], list[list[float]], list[Decimal]]:
    energies = [Decimal(0)] * num_slots
    hr_lists: list[list[float]] = [[] for _ in range(num_slots)]
    step_totals = [Decimal(0)] * num_slots

    for s in samples:
        dt_val = sample_local_dt(s, tz)
        if not _in_bucket_bounds(dt_val, start_dt, end_dt, inclusive_end):
            continue
        idx = int((dt_val - start_dt).total_seconds() // SLOT_SECONDS)
        if not (0 <= idx < num_slots):
            continue
        if s.sample_type in ("active_energy", "active_energy_kcal"):
            energies[idx] += s.value
        elif s.sample_type in ("heart_rate", "resting_hr"):
            hr_lists[idx].append(float(s.value))

    for s in deduped_steps:
        idx = int((sample_local_dt(s, tz) - start_dt).total_seconds() // SLOT_SECONDS)
        if 0 <= idx < num_slots:
            step_totals[idx] += s.value
    return energies, hr_lists, step_totals


def bucket_samples_into_5min_slots(
    samples: Sequence[HealthSample],
    start_dt: datetime,
    end_dt: datetime,
    tz: tzinfo,
    fixed_slot_count: int | None = None,
) -> BucketedWindow:
    """Buckets energy, heart rate, and deduplicated step samples into 5-minute slots."""
    inclusive_end = fixed_slot_count is not None
    raw_steps = [
        s for s in samples
        if s.sample_type in ("step_count", "steps")
        and _in_bucket_bounds(sample_local_dt(s, tz), start_dt, end_dt, inclusive_end)
    ]
    deduped_steps = filter_deduplicated_step_samples(raw_steps)
    num_slots = (
        fixed_slot_count
        if fixed_slot_count is not None
        else max(1, (int((end_dt - start_dt).total_seconds()) + SLOT_SECONDS - 1) // SLOT_SECONDS)
    )
    energies, hr_lists, step_totals = _accumulate_slot_samples(
        samples, deduped_steps, start_dt, end_dt, tz, num_slots, inclusive_end
    )
    slots = tuple(
        SampleSlot(
            started_at=start_dt + timedelta(minutes=i * 5),
            ended_at=min(start_dt + timedelta(minutes=(i + 1) * 5), end_dt),
            energy=energies[i],
            hr_values=tuple(hr_lists[i]),
            steps=step_totals[i],
        )
        for i in range(num_slots)
    )
    return BucketedWindow(slots=slots, deduped_step_samples=tuple(deduped_steps))


def _in_bucket_bounds(
    dt_val: datetime,
    start_dt: datetime,
    end_dt: datetime,
    inclusive_end: bool,
) -> bool:
    if dt_val < start_dt:
        return False
    return dt_val <= end_dt if inclusive_end else dt_val < end_dt


def build_timeline_bins(slots: Sequence[SampleSlot]) -> list[TimelineBinRead]:
    """Converts 5-minute biometric slots into timeline bins with Watch mode flags."""
    bins: list[TimelineBinRead] = []
    for slot in slots:
        hr_count = len(slot.hr_values)
        avg_hr = int(round(sum(slot.hr_values) / hr_count)) if hr_count else None
        peak_hr = int(round(max(slot.hr_values))) if hr_count else None
        bins.append(
            TimelineBinRead(
                started_at=slot.started_at,
                ended_at=slot.ended_at,
                avg_hr_bpm=avg_hr,
                peak_hr_bpm=peak_hr,
                hr_sample_count=hr_count,
                step_count=int(round(slot.steps)),
                active_energy_kcal=int(round(slot.energy)),
                is_active_watch_mode=hr_count >= ACTIVE_WATCH_HR_SAMPLES_MIN,
            )
        )
    return bins


def _is_exertion_bin(bin_item: TimelineBinRead) -> bool:
    return (
        bin_item.active_energy_kcal >= 20
        or (bin_item.avg_hr_bpm is not None and bin_item.avg_hr_bpm >= 120)
        or (
            bin_item.step_count >= 250
            and bin_item.avg_hr_bpm is not None
            and bin_item.avg_hr_bpm >= 95
        )
        or (
            bin_item.active_energy_kcal >= 15
            and bin_item.avg_hr_bpm is not None
            and bin_item.avg_hr_bpm >= 100
        )
    )


def _is_transition_gap_bin(bin_item: TimelineBinRead) -> bool:
    if bin_item.is_active_watch_mode:
        return False
    if not _is_exertion_bin(bin_item):
        return True
    return (bin_item.avg_hr_bpm is None or bin_item.avg_hr_bpm < 105) and bin_item.active_energy_kcal < 20


def _within_split_bounds(
    split_at: datetime,
    resume_at: datetime,
    started_at: datetime,
    ended_at: datetime,
) -> bool:
    min_bound = started_at + timedelta(minutes=10)
    max_bound = ended_at - timedelta(minutes=10)
    return split_at >= min_bound and resume_at <= max_bound and split_at <= resume_at


def _resolve_passive_to_active(bins: Sequence[TimelineBinRead], idx: int) -> SuggestedSplitRead | None:
    split_idx = idx
    checked = 0
    while checked < 2 and split_idx - 1 >= 1 and _is_transition_gap_bin(bins[split_idx - 1]):
        split_idx -= 1
        checked += 1
    if not any(not _is_transition_gap_bin(b) for b in bins[:split_idx]):
        return None
    reason = (
        "Active Watch workout mode started after passive transition gap"
        if split_idx < idx
        else "Active Watch workout mode started after passive monitoring"
    )
    return SuggestedSplitRead(
        split_at=bins[split_idx].started_at,
        resume_at=bins[idx].started_at,
        reason=reason,
        confidence="high",
    )


def _resolve_active_to_passive(
    bins: Sequence[TimelineBinRead],
    idx: int,
) -> tuple[SuggestedSplitRead | None, int]:
    gap_end = idx
    while gap_end < len(bins) and (gap_end - idx) < 2 and _is_transition_gap_bin(bins[gap_end]):
        gap_end += 1

    if gap_end >= len(bins):
        return None, len(bins)

    if bins[gap_end].is_active_watch_mode:
        if gap_end == idx:
            return None, idx + 1
        split = SuggestedSplitRead(
            split_at=bins[idx].started_at,
            resume_at=bins[gap_end].started_at,
            reason="Active Watch workout mode resumed after transition gap",
            confidence="high",
        )
        return split, gap_end + 1

    if gap_end == idx:
        split = SuggestedSplitRead(
            split_at=bins[idx].started_at,
            resume_at=bins[idx].started_at,
            reason="Active Watch workout mode ended while passive exertion continued",
            confidence="high",
        )
        return split, idx + 1

    if not any(not _is_transition_gap_bin(b) for b in bins[gap_end:]):
        return None, len(bins)
    split = SuggestedSplitRead(
        split_at=bins[idx].started_at,
        resume_at=bins[gap_end].started_at,
        reason="Active Watch workout mode ended before transition gap",
        confidence="high",
    )
    return split, gap_end + 1


def suggest_timeline_splits(
    bins: Sequence[TimelineBinRead],
    started_at: datetime,
    ended_at: datetime,
) -> list[SuggestedSplitRead]:
    """Detects passive<->active Watch mode transitions within [started_at + 10m, ended_at - 10m]."""
    if (ended_at - started_at) < timedelta(minutes=20) or len(bins) < 3:
        return []

    suggestions: list[SuggestedSplitRead] = []
    idx = 1
    while idx < len(bins):
        prev_active = bins[idx - 1].is_active_watch_mode
        curr_active = bins[idx].is_active_watch_mode
        candidate: SuggestedSplitRead | None = None
        next_idx = idx + 1

        if not prev_active and curr_active:
            candidate = _resolve_passive_to_active(bins, idx)
        elif prev_active and not curr_active:
            candidate, next_idx = _resolve_active_to_passive(bins, idx)

        if candidate is not None and _within_split_bounds(
            candidate.split_at, candidate.resume_at, started_at, ended_at
        ):
            suggestions.append(candidate)
        idx = next_idx

    return suggestions


def build_workout_timeline(
    slots: Sequence[SampleSlot],
    started_at: datetime,
    ended_at: datetime,
) -> WorkoutTimelineRead:
    """Assembles a WorkoutTimelineRead from 5-minute slots and transition split hints."""
    duration_mins = max(1, int(round((ended_at - started_at).total_seconds() / 60)))
    bins = build_timeline_bins(slots)
    suggested_splits = suggest_timeline_splits(bins, started_at, ended_at)
    return WorkoutTimelineRead(
        started_at=started_at,
        ended_at=ended_at,
        duration_minutes=duration_mins,
        bins=bins,
        suggested_splits=suggested_splits,
    )


def _normalize_dt(dt_val: datetime) -> datetime:
    """Ensures datetime comparisons are consistently timezone-aware."""
    return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)


def _step_sample_interval(sample: HealthSample) -> tuple[datetime, datetime]:
    """Returns normalized (start, end) interval for a step sample."""
    start = _normalize_dt(sample.measured_at)
    if sample.ended_at is not None:
        ended = _normalize_dt(sample.ended_at)
        if ended > start:
            return start, ended
    return start, start + timedelta(seconds=300)


def filter_deduplicated_step_samples(samples: Sequence[HealthSample]) -> list[HealthSample]:
    """Filters multi-device Apple Health step samples with Apple Watch overlap priority.

    When both Apple Watch and iPhone record steps during the same interval, Apple Watch
    takes priority and overlapping iPhone samples are dropped. Non-overlapping iPhone
    intervals (e.g. phone carried while watch is charging) and third-party/unlabeled
    pedometer samples are retained in their original order.
    """
    if not samples:
        return []

    watch_intervals: list[tuple[datetime, datetime]] = []
    classified: list[tuple[HealthSample, str]] = []

    for s in samples:
        dev = str((s.extra or {}).get("device") or "").lower()
        if "watch" in dev:
            watch_intervals.append(_step_sample_interval(s))
            classified.append((s, "watch"))
        elif "phone" in dev:
            classified.append((s, "phone"))
        else:
            classified.append((s, "other"))

    if not watch_intervals:
        return list(samples)

    surviving: list[HealthSample] = []
    for s, kind in classified:
        if kind == "phone":
            p_start, p_end = _step_sample_interval(s)
            if any(
                max(p_start, w_start) <= min(p_end, w_end)
                for w_start, w_end in watch_intervals
            ):
                continue
        surviving.append(s)
    return surviving


def deduplicate_step_samples(samples: Sequence[HealthSample]) -> Decimal:
    """Calculates true total steps across multi-device Apple Health step samples.

    When both Apple Watch and iPhone record steps during the same interval, Apple Watch
    takes priority. Non-overlapping iPhone intervals (e.g. phone carried while watch is charging)
    are retained. Third-party/unlabeled pedometer samples are added cleanly.
    """
    return sum((s.value for s in filter_deduplicated_step_samples(samples)), Decimal(0))


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
    # - sustained elevated HR (Zone 3-5: avg_hr >= 132, or avg_hr >= 128 with peak_hr >= 155)
    # - sustained active court movement:
    #   * cadence >= 40 steps/min, OR
    #   * cadence >= 28 steps/min with burst slots (>= 2 slots with >= 250 steps), OR
    #   * cadence >= 22 steps/min with repeated court bursts (>= 3 burst slots, tolerates post-match cooldown HR tail), OR
    #   * cadence >= 32 steps/min with rest break intervals (game-rest-game pattern)
    # - excludes gym workouts with warmup jog and swimming
    has_badminton_hr = avg_hr is not None and (
        avg_hr >= 132
        or (avg_hr >= 128 and peak_hr is not None and peak_hr >= 155)
    )
    is_badminton = (
        not has_warmup_jog
        and not is_swim
        and (
            cadence >= 40.0
            or (cadence >= 28.0 and burst_slots >= 2)
            or (cadence >= 22.0 and burst_slots >= 3)
            or (cadence >= 32.0 and has_rest_break_interval)
        )
        and has_badminton_hr
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
    age: int | None = None,
    sex: str | None = None,
) -> int:
    """Estimates realistic net active calories for a workout when raw samples are depressed.

    Triggers when Apple Watch background passive monitoring mode causes severe
    undercounting (< 3.5 kcal/min despite elevated heart rate >= 115 bpm).
    """
    if not avg_hr or duration_mins <= 0:
        return 0

    user_age = age if age is not None else settings.user_age
    user_sex = (sex or settings.user_sex).lower()

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
