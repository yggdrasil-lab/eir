"""Window normalization, 5-minute slot bucketing, and unified exertion slot predicate."""

from collections.abc import Sequence
from datetime import datetime, timedelta, tzinfo
from decimal import Decimal

from app.exertion._types import BucketedWindow, InvalidWindowError, SampleSlot
from app.models import HealthSample
from app.services.biometrics import filter_deduplicated_step_samples

SLOT_SECONDS = 300


def _to_local_dt(dt_val: datetime, tz: tzinfo) -> datetime:
    """Normalizes a datetime to the target timezone."""
    return dt_val.astimezone(tz) if dt_val.tzinfo else dt_val.replace(tzinfo=tz)


def sample_local_dt(sample: HealthSample, tz: tzinfo) -> datetime:
    """Normalizes a HealthSample measured_at timestamp to the target timezone."""
    return _to_local_dt(sample.measured_at, tz)


def normalize_window_bounds(
    started_at: datetime,
    ended_at: datetime,
    tz: tzinfo,
) -> tuple[datetime, datetime]:
    """Normalizes window timestamps to tz and validates chronological ordering."""
    s_time = _to_local_dt(started_at, tz)
    e_time = _to_local_dt(ended_at, tz)
    if e_time <= s_time:
        raise InvalidWindowError("ended_at must be after started_at")
    return s_time, e_time


def is_exertion_slot(slot: SampleSlot) -> bool:
    """Single source of truth for whether a 5-minute slot exhibits sustained exertion."""
    avg_hr = (sum(slot.hr_values) / len(slot.hr_values)) if slot.hr_values else None
    return (
        (slot.energy >= Decimal(20))
        or (avg_hr is not None and avg_hr >= 120.0)
        or (slot.steps >= Decimal(250) and avg_hr is not None and avg_hr >= 95.0)
        or (slot.energy >= Decimal(15) and avg_hr is not None and avg_hr >= 100.0)
    )


def _in_bucket_bounds(
    dt_val: datetime,
    start_dt: datetime,
    end_dt: datetime,
    inclusive_end: bool,
) -> bool:
    if dt_val < start_dt:
        return False
    return dt_val <= end_dt if inclusive_end else dt_val < end_dt


def _duration_us(delta: timedelta) -> int:
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _step_sample_in_window(
    sample: HealthSample,
    start_dt: datetime,
    end_dt: datetime,
    tz: tzinfo,
    inclusive_end: bool,
) -> bool:
    s_start = _to_local_dt(sample.measured_at, tz)
    if sample.ended_at is None:
        return _in_bucket_bounds(s_start, start_dt, end_dt, inclusive_end)
    s_end = _to_local_dt(sample.ended_at, tz)
    if s_end <= s_start:
        return _in_bucket_bounds(s_start, start_dt, end_dt, inclusive_end)
    end_ok = s_start <= end_dt if inclusive_end else s_start < end_dt
    return end_ok and s_end > start_dt


def _prorate_step_interval(
    value: Decimal,
    s_start: datetime,
    s_end: datetime,
    step_totals: list[Decimal],
    start_dt: datetime,
    window_end: datetime,
    num_slots: int,
) -> None:
    clamped_start = max(s_start, start_dt)
    clamped_end = min(s_end, window_end)
    if clamped_end <= clamped_start:
        return

    slot_us = SLOT_SECONDS * 1_000_000
    total_us = _duration_us(s_end - s_start)
    overlap_us = _duration_us(clamped_end - clamped_start)
    first_idx = max(0, _duration_us(clamped_start - start_dt) // slot_us)
    last_idx = min(num_slots - 1, (_duration_us(clamped_end - start_dt) - 1) // slot_us)
    if last_idx < first_idx:
        return

    fully_inside = s_start >= start_dt and s_end <= window_end
    target_total = value if fully_inside else (value * Decimal(overlap_us) / Decimal(total_us))
    allocated = Decimal(0)
    for idx in range(first_idx, last_idx + 1):
        if idx == last_idx:
            step_totals[idx] += target_total - allocated
            return
        slot_start = start_dt + timedelta(seconds=idx * SLOT_SECONDS)
        slot_end = min(start_dt + timedelta(seconds=(idx + 1) * SLOT_SECONDS), window_end)
        slot_ov_us = _duration_us(min(clamped_end, slot_end) - max(clamped_start, slot_start))
        if slot_ov_us <= 0:
            continue
        share = value * Decimal(slot_ov_us) / Decimal(total_us)
        step_totals[idx] += share
        allocated += share


def _accumulate_step_sample(
    sample: HealthSample,
    step_totals: list[Decimal],
    start_dt: datetime,
    end_dt: datetime,
    tz: tzinfo,
    num_slots: int,
    inclusive_end: bool,
) -> None:
    s_start = _to_local_dt(sample.measured_at, tz)
    s_end = _to_local_dt(sample.ended_at, tz) if sample.ended_at is not None else None
    if s_end is not None and s_end > s_start:
        window_end = min(end_dt, start_dt + timedelta(seconds=num_slots * SLOT_SECONDS))
        _prorate_step_interval(sample.value, s_start, s_end, step_totals, start_dt, window_end, num_slots)
        return
    if not _in_bucket_bounds(s_start, start_dt, end_dt, inclusive_end):
        return
    idx = int((s_start - start_dt).total_seconds() // SLOT_SECONDS)
    if 0 <= idx < num_slots:
        step_totals[idx] += sample.value


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
        _accumulate_step_sample(s, step_totals, start_dt, end_dt, tz, num_slots, inclusive_end)
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
        and _step_sample_in_window(s, start_dt, end_dt, tz, inclusive_end)
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
