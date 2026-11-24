"""Window normalization, 5-minute slot bucketing, and unified exertion slot predicate."""

from collections.abc import Sequence
from datetime import datetime, timedelta, tzinfo
from decimal import Decimal

from app.exertion._types import BucketedWindow, InvalidWindowError, SampleSlot
from app.models import HealthSample
from app.services.biometrics import filter_deduplicated_step_samples

SLOT_SECONDS = 300


def sample_local_dt(sample: HealthSample, tz: tzinfo) -> datetime:
    """Normalizes a HealthSample measured_at timestamp to the target timezone."""
    if sample.measured_at.tzinfo:
        return sample.measured_at.astimezone(tz)
    return sample.measured_at.replace(tzinfo=tz)


def normalize_window_bounds(
    started_at: datetime,
    ended_at: datetime,
    tz: tzinfo,
) -> tuple[datetime, datetime]:
    """Normalizes window timestamps to tz and validates chronological ordering."""
    s_time = started_at.astimezone(tz) if started_at.tzinfo else started_at.replace(tzinfo=tz)
    e_time = ended_at.astimezone(tz) if ended_at.tzinfo else ended_at.replace(tzinfo=tz)
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
