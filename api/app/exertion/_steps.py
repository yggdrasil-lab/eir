"""Multi-device Apple Health step deduplication helpers."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.models import HealthSample


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


def _intervals_overlap(
    start_a: datetime,
    end_a: datetime,
    watch_intervals: Sequence[tuple[datetime, datetime]],
) -> bool:
    return any(max(start_a, w_start) <= min(end_a, w_end) for w_start, w_end in watch_intervals)


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
        if kind == "phone" and _intervals_overlap(*_step_sample_interval(s), watch_intervals):
            continue
        surviving.append(s)
    return surviving


def deduplicate_step_samples(samples: Sequence[HealthSample]) -> Decimal:
    """Calculates true total steps across multi-device Apple Health step samples."""
    return sum((s.value for s in filter_deduplicated_step_samples(samples)), Decimal(0))
