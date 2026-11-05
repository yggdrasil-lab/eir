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
