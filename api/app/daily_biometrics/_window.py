"""Local biometric day resolution and timezone-aware day window bounds."""

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.daily_biometrics._registry import METRIC_REGISTRY
from app.daily_biometrics._types import BiometricsInputError


def to_local_datetime(dt_val: datetime, tz: ZoneInfo) -> datetime:
    """Normalizes naive datetimes as UTC before converting to tz."""
    aware = dt_val if dt_val.tzinfo is not None else dt_val.replace(tzinfo=timezone.utc)
    return aware.astimezone(tz)


def day_bounds(target_date: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Returns inclusive [00:00:00, 23:59:59.999999] bounds for target_date in tz."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=tz)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=tz)
    return start_dt, end_dt


def sleep_fetch_bounds(target_date: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Returns [target_date - 1d 18:00:00, target_date 23:59:59.999999] in tz."""
    start_dt = datetime.combine(target_date - timedelta(days=1), time(18, 0)).replace(tzinfo=tz)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=tz)
    return start_dt, end_dt


def morning_hrv_bounds(target_date: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Returns [target_date 00:00:00, target_date 10:00:00] in tz."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=tz)
    end_dt = datetime.combine(target_date, time(10, 0)).replace(tzinfo=tz)
    return start_dt, end_dt


def validate_target_date(target_date: object) -> date:
    """Validates a calendar date parameter, rejecting datetimes, bools, and non-dates."""
    if isinstance(target_date, (datetime, bool)) or not isinstance(target_date, date):
        raise BiometricsInputError(f"Expected calendar date, got {type(target_date).__name__}")
    return target_date


def resolve_local_days(targets: Sequence[date | datetime], tz: ZoneInfo) -> tuple[date, ...]:
    """Converts dates/datetimes to deduplicated, sorted local calendar days in tz."""
    days: set[date] = set()
    for target in targets:
        if isinstance(target, bool):
            raise BiometricsInputError("Expected date or datetime, got bool")
        if isinstance(target, datetime):
            days.add(to_local_datetime(target, tz).date())
        elif isinstance(target, date):
            days.add(target)
        else:
            raise BiometricsInputError(f"Expected date or datetime, got {type(target).__name__}")
    return tuple(sorted(days))


def resolve_sleep_affected_days(samples: Sequence[Any], tz: ZoneInfo) -> tuple[date, ...]:
    """Resolves local rollup days for samples, expanding wake-up days for sleep telemetry."""
    days: set[date] = set()
    for sample in samples:
        measured_at = getattr(sample, "measured_at", None)
        if not isinstance(measured_at, datetime):
            raise BiometricsInputError("Sample measured_at must be a datetime")
        measured_local = to_local_datetime(measured_at, tz)
        days.add(measured_local.date())

        raw_type = getattr(sample, "sample_type", None)
        canonical_type = METRIC_REGISTRY.canonicalize(str(raw_type)) if raw_type else ""
        if canonical_type != "sleep":
            continue

        ended_at = getattr(sample, "ended_at", None)
        if isinstance(ended_at, datetime):
            days.add(to_local_datetime(ended_at, tz).date())
        if measured_local.hour >= 18:
            days.add(measured_local.date() + timedelta(days=1))
    return tuple(sorted(days))
