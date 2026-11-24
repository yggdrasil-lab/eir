"""Local biometric day resolution and timezone-aware day window bounds."""

from collections.abc import Sequence
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from app.daily_biometrics._types import BiometricsInputError


def day_bounds(target_date: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Returns inclusive [00:00:00, 23:59:59.999999] bounds for target_date in tz."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=tz)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=tz)
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
            aware = target if target.tzinfo is not None else target.replace(tzinfo=timezone.utc)
            days.add(aware.astimezone(tz).date())
        elif isinstance(target, date):
            days.add(target)
        else:
            raise BiometricsInputError(f"Expected date or datetime, got {type(target).__name__}")
    return tuple(sorted(days))
