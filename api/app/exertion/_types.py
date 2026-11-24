"""Internal value objects, exceptions, and configuration profile for exertion analysis."""

from dataclasses import dataclass
from datetime import datetime, tzinfo
from decimal import Decimal
from typing import Any

from app.config import settings
from app.models import HealthSample


class InvalidWindowError(ValueError):
    """Raised when an exertion analysis window has ended_at <= started_at."""


@dataclass(frozen=True)
class ExertionProfile:
    """Physiological and timezone parameters governing exertion evaluation."""

    tz: tzinfo
    max_hr: int

    @classmethod
    def from_settings(cls) -> "ExertionProfile":
        """Builds an ExertionProfile from application configuration defaults."""
        return cls(tz=settings.tz, max_hr=settings.user_max_hr)


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


@dataclass(frozen=True)
class ActivityPattern:
    """Inferred sport or workout classification labels for an exertion window."""

    classification: str
    activity_type: str
    routine_name: str


@dataclass(frozen=True)
class ExertionScore:
    """Computed heart rate zones, Edwards TRIMP, RPE, and effort rating."""

    effort_rating: str
    rpe: float
    trimp: float
    hr_zones: dict[str, Any] | None
