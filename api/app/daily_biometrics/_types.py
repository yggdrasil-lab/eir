"""Value objects, protocol boundaries, and domain exceptions for daily biometrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.models import BiometricMeasurement
    from app.schemas import DetectedWorkoutRead


class BiometricsInputError(ValueError):
    """Raised when a caller provides invalid date, sample_type, or query parameters."""


class ExertionPort(Protocol):
    """Port decoupling daily biometrics rollups/summaries from exertion analysis."""

    def detect_day(self, target_date: date) -> list[DetectedWorkoutRead]: ...


@dataclass(frozen=True)
class DailyRollupResult:
    """Single-day rollup materialization result and detected workouts."""

    day: date
    measurement: BiometricMeasurement
    detected_workouts: tuple[DetectedWorkoutRead, ...]


@dataclass(frozen=True)
class RollupBatch:
    """Aggregated result across one or more local biometric days."""

    days: tuple[date, ...]
    results: tuple[DailyRollupResult, ...]
    detected_workouts: tuple[DetectedWorkoutRead, ...]
