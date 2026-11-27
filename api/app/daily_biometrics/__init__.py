"""Daily biometrics deep module: canonical registry, daily summaries, and rollups."""

from typing import TYPE_CHECKING, Any

from app.daily_biometrics._registry import METRIC_REGISTRY, MetricAliasRegistry
from app.daily_biometrics._types import (
    BiometricsInputError,
    DailyRollupResult,
    ExertionPort,
    RollupBatch,
)
from app.daily_biometrics._window import resolve_sleep_affected_days

if TYPE_CHECKING:
    from app.daily_biometrics._facade import DailyBiometrics

__all__ = [
    "BiometricsInputError",
    "DailyBiometrics",
    "DailyRollupResult",
    "ExertionPort",
    "METRIC_REGISTRY",
    "MetricAliasRegistry",
    "RollupBatch",
    "resolve_sleep_affected_days",
]


def __getattr__(name: str) -> Any:
    if name == "DailyBiometrics":
        from app.daily_biometrics._facade import DailyBiometrics

        return DailyBiometrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
