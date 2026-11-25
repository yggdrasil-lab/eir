"""Public API for telemetry ingestion, Shortcuts normalization, scale dedup, and Withings sync."""

from typing import TYPE_CHECKING, Any

from app.ingestion._dedup import (
    SCALE_DEDUP_METRICS,
    SCALE_SOURCE_PRIORITY,
    normalize_utc_second,
    upsert_health_samples,
)
from app.ingestion._shortcuts import (
    clean_datetime,
    clean_numeric,
    extract_unexpected_keys,
    normalize_apple_health_payload,
    parse_duration_seconds,
    to_list,
)
from app.ingestion._types import (
    Clock,
    IngestionError,
    IngestResult,
    NormalizedSample,
    RollupPort,
    WithingsAuthStart,
    WithingsCallbackResult,
    WithingsCloudPort,
    WithingsConfig,
    WithingsConfigError,
    WithingsConnectionStatus,
    WithingsMeasure,
    WithingsMeasureGroup,
    WithingsProtocolError,
    WithingsTokenSet,
    WithingsUpstreamError,
    WithingsWebhookResult,
    WorkoutSyncPort,
    utc_now,
)
from app.ingestion._withings_http import HttpWithingsCloudAdapter
from app.ingestion._withings_memory import InMemoryWithingsCloudAdapter

if TYPE_CHECKING:
    from app.ingestion._ingestor import TelemetryIngestor
    from app.ingestion._withings import (
        WITHINGS_MEASURE_TYPES,
        WithingsSync,
        decode_measure_groups,
        resolve_measure_window,
    )

__all__ = [
    "Clock",
    "HttpWithingsCloudAdapter",
    "InMemoryWithingsCloudAdapter",
    "IngestResult",
    "IngestionError",
    "NormalizedSample",
    "RollupPort",
    "SCALE_DEDUP_METRICS",
    "SCALE_SOURCE_PRIORITY",
    "TelemetryIngestor",
    "WITHINGS_MEASURE_TYPES",
    "WithingsAuthStart",
    "WithingsCallbackResult",
    "WithingsCloudPort",
    "WithingsConfig",
    "WithingsConfigError",
    "WithingsConnectionStatus",
    "WithingsMeasure",
    "WithingsMeasureGroup",
    "WithingsProtocolError",
    "WithingsSync",
    "WithingsTokenSet",
    "WithingsUpstreamError",
    "WithingsWebhookResult",
    "WorkoutSyncPort",
    "clean_datetime",
    "clean_numeric",
    "decode_measure_groups",
    "extract_unexpected_keys",
    "normalize_apple_health_payload",
    "normalize_utc_second",
    "parse_duration_seconds",
    "resolve_measure_window",
    "to_list",
    "upsert_health_samples",
    "utc_now",
]


def __getattr__(name: str) -> Any:
    if name == "TelemetryIngestor":
        from app.ingestion._ingestor import TelemetryIngestor

        return TelemetryIngestor
    if name in ("WITHINGS_MEASURE_TYPES", "WithingsSync", "decode_measure_groups", "resolve_measure_window"):
        from app.ingestion import _withings

        return getattr(_withings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
