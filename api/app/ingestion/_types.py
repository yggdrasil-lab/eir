"""Domain value objects, ports, clock, and typed errors for telemetry ingestion."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.daily_biometrics._types import RollupBatch
    from app.schemas import DetectedWorkoutRead

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """Returns the current UTC-aware timestamp."""
    return datetime.now(timezone.utc)


class IngestionError(Exception):
    """Base domain error for telemetry ingestion and external sync."""


class WithingsConfigError(IngestionError):
    """Raised when required Withings OAuth configuration is missing or invalid."""


class WithingsProtocolError(IngestionError):
    """Raised when Withings API returns a non-zero protocol status or malformed payload."""


class WithingsUpstreamError(IngestionError):
    """Raised when Withings HTTP transport fails or returns a non-200 status."""


def _freeze_mapping(raw: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if isinstance(raw, MappingProxyType):
        return raw
    return MappingProxyType(dict(raw) if isinstance(raw, Mapping) else {})


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


@dataclass(frozen=True)
class NormalizedSample:
    """Immutable, unit-resolved telemetry sample ready for deduplication and persistence."""

    sample_type: str
    measured_at: datetime
    value: Decimal
    ended_at: datetime | None = None
    unit: str | None = None
    source: str = "apple_health"
    external_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.ended_at is None:
            object.__setattr__(self, "ended_at", self.measured_at)
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))


@dataclass(frozen=True)
class IngestResult:
    """Immutable summary returned by TelemetryIngestor after persistence and rollups."""

    status: str
    total_samples_ingested: int
    samples_created: int
    metrics_summary: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    days_processed: tuple[date, ...] = ()
    detected_workouts: tuple[DetectedWorkoutRead, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics_summary", _freeze_mapping(self.metrics_summary))


@dataclass(frozen=True)
class WithingsConfig:
    """Validated Withings OAuth2 client configuration without silent fallback URLs."""

    client_id: str | None = None
    client_secret: str | None = None
    redirect_uri: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_id", _clean_optional_str(self.client_id))
        object.__setattr__(self, "client_secret", _clean_optional_str(self.client_secret))
        object.__setattr__(self, "redirect_uri", _clean_optional_str(self.redirect_uri))

    @classmethod
    def from_settings(cls, settings_obj: Any) -> WithingsConfig:
        """Builds WithingsConfig from application settings."""
        return cls(
            client_id=getattr(settings_obj, "withings_client_id", None),
            client_secret=getattr(settings_obj, "withings_client_secret", None),
            redirect_uri=getattr(settings_obj, "withings_redirect_uri", None),
        )

    def require_authorize(self) -> tuple[str, str]:
        """Returns (client_id, redirect_uri) or raises WithingsConfigError."""
        if not self.client_id:
            raise WithingsConfigError("Withings client ID (EIR_WITHINGS_CLIENT_ID) is not configured")
        if not self.redirect_uri:
            raise WithingsConfigError("Withings redirect URI (EIR_WITHINGS_REDIRECT_URI) is not configured")
        return self.client_id, self.redirect_uri

    def require_oauth(self) -> tuple[str, str, str]:
        """Returns (client_id, client_secret, redirect_uri) or raises WithingsConfigError."""
        if not self.client_id or not self.client_secret:
            raise WithingsConfigError(
                "Withings credentials (EIR_WITHINGS_CLIENT_ID / EIR_WITHINGS_CLIENT_SECRET) not configured"
            )
        if not self.redirect_uri:
            raise WithingsConfigError("Withings redirect URI (EIR_WITHINGS_REDIRECT_URI) is not configured")
        return self.client_id, self.client_secret, self.redirect_uri

    def require_refresh(self) -> tuple[str, str]:
        """Returns (client_id, client_secret) or raises WithingsConfigError."""
        if not self.client_id or not self.client_secret:
            raise WithingsConfigError(
                "Withings credentials (EIR_WITHINGS_CLIENT_ID / EIR_WITHINGS_CLIENT_SECRET) not configured"
            )
        return self.client_id, self.client_secret

    @property
    def webhook_callback_url(self) -> str:
        """Derives the Withings webhook callback URL from redirect_uri."""
        if not self.redirect_uri:
            raise WithingsConfigError("Withings redirect URI (EIR_WITHINGS_REDIRECT_URI) is not configured")
        cleaned = self.redirect_uri.rstrip("/")
        if "/" not in cleaned:
            raise WithingsConfigError("Withings redirect URI (EIR_WITHINGS_REDIRECT_URI) must be a valid URL")
        return f"{cleaned.rsplit('/', 1)[0]}/webhook"


@dataclass(frozen=True)
class WithingsTokenSet:
    """OAuth2 token payload returned by Withings authorization or refresh."""

    user_id: str
    access_token: str
    refresh_token: str
    expires_in: int = 10800
    scope: str | None = None


@dataclass(frozen=True)
class WithingsAuthStart:
    """Authorization URL and CSRF state generated for Withings OAuth initiation."""

    authorization_url: str
    state: str


@dataclass(frozen=True)
class WithingsCallbackResult:
    """Result of completing Withings OAuth2 authorization callback."""

    user_id: str
    expires_at: datetime
    subscription: Any = "unattempted"
    status: str = "ok"
    message: str = "Withings account connected successfully"


@dataclass(frozen=True)
class WithingsConnectionStatus:
    """Current Withings OAuth link state stored in the database."""

    status: str
    linked: bool
    provider: str = "withings"
    user_id: str | None = None
    expires_at: datetime | None = None
    scope: str | None = None


@dataclass(frozen=True)
class WithingsWebhookResult:
    """Result returned to Withings webhook notification callbacks."""

    appli: str
    samples_ingested: int
    status: str = "ok"
    received: bool = True


@dataclass(frozen=True)
class WithingsMeasure:
    """Single scaled measurement inside a Withings measure group."""

    type: int
    value: int
    unit: int


@dataclass(frozen=True)
class WithingsMeasureGroup:
    """Group of Withings measurements recorded at a single timestamp."""

    grpid: int | str | None
    date: int
    attrib: int | None = None
    measures: tuple[WithingsMeasure, ...] = ()


class WithingsCloudPort(Protocol):
    """Port decoupling Withings OAuth and measurement APIs from HTTP transport."""

    def exchange_code(self, config: WithingsConfig, code: str) -> WithingsTokenSet: ...

    def refresh_token(self, config: WithingsConfig, refresh_token: str) -> WithingsTokenSet: ...

    def subscribe_notify(
        self,
        access_token: str,
        callback_url: str,
        *,
        appli: int = 1,
        comment: str = "Eir Weight Ingestion",
    ) -> Any: ...

    def fetch_measure_groups(
        self,
        access_token: str,
        *,
        startdate: int | None = None,
        enddate: int | None = None,
    ) -> tuple[WithingsMeasureGroup, ...]: ...


class RollupPort(Protocol):
    """Port abstracting daily biometric rollup materialization."""

    def materialize_rollups(self, targets: Sequence[date | datetime]) -> RollupBatch: ...


class WorkoutSyncPort(Protocol):
    """Port abstracting downstream reconciliation of detected workouts."""

    def sync(self, workouts: Sequence[DetectedWorkoutRead]) -> Any: ...
