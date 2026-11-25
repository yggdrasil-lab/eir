"""Value objects, protocol boundaries, and errors for Kratos workout reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol
import uuid

if TYPE_CHECKING:
    from app.schemas import DetectedWorkoutRead

TOMBSTONE_OVERLAP_THRESHOLD = 0.50
FULL_COVERAGE_RATIO_MIN = 0.80

SyncStatus = Literal[
    "synced",
    "reconciled",
    "skipped_confirmed",
    "skipped_split_overlap",
    "skipped_dismissed",
]


class KratosStoreError(RuntimeError):
    """Raised when the Kratos session store encounters an HTTP, transport, or payload error."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.status_code = status_code


def generate_session_id(started_at_iso: str, ended_at_iso: str) -> uuid.UUID:
    """Generates a deterministic UUIDv5 for an inferred workout window."""
    seed = f"kratos:inferred-session:{started_at_iso}:{ended_at_iso}"
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _optional_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _freeze_extra(extra: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if isinstance(extra, MappingProxyType):
        return extra
    return MappingProxyType(dict(extra) if isinstance(extra, Mapping) else {})


@dataclass(frozen=True)
class KratosSession:
    """Immutable snapshot of a workout session returned by Kratos."""

    id: str
    routine_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    deleted_at: str | None = None
    notes: str | None = None
    items: tuple[Any, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items or ()))

    @classmethod
    def from_api_dict(cls, data: Mapping[str, Any]) -> KratosSession:
        """Normalizes a raw Kratos session JSON dictionary into a frozen KratosSession."""
        raw_id = data.get("id")
        raw_routine_id = data.get("routine_id")
        raw_notes = data.get("notes")
        raw_extra = data.get("extra")
        return cls(
            id=str(raw_id) if raw_id is not None else "",
            routine_id=str(raw_routine_id) if raw_routine_id is not None else None,
            started_at=_optional_iso(data.get("started_at")),
            ended_at=_optional_iso(data.get("ended_at")),
            deleted_at=_optional_iso(data.get("deleted_at")),
            notes=str(raw_notes) if raw_notes is not None else None,
            items=tuple(data.get("items") or ()),
            extra=_freeze_extra(raw_extra if isinstance(raw_extra, Mapping) else None),
        )

    @property
    def is_confirmed(self) -> bool:
        """Returns True if the session is user-confirmed or manually logged."""
        return (
            self.extra.get("confirmed") is True
            or self.extra.get("inferred") is False
            or self.routine_id is not None
            or len(self.items) > 0
        )

    @property
    def is_dismissed_tombstone(self) -> bool:
        """Returns True if a soft-deleted session represents a dismissed or split workout."""
        if self.deleted_at is None:
            return False
        return (
            self.extra.get("dismissed") is True
            or bool(self.extra.get("split_into_ids"))
            or self.extra.get("inferred") is True
        )

    @property
    def is_telemetry_locked(self) -> bool:
        """Returns True when telemetry was manually locked or created via split."""
        return (
            self.extra.get("telemetry_locked") is True
            or self.extra.get("manual_split") is True
            or bool(self.extra.get("split_from_id"))
        )

    @property
    def has_complete_telemetry(self) -> bool:
        """Returns True when reconciled HR zones and RPE telemetry are already present."""
        return (
            self.extra.get("telemetry_reconciled") is True
            and "hr_zones" in self.extra
            and "rpe" in self.extra
        )


@dataclass(frozen=True)
class KratosSessionWrite:
    """Immutable write payload for upserting an inferred or reconciled Kratos session."""

    id: str
    routine_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    items: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items or ()))

    def to_api_dict(self) -> dict[str, Any]:
        """Serializes this write payload to the JSON body expected by PUT /sessions/{id}."""
        return {
            "id": self.id,
            "routine_id": self.routine_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "notes": self.notes,
            "extra": dict(self.extra),
            "items": list(self.items),
        }

    def to_session(self) -> KratosSession:
        """Projects this write payload into a matching active KratosSession snapshot."""
        return KratosSession(
            id=self.id,
            routine_id=self.routine_id,
            started_at=self.started_at,
            ended_at=self.ended_at,
            deleted_at=None,
            notes=self.notes,
            items=self.items,
            extra=self.extra,
        )


@dataclass(frozen=True)
class SyncOutcome:
    """Reconciliation result for a single detected workout."""

    session_id: str
    status: SyncStatus
    routine_name: str | None
    deleted_duplicate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyncBatchResult:
    """Batch reconciliation result returned by KratosReconciler.sync."""

    configured: bool
    outcomes: tuple[SyncOutcome, ...] = ()


class KratosSessionStore(Protocol):
    """Port abstracting session listing, lookup, upsert, and deletion in Kratos."""

    def list_sessions(
        self,
        *,
        limit: int = 200,
        include_deleted: bool = True,
    ) -> tuple[KratosSession, ...]: ...

    def get_session(self, session_id: str) -> KratosSession | None: ...

    def upsert_session(self, session: KratosSessionWrite) -> None: ...

    def delete_session(self, session_id: str) -> bool: ...


class WindowMetricsPort(Protocol):
    """Port abstracting narrow-window exertion metric recomputation."""

    def analyze_window(self, start_dt: datetime, end_dt: datetime) -> DetectedWorkoutRead: ...
