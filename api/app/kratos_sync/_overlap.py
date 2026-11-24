"""Pure overlap detection and session classification for Kratos reconciliation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import uuid

from app.kratos_sync._types import KratosSession


@dataclass(frozen=True)
class _OverlapCandidate:
    session: KratosSession
    started_at: datetime
    ended_at: datetime
    overlap_seconds: float
    is_exact_id: bool


@dataclass
class _ScanAccumulator:
    confirmed: list[_OverlapCandidate] = field(default_factory=list)
    overlapping_inferred_ids: list[str] = field(default_factory=list)
    best_tombstone: KratosSession | None = None
    max_tombstone_overlap: float = 0.0
    best_confirmed: _OverlapCandidate | None = None
    max_confirmed_overlap: float = 0.0
    exact_confirmed_matched: bool = False
    exact_inferred: KratosSession | None = None
    best_inferred: KratosSession | None = None
    max_inferred_overlap: float = 0.0


@dataclass(frozen=True)
class _SyncScanResult:
    confirmed: tuple[_OverlapCandidate, ...]
    overlapping_inferred_ids: tuple[str, ...]
    best_tombstone: KratosSession | None
    max_tombstone_overlap: float
    best_confirmed: _OverlapCandidate | None
    max_confirmed_overlap: float
    exact_inferred: KratosSession | None
    best_inferred: KratosSession | None


def to_utc_dt(val: str | datetime) -> datetime:
    """Converts a datetime or ISO-8601 string into an aware UTC datetime."""
    dt = datetime.fromisoformat(val) if isinstance(val, str) else val
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def session_window_utc(
    session: KratosSession,
    default_start: datetime,
    default_end: datetime,
    default_duration_mins: int,
    is_exact_id: bool,
) -> tuple[datetime, datetime] | None:
    """Resolves a KratosSession's [start, end] UTC window, falling back on exact IDs."""
    if not session.started_at:
        return (default_start, default_end) if is_exact_id else None
    try:
        start_utc = to_utc_dt(session.started_at)
        if session.ended_at:
            return start_utc, to_utc_dt(session.ended_at)
        dur = session.extra.get("duration_minutes") or default_duration_mins or 30
        return start_utc, start_utc + timedelta(minutes=dur)
    except (ValueError, TypeError):
        return (default_start, default_end) if is_exact_id else None


def _record_session_match(
    acc: _ScanAccumulator,
    session: KratosSession,
    window: tuple[datetime, datetime],
    overlap: float,
    is_exact_id: bool,
) -> None:
    if session.deleted_at is not None:
        if session.is_dismissed_tombstone and overlap >= acc.max_tombstone_overlap:
            acc.max_tombstone_overlap = overlap
            acc.best_tombstone = session
        return

    if session.is_confirmed:
        candidate = _OverlapCandidate(session, window[0], window[1], overlap, is_exact_id)
        acc.confirmed.append(candidate)
        if is_exact_id or (not acc.exact_confirmed_matched and overlap > acc.max_confirmed_overlap):
            acc.max_confirmed_overlap = overlap
            acc.best_confirmed = candidate
            if is_exact_id:
                acc.exact_confirmed_matched = True
        return

    if session.id and session.id not in acc.overlapping_inferred_ids:
        acc.overlapping_inferred_ids.append(session.id)
    if is_exact_id:
        acc.exact_inferred = session
    elif overlap > acc.max_inferred_overlap:
        acc.max_inferred_overlap = overlap
        acc.best_inferred = session


def scan_existing_sessions(
    existing_sessions: Sequence[KratosSession],
    session_id: uuid.UUID,
    w_start_utc: datetime,
    w_end_utc: datetime,
    default_duration_mins: int,
) -> _SyncScanResult:
    """Partitions existing Kratos sessions into tombstones, active confirmed, and active inferred overlaps."""
    acc = _ScanAccumulator()
    target_id = str(session_id)
    for session in existing_sessions:
        is_exact_id = bool(session.id and session.id == target_id)
        window = session_window_utc(
            session, w_start_utc, w_end_utc, default_duration_mins, is_exact_id
        )
        if window is None:
            continue
        overlap = (min(w_end_utc, window[1]) - max(w_start_utc, window[0])).total_seconds()
        if overlap <= 0 and not is_exact_id:
            continue
        _record_session_match(acc, session, window, max(overlap, 0.0), is_exact_id)

    return _SyncScanResult(
        confirmed=tuple(acc.confirmed),
        overlapping_inferred_ids=tuple(acc.overlapping_inferred_ids),
        best_tombstone=acc.best_tombstone,
        max_tombstone_overlap=acc.max_tombstone_overlap,
        best_confirmed=acc.best_confirmed,
        max_confirmed_overlap=acc.max_confirmed_overlap,
        exact_inferred=acc.exact_inferred,
        best_inferred=acc.best_inferred,
    )
