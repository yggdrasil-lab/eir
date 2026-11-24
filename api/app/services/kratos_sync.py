"""Synchronization bridge for auto-detected workout sessions to Kratos."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from typing import Any
import uuid

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.exertion import ExertionAnalyzer
from app.schemas import DetectedWorkoutRead

logger = logging.getLogger(__name__)

TOMBSTONE_OVERLAP_THRESHOLD = 0.50
FULL_COVERAGE_RATIO_MIN = 0.80


@dataclass(frozen=True)
class _OverlapCandidate:
    session: dict[str, Any]
    started_at: datetime
    ended_at: datetime
    overlap_seconds: float
    is_exact_id: bool


@dataclass
class _ScanAccumulator:
    confirmed: list[_OverlapCandidate] = field(default_factory=list)
    overlapping_inferred_ids: list[str] = field(default_factory=list)
    best_tombstone: dict[str, Any] | None = None
    max_tombstone_overlap: float = 0.0
    best_confirmed: _OverlapCandidate | None = None
    max_confirmed_overlap: float = 0.0
    exact_confirmed_matched: bool = False
    exact_inferred: dict[str, Any] | None = None
    best_inferred: dict[str, Any] | None = None
    max_inferred_overlap: float = 0.0


@dataclass(frozen=True)
class _SyncScanResult:
    confirmed: tuple[_OverlapCandidate, ...]
    overlapping_inferred_ids: tuple[str, ...]
    best_tombstone: dict[str, Any] | None
    max_tombstone_overlap: float
    best_confirmed: _OverlapCandidate | None
    max_confirmed_overlap: float
    exact_inferred: dict[str, Any] | None
    best_inferred: dict[str, Any] | None


def generate_session_id(started_at_iso: str, ended_at_iso: str) -> uuid.UUID:
    """Generates a deterministic UUIDv5 for an inferred workout window."""
    seed = f"kratos:inferred-session:{started_at_iso}:{ended_at_iso}"
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _to_utc_dt(val: str | datetime) -> datetime:
    dt = datetime.fromisoformat(val) if isinstance(val, str) else val
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_confirmed_session(session: dict[str, Any]) -> bool:
    """Returns True if a Kratos session is user-confirmed or manually logged."""
    extra = session.get("extra") or {}
    return (
        extra.get("confirmed") is True
        or extra.get("inferred") is False
        or session.get("routine_id") is not None
        or len(session.get("items") or []) > 0
    )


def _is_dismissed_tombstone(session: dict[str, Any]) -> bool:
    """Returns True if a soft-deleted Kratos session represents a dismissed or split workout."""
    if session.get("deleted_at") is None:
        return False
    extra = session.get("extra") or {}
    return (
        extra.get("dismissed") is True
        or bool(extra.get("split_into_ids"))
        or extra.get("inferred") is True
    )


def _is_telemetry_locked(session: dict[str, Any]) -> bool:
    """Returns True when a session's telemetry was manually locked or created via split."""
    extra = session.get("extra") or {}
    return (
        extra.get("telemetry_locked") is True
        or extra.get("manual_split") is True
        or bool(extra.get("split_from_id"))
    )


def _session_window_utc(
    session: dict[str, Any],
    default_start: datetime,
    default_end: datetime,
    default_duration_mins: int,
    is_exact_id: bool,
) -> tuple[datetime, datetime] | None:
    s_start_raw = session.get("started_at")
    if not s_start_raw:
        return (default_start, default_end) if is_exact_id else None
    try:
        s_start_utc = _to_utc_dt(s_start_raw)
        s_end_raw = session.get("ended_at")
        if s_end_raw:
            return s_start_utc, _to_utc_dt(s_end_raw)
        dur = (session.get("extra") or {}).get("duration_minutes") or default_duration_mins or 30
        return s_start_utc, s_start_utc + timedelta(minutes=dur)
    except Exception:
        return (default_start, default_end) if is_exact_id else None


def _record_session_match(
    acc: _ScanAccumulator,
    session: dict[str, Any],
    s_id_str: str | None,
    window: tuple[datetime, datetime],
    overlap: float,
    is_exact_id: bool,
) -> None:
    if session.get("deleted_at") is not None:
        if _is_dismissed_tombstone(session) and overlap >= acc.max_tombstone_overlap:
            acc.max_tombstone_overlap = overlap
            acc.best_tombstone = session
        return

    if _is_confirmed_session(session):
        candidate = _OverlapCandidate(session, window[0], window[1], overlap, is_exact_id)
        acc.confirmed.append(candidate)
        if is_exact_id or (not acc.exact_confirmed_matched and overlap > acc.max_confirmed_overlap):
            acc.max_confirmed_overlap = overlap
            acc.best_confirmed = candidate
            if is_exact_id:
                acc.exact_confirmed_matched = True
        return

    if s_id_str and s_id_str not in acc.overlapping_inferred_ids:
        acc.overlapping_inferred_ids.append(s_id_str)
    if is_exact_id:
        acc.exact_inferred = session
    elif overlap > acc.max_inferred_overlap:
        acc.max_inferred_overlap = overlap
        acc.best_inferred = session


def _scan_existing_sessions(
    existing_sessions: list[dict[str, Any]],
    session_id: uuid.UUID,
    w_start_utc: datetime,
    w_end_utc: datetime,
    default_duration_mins: int,
) -> _SyncScanResult:
    """Partitions existing Kratos sessions into tombstones, active confirmed, and active inferred overlaps."""
    acc = _ScanAccumulator()
    for s in existing_sessions:
        s_id_str = str(s["id"]) if s.get("id") else None
        is_exact_id = bool(s_id_str and s_id_str == str(session_id))
        window = _session_window_utc(s, w_start_utc, w_end_utc, default_duration_mins, is_exact_id)
        if window is None:
            continue
        overlap = (min(w_end_utc, window[1]) - max(w_start_utc, window[0])).total_seconds()
        if overlap <= 0 and not is_exact_id:
            continue
        _record_session_match(acc, s, s_id_str, window, max(overlap, 0.0), is_exact_id)

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


def _fetch_kratos_sessions(http_client: httpx.Client, base_url: str) -> list[dict[str, Any]]:
    try:
        resp = http_client.get(
            f"{base_url}/sessions",
            params={"limit": 200, "include_deleted": "true"},
        )
        if resp.status_code == 200:
            s_data = resp.json()
            if isinstance(s_data, list):
                return s_data
            if isinstance(s_data, dict):
                return [s_data]
        elif resp.status_code != 404:
            resp.raise_for_status()
    except httpx.HTTPStatusError:
        raise
    except Exception as exc:
        logger.warning("Unable to fetch Kratos session list for overlap check: %s", exc)
    return []


def _cleanup_stale_inferred_duplicates(
    http_client: httpx.Client,
    base_url: str,
    overlapping_inferred_ids: tuple[str, ...] | list[str],
    target_id: str,
    existing_sessions: list[dict[str, Any]],
) -> None:
    """Deletes stale inferred duplicate sessions after matching a confirmed session."""
    deleted_ids: set[str] = set()
    for dup_id in overlapping_inferred_ids:
        if dup_id == target_id:
            continue
        try:
            del_resp = http_client.delete(f"{base_url}/sessions/{dup_id}")
            if del_resp.status_code in (200, 204, 404):
                deleted_ids.add(dup_id)
            else:
                logger.warning("Unexpected status %s deleting stale inferred Kratos session %s", del_resp.status_code, dup_id)
        except Exception as exc:
            logger.warning("Failed to delete stale inferred Kratos session %s: %s", dup_id, exc)
    if deleted_ids:
        existing_sessions[:] = [
            s for s in existing_sessions if str(s.get("id") or "") not in deleted_ids
        ]


def _apply_metrics_to_extra(
    existing_extra: dict[str, Any],
    metrics: DetectedWorkoutRead,
    *,
    is_confirmed: bool,
) -> dict[str, Any]:
    updated = dict(existing_extra)
    updated["active_energy_kcal"] = metrics.active_energy_kcal
    optional_fields = {
        "avg_hr": metrics.avg_hr_bpm,
        "peak_hr": metrics.peak_hr_bpm,
        "step_count": metrics.step_count,
        "rpe": metrics.rpe,
        "trimp": metrics.trimp,
        "hr_recovery_60s": metrics.hr_recovery_60s,
    }
    for key, val in optional_fields.items():
        if val is not None:
            updated[key] = val
    if metrics.effort_rating and not updated.get("effort_rating"):
        updated["effort_rating"] = metrics.effort_rating
    if metrics.hr_zones:
        updated["hr_zones"] = metrics.hr_zones
    updated["telemetry_source"] = "apple_health_detected"
    updated["telemetry_reconciled"] = True

    if not is_confirmed:
        updated.update({
            "classification": metrics.classification,
            "activity_type": metrics.activity_type,
            "routine_name": metrics.routine_name,
            "duration_minutes": metrics.duration_minutes,
        })
        return updated
    if not updated.get("classification"):
        updated["classification"] = metrics.classification
    if not updated.get("routine_name"):
        updated["routine_name"] = metrics.routine_name
    return updated


def _put_reconciled_session(
    http_client: httpx.Client,
    base_url: str,
    matched_session: dict[str, Any],
    target_id: str,
    metrics: DetectedWorkoutRead,
    *,
    is_confirmed: bool,
    start_iso: str,
    end_iso: str,
) -> dict[str, Any]:
    updated_extra = _apply_metrics_to_extra(
        dict(matched_session.get("extra") or {}),
        metrics,
        is_confirmed=is_confirmed,
    )
    payload = {
        "id": target_id,
        "routine_id": matched_session.get("routine_id"),
        "started_at": matched_session.get("started_at") if is_confirmed else start_iso,
        "ended_at": matched_session.get("ended_at") if is_confirmed else end_iso,
        "notes": (
            matched_session.get("notes")
            if (is_confirmed and matched_session.get("notes"))
            else f"Auto-detected {metrics.routine_name or 'workout'} from Apple Health biometrics"
        ),
        "extra": updated_extra,
        "items": matched_session.get("items") or [],
    }
    put_resp = http_client.put(f"{base_url}/sessions/{target_id}", json=payload)
    put_resp.raise_for_status()
    status_label = "reconciled" if is_confirmed else "synced"
    routine = updated_extra.get("routine_name") or metrics.routine_name
    logger.info("Reconciled detected workout to Kratos: session_id=%s, routine=%s, status=%s", target_id, routine, status_label)
    return {"session_id": target_id, "status": status_label, "routine_name": routine}


def _cleanup_if_needed(
    http_client: httpx.Client,
    base_url: str,
    overlapping_inferred_ids: tuple[str, ...],
    target_id: str,
    existing_sessions: list[dict[str, Any]],
) -> None:
    if overlapping_inferred_ids:
        _cleanup_stale_inferred_duplicates(
            http_client, base_url, overlapping_inferred_ids, target_id, existing_sessions
        )


def _should_skip_single_confirmed(
    primary_extra: dict[str, Any],
    s_duration: float,
    w_duration: float,
    w_energy_kcal: int,
    db: Session | None,
) -> tuple[bool, bool]:
    """Returns (should_skip, is_narrower_window) for an unlocked confirmed session."""
    missing_telemetry = not (
        primary_extra.get("telemetry_reconciled") is True
        and "hr_zones" in primary_extra
        and "rpe" in primary_extra
    )
    is_narrower = s_duration < FULL_COVERAGE_RATIO_MIN * w_duration
    already_complete = not missing_telemetry and (
        is_narrower or (primary_extra.get("active_energy_kcal") or 0) >= w_energy_kcal
    )
    return (already_complete or (is_narrower and db is None)), is_narrower


def _skip_confirmed(
    http_client: httpx.Client,
    base_url: str,
    overlapping_inferred_ids: tuple[str, ...],
    target_id: str,
    existing_sessions: list[dict[str, Any]],
    status: str,
    routine: str | None,
) -> dict[str, Any]:
    _cleanup_if_needed(http_client, base_url, overlapping_inferred_ids, target_id, existing_sessions)
    return {"session_id": target_id, "status": status, "routine_name": routine}


def _reconcile_single_confirmed(
    http_client: httpx.Client,
    base_url: str,
    w: DetectedWorkoutRead,
    best_confirmed: _OverlapCandidate,
    target_id: str,
    primary_extra: dict[str, Any],
    w_duration: float,
    overlapping_inferred_ids: tuple[str, ...],
    existing_sessions: list[dict[str, Any]],
    db: Session | None,
) -> dict[str, Any]:
    s_duration = max(0.0, (best_confirmed.ended_at - best_confirmed.started_at).total_seconds())
    should_skip, is_narrower = _should_skip_single_confirmed(
        primary_extra, s_duration, w_duration, w.active_energy_kcal, db
    )
    if should_skip:
        logger.info("Skipping Kratos session %s: already confirmed and telemetry reconciled", target_id)
        return _skip_confirmed(
            http_client, base_url, overlapping_inferred_ids, target_id,
            existing_sessions, "skipped_confirmed", primary_extra.get("routine_name") or w.routine_name,
        )
    metrics = (
        ExertionAnalyzer(db).analyze_window(best_confirmed.started_at, best_confirmed.ended_at)
        if (is_narrower and db is not None)
        else w
    )
    result = _put_reconciled_session(
        http_client, base_url, best_confirmed.session, target_id, metrics,
        is_confirmed=True, start_iso=w.started_at.isoformat(), end_iso=w.ended_at.isoformat(),
    )
    _cleanup_if_needed(http_client, base_url, overlapping_inferred_ids, target_id, existing_sessions)
    return result


def _handle_confirmed_overlap(
    http_client: httpx.Client,
    base_url: str,
    w: DetectedWorkoutRead,
    session_id: uuid.UUID,
    w_duration: float,
    confirmed: tuple[_OverlapCandidate, ...],
    best_confirmed: _OverlapCandidate,
    overlapping_inferred_ids: tuple[str, ...],
    existing_sessions: list[dict[str, Any]],
    db: Session | None,
) -> dict[str, Any]:
    primary = best_confirmed.session
    target_id = str(primary.get("id") or session_id)
    primary_extra = dict(primary.get("extra") or {})
    routine = primary_extra.get("routine_name") or w.routine_name

    if len(confirmed) >= 2:
        return _skip_confirmed(
            http_client, base_url, overlapping_inferred_ids, target_id,
            existing_sessions, "skipped_split_overlap", routine,
        )
    if _is_telemetry_locked(primary):
        return _skip_confirmed(
            http_client, base_url, overlapping_inferred_ids, target_id,
            existing_sessions, "skipped_confirmed", routine,
        )
    return _reconcile_single_confirmed(
        http_client, base_url, w, best_confirmed, target_id, primary_extra,
        w_duration, overlapping_inferred_ids, existing_sessions, db,
    )


def _create_inferred_session(
    http_client: httpx.Client,
    base_url: str,
    w: DetectedWorkoutRead,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    extra_payload = {
        "inferred": True,
        "source": "apple_health_detected",
        "activity_type": w.activity_type,
        "classification": w.classification,
        "routine_name": w.routine_name,
        "duration_minutes": w.duration_minutes,
        "active_energy_kcal": w.active_energy_kcal,
        "avg_hr": w.avg_hr_bpm,
        "peak_hr": w.peak_hr_bpm,
        "step_count": w.step_count,
        "effort_rating": w.effort_rating,
        "rpe": w.rpe,
        "trimp": w.trimp,
        "hr_zones": w.hr_zones,
        "hr_recovery_60s": w.hr_recovery_60s,
        "confidence": w.confidence,
        "telemetry_source": "apple_health_detected",
        "telemetry_reconciled": True,
    }
    payload = {
        "id": str(session_id),
        "routine_id": None,
        "started_at": w.started_at.isoformat(),
        "ended_at": w.ended_at.isoformat(),
        "notes": f"Auto-detected {w.routine_name or 'workout'} from Apple Health biometrics",
        "extra": extra_payload,
        "items": [],
    }
    put_resp = http_client.put(f"{base_url}/sessions/{session_id}", json=payload)
    put_resp.raise_for_status()
    logger.info("Synced auto-detected workout to Kratos: session_id=%s, routine=%s", session_id, w.routine_name)
    return {"session_id": str(session_id), "status": "synced", "routine_name": w.routine_name}


def _fetch_direct_session(
    http_client: httpx.Client,
    base_url: str,
    session_id: uuid.UUID,
) -> dict[str, Any] | None:
    check_resp = http_client.get(f"{base_url}/sessions/{session_id}")
    if check_resp.status_code == 200:
        return check_resp.json()
    if check_resp.status_code != 404:
        check_resp.raise_for_status()
    return None


def _sync_inferred_or_fallback(
    http_client: httpx.Client,
    base_url: str,
    w: DetectedWorkoutRead,
    session_id: uuid.UUID,
    w_start_utc: datetime,
    w_end_utc: datetime,
    w_duration: float,
    scan: _SyncScanResult,
    existing_sessions: list[dict[str, Any]],
    db: Session | None,
) -> dict[str, Any]:
    start_iso, end_iso = w.started_at.isoformat(), w.ended_at.isoformat()
    matched = scan.exact_inferred or scan.best_inferred or _fetch_direct_session(http_client, base_url, session_id)
    if matched is None:
        return _create_inferred_session(http_client, base_url, w, session_id)
    if _is_confirmed_session(matched):
        win = _session_window_utc(matched, w_start_utc, w_end_utc, w.duration_minutes, True) or (w_start_utc, w_end_utc)
        cand = _OverlapCandidate(matched, win[0], win[1], w_duration, True)
        return _handle_confirmed_overlap(
            http_client, base_url, w, session_id, w_duration, (cand,), cand, (), existing_sessions, db
        )
    return _put_reconciled_session(
        http_client, base_url, matched, str(matched.get("id") or session_id), w,
        is_confirmed=False, start_iso=start_iso, end_iso=end_iso,
    )


def _sync_single_workout(
    http_client: httpx.Client,
    base_url: str,
    w: DetectedWorkoutRead,
    existing_sessions: list[dict[str, Any]],
    db: Session | None,
) -> dict[str, Any]:
    session_id = generate_session_id(w.started_at.isoformat(), w.ended_at.isoformat())
    w_start_utc, w_end_utc = _to_utc_dt(w.started_at), _to_utc_dt(w.ended_at)
    w_duration = max(1.0, (w_end_utc - w_start_utc).total_seconds())
    scan = _scan_existing_sessions(existing_sessions, session_id, w_start_utc, w_end_utc, w.duration_minutes)

    if (
        scan.best_tombstone is not None
        and scan.max_tombstone_overlap >= TOMBSTONE_OVERLAP_THRESHOLD * w_duration
        and scan.max_confirmed_overlap <= scan.max_tombstone_overlap
    ):
        tombstone_id = str(scan.best_tombstone.get("id") or session_id)
        if scan.confirmed:
            _cleanup_if_needed(http_client, base_url, scan.overlapping_inferred_ids, tombstone_id, existing_sessions)
        return {"session_id": tombstone_id, "status": "skipped_dismissed", "routine_name": w.routine_name}

    if scan.confirmed and scan.best_confirmed is not None:
        return _handle_confirmed_overlap(
            http_client, base_url, w, session_id, w_duration,
            scan.confirmed, scan.best_confirmed, scan.overlapping_inferred_ids, existing_sessions, db,
        )

    return _sync_inferred_or_fallback(
        http_client, base_url, w, session_id, w_start_utc, w_end_utc, w_duration, scan, existing_sessions, db
    )


def sync_detected_workouts_to_kratos(
    workouts: list[DetectedWorkoutRead],
    kratos_url: str | None = None,
    client: httpx.Client | None = None,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    """Idempotently syncs detected workouts to Kratos /sessions/{id} with split & tombstone guards."""
    if not workouts:
        return []

    base_url = (kratos_url or settings.kratos_api_url or "").rstrip("/")
    if not base_url:
        logger.warning("Kratos API URL not configured; skipping workout sync")
        return []

    results: list[dict[str, Any]] = []
    owns_client = client is None
    http_client = client or httpx.Client(timeout=10.0)
    try:
        existing_sessions = _fetch_kratos_sessions(http_client, base_url)
        for w in workouts:
            results.append(_sync_single_workout(http_client, base_url, w, existing_sessions, db))
    finally:
        if owns_client:
            http_client.close()

    return results
