"""Reconciler facade orchestrating Kratos session deduplication, guards, and telemetry sync."""

from __future__ import annotations

from collections.abc import Sequence
import logging
import uuid

from sqlalchemy.orm import Session

from app.config import settings
from app.exertion import ExertionAnalyzer
from app.kratos_sync._http_store import HttpKratosSessionStore
from app.kratos_sync._overlap import (
    _OverlapCandidate,
    scan_existing_sessions,
    session_window_utc,
    to_utc_dt,
)
from app.kratos_sync._payloads import build_inferred_write, build_reconciled_write
from app.kratos_sync._types import (
    FULL_COVERAGE_RATIO_MIN,
    TOMBSTONE_OVERLAP_THRESHOLD,
    KratosSession,
    KratosSessionStore,
    KratosSessionWrite,
    KratosStoreError,
    SyncBatchResult,
    SyncOutcome,
    SyncStatus,
    WindowMetricsPort,
    generate_session_id,
)
from app.schemas import DetectedWorkoutRead

logger = logging.getLogger(__name__)


def _upsert_working_session(
    working_sessions: list[KratosSession],
    write: KratosSessionWrite,
) -> None:
    updated = write.to_session()
    for idx, existing in enumerate(working_sessions):
        if existing.id == updated.id:
            working_sessions[idx] = updated
            return
    working_sessions.append(updated)


def _cleanup_stale_inferred(
    store: KratosSessionStore,
    overlapping_inferred_ids: tuple[str, ...],
    target_id: str,
    working_sessions: list[KratosSession],
) -> tuple[str, ...]:
    if not overlapping_inferred_ids:
        return ()
    deleted_ids: list[str] = []
    for dup_id in overlapping_inferred_ids:
        if dup_id == target_id:
            continue
        try:
            if store.delete_session(dup_id):
                deleted_ids.append(dup_id)
        except KratosStoreError as exc:
            logger.warning("Failed to delete stale inferred Kratos session %s: %s", dup_id, exc)
    if deleted_ids:
        deleted_set = set(deleted_ids)
        working_sessions[:] = [s for s in working_sessions if s.id not in deleted_set]
    return tuple(deleted_ids)


class KratosReconciler:
    """Deep module synchronizing detected Apple Health workouts to Kratos sessions."""

    def __init__(
        self,
        db: Session | None = None,
        *,
        store: KratosSessionStore | None = None,
        window_metrics: WindowMetricsPort | None = None,
        kratos_url: str | None = None,
    ) -> None:
        self._store = store
        if window_metrics is not None:
            self._window_metrics: WindowMetricsPort | None = window_metrics
        elif db is not None:
            self._window_metrics = ExertionAnalyzer(db)
        else:
            self._window_metrics = None
        raw_url = settings.kratos_api_url if kratos_url is None else kratos_url
        self._kratos_url = (raw_url or "").strip().rstrip("/")

    def sync(self, workouts: Sequence[DetectedWorkoutRead]) -> SyncBatchResult:
        """Idempotently reconciles detected workouts against Kratos sessions."""
        if self._store is None and not self._kratos_url:
            logger.warning("Kratos API URL not configured; skipping workout sync")
            return SyncBatchResult(configured=False, outcomes=())
        if not workouts:
            return SyncBatchResult(configured=True, outcomes=())

        if self._store is not None:
            return self._sync_with_store(self._store, workouts)
        with HttpKratosSessionStore(self._kratos_url) as http_store:
            return self._sync_with_store(http_store, workouts)

    def _sync_with_store(
        self,
        store: KratosSessionStore,
        workouts: Sequence[DetectedWorkoutRead],
    ) -> SyncBatchResult:
        working_sessions = list(store.list_sessions(limit=200, include_deleted=True))
        outcomes = tuple(
            self._sync_single_workout(store, workout, working_sessions) for workout in workouts
        )
        return SyncBatchResult(configured=True, outcomes=outcomes)

    def _sync_single_workout(
        self,
        store: KratosSessionStore,
        workout: DetectedWorkoutRead,
        working_sessions: list[KratosSession],
    ) -> SyncOutcome:
        session_id = generate_session_id(
            workout.started_at.isoformat(), workout.ended_at.isoformat()
        )
        w_start_utc, w_end_utc = to_utc_dt(workout.started_at), to_utc_dt(workout.ended_at)
        w_duration = max(1.0, (w_end_utc - w_start_utc).total_seconds())
        scan = scan_existing_sessions(
            working_sessions, session_id, w_start_utc, w_end_utc, workout.duration_minutes
        )

        if (
            scan.best_tombstone is not None
            and scan.max_tombstone_overlap >= TOMBSTONE_OVERLAP_THRESHOLD * w_duration
            and scan.max_confirmed_overlap <= scan.max_tombstone_overlap
        ):
            tombstone_id = scan.best_tombstone.id or str(session_id)
            deleted_ids = (
                _cleanup_stale_inferred(
                    store, scan.overlapping_inferred_ids, tombstone_id, working_sessions
                )
                if scan.confirmed
                else ()
            )
            return SyncOutcome(
                session_id=tombstone_id,
                status="skipped_dismissed",
                routine_name=workout.routine_name,
                deleted_duplicate_ids=deleted_ids,
            )

        if scan.confirmed and scan.best_confirmed is not None:
            return self._handle_confirmed_overlap(
                store,
                workout,
                session_id,
                w_duration,
                scan.confirmed,
                scan.best_confirmed,
                scan.overlapping_inferred_ids,
                working_sessions,
            )

        return self._sync_inferred_or_fallback(
            store, workout, session_id, w_start_utc, w_end_utc, w_duration, scan, working_sessions
        )

    def _handle_confirmed_overlap(
        self,
        store: KratosSessionStore,
        workout: DetectedWorkoutRead,
        session_id: uuid.UUID,
        w_duration: float,
        confirmed: tuple[_OverlapCandidate, ...],
        best_confirmed: _OverlapCandidate,
        overlapping_inferred_ids: tuple[str, ...],
        working_sessions: list[KratosSession],
    ) -> SyncOutcome:
        primary = best_confirmed.session
        target_id = primary.id or str(session_id)
        routine = primary.extra.get("routine_name") or workout.routine_name

        if len(confirmed) >= 2:
            deleted_ids = _cleanup_stale_inferred(
                store, overlapping_inferred_ids, target_id, working_sessions
            )
            return SyncOutcome(target_id, "skipped_split_overlap", routine, deleted_ids)
        if primary.is_telemetry_locked:
            deleted_ids = _cleanup_stale_inferred(
                store, overlapping_inferred_ids, target_id, working_sessions
            )
            return SyncOutcome(target_id, "skipped_confirmed", routine, deleted_ids)
        return self._reconcile_single_confirmed(
            store,
            workout,
            best_confirmed,
            target_id,
            routine,
            w_duration,
            overlapping_inferred_ids,
            working_sessions,
        )

    def _reconcile_single_confirmed(
        self,
        store: KratosSessionStore,
        workout: DetectedWorkoutRead,
        best_confirmed: _OverlapCandidate,
        target_id: str,
        routine: str | None,
        w_duration: float,
        overlapping_inferred_ids: tuple[str, ...],
        working_sessions: list[KratosSession],
    ) -> SyncOutcome:
        primary = best_confirmed.session
        s_duration = max(0.0, (best_confirmed.ended_at - best_confirmed.started_at).total_seconds())
        is_narrower = s_duration < FULL_COVERAGE_RATIO_MIN * w_duration
        already_complete = primary.has_complete_telemetry and (
            is_narrower
            or (primary.extra.get("active_energy_kcal") or 0) >= workout.active_energy_kcal
        )
        if already_complete or (is_narrower and self._window_metrics is None):
            logger.info(
                "Skipping Kratos session %s: already confirmed and telemetry reconciled",
                target_id,
            )
            deleted_ids = _cleanup_stale_inferred(
                store, overlapping_inferred_ids, target_id, working_sessions
            )
            return SyncOutcome(target_id, "skipped_confirmed", routine, deleted_ids)

        metrics = (
            self._window_metrics.analyze_window(best_confirmed.started_at, best_confirmed.ended_at)
            if (is_narrower and self._window_metrics is not None)
            else workout
        )
        write = build_reconciled_write(
            primary,
            target_id,
            metrics,
            is_confirmed=True,
            start_iso=workout.started_at.isoformat(),
            end_iso=workout.ended_at.isoformat(),
        )
        return self._commit_write(
            store,
            write,
            status="reconciled",
            fallback_routine=metrics.routine_name,
            working_sessions=working_sessions,
            overlapping_inferred_ids=overlapping_inferred_ids,
        )

    def _sync_inferred_or_fallback(
        self,
        store: KratosSessionStore,
        workout: DetectedWorkoutRead,
        session_id: uuid.UUID,
        w_start_utc: Any,
        w_end_utc: Any,
        w_duration: float,
        scan: Any,
        working_sessions: list[KratosSession],
    ) -> SyncOutcome:
        matched = scan.exact_inferred or scan.best_inferred or store.get_session(str(session_id))
        if matched is None:
            write = build_inferred_write(workout, session_id)
            return self._commit_write(
                store,
                write,
                status="synced",
                fallback_routine=workout.routine_name,
                working_sessions=working_sessions,
                overlapping_inferred_ids=(),
            )
        if matched.is_confirmed:
            win = session_window_utc(
                matched, w_start_utc, w_end_utc, workout.duration_minutes, True
            ) or (w_start_utc, w_end_utc)
            cand = _OverlapCandidate(matched, win[0], win[1], w_duration, True)
            return self._handle_confirmed_overlap(
                store, workout, session_id, w_duration, (cand,), cand, (), working_sessions
            )

        target_id = matched.id or str(session_id)
        write = build_reconciled_write(
            matched,
            target_id,
            workout,
            is_confirmed=False,
            start_iso=workout.started_at.isoformat(),
            end_iso=workout.ended_at.isoformat(),
        )
        return self._commit_write(
            store,
            write,
            status="synced",
            fallback_routine=workout.routine_name,
            working_sessions=working_sessions,
            overlapping_inferred_ids=(),
        )

    @staticmethod
    def _commit_write(
        store: KratosSessionStore,
        write: KratosSessionWrite,
        *,
        status: SyncStatus,
        fallback_routine: str | None,
        working_sessions: list[KratosSession],
        overlapping_inferred_ids: tuple[str, ...],
    ) -> SyncOutcome:
        store.upsert_session(write)
        _upsert_working_session(working_sessions, write)
        deleted_ids = _cleanup_stale_inferred(
            store, overlapping_inferred_ids, write.id, working_sessions
        )
        routine = write.extra.get("routine_name") or fallback_routine
        logger.info(
            "Reconciled detected workout to Kratos: session_id=%s, routine=%s, status=%s",
            write.id,
            routine,
            status,
        )
        return SyncOutcome(
            session_id=write.id,
            status=status,
            routine_name=routine,
            deleted_duplicate_ids=deleted_ids,
        )
