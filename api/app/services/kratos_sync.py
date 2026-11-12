"""Synchronization bridge for auto-detected workout sessions to Kratos."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.config import settings
from app.schemas import DetectedWorkoutRead

logger = logging.getLogger(__name__)


def generate_session_id(started_at_iso: str, ended_at_iso: str) -> uuid.UUID:
    """Generates a deterministic UUIDv5 for an inferred workout window."""
    seed = f"kratos:inferred-session:{started_at_iso}:{ended_at_iso}"
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _to_utc_dt(val: str | datetime) -> datetime:
    if isinstance(val, str):
        dt = datetime.fromisoformat(val)
    else:
        dt = val
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def sync_detected_workouts_to_kratos(
    workouts: list[DetectedWorkoutRead],
    kratos_url: str | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Idempotently syncs detected workouts to Kratos /sessions/{id}.

    1. Derives a deterministic UUIDv5 per workout window to ensure repeated calls are idempotent.
    2. Queries Kratos for existing/confirmed sessions in candidate window to prevent duplicate
       suggestions and reconcile biometrics/telemetry onto existing sessions.
    3. Upserts via PUT /sessions/{id} with structured inferred extra metadata when new.
    """
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
        # Query Kratos for existing sessions to guard against overlapping duplicates
        existing_sessions: list[dict[str, Any]] = []
        try:
            sessions_resp = http_client.get(f"{base_url}/sessions", params={"limit": 200})
            if sessions_resp.status_code == 200:
                s_data = sessions_resp.json()
                if isinstance(s_data, list):
                    existing_sessions = s_data
                elif isinstance(s_data, dict):
                    existing_sessions = [s_data]
            elif sessions_resp.status_code != 404:
                sessions_resp.raise_for_status()
        except httpx.HTTPStatusError:
            raise
        except Exception as exc:
            logger.warning("Unable to fetch Kratos session list for overlap check: %s", exc)

        for w in workouts:
            start_iso = w.started_at.isoformat()
            end_iso = w.ended_at.isoformat()
            session_id = generate_session_id(start_iso, end_iso)

            w_start_utc = _to_utc_dt(w.started_at)
            w_end_utc = _to_utc_dt(w.ended_at)

            # Find matching or overlapping session in Kratos
            matched_session: dict[str, Any] | None = None
            max_overlap_s = 0.0

            for s in existing_sessions:
                s_id = s.get("id")
                if s_id and str(s_id) == str(session_id):
                    matched_session = s
                    break

                s_start_raw = s.get("started_at")
                if not s_start_raw:
                    continue
                try:
                    s_start_utc = _to_utc_dt(s_start_raw)
                    s_end_raw = s.get("ended_at")
                    if s_end_raw:
                        s_end_utc = _to_utc_dt(s_end_raw)
                    else:
                        dur = (s.get("extra") or {}).get("duration_minutes") or w.duration_minutes or 30
                        s_end_utc = s_start_utc + timedelta(minutes=dur)

                    overlap = (min(w_end_utc, s_end_utc) - max(w_start_utc, s_start_utc)).total_seconds()
                    if overlap > max_overlap_s:
                        max_overlap_s = overlap
                        matched_session = s
                except Exception:
                    continue

            # Fallback direct lookup by deterministic UUID if not found in list
            if matched_session is None:
                check_resp = http_client.get(f"{base_url}/sessions/{session_id}")
                if check_resp.status_code == 200:
                    matched_session = check_resp.json()
                elif check_resp.status_code not in (404, 200):
                    check_resp.raise_for_status()

            if matched_session is not None:
                target_id = str(matched_session.get("id") or session_id)
                existing_extra = dict(matched_session.get("extra") or {})
                is_confirmed = (
                    existing_extra.get("confirmed") is True
                    or existing_extra.get("inferred") is False
                    or matched_session.get("routine_id") is not None
                    or len(matched_session.get("items") or []) > 0
                )

                # If already reconciled and confirmed, skip to avoid redundant writes
                if is_confirmed and existing_extra.get("telemetry_reconciled") is True:
                    logger.info(
                        "Skipping Kratos session %s: already confirmed and telemetry reconciled",
                        target_id,
                    )
                    results.append({
                        "session_id": target_id,
                        "status": "skipped_confirmed",
                        "routine_name": existing_extra.get("routine_name") or w.routine_name,
                    })
                    continue

                # Reconcile biometrics telemetry onto existing session
                existing_extra["active_energy_kcal"] = w.active_energy_kcal
                if w.avg_hr_bpm is not None:
                    existing_extra["avg_hr"] = w.avg_hr_bpm
                if w.peak_hr_bpm is not None:
                    existing_extra["peak_hr"] = w.peak_hr_bpm
                if w.step_count is not None:
                    existing_extra["step_count"] = w.step_count
                existing_extra["telemetry_source"] = "apple_health_detected"
                existing_extra["telemetry_reconciled"] = True
                if not is_confirmed:
                    existing_extra["classification"] = w.classification
                    existing_extra["activity_type"] = w.activity_type
                    existing_extra["routine_name"] = w.routine_name
                    existing_extra["duration_minutes"] = w.duration_minutes
                else:
                    if not existing_extra.get("classification"):
                        existing_extra["classification"] = w.classification
                    if not existing_extra.get("routine_name"):
                        existing_extra["routine_name"] = w.routine_name

                reconciled_payload = {
                    "id": target_id,
                    "routine_id": matched_session.get("routine_id"),
                    "started_at": matched_session.get("started_at") if is_confirmed else start_iso,
                    "ended_at": matched_session.get("ended_at") if is_confirmed else end_iso,
                    "notes": (
                        matched_session.get("notes")
                        if (is_confirmed and matched_session.get("notes"))
                        else f"Auto-detected {w.routine_name or 'workout'} from Apple Health biometrics"
                    ),
                    "extra": existing_extra,
                    "items": matched_session.get("items") or [],
                }
                put_resp = http_client.put(f"{base_url}/sessions/{target_id}", json=reconciled_payload)
                put_resp.raise_for_status()

                status_label = "reconciled" if is_confirmed else "synced"
                logger.info(
                    "Reconciled detected workout to Kratos: session_id=%s, routine=%s, status=%s",
                    target_id,
                    existing_extra.get("routine_name") or w.routine_name,
                    status_label,
                )
                results.append({
                    "session_id": target_id,
                    "status": status_label,
                    "routine_name": existing_extra.get("routine_name") or w.routine_name,
                })
                continue

            # New session - insert inferred workout suggestion
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
                "confidence": w.confidence,
                "telemetry_source": "apple_health_detected",
                "telemetry_reconciled": True,
            }

            payload = {
                "id": str(session_id),
                "routine_id": None,
                "started_at": start_iso,
                "ended_at": end_iso,
                "notes": f"Auto-detected {w.routine_name or 'workout'} from Apple Health biometrics",
                "extra": extra_payload,
                "items": [],
            }

            put_resp = http_client.put(f"{base_url}/sessions/{session_id}", json=payload)
            put_resp.raise_for_status()
            logger.info(
                "Synced auto-detected workout to Kratos: session_id=%s, routine=%s",
                session_id,
                w.routine_name,
            )
            results.append({
                "session_id": str(session_id),
                "status": "synced",
                "routine_name": w.routine_name,
            })
    finally:
        if owns_client:
            http_client.close()

    return results
