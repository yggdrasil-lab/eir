"""Synchronization bridge for auto-detected workout sessions to Kratos."""

import logging
import uuid
from typing import Any

import httpx

from app.config import settings
from app.schemas import DetectedWorkoutRead

logger = logging.getLogger(__name__)


def generate_session_id(started_at_iso: str, ended_at_iso: str) -> uuid.UUID:
    """Generates a deterministic UUIDv5 for an inferred workout window."""
    seed = f"kratos:inferred-session:{started_at_iso}:{ended_at_iso}"
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def sync_detected_workouts_to_kratos(
    workouts: list[DetectedWorkoutRead],
    kratos_url: str | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Idempotently syncs detected workouts to Kratos /sessions/{id}.

    1. Derives a deterministic UUIDv5 per workout window to ensure repeated calls are idempotent.
    2. Queries Kratos to avoid clobbering workouts already reviewed or confirmed by the user.
    3. Upserts via PUT /sessions/{id} with structured inferred extra metadata.
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
        for w in workouts:
            start_iso = w.started_at.isoformat()
            end_iso = w.ended_at.isoformat()
            session_id = generate_session_id(start_iso, end_iso)

            # Check if session already exists
            check_resp = http_client.get(f"{base_url}/sessions/{session_id}")
            if check_resp.status_code == 200:
                existing_extra = check_resp.json().get("extra") or {}
                if existing_extra.get("confirmed") is True or existing_extra.get("inferred") is False:
                    logger.info(
                        "Skipping Kratos session %s: already confirmed or edited by user",
                        session_id,
                    )
                    results.append({
                        "session_id": str(session_id),
                        "status": "skipped_confirmed",
                        "routine_name": w.routine_name,
                    })
                    continue
            elif check_resp.status_code not in (404, 200):
                check_resp.raise_for_status()

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
