"""Pure payload builders preserving user-authored Kratos metadata and telemetry fields."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
import uuid

from app.kratos_sync._types import KratosSession, KratosSessionWrite
from app.schemas import DetectedWorkoutRead


def apply_metrics_to_extra(
    existing_extra: Mapping[str, Any],
    metrics: DetectedWorkoutRead,
    *,
    is_confirmed: bool,
) -> dict[str, Any]:
    """Merges biometric telemetry into an existing session extra mapping without clobbering user fields."""
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
        updated.update(
            {
                "classification": metrics.classification,
                "activity_type": metrics.activity_type,
                "routine_name": metrics.routine_name,
                "duration_minutes": metrics.duration_minutes,
            }
        )
        return updated
    if not updated.get("classification"):
        updated["classification"] = metrics.classification
    if not updated.get("routine_name"):
        updated["routine_name"] = metrics.routine_name
    return updated


def build_reconciled_write(
    matched_session: KratosSession,
    target_id: str,
    metrics: DetectedWorkoutRead,
    *,
    is_confirmed: bool,
    start_iso: str,
    end_iso: str,
) -> KratosSessionWrite:
    """Builds an immutable KratosSessionWrite reconciling telemetry onto an existing session."""
    updated_extra = apply_metrics_to_extra(
        matched_session.extra,
        metrics,
        is_confirmed=is_confirmed,
    )
    notes = (
        matched_session.notes
        if (is_confirmed and matched_session.notes)
        else f"Auto-detected {metrics.routine_name or 'workout'} from Apple Health biometrics"
    )
    return KratosSessionWrite(
        id=target_id,
        routine_id=matched_session.routine_id,
        started_at=matched_session.started_at if is_confirmed else start_iso,
        ended_at=matched_session.ended_at if is_confirmed else end_iso,
        notes=notes,
        extra=MappingProxyType(updated_extra),
        items=matched_session.items,
    )


def build_inferred_write(
    workout: DetectedWorkoutRead,
    session_id: uuid.UUID,
) -> KratosSessionWrite:
    """Builds an immutable KratosSessionWrite for a newly auto-detected workout."""
    extra_payload = {
        "inferred": True,
        "source": "apple_health_detected",
        "activity_type": workout.activity_type,
        "classification": workout.classification,
        "routine_name": workout.routine_name,
        "duration_minutes": workout.duration_minutes,
        "active_energy_kcal": workout.active_energy_kcal,
        "avg_hr": workout.avg_hr_bpm,
        "peak_hr": workout.peak_hr_bpm,
        "step_count": workout.step_count,
        "effort_rating": workout.effort_rating,
        "rpe": workout.rpe,
        "trimp": workout.trimp,
        "hr_zones": workout.hr_zones,
        "hr_recovery_60s": workout.hr_recovery_60s,
        "confidence": workout.confidence,
        "telemetry_source": "apple_health_detected",
        "telemetry_reconciled": True,
    }
    return KratosSessionWrite(
        id=str(session_id),
        routine_id=None,
        started_at=workout.started_at.isoformat(),
        ended_at=workout.ended_at.isoformat(),
        notes=f"Auto-detected {workout.routine_name or 'workout'} from Apple Health biometrics",
        extra=MappingProxyType(extra_payload),
        items=(),
    )
