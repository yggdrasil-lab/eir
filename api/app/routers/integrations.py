"""Health integrations router for Withings scales and Apple Watch telemetry.

Receives webhook payloads from Withings Cloud and iOS Shortcuts (HealthKit),
persisting biometrics and offering optional forward routing to Kratos.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import BiometricMeasurement
from app.schemas import (
    AppleBiometricsPayload,
    AppleWorkoutPayload,
    BiometricRead,
    WithingsMeasurementPayload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrations", tags=["integrations"])


def verify_webhook_token(
    authorization: str | None = Header(default=None),
    x_webhook_secret: str | None = Header(default=None),
    secret: str | None = Query(default=None),
) -> None:
    """Validate webhook secret from header or query param.

    If EIR_WEBHOOK_SECRET or EIR_MCP_TOKEN is configured, machine-to-machine
    requests must present a matching secret.
    """
    expected = settings.webhook_secret or settings.mcp_token
    if not expected:
        return

    token = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
        elif len(parts) == 1:
            token = parts[0]
    elif x_webhook_secret:
        token = x_webhook_secret
    elif secret:
        token = secret

    if not token or token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing integration webhook token",
        )


# ------------------------------------------------------------------ Apple Health
@router.post(
    "/apple-health/workout",
    response_model=BiometricRead,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest workout completed on Apple Watch via iOS Shortcuts",
)
async def ingest_apple_workout(
    payload: AppleWorkoutPayload,
    db: Session = Depends(get_db),
    _: None = Depends(verify_webhook_token),
) -> BiometricMeasurement:
    """Records Apple Watch workout telemetry and optionally forwards to Kratos."""
    now = datetime.now(timezone.utc)
    ended_at = payload.ended_at or now
    started_at = payload.started_at or (ended_at - timedelta(minutes=float(payload.duration_minutes or 30)))

    duration_s = payload.duration_seconds
    if duration_s is None and payload.duration_minutes is not None:
        duration_s = int(payload.duration_minutes * 60)
    elif duration_s is None and started_at and ended_at:
        duration_s = max(0, int((ended_at - started_at).total_seconds()))

    workout_extra = {
        "workout_type": payload.workout_type,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": duration_s,
        "avg_heart_rate_bpm": payload.avg_heart_rate_bpm,
        "max_heart_rate_bpm": payload.max_heart_rate_bpm,
        "distance_meters": float(payload.distance_meters) if payload.distance_meters else None,
        "pool_length_meters": float(payload.pool_length_meters) if payload.pool_length_meters else None,
        "strokes": payload.strokes,
        **payload.extra,
    }

    # Idempotent deduplication:
    # If explicit ended_at was given, match by exact ±60s window.
    # If ended_at was omitted (periodic sync of today's exercise), match any workout for today.
    if payload.ended_at:
        match_stmt = select(BiometricMeasurement).where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.source == "apple_health_workout",
            BiometricMeasurement.measured_at >= ended_at - timedelta(seconds=60),
            BiometricMeasurement.measured_at <= ended_at + timedelta(seconds=60),
        )
    else:
        # Match within the same calendar day (UTC/local boundary)
        start_of_day = ended_at.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        match_stmt = select(BiometricMeasurement).where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.source == "apple_health_workout",
            BiometricMeasurement.measured_at >= start_of_day,
            BiometricMeasurement.measured_at < end_of_day,
        )

    measurement = db.scalar(match_stmt.order_by(BiometricMeasurement.measured_at.desc()).limit(1))

    if measurement:
        # Idempotent update in-place if already recorded
        measurement.resting_hr_bpm = payload.avg_heart_rate_bpm
        measurement.active_energy_kcal = int(payload.active_energy_kcal) if payload.active_energy_kcal else None
        measurement.notes = payload.notes or f"Workout: {payload.workout_type}"
        measurement.extra = workout_extra
    else:
        measurement = BiometricMeasurement(
            id=uuid.uuid4(),
            measured_at=ended_at,
            source="apple_health_workout",
            resting_hr_bpm=payload.avg_heart_rate_bpm,
            active_energy_kcal=int(payload.active_energy_kcal) if payload.active_energy_kcal else None,
            notes=payload.notes or f"Workout: {payload.workout_type}",
            extra=workout_extra,
        )
        db.add(measurement)

    db.commit()
    db.refresh(measurement)

    # Optional forwarding to Kratos API over Swarm overlay network
    if payload.forward_to_kratos and settings.kratos_api_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {}
                if settings.mcp_token:
                    headers["Authorization"] = f"Bearer {settings.mcp_token}"
                session_payload = {
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "notes": payload.notes,
                    "extra": {
                        "source": "apple_health",
                        "session_type": "sport",
                        "activity_name": payload.workout_type,
                        "duration_minutes": round(duration_s / 60) if duration_s else None,
                        "active_energy_kcal": float(payload.active_energy_kcal) if payload.active_energy_kcal else None,
                    },
                }
                res = await client.post(f"{settings.kratos_api_url}/sessions", json=session_payload, headers=headers)
                if res.status_code not in (200, 201):
                    logger.warning("Failed to forward workout to Kratos: status=%s", res.status_code)
        except Exception as exc:
            logger.warning("Error forwarding workout to Kratos: %s", exc)

    return measurement


@router.post(
    "/apple-health/biometrics",
    response_model=BiometricRead,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest morning or daily biometrics from Apple Health via iOS Shortcuts",
)
def ingest_apple_biometrics(
    payload: AppleBiometricsPayload,
    db: Session = Depends(get_db),
    _: None = Depends(verify_webhook_token),
) -> BiometricMeasurement:
    """Ingests Apple Health vitals: HRV, resting heart rate, sleep duration, and weight."""
    # Idempotent deduplication: match on same source within 60s window
    measurement = db.scalar(
        select(BiometricMeasurement).where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.source == "apple_health",
            BiometricMeasurement.measured_at >= payload.measured_at - timedelta(seconds=60),
            BiometricMeasurement.measured_at <= payload.measured_at + timedelta(seconds=60),
        )
    )

    if measurement:
        if payload.weight_kg is not None:
            measurement.weight_kg = payload.weight_kg
        if payload.body_fat_pct is not None:
            measurement.body_fat_pct = payload.body_fat_pct
        if payload.resting_hr_bpm is not None:
            measurement.resting_hr_bpm = payload.resting_hr_bpm
        if payload.hrv_ms is not None:
            measurement.hrv_ms = payload.hrv_ms
        if payload.sleep_duration_s is not None:
            measurement.sleep_duration_s = payload.sleep_duration_s
        if payload.sleep_deep_s is not None:
            measurement.sleep_deep_s = payload.sleep_deep_s
        if payload.sleep_rem_s is not None:
            measurement.sleep_rem_s = payload.sleep_rem_s
        if payload.sleep_score is not None:
            measurement.sleep_score = payload.sleep_score
        if payload.step_count is not None:
            measurement.step_count = payload.step_count
        if payload.active_energy_kcal is not None:
            measurement.active_energy_kcal = payload.active_energy_kcal
        if payload.notes:
            measurement.notes = payload.notes
        if payload.extra:
            measurement.extra = {**measurement.extra, **payload.extra}
    else:
        measurement = BiometricMeasurement(
            id=uuid.uuid4(),
            measured_at=payload.measured_at,
            source="apple_health",
            weight_kg=payload.weight_kg,
            body_fat_pct=payload.body_fat_pct,
            resting_hr_bpm=payload.resting_hr_bpm,
            hrv_ms=payload.hrv_ms,
            sleep_duration_s=payload.sleep_duration_s,
            sleep_deep_s=payload.sleep_deep_s,
            sleep_rem_s=payload.sleep_rem_s,
            sleep_score=payload.sleep_score,
            step_count=payload.step_count,
            active_energy_kcal=payload.active_energy_kcal,
            notes=payload.notes,
            extra=payload.extra,
        )
        db.add(measurement)

    db.commit()
    db.refresh(measurement)
    return measurement


# --------------------------------------------------------------------- Withings
@router.api_route(
    "/withings/webhook",
    methods=["GET", "HEAD"],
    summary="Withings subscription verification challenge endpoint",
)
def withings_webhook_verification():
    """Responds with 200 OK to Withings developer verification probes."""
    return {"status": "ok", "service": "eir-withings"}


@router.post(
    "/withings/webhook",
    summary="Withings weigh-in notification callback",
)
async def withings_webhook_callback(
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_webhook_token),
):
    """Receives Withings notifications when a user steps on the scale."""
    payload = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
    else:
        form = await request.form()
        payload = dict(form)

    measurement = BiometricMeasurement(
        id=uuid.uuid4(),
        measured_at=datetime.now(timezone.utc),
        source="withings_webhook",
        notes=f"Notification: appli={payload.get('appli')}, userid={payload.get('userid')}",
        extra=payload,
    )
    db.add(measurement)
    db.commit()

    return {"status": "ok", "received": True}


@router.post(
    "/withings/measurements",
    response_model=BiometricRead,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest body metrics from Withings Body Smart scale",
)
def ingest_withings_measurement(
    payload: WithingsMeasurementPayload,
    db: Session = Depends(get_db),
    _: None = Depends(verify_webhook_token),
) -> BiometricMeasurement:
    """Ingests weight, body fat %, muscle mass, and water % from Withings scale."""
    # Idempotent deduplication for Withings scale measurements
    measurement = db.scalar(
        select(BiometricMeasurement).where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.source == "withings",
            BiometricMeasurement.measured_at >= payload.measured_at - timedelta(seconds=60),
            BiometricMeasurement.measured_at <= payload.measured_at + timedelta(seconds=60),
        )
    )

    if measurement:
        measurement.weight_kg = payload.weight_kg
        if payload.body_fat_pct is not None:
            measurement.body_fat_pct = payload.body_fat_pct
        if payload.muscle_mass_kg is not None:
            measurement.muscle_mass_kg = payload.muscle_mass_kg
        if payload.water_pct is not None:
            measurement.water_pct = payload.water_pct
        if payload.bone_mass_kg is not None:
            measurement.bone_mass_kg = payload.bone_mass_kg
        if payload.resting_hr_bpm is not None:
            measurement.resting_hr_bpm = payload.resting_hr_bpm
        if payload.notes:
            measurement.notes = payload.notes
        if payload.extra:
            measurement.extra = {**measurement.extra, **payload.extra}
    else:
        measurement = BiometricMeasurement(
            id=uuid.uuid4(),
            measured_at=payload.measured_at,
            source="withings",
            weight_kg=payload.weight_kg,
            body_fat_pct=payload.body_fat_pct,
            muscle_mass_kg=payload.muscle_mass_kg,
            water_pct=payload.water_pct,
            bone_mass_kg=payload.bone_mass_kg,
            resting_hr_bpm=payload.resting_hr_bpm,
            notes=payload.notes,
            extra=payload.extra,
        )
        db.add(measurement)

    db.commit()
    db.refresh(measurement)
    return measurement
