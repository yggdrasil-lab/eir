"""Health integrations router for Withings scales and Apple Watch telemetry.

Receives webhook payloads from Withings Cloud and iOS Shortcuts (HealthKit),
persisting biometrics and offering optional forward routing to Kratos.
"""

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import BiometricMeasurement, HealthSample
from app.schemas import (
    AppleBiometricsPayload,
    AppleHealthComprehensiveSyncPayload,
    AppleHealthSyncResponse,
    AppleWorkoutPayload,
    BiometricRead,
    HealthSampleBulkPayload,
    HealthSampleBulkResponse,
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


def _normalize_key_dt(dt_: datetime | None) -> str:
    if dt_ is None:
        return ""
    if dt_.tzinfo is not None:
        return dt_.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return dt_.strftime("%Y-%m-%d %H:%M:%S")


@router.post(
    "/apple-health/samples",
    response_model=HealthSampleBulkResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest granular health samples (active energy, steps) from Apple Watch",
)
@router.post(
    "/apple-health/bulk-samples",
    response_model=HealthSampleBulkResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
def ingest_apple_health_samples(
    payload: HealthSampleBulkPayload,
    db: Session = Depends(get_db),
    _: None = Depends(verify_webhook_token),
) -> HealthSampleBulkResponse:
    """Bulk ingest granular time-series samples with idempotent upsert."""
    items = payload.to_sample_items()
    if not items:
        return HealthSampleBulkResponse(
            status="empty",
            ingested=0,
            sample_type=payload.sample_type,
            daily_total=None,
        )

    min_date = min(it.measured_at for it in items)
    max_date = max((it.ended_at or it.measured_at) for it in items)

    # Batch query existing samples in range for fast in-memory upsert
    existing_stmt = select(HealthSample).where(
        HealthSample.deleted_at.is_(None),
        HealthSample.source == payload.source,
        HealthSample.sample_type == payload.sample_type,
        HealthSample.measured_at >= min_date,
        HealthSample.measured_at <= max_date,
    )
    existing_map = {
        (_normalize_key_dt(s.measured_at), _normalize_key_dt(s.ended_at)): s
        for s in db.scalars(existing_stmt)
    }

    ingested_count = 0
    for item in items:
        ended_at = item.ended_at or item.measured_at
        key = (_normalize_key_dt(item.measured_at), _normalize_key_dt(ended_at))
        if key in existing_map:
            existing_sample = existing_map[key]
            existing_sample.value = item.value
            existing_sample.unit = item.unit
            if item.extra:
                existing_sample.extra = item.extra
        else:
            sample = HealthSample(
                id=uuid.uuid4(),
                source=item.source or payload.source,
                sample_type=item.sample_type or payload.sample_type,
                measured_at=item.measured_at,
                ended_at=ended_at,
                value=item.value,
                unit=item.unit or payload.unit,
                external_id=item.external_id,
                extra=item.extra,
            )
            db.add(sample)
            existing_map[key] = sample
        ingested_count += 1

    db.commit()

    # Calculate daily sum for the target date
    ref_date = items[0].measured_at.date()
    start_of_day = datetime.combine(ref_date, time.min).replace(tzinfo=timezone.utc)
    end_of_day = start_of_day + timedelta(days=1)

    sum_stmt = select(func.sum(HealthSample.value)).where(
        HealthSample.deleted_at.is_(None),
        HealthSample.source == payload.source,
        HealthSample.sample_type == payload.sample_type,
        HealthSample.measured_at >= start_of_day,
        HealthSample.measured_at < end_of_day,
    )
    daily_total = db.scalar(sum_stmt)

    return HealthSampleBulkResponse(
        status="ok",
        ingested=ingested_count,
        sample_type=payload.sample_type,
        daily_total=daily_total,
    )


def _sync_daily_rollups(db: Session, target_date: date) -> BiometricMeasurement:
    """Calculates and persists daily rollup metrics from health_sample into biometric_measurement."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=timezone.utc)

    # 1. Latest weight
    w = db.scalar(
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["weight", "weight_kg"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )

    # 2. Latest body fat
    bf = db.scalar(
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["body_fat_pct", "body_fat", "body_fat_percentage"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )

    # 3. Resting HR
    rhr = db.scalar(
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["resting_hr", "resting_heart_rate", "resting_hr_bpm"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )

    # 4. HRV
    hrv = db.scalar(
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["hrv", "heart_rate_variability", "hrv_ms", "hrv_sdnn"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )

    # 5. Sleep duration (sum of seconds)
    sleep_s = db.scalar(
        select(func.sum(HealthSample.value)).where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["sleep", "sleep_duration", "sleep_duration_s", "sleep_analysis"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
    )

    # 6. Active energy
    energy_kcal = db.scalar(
        select(func.sum(HealthSample.value)).where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["active_energy", "active_energy_kcal"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
    )

    # 7. Step count
    steps = db.scalar(
        select(func.sum(HealthSample.value)).where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["step_count", "steps"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
    )

    # Find or create daily biometric_measurement
    measurement = db.scalar(
        select(BiometricMeasurement).where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.source == "apple_health",
            BiometricMeasurement.measured_at >= start_dt,
            BiometricMeasurement.measured_at <= end_dt,
        ).order_by(BiometricMeasurement.measured_at.desc()).limit(1)
    )

    if not measurement:
        measurement = BiometricMeasurement(
            id=uuid.uuid4(),
            measured_at=datetime.combine(target_date, time(8, 0)).replace(tzinfo=timezone.utc),
            source="apple_health",
            notes=f"Apple Health rollup for {target_date.isoformat()}",
        )
        db.add(measurement)

    if w is not None:
        measurement.weight_kg = w
    if bf is not None:
        measurement.body_fat_pct = bf
    if rhr is not None:
        measurement.resting_hr_bpm = int(round(rhr))
    if hrv is not None:
        measurement.hrv_ms = hrv
    if sleep_s is not None:
        measurement.sleep_duration_s = int(round(sleep_s))
    if energy_kcal is not None:
        measurement.active_energy_kcal = int(round(energy_kcal))
    if steps is not None:
        measurement.step_count = int(round(steps))

    return measurement


@router.post(
    "/apple-health/sync",
    response_model=AppleHealthSyncResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Comprehensive high-resolution Apple Health multi-metric synchronization (2-day rollups)",
)
def sync_apple_health_comprehensive(
    payload: AppleHealthComprehensiveSyncPayload,
    db: Session = Depends(get_db),
    _: None = Depends(verify_webhook_token),
) -> AppleHealthSyncResponse:
    """Ingests multi-metric health samples with idempotent upsert and multi-day rollups."""
    items = payload.extract_all_samples()
    if not items:
        return AppleHealthSyncResponse(
            status="empty",
            total_samples_ingested=0,
            metrics_summary={},
            days_processed=[],
        )

    min_date = min(it.measured_at for it in items)
    max_date = max((it.ended_at or it.measured_at) for it in items)

    # Batch query existing samples in range for fast in-memory upsert
    existing_stmt = select(HealthSample).where(
        HealthSample.deleted_at.is_(None),
        HealthSample.source == payload.source,
        HealthSample.measured_at >= min_date,
        HealthSample.measured_at <= max_date,
    )
    existing_map = {
        (s.source, s.sample_type, _normalize_key_dt(s.measured_at), _normalize_key_dt(s.ended_at)): s
        for s in db.scalars(existing_stmt)
    }

    metrics_counts: dict[str, int] = {}
    for item in items:
        st = item.sample_type or "unknown"
        metrics_counts[st] = metrics_counts.get(st, 0) + 1
        ended_at = item.ended_at or item.measured_at
        key = (item.source or payload.source, st, _normalize_key_dt(item.measured_at), _normalize_key_dt(ended_at))

        if key in existing_map:
            existing_sample = existing_map[key]
            existing_sample.value = item.value
            existing_sample.unit = item.unit
            if item.extra:
                existing_sample.extra = item.extra
        else:
            sample = HealthSample(
                id=uuid.uuid4(),
                source=item.source or payload.source,
                sample_type=st,
                measured_at=item.measured_at,
                ended_at=ended_at,
                value=item.value,
                unit=item.unit,
                external_id=item.external_id,
                extra=item.extra,
            )
            db.add(sample)
            existing_map[key] = sample

    db.commit()

    # Recalculate and update daily rollups for each distinct calendar day
    days = sorted(set(it.measured_at.date() for it in items))
    for d in days:
        _sync_daily_rollups(db, d)

    db.commit()

    return AppleHealthSyncResponse(
        status="ok",
        total_samples_ingested=len(items),
        metrics_summary=metrics_counts,
        days_processed=days,
    )


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
