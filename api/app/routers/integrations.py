"""Health integrations router for Withings scales and Apple Health telemetry.

Receives webhook payloads from Withings Cloud and iOS Shortcuts (HealthKit),
persisting biometrics and high-resolution health samples.
"""

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import logging
import urllib.parse
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import BiometricMeasurement, HealthSample, OAuthToken
from app.schemas import (
    AppleHealthComprehensiveSyncPayload,
    AppleHealthSyncResponse,
    BiometricRead,
    WithingsAuthStatusResponse,
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
def _normalize_key_dt(dt_: datetime | None) -> str:
    if dt_ is None:
        return ""
    if dt_.tzinfo is not None:
        return dt_.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return dt_.strftime("%Y-%m-%d %H:%M:%S")


def _sync_daily_rollups(db: Session, target_date: date) -> BiometricMeasurement:
    """Calculates and persists daily rollup metrics from health_sample into biometric_measurement."""
    start_dt = datetime.combine(target_date, time.min).replace(tzinfo=settings.tz)
    end_dt = datetime.combine(target_date, time.max).replace(tzinfo=settings.tz)

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

    # 2b. Latest muscle mass
    muscle = db.scalar(
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["muscle_mass_kg", "muscle_mass", "lean_body_mass"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )

    # 2c. Latest water %
    water = db.scalar(
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["water_pct", "water_percentage", "hydration_pct"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )

    # 2d. Latest bone mass
    bone = db.scalar(
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["bone_mass_kg", "bone_mass"]),
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

    # 5. Sleep duration (sum of seconds, stage-aware)
    sleep_samples = db.scalars(
        select(HealthSample).where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["sleep", "sleep_duration", "sleep_duration_s", "sleep_analysis"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
    ).all()

    sleep_total_s = 0
    sleep_deep_s = 0
    sleep_rem_s = 0
    has_sleep = len(sleep_samples) > 0
    for s in sleep_samples:
        stage = str((s.extra or {}).get("stage") or "").strip().lower()
        if stage in ("awake", "inbed", "in_bed", "in bed"):
            continue
        val = int(round(s.value))
        sleep_total_s += val
        if stage == "deep":
            sleep_deep_s += val
        elif stage == "rem":
            sleep_rem_s += val

    # 6. Active energy
    energy_kcal = db.scalar(
        select(func.sum(HealthSample.value)).where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["active_energy", "active_energy_kcal"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
    )

    # 7. Step count with smart multi-device deduplication (Apple Watch vs iPhone)
    step_samples = db.scalars(
        select(HealthSample).where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["step_count", "steps"]),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        ).order_by(HealthSample.measured_at.asc())
    ).all()

    watch_step_samples: list[HealthSample] = []
    phone_step_samples: list[HealthSample] = []
    other_step_samples: list[HealthSample] = []

    for s in step_samples:
        dev = str((s.extra or {}).get("device") or "").lower()
        if "watch" in dev:
            watch_step_samples.append(s)
        elif "phone" in dev:
            phone_step_samples.append(s)
        else:
            other_step_samples.append(s)

    total_steps: Decimal = Decimal(0)
    if watch_step_samples and phone_step_samples:
        total_steps += sum((s.value for s in watch_step_samples), Decimal(0))
        for p in phone_step_samples:
            p_start = p.measured_at
            p_end = p.ended_at if (p.ended_at and p.ended_at > p.measured_at) else p.measured_at + timedelta(seconds=300)
            overlaps = any(
                max(p_start, w.measured_at) <= min(p_end, w.ended_at if (w.ended_at and w.ended_at > w.measured_at) else w.measured_at + timedelta(seconds=300))
                for w in watch_step_samples
            )
            if not overlaps:
                total_steps += p.value
        total_steps += sum((s.value for s in other_step_samples), Decimal(0))
    elif watch_step_samples:
        total_steps = sum((s.value for s in watch_step_samples), Decimal(0)) + sum((s.value for s in other_step_samples), Decimal(0))
    else:
        total_steps = sum((s.value for s in step_samples), Decimal(0))

    steps = total_steps if step_samples else None

    # Find or create daily biometric_measurement (unified across all sources for this date)
    measurement = db.scalar(
        select(BiometricMeasurement).where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.measured_at >= start_dt,
            BiometricMeasurement.measured_at <= end_dt,
        ).order_by(BiometricMeasurement.measured_at.desc()).limit(1)
    )

    if not measurement:
        measurement = BiometricMeasurement(
            id=uuid.uuid4(),
            measured_at=datetime.combine(target_date, time(8, 0)).replace(tzinfo=settings.tz),
            source="daily_rollup",
            notes=f"Daily health rollup for {target_date.isoformat()}",
        )
        db.add(measurement)

    if w is not None:
        measurement.weight_kg = w
    if bf is not None:
        measurement.body_fat_pct = bf
    if muscle is not None:
        measurement.muscle_mass_kg = muscle
    if water is not None:
        measurement.water_pct = water
    if bone is not None:
        measurement.bone_mass_kg = bone
    if rhr is not None:
        measurement.resting_hr_bpm = int(round(rhr))
    if hrv is not None:
        measurement.hrv_ms = hrv
    if has_sleep:
        measurement.sleep_duration_s = sleep_total_s
        if sleep_deep_s > 0:
            measurement.sleep_deep_s = sleep_deep_s
        if sleep_rem_s > 0:
            measurement.sleep_rem_s = sleep_rem_s
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
    extra = getattr(payload, "__pydantic_extra__", {}) or {}
    logger.info("Apple Health sync received. Extra keys: %s", list(extra.keys()))
    for k, v in extra.items():
        val_str = str(v)
        preview = val_str[:120] + "..." if len(val_str) > 120 else val_str
        logger.info("  Key [%s] (type %s): %s", k, type(v).__name__, preview)

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

    # Batch query existing samples in range for fast in-memory upsert (including cross-source scale samples)
    existing_samples_in_range = list(
        db.scalars(
            select(HealthSample).where(
                HealthSample.deleted_at.is_(None),
                HealthSample.measured_at >= min_date - timedelta(seconds=60),
                HealthSample.measured_at <= max_date + timedelta(seconds=60),
            )
        )
    )
    existing_map = {
        (s.source, s.sample_type, _normalize_key_dt(s.measured_at), _normalize_key_dt(s.ended_at)): s
        for s in existing_samples_in_range
    }

    scale_types = {"weight", "weight_kg", "body_fat_pct", "body_fat", "muscle_mass_kg", "water_pct", "bone_mass_kg"}

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
            continue

        # Check cross-source scale deduplication (e.g. sample already exists from Withings within +-60s)
        matched_scale_sample = None
        if st in scale_types:
            for s in existing_samples_in_range:
                if s.sample_type == st and abs((s.measured_at - item.measured_at).total_seconds()) <= 60:
                    matched_scale_sample = s
                    break

        if matched_scale_sample:
            matched_scale_sample.value = item.value
            matched_scale_sample.unit = item.unit
            existing_map[key] = matched_scale_sample
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
            existing_samples_in_range.append(sample)

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


@router.get(
    "/withings/authorize",
    summary="Initiate Withings OAuth2 authorization flow",
)
def withings_authorize(
    format: str | None = Query(default=None, description="Set to 'json' to return authorization URL"),
):
    """Generates Withings OAuth2 authorization URL and redirects the user."""
    if not settings.withings_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Withings client ID (EIR_WITHINGS_CLIENT_ID) is not configured",
        )
    redirect_uri = settings.withings_redirect_uri or "https://eir.tienzo.net/integrations/withings/callback"
    state = uuid.uuid4().hex
    params = {
        "response_type": "code",
        "client_id": settings.withings_client_id,
        "scope": "user.info,user.metrics,user.activity",
        "redirect_uri": redirect_uri,
        "state": state,
    }
    url = f"https://account.withings.com/oauth2_user/authorize2?{urllib.parse.urlencode(params)}"
    if format == "json":
        return {"authorization_url": url, "state": state}
    return RedirectResponse(url=url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get(
    "/withings/callback",
    summary="Withings OAuth2 callback endpoint",
)
async def withings_oauth_callback(
    code: str = Query(..., description="Authorization code from Withings"),
    state: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Exchanges authorization code for access and refresh tokens, saves them, and subscribes to webhooks."""
    if not settings.withings_client_id or not settings.withings_client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Withings credentials (EIR_WITHINGS_CLIENT_ID / EIR_WITHINGS_CLIENT_SECRET) not configured",
        )
    redirect_uri = settings.withings_redirect_uri or "https://eir.tienzo.net/integrations/withings/callback"

    async with httpx.AsyncClient(timeout=15.0) as client:
        token_resp = await client.post(
            "https://wbsapi.withings.net/v2/oauth2",
            data={
                "action": "requesttoken",
                "grant_type": "authorization_code",
                "client_id": settings.withings_client_id,
                "client_secret": settings.withings_client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Withings token exchange failed: {token_resp.text}",
            )
        data = token_resp.json()
        if data.get("status") != 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Withings error {data.get('status')}: {data.get('error')}",
            )
        body = data["body"]
        user_id = str(body["userid"])
        access_token = body["access_token"]
        refresh_token = body["refresh_token"]
        expires_in = int(body.get("expires_in", 10800))
        scope = body.get("scope")
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        token_rec = db.scalar(
            select(OAuthToken).where(
                OAuthToken.deleted_at.is_(None),
                OAuthToken.provider == "withings",
                OAuthToken.user_id == user_id,
            )
        )
        if not token_rec:
            token_rec = OAuthToken(
                id=uuid.uuid4(),
                provider="withings",
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token,
                scope=scope,
                expires_at=expires_at,
            )
            db.add(token_rec)
        else:
            token_rec.access_token = access_token
            token_rec.refresh_token = refresh_token
            token_rec.scope = scope
            token_rec.expires_at = expires_at
        db.commit()

        # Automatically register webhook subscription for weight & body metrics (appli=1)
        sub_status = "unattempted"
        try:
            callback_url = f"{redirect_uri.rsplit('/', 1)[0]}/webhook"
            sub_resp = await client.post(
                "https://wbsapi.withings.net/notify",
                headers={"Authorization": f"Bearer {access_token}"},
                data={
                    "action": "subscribe",
                    "callbackurl": callback_url,
                    "appli": 1,
                    "comment": "Eir Weight Ingestion",
                },
            )
            if sub_resp.status_code == 200:
                sub_status = sub_resp.json()
        except Exception as exc:
            logger.warning("Withings notification auto-subscription failed: %s", exc)

    return {
        "status": "ok",
        "message": "Withings account connected successfully",
        "user_id": user_id,
        "expires_at": expires_at.isoformat(),
        "subscription": sub_status,
    }


@router.get(
    "/withings/status",
    response_model=WithingsAuthStatusResponse,
    summary="Check Withings OAuth connection status",
)
def withings_status(db: Session = Depends(get_db)):
    """Returns whether Withings is authorized and token expiry info."""
    token = db.scalar(
        select(OAuthToken)
        .where(OAuthToken.deleted_at.is_(None), OAuthToken.provider == "withings")
        .order_by(OAuthToken.updated_at.desc())
        .limit(1)
    )
    if not token:
        return WithingsAuthStatusResponse(status="disconnected", linked=False)
    return WithingsAuthStatusResponse(
        status="connected",
        linked=True,
        user_id=token.user_id,
        expires_at=token.expires_at,
        scope=token.scope,
    )


async def _get_valid_withings_token(db: Session, user_id: str | None = None) -> str | None:
    """Retrieves a valid Withings access token, refreshing it if expired."""
    stmt = select(OAuthToken).where(
        OAuthToken.deleted_at.is_(None),
        OAuthToken.provider == "withings",
    )
    if user_id:
        stmt = stmt.where(OAuthToken.user_id == user_id)
    token = db.scalar(stmt.order_by(OAuthToken.updated_at.desc()).limit(1))
    if not token:
        return None

    # Refresh if within 5 minutes of expiration
    if token.expires_at <= datetime.now(timezone.utc) + timedelta(minutes=5):
        if not settings.withings_client_id or not settings.withings_client_secret:
            logger.error("Cannot refresh Withings token: missing client credentials")
            return None
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://wbsapi.withings.net/v2/oauth2",
                    data={
                        "action": "requesttoken",
                        "grant_type": "refresh_token",
                        "client_id": settings.withings_client_id,
                        "client_secret": settings.withings_client_secret,
                        "refresh_token": token.refresh_token,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == 0:
                        b = data["body"]
                        token.access_token = b["access_token"]
                        token.refresh_token = b["refresh_token"]
                        token.expires_at = datetime.now(timezone.utc) + timedelta(
                            seconds=int(b.get("expires_in", 10800))
                        )
                        db.commit()
                    else:
                        logger.error("Withings token refresh returned status %s", data.get("status"))
                        return None
                else:
                    logger.error("Withings token refresh failed with HTTP %s", resp.status_code)
                    return None
        except Exception as exc:
            logger.error("Withings token refresh exception: %s", exc)
            return None

    return token.access_token


@router.post(
    "/withings/webhook",
    summary="Withings weigh-in notification callback",
)
async def withings_webhook_callback(
    request: Request,
    db: Session = Depends(get_db),
):
    """Receives Withings notifications when a user steps on the scale, then fetches and ingests measurements."""
    payload = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
    else:
        form = await request.form()
        payload = dict(form)

    logger.info("Withings webhook callback received: %s", payload)
    appli = str(payload.get("appli", ""))
    user_id = str(payload.get("userid", ""))
    startdate = payload.get("startdate")
    enddate = payload.get("enddate")

    samples_created = 0
    if appli == "1":
        access_token = await _get_valid_withings_token(db, user_id=user_id if user_id else None)
        if access_token:
            meas_params = {
                "action": "getmeas",
                "category": 1,
            }
            if startdate:
                meas_params["startdate"] = int(startdate)
            if enddate:
                meas_params["enddate"] = int(enddate)
            else:
                meas_params["startdate"] = int(datetime.now(timezone.utc).timestamp()) - 86400

            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    meas_resp = await client.post(
                        "https://wbsapi.withings.net/measure",
                        headers={"Authorization": f"Bearer {access_token}"},
                        data=meas_params,
                    )
                    if meas_resp.status_code == 200:
                        meas_data = meas_resp.json()
                        if meas_data.get("status") == 0:
                            body = meas_data.get("body", {})
                            grps = body.get("measuregrps", [])
                            days_to_rollup = set()
                            type_map = {
                                1: ("weight", "kg"),
                                6: ("body_fat_pct", "%"),
                                76: ("muscle_mass_kg", "kg"),
                                77: ("water_pct", "%"),
                                88: ("bone_mass_kg", "kg"),
                                11: ("resting_hr", "bpm"),
                            }
                            for grp in grps:
                                grp_date = datetime.fromtimestamp(grp["date"], tz=timezone.utc)
                                grpid = grp.get("grpid")
                                days_to_rollup.add(grp_date.astimezone(settings.tz).date())

                                for m in grp.get("measures", []):
                                    m_type = m.get("type")
                                    if m_type not in type_map:
                                        continue
                                    st_name, unit_str = type_map[m_type]
                                    val_raw = m.get("value", 0)
                                    unit_pow = m.get("unit", 0)
                                    calc_val = Decimal(str(val_raw)) * (Decimal(10) ** Decimal(str(unit_pow)))

                                    # Cross-source deduplication (+-60s window)
                                    existing = db.scalar(
                                        select(HealthSample).where(
                                            HealthSample.deleted_at.is_(None),
                                            HealthSample.sample_type == st_name,
                                            HealthSample.measured_at >= grp_date - timedelta(seconds=60),
                                            HealthSample.measured_at <= grp_date + timedelta(seconds=60),
                                        ).order_by(HealthSample.measured_at.desc()).limit(1)
                                    )
                                    if existing:
                                        existing.value = calc_val
                                        existing.unit = unit_str
                                        existing.source = "withings"
                                        if grpid:
                                            existing.external_id = f"withings_{grpid}_{m_type}"
                                    else:
                                        sample = HealthSample(
                                            id=uuid.uuid4(),
                                            source="withings",
                                            sample_type=st_name,
                                            measured_at=grp_date,
                                            ended_at=grp_date,
                                            value=calc_val,
                                            unit=unit_str,
                                            external_id=f"withings_{grpid}_{m_type}" if grpid else None,
                                            extra={"grpid": grpid, "attrib": grp.get("attrib")},
                                        )
                                        db.add(sample)
                                        samples_created += 1

                            db.commit()
                            for d in days_to_rollup:
                                _sync_daily_rollups(db, d)
                            db.commit()
            except Exception as exc:
                logger.error("Failed to query Withings measure API: %s", exc)

    return {
        "status": "ok",
        "received": True,
        "appli": appli,
        "samples_ingested": samples_created,
    }


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
