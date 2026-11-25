"""Thin HTTP router for Withings scale and Apple Health telemetry integrations."""

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.ingestion import (
    IngestionError,
    TelemetryIngestor,
    WithingsConfigError,
    WithingsProtocolError,
    WithingsSync,
    WithingsUpstreamError,
)
from app.models import BiometricMeasurement
from app.schemas import (
    AppleHealthComprehensiveSyncPayload,
    AppleHealthSyncResponse,
    BiometricRead,
    WithingsAuthStatusResponse,
    WithingsMeasurementPayload,
)

router = APIRouter(prefix="/integrations", tags=["integrations"])


def verify_webhook_token(
    authorization: str | None = Header(default=None),
    x_webhook_secret: str | None = Header(default=None),
    secret: str | None = Query(default=None),
) -> None:
    """Validate webhook secret from header or query param."""
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


def _map_withings_http_error(exc: IngestionError) -> HTTPException:
    if isinstance(exc, WithingsConfigError):
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    if isinstance(exc, WithingsProtocolError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, WithingsUpstreamError):
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


async def _parse_webhook_body(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    return dict(await request.form())


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
    try:
        result = TelemetryIngestor(db).ingest_apple_health(payload)
    except IngestionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return AppleHealthSyncResponse(
        status=result.status,
        total_samples_ingested=result.total_samples_ingested,
        metrics_summary=dict(result.metrics_summary),
        days_processed=list(result.days_processed),
        detected_workouts=list(result.detected_workouts),
    )


@router.api_route(
    "/withings/webhook",
    methods=["GET", "HEAD"],
    summary="Withings subscription verification challenge endpoint",
)
def withings_webhook_verification() -> dict[str, str]:
    """Responds with 200 OK to Withings developer verification probes."""
    return {"status": "ok", "service": "eir-withings"}


def get_withings_sync(db: Session = Depends(get_db)) -> WithingsSync:
    """Dependency provider for WithingsSync bound to the request database session."""
    return WithingsSync(db)


@router.get(
    "/withings/authorize",
    summary="Initiate Withings OAuth2 authorization flow",
)
def withings_authorize(
    format: str | None = Query(default=None, description="Set to 'json' to return authorization URL"),
    sync: WithingsSync = Depends(get_withings_sync),
):
    """Generates Withings OAuth2 authorization URL and redirects the user."""
    try:
        auth = sync.build_authorize_url()
    except WithingsConfigError as exc:
        raise _map_withings_http_error(exc) from exc
    if format == "json":
        return {"authorization_url": auth.authorization_url, "state": auth.state}
    return RedirectResponse(url=auth.authorization_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get(
    "/withings/callback",
    summary="Withings OAuth2 callback endpoint",
)
def withings_oauth_callback(
    code: str = Query(..., description="Authorization code from Withings"),
    state: str | None = Query(default=None),
    sync: WithingsSync = Depends(get_withings_sync),
) -> dict[str, Any]:
    """Exchanges authorization code for access and refresh tokens, saves them, and subscribes."""
    try:
        result = sync.complete_oauth_callback(code)
    except (WithingsConfigError, WithingsProtocolError, WithingsUpstreamError) as exc:
        raise _map_withings_http_error(exc) from exc
    return {
        "status": result.status,
        "message": result.message,
        "user_id": result.user_id,
        "expires_at": result.expires_at.isoformat(),
        "subscription": result.subscription,
    }


@router.get(
    "/withings/status",
    response_model=WithingsAuthStatusResponse,
    summary="Check Withings OAuth connection status",
)
def withings_status(sync: WithingsSync = Depends(get_withings_sync)) -> WithingsAuthStatusResponse:
    """Returns whether Withings is authorized and token expiry info."""
    res = sync.connection_status()
    return WithingsAuthStatusResponse(
        status=res.status,
        linked=res.linked,
        provider=res.provider,
        user_id=res.user_id,
        expires_at=res.expires_at,
        scope=res.scope,
    )


@router.post(
    "/withings/webhook",
    summary="Withings weigh-in notification callback",
)
async def withings_webhook_callback(
    request: Request,
    sync: WithingsSync = Depends(get_withings_sync),
) -> dict[str, Any]:
    """Receives Withings weigh-in notifications, refreshes token if needed, and ingests measures."""
    payload = await _parse_webhook_body(request)
    result = await run_in_threadpool(sync.handle_webhook, payload)
    return {
        "status": result.status,
        "received": result.received,
        "appli": result.appli,
        "samples_ingested": result.samples_ingested,
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
    """Ingests body scale metrics through HealthSample dedup and unified daily rollup."""
    return TelemetryIngestor(db).ingest_direct_scale(payload)
