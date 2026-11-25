"""Withings OAuth2 lifecycle, measure decoding, and webhook sync orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
from typing import Any
import urllib.parse
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion._ingestor import TelemetryIngestor
from app.ingestion._types import (
    Clock,
    NormalizedSample,
    WithingsAuthStart,
    WithingsCallbackResult,
    WithingsCloudPort,
    WithingsConfig,
    WithingsConnectionStatus,
    WithingsMeasureGroup,
    WithingsProtocolError,
    WithingsTokenSet,
    WithingsWebhookResult,
    utc_now,
)
from app.ingestion._withings_http import HttpWithingsCloudAdapter
from app.models import OAuthToken

logger = logging.getLogger(__name__)

WITHINGS_MEASURE_TYPES: dict[int, tuple[str, str]] = {
    1: ("weight", "kg"),
    6: ("body_fat_pct", "%"),
    76: ("muscle_mass_kg", "kg"),
    77: ("water_pct", "%"),
    88: ("bone_mass_kg", "kg"),
    11: ("resting_hr", "bpm"),
}


def _as_utc(dt: datetime) -> datetime:
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc)


def _parse_optional_int(value: Any, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WithingsProtocolError(f"Invalid integer for {field_name}: {value!r}") from exc


def resolve_measure_window(
    startdate: Any,
    enddate: Any,
    now: datetime,
) -> tuple[int | None, int | None]:
    """Resolves Withings getmeas bounds, preserving explicit startdate when enddate is omitted."""
    start_ts = _parse_optional_int(startdate, "startdate")
    end_ts = _parse_optional_int(enddate, "enddate")
    if start_ts is None:
        start_ts = int(_as_utc(now).timestamp()) - 86400
    return start_ts, end_ts


def decode_measure_groups(groups: Sequence[WithingsMeasureGroup]) -> tuple[NormalizedSample, ...]:
    """Converts Withings measure groups into NormalizedSample objects scaled by 10**unit."""
    samples: list[NormalizedSample] = []
    for group in groups:
        grp_date = datetime.fromtimestamp(group.date, tz=timezone.utc)
        for measure in group.measures:
            mapping = WITHINGS_MEASURE_TYPES.get(measure.type)
            if mapping is None:
                continue
            sample_type, unit = mapping
            scaled = Decimal(str(measure.value)) * (Decimal(10) ** Decimal(str(measure.unit)))
            external_id = f"withings_{group.grpid}_{measure.type}" if group.grpid else None
            samples.append(
                NormalizedSample(
                    sample_type=sample_type,
                    measured_at=grp_date,
                    ended_at=grp_date,
                    value=scaled,
                    unit=unit,
                    source="withings",
                    external_id=external_id,
                    extra={"grpid": group.grpid, "attrib": group.attrib},
                )
            )
    return tuple(samples)


class WithingsSync:
    """Synchronous deep module managing Withings OAuth2, status, and webhook ingestion."""

    def __init__(
        self,
        db: Session,
        *,
        config: WithingsConfig | None = None,
        cloud: WithingsCloudPort | None = None,
        ingestor: TelemetryIngestor | None = None,
        clock: Clock = utc_now,
    ) -> None:
        self._db = db
        self._config = config if config is not None else WithingsConfig.from_settings(settings)
        self._cloud = cloud
        self._ingestor = ingestor if ingestor is not None else TelemetryIngestor(db)
        self._clock = clock

    def _cloud_port(self) -> WithingsCloudPort:
        if self._cloud is None:
            self._cloud = HttpWithingsCloudAdapter()
        return self._cloud

    def _now_utc(self) -> datetime:
        return _as_utc(self._clock())

    def build_authorize_url(self) -> WithingsAuthStart:
        """Builds the Withings OAuth2 authorization URL and CSRF state."""
        client_id, redirect_uri = self._config.require_authorize()
        state = uuid.uuid4().hex
        params = {
            "response_type": "code",
            "client_id": client_id,
            "scope": "user.info,user.metrics,user.activity",
            "redirect_uri": redirect_uri,
            "state": state,
        }
        url = f"https://account.withings.com/oauth2_user/authorize2?{urllib.parse.urlencode(params)}"
        return WithingsAuthStart(authorization_url=url, state=state)

    def complete_oauth_callback(self, code: str) -> WithingsCallbackResult:
        """Exchanges auth code, saves OAuthToken, and best-effort subscribes to webhooks."""
        self._config.require_oauth()
        cloud = self._cloud_port()
        token_set = cloud.exchange_code(self._config, code)
        expires_at = self._now_utc() + timedelta(seconds=token_set.expires_in)
        self._upsert_oauth_token(token_set, expires_at)
        self._db.commit()

        subscription = self._subscribe_best_effort(cloud, token_set.access_token)
        return WithingsCallbackResult(
            user_id=token_set.user_id,
            expires_at=expires_at,
            subscription=subscription,
        )

    def connection_status(self) -> WithingsConnectionStatus:
        """Returns database-backed Withings OAuth connection status without env config."""
        token = self._fetch_latest_token()
        if token is None:
            return WithingsConnectionStatus(status="disconnected", linked=False)
        return WithingsConnectionStatus(
            status="connected",
            linked=True,
            user_id=token.user_id,
            expires_at=token.expires_at,
            scope=token.scope,
        )

    def handle_webhook(self, payload: Mapping[str, Any]) -> WithingsWebhookResult:
        """Handles Withings webhook trigger, refreshing tokens and ingesting measures safely."""
        appli = str(payload.get("appli", ""))
        user_id = str(payload.get("userid", ""))
        if appli != "1":
            return WithingsWebhookResult(appli=appli, samples_ingested=0)

        access_token = self._get_valid_access_token(user_id=user_id or None)
        if not access_token:
            return WithingsWebhookResult(appli=appli, samples_ingested=0)

        samples_ingested = 0
        try:
            startdate, enddate = resolve_measure_window(
                payload.get("startdate"),
                payload.get("enddate"),
                self._now_utc(),
            )
            groups = self._cloud_port().fetch_measure_groups(
                access_token,
                startdate=startdate,
                enddate=enddate,
            )
            samples = decode_measure_groups(groups)
            result = self._ingestor.ingest_samples(samples)
            samples_ingested = result.samples_created
        except Exception as exc:
            logger.error("Failed to query Withings measure API: %s", exc)

        return WithingsWebhookResult(appli=appli, samples_ingested=samples_ingested)

    def _fetch_latest_token(self, user_id: str | None = None) -> OAuthToken | None:
        stmt = select(OAuthToken).where(
            OAuthToken.deleted_at.is_(None),
            OAuthToken.provider == "withings",
        )
        if user_id:
            stmt = stmt.where(OAuthToken.user_id == user_id)
        return self._db.scalar(stmt.order_by(OAuthToken.updated_at.desc()).limit(1))

    def _upsert_oauth_token(self, token_set: WithingsTokenSet, expires_at: datetime) -> None:
        existing = self._fetch_latest_token(user_id=token_set.user_id)
        if existing is None:
            self._db.add(
                OAuthToken(
                    id=uuid.uuid4(),
                    provider="withings",
                    user_id=token_set.user_id,
                    access_token=token_set.access_token,
                    refresh_token=token_set.refresh_token,
                    scope=token_set.scope,
                    expires_at=expires_at,
                )
            )
            return
        existing.access_token = token_set.access_token
        existing.refresh_token = token_set.refresh_token
        existing.scope = token_set.scope
        existing.expires_at = expires_at

    def _subscribe_best_effort(self, cloud: WithingsCloudPort, access_token: str) -> Any:
        try:
            return cloud.subscribe_notify(
                access_token,
                self._config.webhook_callback_url,
                appli=1,
                comment="Eir Weight Ingestion",
            )
        except Exception as exc:
            logger.warning("Withings notification auto-subscription failed: %s", exc)
            return "unattempted"

    def _get_valid_access_token(self, *, user_id: str | None = None) -> str | None:
        token = self._fetch_latest_token(user_id=user_id)
        if token is None:
            return None
        now = self._now_utc()
        if _as_utc(token.expires_at) > now + timedelta(minutes=5):
            return token.access_token
        try:
            self._config.require_refresh()
            refreshed = self._cloud_port().refresh_token(self._config, token.refresh_token)
        except Exception as exc:
            logger.error("Withings token refresh failed: %s", exc)
            return None

        token.access_token = refreshed.access_token
        token.refresh_token = refreshed.refresh_token
        token.expires_at = now + timedelta(seconds=refreshed.expires_in)
        self._db.commit()
        return token.access_token
