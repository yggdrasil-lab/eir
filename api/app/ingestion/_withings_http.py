"""Synchronous httpx adapter implementing WithingsCloudPort against wbsapi.withings.net."""

from __future__ import annotations

from typing import Any

import httpx

from app.ingestion._types import (
    WithingsConfig,
    WithingsMeasure,
    WithingsMeasureGroup,
    WithingsProtocolError,
    WithingsTokenSet,
    WithingsUpstreamError,
)

_OAUTH_URL = "https://wbsapi.withings.net/v2/oauth2"
_NOTIFY_URL = "https://wbsapi.withings.net/notify"
_MEASURE_URL = "https://wbsapi.withings.net/measure"


def _parse_token_body(body: Any, *, fallback_user_id: str = "") -> WithingsTokenSet:
    if not isinstance(body, dict):
        raise WithingsProtocolError("Withings token response missing body object")
    try:
        return WithingsTokenSet(
            user_id=str(body.get("userid", fallback_user_id)),
            access_token=str(body["access_token"]),
            refresh_token=str(body["refresh_token"]),
            expires_in=int(body.get("expires_in", 10800)),
            scope=body.get("scope"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WithingsProtocolError(f"Malformed Withings token body: {exc}") from exc


def _parse_measure_group(raw: Any) -> WithingsMeasureGroup:
    if not isinstance(raw, dict) or "date" not in raw:
        raise WithingsProtocolError("Malformed Withings measure group payload")
    measures: list[WithingsMeasure] = []
    for item in raw.get("measures", []):
        if not isinstance(item, dict):
            continue
        measures.append(
            WithingsMeasure(
                type=int(item["type"]),
                value=int(item.get("value", 0)),
                unit=int(item.get("unit", 0)),
            )
        )
    return WithingsMeasureGroup(
        grpid=raw.get("grpid"),
        date=int(raw["date"]),
        attrib=raw.get("attrib"),
        measures=tuple(measures),
    )


class HttpWithingsCloudAdapter:
    """HTTP adapter implementing WithingsCloudPort via synchronous httpx.Client."""

    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 15.0) -> None:
        self._client = client
        self._timeout = timeout

    def exchange_code(self, config: WithingsConfig, code: str) -> WithingsTokenSet:
        """Exchanges an OAuth2 authorization code for Withings access and refresh tokens."""
        client_id, client_secret, redirect_uri = config.require_oauth()
        data = self._post_checked(
            "token exchange",
            _OAUTH_URL,
            data={
                "action": "requesttoken",
                "grant_type": "authorization_code",
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        return _parse_token_body(data.get("body"))

    def refresh_token(self, config: WithingsConfig, refresh_token: str) -> WithingsTokenSet:
        """Refreshes an expired or expiring Withings OAuth2 access token."""
        client_id, client_secret = config.require_refresh()
        data = self._post_checked(
            "token refresh",
            _OAUTH_URL,
            data={
                "action": "requesttoken",
                "grant_type": "refresh_token",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
            },
        )
        return _parse_token_body(data.get("body"))

    def subscribe_notify(
        self,
        access_token: str,
        callback_url: str,
        *,
        appli: int = 1,
        comment: str = "Eir Weight Ingestion",
    ) -> Any:
        """Subscribes Withings notification webhook for the specified application ID."""
        return self._post_checked(
            "notify subscribe",
            _NOTIFY_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            data={
                "action": "subscribe",
                "callbackurl": callback_url,
                "appli": appli,
                "comment": comment,
            },
        )

    def fetch_measure_groups(
        self,
        access_token: str,
        *,
        startdate: int | None = None,
        enddate: int | None = None,
    ) -> tuple[WithingsMeasureGroup, ...]:
        """Fetches Withings measurement groups within optional [startdate, enddate] epoch bounds."""
        params: dict[str, Any] = {"action": "getmeas", "category": 1}
        if startdate is not None:
            params["startdate"] = int(startdate)
        if enddate is not None:
            params["enddate"] = int(enddate)

        data = self._post_checked(
            "getmeas",
            _MEASURE_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            data=params,
        )
        raw_groups = (data.get("body") or {}).get("measuregrps", [])
        return tuple(_parse_measure_group(group) for group in raw_groups)

    def _post_checked(self, operation: str, url: str, **kwargs: Any) -> dict[str, Any]:
        resp = self._post(operation, url, **kwargs)
        if resp.status_code != 200:
            raise WithingsUpstreamError(
                f"Withings {operation} failed with HTTP {resp.status_code}: {resp.text}"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise WithingsUpstreamError(f"Withings {operation} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise WithingsProtocolError(f"Withings {operation} expected JSON object")
        if payload.get("status") != 0:
            raise WithingsProtocolError(
                f"Withings error {payload.get('status')}: {payload.get('error')}"
            )
        return payload

    def _post(self, operation: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            if self._client is not None:
                return self._client.post(url, **kwargs)
            with httpx.Client(timeout=self._timeout) as client:
                return client.post(url, **kwargs)
        except httpx.HTTPError as exc:
            raise WithingsUpstreamError(f"Withings {operation} transport error: {exc}") from exc
