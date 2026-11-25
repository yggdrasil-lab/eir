"""Deterministic in-memory WithingsCloudPort adapter for unit tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.ingestion._types import (
    WithingsConfig,
    WithingsMeasureGroup,
    WithingsTokenSet,
)


class InMemoryWithingsCloudAdapter:
    """In-memory double implementing WithingsCloudPort without mocking frameworks."""

    def __init__(
        self,
        *,
        token_set: WithingsTokenSet | None = None,
        refreshed_token_set: WithingsTokenSet | None = None,
        measure_groups: Sequence[WithingsMeasureGroup] = (),
        subscription_response: Any = None,
        exchange_error: Exception | None = None,
        refresh_error: Exception | None = None,
        subscribe_error: Exception | None = None,
        measure_error: Exception | None = None,
    ) -> None:
        self._token_set = token_set or WithingsTokenSet(
            user_id="withings-user-1",
            access_token="access-token-initial",
            refresh_token="refresh-token-initial",
            expires_in=10800,
            scope="user.info,user.metrics,user.activity",
        )
        self._refreshed_token_set = refreshed_token_set or WithingsTokenSet(
            user_id=self._token_set.user_id,
            access_token="access-token-refreshed",
            refresh_token="refresh-token-rotated",
            expires_in=10800,
            scope=self._token_set.scope,
        )
        self._measure_groups = tuple(measure_groups)
        self._subscription_response = (
            {"status": 0, "body": {}} if subscription_response is None else subscription_response
        )
        self._exchange_error = exchange_error
        self._refresh_error = refresh_error
        self._subscribe_error = subscribe_error
        self._measure_error = measure_error

        self.exchange_calls: list[tuple[WithingsConfig, str]] = []
        self.refresh_calls: list[tuple[WithingsConfig, str]] = []
        self.subscribe_calls: list[tuple[str, str, int, str]] = []
        self.measure_calls: list[tuple[str, int | None, int | None]] = []

    def exchange_code(self, config: WithingsConfig, code: str) -> WithingsTokenSet:
        self.exchange_calls.append((config, code))
        if self._exchange_error is not None:
            raise self._exchange_error
        return self._token_set

    def refresh_token(self, config: WithingsConfig, refresh_token: str) -> WithingsTokenSet:
        self.refresh_calls.append((config, refresh_token))
        if self._refresh_error is not None:
            raise self._refresh_error
        return self._refreshed_token_set

    def subscribe_notify(
        self,
        access_token: str,
        callback_url: str,
        *,
        appli: int = 1,
        comment: str = "Eir Weight Ingestion",
    ) -> Any:
        self.subscribe_calls.append((access_token, callback_url, appli, comment))
        if self._subscribe_error is not None:
            raise self._subscribe_error
        return self._subscription_response

    def fetch_measure_groups(
        self,
        access_token: str,
        *,
        startdate: int | None = None,
        enddate: int | None = None,
    ) -> tuple[WithingsMeasureGroup, ...]:
        self.measure_calls.append((access_token, startdate, enddate))
        if self._measure_error is not None:
            raise self._measure_error
        return self._measure_groups
