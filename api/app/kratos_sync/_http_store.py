"""Fail-loud HTTP adapter implementing KratosSessionStore over Kratos REST endpoints."""

from __future__ import annotations

from typing import Any

import httpx

from app.kratos_sync._types import KratosSession, KratosSessionWrite, KratosStoreError


class HttpKratosSessionStore:
    """HTTP session-store adapter backed by an httpx.Client."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        cleaned = base_url.strip().rstrip("/")
        if not cleaned:
            raise ValueError("base_url must be a non-empty URL")
        self._base_url = cleaned
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def __enter__(self) -> HttpKratosSessionStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Closes the underlying httpx.Client when owned by this store."""
        if self._owns_client:
            self._client.close()

    def list_sessions(
        self,
        *,
        limit: int = 200,
        include_deleted: bool = True,
    ) -> tuple[KratosSession, ...]:
        """Lists existing Kratos sessions, returning () on 404 or failing loudly on errors."""
        resp = self._request(
            "list_sessions",
            "GET",
            "/sessions",
            params={"limit": limit, "include_deleted": "true" if include_deleted else "false"},
        )
        if resp.status_code == 404:
            return ()
        self._ensure_success("list_sessions", resp)
        data = self._decode_json("list_sessions", resp)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise KratosStoreError(
                "Kratos list_sessions expected a JSON list of session objects",
                operation="list_sessions",
                status_code=resp.status_code,
            )
        return tuple(KratosSession.from_api_dict(item) for item in data)

    def get_session(self, session_id: str) -> KratosSession | None:
        """Fetches a single Kratos session by ID, returning None on 404."""
        resp = self._request("get_session", "GET", f"/sessions/{session_id}")
        if resp.status_code == 404:
            return None
        self._ensure_success("get_session", resp)
        data = self._decode_json("get_session", resp)
        if not isinstance(data, dict):
            raise KratosStoreError(
                "Kratos get_session expected a JSON object",
                operation="get_session",
                status_code=resp.status_code,
            )
        return KratosSession.from_api_dict(data)

    def upsert_session(self, session: KratosSessionWrite) -> None:
        """Creates or updates a Kratos session via PUT /sessions/{id}."""
        resp = self._request(
            "upsert_session",
            "PUT",
            f"/sessions/{session.id}",
            json=session.to_api_dict(),
        )
        self._ensure_success("upsert_session", resp)

    def delete_session(self, session_id: str) -> bool:
        """Deletes a Kratos session via DELETE /sessions/{id}; treats 200/204/404 as success."""
        resp = self._request("delete_session", "DELETE", f"/sessions/{session_id}")
        if resp.status_code in (200, 204, 404):
            return True
        raise KratosStoreError(
            f"Kratos delete_session failed with HTTP {resp.status_code}",
            operation="delete_session",
            status_code=resp.status_code,
        )

    def _request(self, operation: str, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._client.request(method, f"{self._base_url}{path}", **kwargs)
        except httpx.HTTPStatusError as exc:
            raise KratosStoreError(
                f"Kratos {operation} failed with HTTP {exc.response.status_code}",
                operation=operation,
                status_code=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise KratosStoreError(
                f"Kratos {operation} transport error: {exc}",
                operation=operation,
                status_code=None,
            ) from exc

    @staticmethod
    def _ensure_success(operation: str, resp: httpx.Response) -> None:
        if not (200 <= resp.status_code < 300):
            raise KratosStoreError(
                f"Kratos {operation} failed with HTTP {resp.status_code}",
                operation=operation,
                status_code=resp.status_code,
            )

    @staticmethod
    def _decode_json(operation: str, resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except ValueError as exc:
            raise KratosStoreError(
                f"Kratos {operation} returned invalid JSON",
                operation=operation,
                status_code=resp.status_code,
            ) from exc
