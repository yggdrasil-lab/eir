"""Shared unittest harness providing isolated SQLite DB, TestClient, and DI overrides."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from typing import Any
import unittest

from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import db as db_module
from app.config import settings
from app.db import Base, get_db
from app.ingestion import Clock, TelemetryIngestor, WithingsCloudPort, WithingsConfig, WithingsSync, utc_now
from app.main import app
from app.routers.integrations import get_withings_sync

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


class EirTestCase(unittest.TestCase):
    """Base TestCase with isolated in-memory SQLite StaticPool DB and FastAPI DI helpers."""

    def setUp(self) -> None:
        super().setUp()
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self._engine)
        self.session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            bind=self._engine,
        )
        self._orig_session_local = db_module.SessionLocal
        db_module.SessionLocal = self.session_factory
        self.db: Session = self.session_factory()
        self._client: TestClient | None = None
        app.dependency_overrides[get_db] = self._override_get_db

    def tearDown(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
                self._client = None
            self.db.close()
        finally:
            db_module.SessionLocal = self._orig_session_local
            app.dependency_overrides.clear()
            Base.metadata.drop_all(bind=self._engine)
            self._engine.dispose()
            super().tearDown()

    def _override_get_db(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    @property
    def client(self) -> TestClient:
        """Lazily instantiated FastAPI TestClient wired to the isolated test DB."""
        if self._client is None:
            self._client = TestClient(app)
        return self._client

    @contextmanager
    def open_lifespan_client(self) -> Iterator[TestClient]:
        """Opens a TestClient running ASGI lifespan hooks (required for /mcp)."""
        with TestClient(app) as lifespan_client:
            yield lifespan_client

    def patch_settings(self, **overrides: Any) -> None:
        """Temporarily overrides attributes on app.config.settings with addCleanup restore."""
        sentinel = object()
        previous = {key: getattr(settings, key, sentinel) for key in overrides}
        for key, value in overrides.items():
            setattr(settings, key, value)

        def _restore() -> None:
            for key, old_value in previous.items():
                if old_value is sentinel:
                    delattr(settings, key)
                else:
                    setattr(settings, key, old_value)

        self.addCleanup(_restore)

    def override_withings_sync(
        self,
        cloud: WithingsCloudPort,
        *,
        config: WithingsConfig | None = None,
        ingestor: TelemetryIngestor | None = None,
        clock: Clock = utc_now,
    ) -> None:
        """Overrides FastAPI get_withings_sync dependency with a deterministic cloud adapter."""

        def _override_sync(db: Session = Depends(get_db)) -> WithingsSync:
            return WithingsSync(db, config=config, cloud=cloud, ingestor=ingestor, clock=clock)

        app.dependency_overrides[get_withings_sync] = _override_sync
        self.addCleanup(app.dependency_overrides.pop, get_withings_sync, None)


def post_mcp_rpc(
    client: TestClient,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    token: str,
    request_id: int = 1,
) -> dict[str, Any]:
    """Posts a JSON-RPC request to /mcp and returns the parsed JSON response body."""
    headers = {**_MCP_HEADERS, "Authorization": f"Bearer {token}"}
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    response = client.post("/mcp", headers=headers, json=body)
    if response.status_code != 200:
        raise AssertionError(f"Expected 200 from /mcp, got {response.status_code}: {response.text}")
    return response.json()


def call_mcp_tool(
    client: TestClient,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    token: str,
    request_id: int = 1,
) -> tuple[bool, Any]:
    """Invokes an MCP tool via /mcp and returns (is_error, parsed_json_or_text)."""
    envelope = post_mcp_rpc(
        client,
        "tools/call",
        {"name": name, "arguments": arguments or {}},
        token=token,
        request_id=request_id,
    )
    result = envelope["result"]
    is_error = bool(result.get("isError", False))
    text = result["content"][0]["text"]
    if is_error:
        return True, text
    try:
        return False, json.loads(text)
    except json.JSONDecodeError:
        return False, text
