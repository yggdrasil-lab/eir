"""MCP endpoint for the Eir Health API.

A streamable-HTTP MCP server mounted at /mcp on the FastAPI app, built on
the official MCP Python SDK (mcp.server.mcpserver.MCPServer).
"""

from __future__ import annotations

import contextlib
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
from typing import Any, Iterator
import uuid

from fastapi import FastAPI, HTTPException
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.config import settings
from app.models import BiometricMeasurement
from app.schemas import BiometricCreate

SOURCE = "agent"


class McpAuthMiddleware(BaseHTTPMiddleware):
    """Guard /mcp with a bearer token; fail closed when unconfigured."""

    async def dispatch(self, request, call_next):
        if request.url.path.rstrip("/") == "/mcp":
            token = settings.mcp_token
            if not token:
                return JSONResponse(
                    {"error": "mcp token not configured"}, status_code=503
                )
            scheme, _, presented = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or presented != token:
                return JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)


def mcp_session() -> Session:
    from app.db import SessionLocal

    return SessionLocal()


def _serialise_measurement(m: BiometricMeasurement) -> dict[str, Any]:
    return {
        "id": str(m.id),
        "measured_at": m.measured_at.isoformat(),
        "source": m.source,
        "weight_kg": float(m.weight_kg) if m.weight_kg is not None else None,
        "body_fat_pct": float(m.body_fat_pct) if m.body_fat_pct is not None else None,
        "muscle_mass_kg": float(m.muscle_mass_kg) if m.muscle_mass_kg is not None else None,
        "resting_hr_bpm": m.resting_hr_bpm,
        "hrv_ms": float(m.hrv_ms) if m.hrv_ms is not None else None,
        "sleep_duration_hours": round(m.sleep_duration_s / 3600.0, 2) if m.sleep_duration_s else None,
        "sleep_score": m.sleep_score,
        "step_count": m.step_count,
        "active_energy_kcal": m.active_energy_kcal,
        "notes": m.notes,
        "created_at": m.created_at.isoformat(),
    }


def build_mcp_server() -> MCPServer:
    server = MCPServer(
        name="eir",
        version="0.1.0",
        instructions=(
            "Eir is the homelab biometrics and health telemetry service. Use it to "
            "retrieve morning weigh-ins, body composition, sleep metrics, and recovery "
            "vitals (HRV and resting heart rate), or log health measurements."
        ),
    )

    @server.tool(
        name="eir_get_latest_biometrics",
        description="Fetch the most recent biometric reading (weight, body fat %, HRV, resting HR, sleep).",
    )
    def get_latest_biometrics(source: str | None = None) -> str:
        with mcp_session() as db:
            stmt = select(BiometricMeasurement).where(BiometricMeasurement.deleted_at.is_(None))
            if source:
                stmt = stmt.where(BiometricMeasurement.source == source)
            stmt = stmt.order_by(BiometricMeasurement.measured_at.desc()).limit(1)
            reading = db.scalar(stmt)
            if not reading:
                return json.dumps({"status": "empty", "message": "No biometric records found."})
            return json.dumps(_serialise_measurement(reading))

    @server.tool(
        name="eir_get_daily_summary",
        description="Get aggregated health and vitals summary for a specific calendar date (YYYY-MM-DD).",
    )
    def get_daily_summary(target_date: str) -> str:
        try:
            d = date.fromisoformat(target_date)
        except ValueError as err:
            raise ToolError(f"Invalid date format: {target_date}. Expected YYYY-MM-DD.") from err

        start_dt = datetime.combine(d, time.min).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(d, time.max).replace(tzinfo=timezone.utc)

        with mcp_session() as db:
            stmt = (
                select(BiometricMeasurement)
                .where(
                    BiometricMeasurement.deleted_at.is_(None),
                    BiometricMeasurement.measured_at >= start_dt,
                    BiometricMeasurement.measured_at <= end_dt,
                )
                .order_by(BiometricMeasurement.measured_at.asc())
            )
            records = list(db.scalars(stmt))

            summary = {
                "date": target_date,
                "latest_weight_kg": None,
                "latest_body_fat_pct": None,
                "resting_hr_bpm": None,
                "hrv_ms": None,
                "sleep_duration_hours": None,
                "sleep_score": None,
                "step_count": 0,
                "active_energy_kcal": 0,
                "readings_count": len(records),
            }

            for r in records:
                if r.weight_kg is not None:
                    summary["latest_weight_kg"] = float(r.weight_kg)
                if r.body_fat_pct is not None:
                    summary["latest_body_fat_pct"] = float(r.body_fat_pct)
                if r.resting_hr_bpm is not None:
                    summary["resting_hr_bpm"] = r.resting_hr_bpm
                if r.hrv_ms is not None:
                    summary["hrv_ms"] = float(r.hrv_ms)
                if r.sleep_duration_s is not None:
                    summary["sleep_duration_hours"] = round(r.sleep_duration_s / 3600.0, 2)
                if r.sleep_score is not None:
                    summary["sleep_score"] = r.sleep_score
                if r.step_count is not None:
                    summary["step_count"] += r.step_count
                if r.active_energy_kcal is not None:
                    summary["active_energy_kcal"] += r.active_energy_kcal

            return json.dumps(summary)

    @server.tool(
        name="eir_log_measurement",
        description="Log a biometric measurement (weight, body fat %, resting HR, HRV, sleep, steps) from an agent.",
    )
    def log_measurement(
        weight_kg: float | None = None,
        body_fat_pct: float | None = None,
        resting_hr_bpm: int | None = None,
        hrv_ms: float | None = None,
        sleep_duration_hours: float | None = None,
        sleep_score: int | None = None,
        step_count: int | None = None,
        notes: str | None = None,
    ) -> str:
        sleep_s = int(sleep_duration_hours * 3600) if sleep_duration_hours is not None else None
        now = datetime.now(timezone.utc)

        measurement = BiometricMeasurement(
            id=uuid.uuid4(),
            measured_at=now,
            source="agent",
            weight_kg=Decimal(str(weight_kg)) if weight_kg is not None else None,
            body_fat_pct=Decimal(str(body_fat_pct)) if body_fat_pct is not None else None,
            resting_hr_bpm=resting_hr_bpm,
            hrv_ms=Decimal(str(hrv_ms)) if hrv_ms is not None else None,
            sleep_duration_s=sleep_s,
            sleep_score=sleep_score,
            step_count=step_count,
            notes=notes,
            extra={"source": SOURCE, "logged_by": "mcp"},
        )

        with mcp_session() as db:
            db.add(measurement)
            db.commit()
            db.refresh(measurement)
            return json.dumps(_serialise_measurement(measurement))

    return server


class _RebuildableApp:
    def __init__(self, server: MCPServer) -> None:
        self._server = server
        self.app: Starlette | None = None

    def _build(self) -> Starlette:
        return self._server.streamable_http_app(
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            ),
        )

    async def __call__(self, scope, receive, send):
        if self.app is None:
            raise RuntimeError("MCP sub-app is not running; lifespan did not start")
        await self.app(scope, receive, send)


def install_mcp(app: FastAPI) -> None:
    """Attach the MCP server's route and auth middleware to app."""
    server = build_mcp_server()
    holder = _RebuildableApp(server)

    app.router.routes.append(Route("/mcp", endpoint=holder))

    original_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app_):
        sub_app = holder._build()
        holder.app = sub_app
        async with contextlib.AsyncExitStack() as stack:
            if original_lifespan is not None:
                await stack.enter_async_context(original_lifespan(app_))
            await stack.enter_async_context(sub_app.router.lifespan_context(app_))
            yield
        holder.app = None

    app.router.lifespan_context = lifespan
    app.add_middleware(McpAuthMiddleware)
