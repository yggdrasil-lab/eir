"""HTTP transport, auth middleware, and tool contract tests for /mcp."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import uuid

from app.config import settings
from app.models import HealthSample
from tests._harness import EirTestCase, call_mcp_tool, post_mcp_rpc

_MCP_ACCEPT_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


class TestMcpHttpTransport(EirTestCase):
    """Exercises the mounted Streamable HTTP MCP server through TestClient lifespan."""

    def test_auth_middleware_503_when_unconfigured_and_401_on_invalid_bearer(self) -> None:
        self.patch_settings(mcp_token="")
        with self.open_lifespan_client() as client:
            unconfigured = client.post("/mcp", headers=_MCP_ACCEPT_HEADERS, json={})
            self.assertEqual(unconfigured.status_code, 503)
            self.assertEqual(unconfigured.json(), {"error": "mcp token not configured"})

        self.patch_settings(mcp_token="secret-mcp-token")
        with self.open_lifespan_client() as client:
            for auth_headers in (
                {},
                {"Authorization": "Basic abc123"},
                {"Authorization": "secret-mcp-token"},
                {"Authorization": "Bearer wrong-token"},
            ):
                resp = client.post(
                    "/mcp",
                    headers={**_MCP_ACCEPT_HEADERS, **auth_headers},
                    json={},
                )
                self.assertEqual(resp.status_code, 401)
                self.assertEqual(resp.headers.get("www-authenticate"), "Bearer")

    def test_initialize_handshake_and_tools_list(self) -> None:
        token = "secret-mcp-token"
        self.patch_settings(mcp_token=token)
        with self.open_lifespan_client() as client:
            init_res = post_mcp_rpc(
                client,
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "eir-test", "version": "1.0"},
                },
                token=token,
                request_id=1,
            )
            self.assertEqual(init_res["result"]["serverInfo"]["name"], "eir")

            tools_res = post_mcp_rpc(client, "tools/list", {}, token=token, request_id=2)
            names = {tool["name"] for tool in tools_res["result"]["tools"]}
            self.assertEqual(
                names,
                {
                    "eir_get_latest_biometrics",
                    "eir_get_daily_summary",
                    "eir_get_granular_samples",
                    "eir_log_measurement",
                    "eir_get_workout_metrics",
                },
            )

    def test_biometrics_mcp_tools_happy_and_invalid_input_paths(self) -> None:
        token = "secret-mcp-token"
        self.patch_settings(mcp_token=token)
        target_day = date(2026, 11, 24)
        sample_dt = datetime(2026, 11, 24, 7, 15, tzinfo=settings.tz)
        self.db.add(
            HealthSample(
                id=uuid.uuid4(),
                source="apple_health",
                sample_type="active_energy",
                measured_at=sample_dt,
                ended_at=sample_dt + timedelta(minutes=5),
                value=Decimal("185"),
                unit="kcal",
            )
        )
        self.db.commit()

        with self.open_lifespan_client() as client:
            is_err, empty_latest = call_mcp_tool(client, "eir_get_latest_biometrics", {}, token=token)
            self.assertFalse(is_err)
            self.assertEqual(empty_latest["status"], "empty")

            is_err, logged = call_mcp_tool(
                client,
                "eir_log_measurement",
                {
                    "weight_kg": 80.5,
                    "body_fat_pct": 16.0,
                    "resting_hr_bpm": 53,
                    "hrv_ms": 70.0,
                    "sleep_duration_hours": 7.5,
                    "sleep_score": 88,
                    "step_count": 6200,
                    "notes": "MCP check-in",
                },
                token=token,
            )
            self.assertFalse(is_err)
            self.assertEqual(logged["source"], "agent")
            self.assertEqual(logged["weight_kg"], 80.5)

            is_err, latest = call_mcp_tool(
                client, "eir_get_latest_biometrics", {"source": "agent"}, token=token
            )
            self.assertFalse(is_err)
            self.assertEqual(latest["id"], logged["id"])

            is_err, summary = call_mcp_tool(
                client, "eir_get_daily_summary", {"target_date": target_day.isoformat()}, token=token
            )
            self.assertFalse(is_err)
            self.assertEqual(summary["date"], target_day.isoformat())
            self.assertEqual(summary["active_energy_kcal"], 185)

            is_err, granular = call_mcp_tool(
                client,
                "eir_get_granular_samples",
                {"target_date": target_day.isoformat(), "sample_type": "active_energy"},
                token=token,
            )
            self.assertFalse(is_err)
            self.assertEqual(granular["count"], 1)
            self.assertEqual(granular["samples"][0]["value"], 185.0)

            invalid_calls = [
                ("eir_get_latest_biometrics", {"source": 123}),
                ("eir_log_measurement", {"weight_kg": "not-a-number"}),
                ("eir_get_daily_summary", {"target_date": "bad-date"}),
                ("eir_get_granular_samples", {"target_date": target_day.isoformat(), "sample_type": "   "}),
            ]
            for tool_name, args in invalid_calls:
                with self.subTest(tool=tool_name, args=args):
                    err_flag, _ = call_mcp_tool(client, tool_name, args, token=token)
                    self.assertTrue(err_flag)

    def test_workout_metrics_mcp_tool_happy_and_invalid_input_paths(self) -> None:
        token = "secret-mcp-token"
        self.patch_settings(mcp_token=token)
        start = datetime(2026, 11, 23, 19, 0, tzinfo=settings.tz)
        end = start + timedelta(minutes=30)
        for i in range(6):
            t = start + timedelta(minutes=i * 5)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + timedelta(minutes=5),
                    value=Decimal("30"),
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t + timedelta(minutes=5),
                    value=Decimal("135"),
                    unit="bpm",
                )
            )
        self.db.commit()

        with self.open_lifespan_client() as client:
            is_err, workout = call_mcp_tool(
                client,
                "eir_get_workout_metrics",
                {
                    "started_at": start.isoformat(),
                    "ended_at": end.isoformat(),
                    "classification": "gym",
                },
                token=token,
            )
            self.assertFalse(is_err)
            self.assertEqual(workout["duration_minutes"], 30)
            self.assertEqual(workout["classification"], "gym")
            self.assertEqual(workout["active_energy_kcal"], 180)
            self.assertEqual(workout["avg_hr_bpm"], 135)

            for args in (
                {"started_at": "bad-ts", "ended_at": end.isoformat()},
                {"started_at": end.astimezone(timezone.utc).isoformat(), "ended_at": start.isoformat()},
            ):
                is_err, _ = call_mcp_tool(client, "eir_get_workout_metrics", args, token=token)
                self.assertTrue(is_err)
