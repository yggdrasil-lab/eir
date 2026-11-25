"""Comprehensive unit tests for the app.ingestion deep module and Withings sync."""

from __future__ import annotations

import ast
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import select

from app.daily_biometrics import DailyBiometrics
from app.ingestion import (
    SCALE_DEDUP_METRICS,
    HttpWithingsCloudAdapter,
    InMemoryWithingsCloudAdapter,
    NormalizedSample,
    TelemetryIngestor,
    WithingsConfig,
    WithingsConfigError,
    WithingsMeasure,
    WithingsMeasureGroup,
    WithingsProtocolError,
    WithingsSync,
    WithingsTokenSet,
    WithingsUpstreamError,
    clean_datetime,
    clean_numeric,
    decode_measure_groups,
    extract_unexpected_keys,
    normalize_apple_health_payload,
    normalize_utc_second,
    parse_duration_seconds,
    resolve_measure_window,
    upsert_health_samples,
)
from app.models import BiometricMeasurement, HealthSample, OAuthToken
from app.schemas import DetectedWorkoutRead, WithingsMeasurementPayload
from tests._harness import EirTestCase


class _FakeWorkoutSync:
    """In-memory WorkoutSyncPort double recording synced workouts."""

    def __init__(self) -> None:
        self.calls: list[list[DetectedWorkoutRead]] = []

    def sync(self, workouts) -> None:
        self.calls.append(list(workouts))


class _FakeRollups:
    """In-memory RollupPort double returning deterministic RollupBatch."""

    def __init__(self, db, tz: ZoneInfo) -> None:
        self._inner = DailyBiometrics(db, tz=tz)
        self.calls: list[list[date | datetime]] = []

    def materialize_rollups(self, targets):
        self.calls.append(list(targets))
        return self._inner.materialize_rollups(targets)


class TestShortcutsParsingAndNormalization(unittest.TestCase):
    def test_clean_numeric_and_duration_helpers(self) -> None:
        self.assertEqual(clean_numeric("2 kcal\n1.5 kcal\n3.0 kcal"), Decimal("6.5"))
        self.assertEqual(clean_numeric(["10", "20.5"]), Decimal("30.5"))
        self.assertEqual(clean_numeric({"value": "72.5 kg"}), Decimal("72.5"))
        self.assertIsNone(clean_numeric("invalid"))
        self.assertIsNone(clean_numeric(None))
        self.assertIsNone(clean_numeric(True))

        self.assertEqual(parse_duration_seconds("19:25"), Decimal("1165"))
        self.assertEqual(parse_duration_seconds("1:18:43"), Decimal("4723"))
        self.assertEqual(parse_duration_seconds(900), Decimal("900"))
        self.assertIsNone(parse_duration_seconds(None))

    def test_clean_datetime_iso_and_ios_locale(self) -> None:
        dt_val = clean_datetime("29 Oct 2026 at 6:35\u202fPM")
        self.assertIsInstance(dt_val, datetime)
        self.assertEqual(dt_val, datetime(2026, 10, 29, 18, 35))

        iso_val = clean_datetime("2026-11-24T07:30:00+00:00")
        self.assertEqual(iso_val, datetime(2026, 11, 24, 7, 30, tzinfo=timezone.utc))

    def test_normalize_all_payload_shapes_and_unexpected_keys(self) -> None:
        payload = {
            "source": "apple_health",
            "samples": [
                {
                    "measured_at": "2026-11-24T06:00:00+00:00",
                    "value": "65 ms",
                    "sample_type": "heart_rate_variability",
                    "unit": "ms",
                }
            ],
            "metrics": {
                "active_energy_kcal": {
                    "start_dates": ["2026-11-24T08:00:00+00:00"],
                    "values": ["120 kcal"],
                    "sources": ["Alex Watch"],
                }
            },
            "": {
                "sleep_start_dates": "2026-11-24T01:00:00+00:00\n2026-11-24T02:00:00+00:00",
                "sleep_end_dates": "2026-11-24T01:30:00+00:00\n2026-11-24T03:00:00+00:00",
                "sleep_values": "Deep\nCore",
            },
            "sample_type": "walking_ asymmetry",
            "dates": ["2026-11-24T09:00:00+00:00"],
            "values": ["1.5 %"],
        }

        samples = normalize_apple_health_payload(payload)
        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0].sample_type, "hrv")
        self.assertEqual(samples[1].sample_type, "active_energy")
        self.assertEqual(samples[1].extra["device"], "Alex Watch")
        self.assertEqual(samples[2].sample_type, "sleep")
        self.assertEqual(samples[2].extra["stage"], "Deep")
        self.assertEqual(samples[2].value, Decimal("1800"))
        self.assertEqual(samples[4].sample_type, "walking_asymmetry")
        self.assertEqual(samples[4].unit, "%")

        unexpected = extract_unexpected_keys(payload)
        self.assertIn("", unexpected)
        self.assertIn("sample_type", unexpected)


class TestScaleDedupAndTelemetryIngestor(EirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.nz_tz = ZoneInfo("Pacific/Auckland")

    def test_scale_dedup_withings_wins_both_orders_and_all_five_metrics(self) -> None:
        t_withings = datetime(2026, 11, 25, 7, 30, 15, tzinfo=timezone.utc)
        t_apple = datetime(2026, 11, 25, 7, 30, 0, tzinfo=timezone.utc)

        # Case 1: Withings inserted first, Apple Health arrives within +-60s -> Withings untouched
        withings_samples = [
            NormalizedSample(
                sample_type=metric,
                measured_at=t_withings,
                value=Decimal("50.5"),
                unit="kg" if "kg" in metric or metric == "weight" else "%",
                source="withings",
                external_id=f"withings_1_{metric}",
            )
            for metric in sorted(SCALE_DEDUP_METRICS)
        ]
        _, created = upsert_health_samples(self.db, withings_samples)
        self.db.commit()
        self.assertEqual(created, 5)

        apple_overwrite = [
            NormalizedSample(
                sample_type=metric,
                measured_at=t_apple,
                value=Decimal("99.9"),
                unit="lb",
                source="apple_health",
                external_id=f"apple_{metric}",
            )
            for metric in sorted(SCALE_DEDUP_METRICS)
        ]
        _, created_after = upsert_health_samples(self.db, apple_overwrite)
        self.db.commit()
        self.assertEqual(created_after, 0)

        rows = list(self.db.scalars(select(HealthSample)))
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(row.source, "withings")
            self.assertEqual(row.value, Decimal("50.5"))
            self.assertNotEqual(row.unit, "lb")
            self.assertTrue(row.external_id.startswith("withings_1_"))

    def test_scale_dedup_promotes_apple_health_when_withings_arrives_later(self) -> None:
        t_apple = datetime(2026, 11, 25, 8, 0, 0, tzinfo=timezone.utc)
        t_withings = datetime(2026, 11, 25, 8, 0, 45, tzinfo=timezone.utc)

        upsert_health_samples(
            self.db,
            [
                NormalizedSample(
                    sample_type="weight",
                    measured_at=t_apple,
                    value=Decimal("80.0"),
                    unit="kg",
                    source="apple_health",
                    external_id="apple_w1",
                    extra={"device": "Alex Scale"},
                )
            ],
        )
        self.db.commit()

        _, created = upsert_health_samples(
            self.db,
            [
                NormalizedSample(
                    sample_type="weight",
                    measured_at=t_withings,
                    value=Decimal("79.85"),
                    unit="kg",
                    source="withings",
                    external_id="withings_99_1",
                    extra={"grpid": 99},
                )
            ],
        )
        self.db.commit()
        self.assertEqual(created, 0)

        rows = list(self.db.scalars(select(HealthSample).where(HealthSample.sample_type == "weight")))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source, "withings")
        self.assertEqual(rows[0].value, Decimal("79.85"))
        self.assertEqual(rows[0].external_id, "withings_99_1")
        self.assertEqual(rows[0].extra["device"], "Alex Scale")
        self.assertEqual(rows[0].extra["grpid"], 99)

    def test_scale_dedup_61s_boundary_and_resting_hr_exclusion(self) -> None:
        t0 = datetime(2026, 11, 25, 7, 0, 0, tzinfo=timezone.utc)
        t_61s = t0 + timedelta(seconds=61)
        t_15s = t0 + timedelta(seconds=15)

        upsert_health_samples(
            self.db,
            [
                NormalizedSample("weight", t0, Decimal("81.0"), unit="kg", source="withings"),
                NormalizedSample("weight", t_61s, Decimal("81.2"), unit="kg", source="apple_health"),
                NormalizedSample("resting_hr", t0, Decimal("54"), unit="bpm", source="withings"),
                NormalizedSample("resting_hr", t_15s, Decimal("55"), unit="bpm", source="apple_health"),
            ],
        )
        self.db.commit()

        weights = list(self.db.scalars(select(HealthSample).where(HealthSample.sample_type == "weight")))
        rhrs = list(self.db.scalars(select(HealthSample).where(HealthSample.sample_type == "resting_hr")))
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(rhrs), 2)

    def test_normalize_utc_second_handles_naive_and_offset_datetimes(self) -> None:
        naive = datetime(2026, 11, 25, 7, 30, 15, 500000)
        aware_utc = datetime(2026, 11, 25, 7, 30, 15, 123000, tzinfo=timezone.utc)
        aware_nz = datetime(2026, 11, 25, 20, 30, 15, tzinfo=ZoneInfo("Pacific/Auckland"))
        self.assertEqual(normalize_utc_second(naive), normalize_utc_second(aware_utc))
        self.assertEqual(normalize_utc_second(aware_nz), normalize_utc_second(aware_utc))

    def test_ingestor_empty_workout_sync_and_direct_scale_unification(self) -> None:
        fake_sync = _FakeWorkoutSync()
        fake_rollups = _FakeRollups(self.db, self.nz_tz)
        ingestor = TelemetryIngestor(self.db, rollups=fake_rollups, workout_sync=fake_sync)

        empty_res = ingestor.ingest_apple_health({"source": "apple_health"})
        self.assertEqual(empty_res.status, "empty")
        self.assertEqual(empty_res.total_samples_ingested, 0)
        self.assertEqual(fake_sync.calls, [])

        # Apple Watch steps earlier in the day
        day_dt = datetime(2026, 11, 25, 8, 0, tzinfo=self.nz_tz)
        ingestor.ingest_samples(
            [
                NormalizedSample(
                    sample_type="step_count",
                    measured_at=day_dt,
                    ended_at=day_dt + timedelta(minutes=15),
                    value=Decimal("1200"),
                    unit="count",
                    source="apple_health",
                )
            ],
            sync_workouts=True,
        )
        self.assertEqual(fake_sync.calls, [])

        # Direct Withings scale reading unifies into the same BiometricMeasurement row
        scale_payload = WithingsMeasurementPayload(
            measured_at=day_dt + timedelta(minutes=30),
            weight_kg=Decimal("82.4"),
            body_fat_pct=Decimal("16.5"),
            muscle_mass_kg=Decimal("64.9"),
            water_pct=Decimal("58.1"),
            bone_mass_kg=Decimal("3.4"),
            resting_hr_bpm=52,
            notes="Morning weigh-in",
            extra={"scale_model": "Body Smart"},
        )
        measurement = ingestor.ingest_direct_scale(scale_payload)
        self.assertEqual(measurement.source, "withings")
        self.assertEqual(measurement.weight_kg, Decimal("82.4"))
        self.assertEqual(measurement.step_count, 1200)
        self.assertEqual(measurement.resting_hr_bpm, 52)
        self.assertEqual(measurement.notes, "Morning weigh-in")
        self.assertEqual(measurement.extra["scale_model"], "Body Smart")

        all_measurements = list(self.db.scalars(select(BiometricMeasurement)))
        self.assertEqual(len(all_measurements), 1)


class TestWithingsSyncAndHttpAdapter(EirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config = WithingsConfig(
            client_id="cid-1",
            client_secret="csec-1",
            redirect_uri="https://eir.example.test/integrations/withings/callback",
        )
        self.fixed_now = datetime(2026, 11, 25, 10, 0, 0, tzinfo=timezone.utc)

    def test_config_requires_explicit_values_without_hardcoded_fallback(self) -> None:
        cfg_missing_redirect = WithingsConfig(client_id="cid-1", client_secret="sec", redirect_uri="  ")
        with self.assertRaises(WithingsConfigError):
            cfg_missing_redirect.require_authorize()
        with self.assertRaises(WithingsConfigError):
            cfg_missing_redirect.require_oauth()
        with self.assertRaises(WithingsConfigError):
            _ = cfg_missing_redirect.webhook_callback_url
        self.assertEqual(cfg_missing_redirect.require_refresh(), ("cid-1", "sec"))
        self.assertEqual(
            self.config.webhook_callback_url,
            "https://eir.example.test/integrations/withings/webhook",
        )

    def test_decode_measure_groups_scaling_and_window_resolution(self) -> None:
        now = self.fixed_now
        self.assertEqual(resolve_measure_window("1700000100", None, now), (1700000100, None))
        self.assertEqual(
            resolve_measure_window(None, None, now),
            (int(now.timestamp()) - 86400, None),
        )
        self.assertEqual(resolve_measure_window("1700000100", "1700000200", now), (1700000100, 1700000200))

        groups = (
            WithingsMeasureGroup(
                grpid=501,
                date=int(now.timestamp()),
                attrib=0,
                measures=(
                    WithingsMeasure(type=1, value=82450, unit=-3),
                    WithingsMeasure(type=6, value=1685, unit=-2),
                    WithingsMeasure(type=11, value=54, unit=0),
                    WithingsMeasure(type=999, value=123, unit=0),
                ),
            ),
        )
        decoded = decode_measure_groups(groups)
        self.assertEqual(len(decoded), 3)
        self.assertEqual(decoded[0].sample_type, "weight")
        self.assertEqual(decoded[0].value, Decimal("82.450"))
        self.assertEqual(decoded[0].external_id, "withings_501_1")
        self.assertEqual(decoded[1].sample_type, "body_fat_pct")
        self.assertEqual(decoded[1].value, Decimal("16.85"))
        self.assertEqual(decoded[2].sample_type, "resting_hr")
        self.assertEqual(decoded[2].value, Decimal("54"))

    def test_oauth_callback_persists_token_and_best_effort_subscribe(self) -> None:
        cloud = InMemoryWithingsCloudAdapter(
            token_set=WithingsTokenSet(
                user_id="u-100",
                access_token="acc-1",
                refresh_token="ref-1",
                expires_in=3600,
                scope="user.metrics",
            ),
            subscribe_error=WithingsUpstreamError("notify down"),
        )
        sync = WithingsSync(self.db, config=self.config, cloud=cloud, clock=lambda: self.fixed_now)

        auth = sync.build_authorize_url()
        self.assertIn("client_id=cid-1", auth.authorization_url)
        self.assertIn("eir.example.test", auth.authorization_url)

        res = sync.complete_oauth_callback("auth-code-1")
        self.assertEqual(res.user_id, "u-100")
        self.assertEqual(res.subscription, "unattempted")
        self.assertEqual(
            cloud.subscribe_calls[0][1],
            "https://eir.example.test/integrations/withings/webhook",
        )

        # connection_status works purely from DB without OAuth config
        status_only = WithingsSync(self.db, config=WithingsConfig()).connection_status()
        self.assertTrue(status_only.linked)
        self.assertEqual(status_only.user_id, "u-100")

    def test_webhook_refreshes_expiring_token_and_ingests_measures(self) -> None:
        # Insert naive datetime (SQLite behavior) expiring within 4 minutes of fixed_now
        self.db.add(
            OAuthToken(
                provider="withings",
                user_id="u-100",
                access_token="stale-acc",
                refresh_token="old-ref",
                scope="user.metrics",
                expires_at=(self.fixed_now + timedelta(minutes=4)).replace(tzinfo=None),
            )
        )
        self.db.commit()

        groups = (
            WithingsMeasureGroup(
                grpid=777,
                date=int(self.fixed_now.timestamp()),
                attrib=0,
                measures=(WithingsMeasure(type=1, value=8120, unit=-2),),
            ),
        )
        cloud = InMemoryWithingsCloudAdapter(
            refreshed_token_set=WithingsTokenSet(
                user_id="u-100",
                access_token="fresh-acc",
                refresh_token="fresh-ref",
                expires_in=7200,
            ),
            measure_groups=groups,
        )
        sync = WithingsSync(self.db, config=self.config, cloud=cloud, clock=lambda: self.fixed_now)

        # Ignore appli != 1
        ignored = sync.handle_webhook({"appli": "44", "userid": "u-100"})
        self.assertEqual(ignored.samples_ingested, 0)
        self.assertEqual(cloud.refresh_calls, [])

        # Refresh + poll when appli == 1
        res = sync.handle_webhook({"appli": "1", "userid": "u-100", "startdate": "1700000000"})
        self.assertEqual(res.samples_ingested, 1)
        self.assertEqual(len(cloud.refresh_calls), 1)
        self.assertEqual(cloud.measure_calls, [("fresh-acc", 1700000000, None)])

        stored_token = self.db.scalar(select(OAuthToken))
        self.assertEqual(stored_token.access_token, "fresh-acc")
        self.assertEqual(stored_token.refresh_token, "fresh-ref")

        sample = self.db.scalar(select(HealthSample).where(HealthSample.sample_type == "weight"))
        self.assertIsNotNone(sample)
        self.assertEqual(sample.value, Decimal("81.20"))

    def test_webhook_unlinked_and_upstream_errors_return_zero_safely(self) -> None:
        cloud = InMemoryWithingsCloudAdapter(measure_error=WithingsUpstreamError("timeout"))
        sync = WithingsSync(self.db, config=self.config, cloud=cloud, clock=lambda: self.fixed_now)

        # Unlinked user
        self.assertEqual(sync.handle_webhook({"appli": "1"}).samples_ingested, 0)

        # Linked user with upstream error on getmeas -> catches and returns 0
        self.db.add(
            OAuthToken(
                provider="withings",
                user_id="u-200",
                access_token="valid-acc",
                refresh_token="valid-ref",
                expires_at=self.fixed_now + timedelta(hours=2),
            )
        )
        self.db.commit()
        res = sync.handle_webhook({"appli": "1", "userid": "u-200"})
        self.assertEqual(res.samples_ingested, 0)

    def test_http_adapter_maps_upstream_and_protocol_errors(self) -> None:
        def error_handler(request: httpx.Request) -> httpx.Response:
            if "protocol-err" in request.content.decode():
                return httpx.Response(200, json={"status": 293, "error": "invalid code"})
            return httpx.Response(503, text="service unavailable")

        with httpx.Client(transport=httpx.MockTransport(error_handler)) as client:
            adapter = HttpWithingsCloudAdapter(client=client)
            with self.assertRaises(WithingsProtocolError):
                adapter.exchange_code(self.config, "protocol-err")
            with self.assertRaises(WithingsUpstreamError):
                adapter.exchange_code(self.config, "bad-gateway")

    def test_router_callback_maps_config_protocol_and_upstream_http_codes(self) -> None:
        self.patch_settings(
            withings_client_id=None,
            withings_client_secret=None,
            withings_redirect_uri=None,
        )
        r500 = self.client.get("/integrations/withings/callback?code=abc")
        self.assertEqual(r500.status_code, 500)

        self.patch_settings(
            withings_client_id="cid",
            withings_client_secret="csec",
            withings_redirect_uri="https://eir.example.test/integrations/withings/callback",
        )
        self.override_withings_sync(
            InMemoryWithingsCloudAdapter(exchange_error=WithingsProtocolError("Withings error 503: bad code"))
        )
        r400 = self.client.get("/integrations/withings/callback?code=proto")
        self.assertEqual(r400.status_code, 400)

        self.override_withings_sync(
            InMemoryWithingsCloudAdapter(exchange_error=WithingsUpstreamError("Withings token exchange failed with HTTP 502"))
        )
        r502 = self.client.get("/integrations/withings/callback?code=up")
        self.assertEqual(r502.status_code, 502)

    def test_ingestion_package_acyclic_dag_and_zero_dashes(self) -> None:
        app_root = Path(__file__).resolve().parents[1] / "app"
        for py_path in sorted((app_root / "ingestion").glob("*.py")):
            text = py_path.read_text(encoding="utf-8")
            self.assertNotIn("\u2014", text, f"Em dash found in {py_path}")
            self.assertNotIn("\u2013", text, f"En dash found in {py_path}")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith("fastapi"))
                    self.assertFalse(node.module.startswith("starlette"))
                    self.assertFalse(node.module.startswith("app.routers"))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith("fastapi"))
                        self.assertFalse(alias.name.startswith("starlette"))


if __name__ == "__main__":
    unittest.main()
