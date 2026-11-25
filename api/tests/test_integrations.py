"""Tests for Withings and Apple Health integration webhooks in Eir."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import select

from app.ingestion import (
    InMemoryWithingsCloudAdapter,
    WithingsMeasure,
    WithingsMeasureGroup,
    WithingsTokenSet,
)
from app.models import BiometricMeasurement, HealthSample, OAuthToken
from tests._harness import EirTestCase


class TestIntegrationsRouter(EirTestCase):
    """Router and end-to-end integration tests for /integrations/*."""

    def test_withings_webhook_verification_get_and_head(self) -> None:
        get_res = self.client.get("/integrations/withings/webhook")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["status"], "ok")

        head_res = self.client.head("/integrations/withings/webhook")
        self.assertEqual(head_res.status_code, 200)

    def test_withings_measurements_ingestion(self) -> None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        response = self.client.post(
            "/integrations/withings/measurements",
            json={
                "measured_at": now,
                "weight_kg": 87.2,
                "body_fat_pct": 17.8,
                "muscle_mass_kg": 68.4,
                "water_pct": 58.2,
                "bone_mass_kg": 3.6,
                "resting_hr_bpm": 53,
                "notes": "Post-waking weigh-in",
            },
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["source"], "withings")
        self.assertEqual(float(data["weight_kg"]), 87.2)
        self.assertEqual(float(data["body_fat_pct"]), 17.8)

    def test_webhook_token_authentication_and_mcp_fallback(self) -> None:
        self.patch_settings(webhook_secret="secret-test-token-123")
        payload = {"measured_at": dt.datetime.now(dt.timezone.utc).isoformat(), "weight_kg": 85.0}

        self.assertEqual(self.client.post("/integrations/withings/measurements", json=payload).status_code, 401)
        self.assertEqual(
            self.client.post(
                "/integrations/withings/measurements",
                json=payload,
                headers={"Authorization": "Bearer wrong-token"},
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.post(
                "/integrations/withings/measurements",
                json=payload,
                headers={"Authorization": "Bearer secret-test-token-123"},
            ).status_code,
            201,
        )
        self.assertEqual(
            self.client.post(
                "/integrations/withings/measurements",
                json=payload,
                headers={"X-Webhook-Secret": "secret-test-token-123"},
            ).status_code,
            201,
        )
        self.assertEqual(
            self.client.post(
                "/integrations/withings/measurements?secret=secret-test-token-123",
                json=payload,
            ).status_code,
            201,
        )

        # Fallback to mcp_token and bare Authorization header when webhook_secret is unset
        self.patch_settings(webhook_secret=None, mcp_token="mcp-fallback-token")
        self.assertEqual(
            self.client.post(
                "/integrations/withings/measurements",
                json=payload,
                headers={"Authorization": "mcp-fallback-token"},
            ).status_code,
            201,
        )
        self.assertEqual(
            self.client.post(
                "/integrations/withings/measurements",
                json=payload,
                headers={"Authorization": "wrong-bare-token"},
            ).status_code,
            401,
        )

    def test_apple_health_comprehensive_sync_two_days(self) -> None:
        today = dt.date.today()
        yesterday = today - dt.timedelta(days=1)

        t_yest_morn = dt.datetime.combine(yesterday, dt.time(7, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_today_morn = dt.datetime.combine(today, dt.time(7, 45)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_yest_hr1 = dt.datetime.combine(yesterday, dt.time(12, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_yest_hr2 = dt.datetime.combine(yesterday, dt.time(12, 5)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_today_hr1 = dt.datetime.combine(today, dt.time(10, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_today_hr2 = dt.datetime.combine(today, dt.time(10, 5)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_yest_sleep_st = (
            dt.datetime.combine(yesterday - dt.timedelta(days=1), dt.time(23, 0))
            .replace(tzinfo=dt.timezone.utc)
            .isoformat()
        )
        t_yest_sleep_end = dt.datetime.combine(yesterday, dt.time(7, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_today_sleep_st = dt.datetime.combine(yesterday, dt.time(23, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_today_sleep_end = dt.datetime.combine(today, dt.time(7, 15)).replace(tzinfo=dt.timezone.utc).isoformat()

        payload = {
            "weight_dates": [t_yest_morn, t_today_morn],
            "weight_values": ["88.0 kg", "87.5 kg"],
            "resting_hr_dates": [t_yest_morn, t_today_morn],
            "resting_hr_values": [52, 50],
            "hrv_dates": [t_yest_morn, t_today_morn],
            "hrv_values": ["62 ms", "68 ms"],
            "heart_rate_dates": [t_yest_hr1, t_yest_hr2, t_today_hr1, t_today_hr2],
            "heart_rate_values": [140, 145, 130, 135],
            "sleep_start_dates": [t_yest_sleep_st, t_today_sleep_st],
            "sleep_end_dates": [t_yest_sleep_end, t_today_sleep_end],
            "sleep_durations": [28800, 27900],
            "active_energy_dates": [t_yest_hr1, t_today_hr1],
            "active_energy_values": [250, 300],
        }

        res = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["total_samples_ingested"], 14)
        self.assertEqual(data["metrics_summary"]["heart_rate"], 4)
        self.assertEqual(data["metrics_summary"]["weight"], 2)
        self.assertGreaterEqual(len(data["days_processed"]), 2)

        gran_res = self.client.get(f"/biometrics/granular?target_date={today.isoformat()}&sample_type=heart_rate")
        self.assertEqual(gran_res.status_code, 200)
        gran_data = gran_res.json()
        self.assertEqual(len(gran_data), 2)
        self.assertEqual(float(gran_data[0]["value"]), 130.0)
        self.assertEqual(float(gran_data[1]["value"]), 135.0)

        res_repeat = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res_repeat.status_code, 201)
        self.assertEqual(res_repeat.json()["total_samples_ingested"], 14)

        today_summary = self.client.get(f"/biometrics/daily/{today.isoformat()}").json()
        self.assertEqual(float(today_summary["latest_weight_kg"]), 87.5)
        self.assertEqual(today_summary["resting_hr_bpm"], 50)
        self.assertEqual(today_summary["active_energy_kcal"], 300)

    def test_apple_health_sync_multiline_strings_support(self) -> None:
        today = dt.date.today()
        t1 = dt.datetime.combine(today, dt.time(10, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t2 = dt.datetime.combine(today, dt.time(10, 5)).replace(tzinfo=dt.timezone.utc).isoformat()

        payload = {
            "active_energy_dates": f"{t1}\n{t2}",
            "active_energy_values": "5.5 kcal\n4.5 kcal",
        }
        res = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["total_samples_ingested"], 2)
        self.assertEqual(res.json()["metrics_summary"]["active_energy"], 2)

    def test_apple_health_sync_keyless_nested_dictionary_support(self) -> None:
        today = dt.date.today()
        t1 = dt.datetime.combine(today, dt.time(12, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t2 = dt.datetime.combine(today, dt.time(12, 10)).replace(tzinfo=dt.timezone.utc).isoformat()

        payload = {
            "": {
                "active_energy_dates": f"{t1}\n{t2}",
                "active_energy_values": "8.0 kcal\n9.5 kcal",
            }
        }
        res = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["total_samples_ingested"], 2)
        self.assertEqual(res.json()["metrics_summary"]["active_energy"], 2)

    def test_apple_health_sync_root_level_metric_support(self) -> None:
        today = dt.date.today()
        t1 = dt.datetime.combine(today, dt.time(11, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t2 = dt.datetime.combine(today, dt.time(11, 15)).replace(tzinfo=dt.timezone.utc).isoformat()

        payload = {
            "sample_type": "active_energy",
            "dates": [t1, t2],
            "values": [12.0, 18.0],
        }
        res = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["total_samples_ingested"], 2)
        self.assertEqual(res.json()["metrics_summary"]["active_energy"], 2)

    def test_apple_health_sync_human_locale_date_format(self) -> None:
        payload = {
            "active_energy_dates": "29 Oct 2026 at 6:35\u202fPM\n29 Oct 2026 at 6:40\u202fPM",
            "active_energy_values": "15.0\n20.0",
        }
        res = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["total_samples_ingested"], 2)
        self.assertEqual(res.json()["metrics_summary"]["active_energy"], 2)

    def test_apple_health_sync_invalid_payload_returns_422(self) -> None:
        res = self.client.post(
            "/integrations/apple-health/sync",
            json={
                "active_energy_dates": "not-a-valid-date",
                "active_energy_values": "15.0",
            },
        )
        self.assertEqual(res.status_code, 422)

    def test_apple_health_sync_sleep_stages_and_duration(self) -> None:
        today = dt.date.today()
        t1_s = dt.datetime.combine(today, dt.time(1, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t1_e = dt.datetime.combine(today, dt.time(1, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
        t2_s = dt.datetime.combine(today, dt.time(1, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
        t2_e = dt.datetime.combine(today, dt.time(2, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
        t3_s = dt.datetime.combine(today, dt.time(2, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
        t3_e = dt.datetime.combine(today, dt.time(2, 45)).replace(tzinfo=dt.timezone.utc).isoformat()

        payload = {
            "sleep_start_dates": f"{t1_s}\n{t2_s}\n{t3_s}",
            "sleep_end_dates": f"{t1_e}\n{t2_e}\n{t3_e}",
            "sleep_values": "Deep\nCore\nAwake",
        }
        res = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["total_samples_ingested"], 3)
        self.assertEqual(res.json()["metrics_summary"]["sleep"], 3)

        samples = self.db.scalars(
            select(HealthSample).where(HealthSample.sample_type == "sleep").order_by(HealthSample.measured_at.asc())
        ).all()
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].extra.get("stage"), "Deep")
        self.assertEqual(samples[0].value, Decimal("1800"))
        self.assertEqual(samples[1].extra.get("stage"), "Core")
        self.assertEqual(samples[1].value, Decimal("3600"))
        self.assertEqual(samples[2].extra.get("stage"), "Awake")
        self.assertEqual(samples[2].value, Decimal("900"))

        measurement = self.db.scalar(select(BiometricMeasurement))
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.sleep_duration_s, 5400)
        self.assertEqual(measurement.sleep_deep_s, 1800)
        self.assertIsNone(measurement.sleep_rem_s)

    def test_apple_health_sync_duration_string_parsing(self) -> None:
        today = dt.date.today()
        t1_s = dt.datetime.combine(today, dt.time(3, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t2_s = dt.datetime.combine(today, dt.time(3, 30)).replace(tzinfo=dt.timezone.utc).isoformat()

        payload = {
            "sleep_start_dates": f"{t1_s}\n{t2_s}",
            "sleep_durations": "19:25\n1:18:43",
            "sleep_values": "REM\nDeep",
        }
        res = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["total_samples_ingested"], 2)

        samples = self.db.scalars(
            select(HealthSample).where(HealthSample.sample_type == "sleep").order_by(HealthSample.measured_at.asc())
        ).all()
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].value, Decimal("1165"))
        self.assertEqual(samples[0].extra.get("stage"), "REM")
        self.assertEqual(samples[1].value, Decimal("4723"))
        self.assertEqual(samples[1].extra.get("stage"), "Deep")

    def test_apple_health_sync_smart_step_deduplication(self) -> None:
        today = dt.date.today()
        w1_s = dt.datetime.combine(today, dt.time(12, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        w1_e = dt.datetime.combine(today, dt.time(12, 10)).replace(tzinfo=dt.timezone.utc).isoformat()
        w2_s = dt.datetime.combine(today, dt.time(12, 10)).replace(tzinfo=dt.timezone.utc).isoformat()
        w2_e = dt.datetime.combine(today, dt.time(12, 20)).replace(tzinfo=dt.timezone.utc).isoformat()
        p_ov_s = dt.datetime.combine(today, dt.time(12, 5)).replace(tzinfo=dt.timezone.utc).isoformat()
        p_ov_e = dt.datetime.combine(today, dt.time(12, 15)).replace(tzinfo=dt.timezone.utc).isoformat()
        p_ch_s = dt.datetime.combine(today, dt.time(8, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        p_ch_e = dt.datetime.combine(today, dt.time(8, 15)).replace(tzinfo=dt.timezone.utc).isoformat()

        payload = {
            "step_start_dates": f"{w1_s}\n{w2_s}\n{p_ov_s}\n{p_ch_s}",
            "step_end_dates": f"{w1_e}\n{w2_e}\n{p_ov_e}\n{p_ch_e}",
            "step_values": "200\n300\n250\n80",
            "step_sources": "Alex's Apple Watch\nAlex's Apple Watch\nAlex's iPhone\nAlex's iPhone",
        }
        res = self.client.post("/integrations/apple-health/sync", json=payload)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["total_samples_ingested"], 4)

        samples = self.db.scalars(
            select(HealthSample).where(HealthSample.sample_type == "step_count").order_by(HealthSample.measured_at.asc())
        ).all()
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].extra.get("device"), "Alex's iPhone")
        self.assertEqual(samples[1].extra.get("device"), "Alex's Apple Watch")

        measurement = self.db.scalar(select(BiometricMeasurement))
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.step_count, 580)

    def test_withings_authorize_json_redirect_and_missing_config(self) -> None:
        self.patch_settings(
            withings_client_id="test-client-id-123",
            withings_redirect_uri="https://eir.example.test/integrations/withings/callback",
        )
        json_res = self.client.get("/integrations/withings/authorize?format=json")
        self.assertEqual(json_res.status_code, 200)
        data = json_res.json()
        self.assertIn("https://account.withings.com/oauth2_user/authorize2", data["authorization_url"])
        self.assertIn("client_id=test-client-id-123", data["authorization_url"])

        redir_res = self.client.get("/integrations/withings/authorize", follow_redirects=False)
        self.assertEqual(redir_res.status_code, 307)
        self.assertIn("client_id=test-client-id-123", redir_res.headers["location"])

        self.patch_settings(withings_redirect_uri=None)
        err_res = self.client.get("/integrations/withings/authorize?format=json")
        self.assertEqual(err_res.status_code, 500)
        self.assertIn("EIR_WITHINGS_REDIRECT_URI", err_res.json()["detail"])

    def test_withings_oauth_callback_200_via_di_override(self) -> None:
        self.patch_settings(
            withings_client_id="cid-123",
            withings_client_secret="csec-123",
            withings_redirect_uri="https://eir.example.test/integrations/withings/callback",
        )
        cloud = InMemoryWithingsCloudAdapter(
            token_set=WithingsTokenSet(
                user_id="withings-user-42",
                access_token="acc-ok",
                refresh_token="ref-ok",
                expires_in=3600,
                scope="user.metrics",
            )
        )
        self.override_withings_sync(cloud)

        res = self.client.get("/integrations/withings/callback?code=auth-ok&state=csrf-1")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["user_id"], "withings-user-42")
        self.assertEqual(body["subscription"], {"status": 0, "body": {}})

    def test_withings_webhook_post_form_and_json(self) -> None:
        self.patch_settings(
            withings_client_id="cid-123",
            withings_client_secret="csec-123",
            withings_redirect_uri="https://eir.example.test/integrations/withings/callback",
        )
        now = dt.datetime.now(dt.timezone.utc)
        self.db.add(
            OAuthToken(
                provider="withings",
                user_id="u-1",
                access_token="acc-valid",
                refresh_token="ref-valid",
                scope="user.metrics",
                expires_at=now + dt.timedelta(hours=2),
            )
        )
        self.db.commit()

        cloud = InMemoryWithingsCloudAdapter(
            measure_groups=(
                WithingsMeasureGroup(
                    grpid=901,
                    date=int(now.timestamp()),
                    attrib=0,
                    measures=(WithingsMeasure(type=1, value=81450, unit=-3),),
                ),
            )
        )
        self.override_withings_sync(cloud)

        ignored = self.client.post("/integrations/withings/webhook", json={"appli": "44", "userid": "u-1"})
        self.assertEqual(ignored.status_code, 200)
        self.assertEqual(ignored.json()["samples_ingested"], 0)

        form_res = self.client.post("/integrations/withings/webhook", data={"appli": "1", "userid": "u-1"})
        self.assertEqual(form_res.status_code, 200)
        self.assertEqual(form_res.json()["samples_ingested"], 1)

        sample = self.db.scalar(select(HealthSample).where(HealthSample.sample_type == "weight"))
        self.assertIsNotNone(sample)
        self.assertEqual(sample.value, Decimal("81.450"))

    def test_withings_status_endpoint(self) -> None:
        res = self.client.get("/integrations/withings/status")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "disconnected")
        self.assertFalse(res.json()["linked"])

        self.db.add(
            OAuthToken(
                provider="withings",
                user_id="user-999",
                access_token="acc-123",
                refresh_token="ref-123",
                scope="user.metrics",
                expires_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3),
            )
        )
        self.db.commit()

        res = self.client.get("/integrations/withings/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "connected")
        self.assertTrue(data["linked"])
        self.assertEqual(data["user_id"], "user-999")

    def test_cross_source_scale_deduplication(self) -> None:
        today = dt.date.today()
        t_withings = dt.datetime.combine(today, dt.time(7, 30, 15)).replace(tzinfo=dt.timezone.utc)
        self.db.add(
            HealthSample(
                source="withings",
                sample_type="weight",
                measured_at=t_withings,
                ended_at=t_withings,
                value=Decimal("82.5"),
                unit="kg",
                external_id="withings_12345_1",
            )
        )
        self.db.commit()

        t_apple = dt.datetime.combine(today, dt.time(7, 30, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        res = self.client.post(
            "/integrations/apple-health/sync",
            json={"weight_dates": t_apple, "weight_values": "82.5"},
        )
        self.assertEqual(res.status_code, 201)

        samples = self.db.scalars(select(HealthSample).where(HealthSample.sample_type == "weight")).all()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].value, Decimal("82.5"))

    def test_unified_daily_rollup_merges_scale_and_wearables(self) -> None:
        today = dt.date.today()
        t_morning = dt.datetime.combine(today, dt.time(7, 0)).replace(tzinfo=dt.timezone.utc)
        for sample_type, val, unit in (
            ("weight", "81.4", "kg"),
            ("body_fat_pct", "16.8", "%"),
            ("muscle_mass_kg", "64.2", "kg"),
            ("water_pct", "58.5", "%"),
        ):
            self.db.add(
                HealthSample(
                    source="withings",
                    sample_type=sample_type,
                    measured_at=t_morning,
                    ended_at=t_morning,
                    value=Decimal(val),
                    unit=unit,
                )
            )
        self.db.commit()

        t_sleep_s = dt.datetime.combine(today, dt.time(0, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_sleep_e = dt.datetime.combine(today, dt.time(7, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_step_s = dt.datetime.combine(today, dt.time(8, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
        t_step_e = dt.datetime.combine(today, dt.time(8, 30)).replace(tzinfo=dt.timezone.utc).isoformat()

        res = self.client.post(
            "/integrations/apple-health/sync",
            json={
                "sleep_start_dates": t_sleep_s,
                "sleep_end_dates": t_sleep_e,
                "sleep_values": "Deep",
                "step_start_dates": t_step_s,
                "step_end_dates": t_step_e,
                "step_values": "1500",
            },
        )
        self.assertEqual(res.status_code, 201)

        measurements = self.db.scalars(select(BiometricMeasurement)).all()
        self.assertEqual(len(measurements), 1)
        m = measurements[0]
        self.assertEqual(float(m.weight_kg), 81.4)
        self.assertEqual(float(m.body_fat_pct), 16.8)
        self.assertEqual(float(m.muscle_mass_kg), 64.2)
        self.assertEqual(float(m.water_pct), 58.5)
        self.assertEqual(m.sleep_duration_s, 23400)
        self.assertEqual(m.sleep_deep_s, 23400)
        self.assertEqual(m.step_count, 1500)

    def test_withings_measurements_repeated_within_window_dedups_and_unifies(self) -> None:
        t1 = dt.datetime(2026, 11, 25, 7, 30, 0, tzinfo=dt.timezone.utc)
        t2 = t1 + dt.timedelta(seconds=30)

        res1 = self.client.post(
            "/integrations/withings/measurements",
            json={
                "measured_at": t1.isoformat(),
                "weight_kg": 82.4,
                "body_fat_pct": 16.5,
                "notes": "First reading",
                "extra": {"scale": "body-smart"},
            },
        )
        self.assertEqual(res1.status_code, 201)

        res2 = self.client.post(
            "/integrations/withings/measurements",
            json={
                "measured_at": t2.isoformat(),
                "weight_kg": 82.1,
                "body_fat_pct": 16.3,
                "notes": "Settled reading",
                "extra": {"firmware": "v2"},
            },
        )
        self.assertEqual(res2.status_code, 201)
        body = res2.json()
        self.assertEqual(body["source"], "withings")
        self.assertEqual(float(body["weight_kg"]), 82.1)
        self.assertEqual(float(body["body_fat_pct"]), 16.3)
        self.assertEqual(body["notes"], "Settled reading")
        self.assertEqual(body["extra"]["scale"], "body-smart")
        self.assertEqual(body["extra"]["firmware"], "v2")

        weights = self.db.scalars(select(HealthSample).where(HealthSample.sample_type == "weight")).all()
        fats = self.db.scalars(select(HealthSample).where(HealthSample.sample_type == "body_fat_pct")).all()
        measurements = self.db.scalars(select(BiometricMeasurement)).all()
        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0].value, Decimal("82.1"))
        self.assertEqual(len(fats), 1)
        self.assertEqual(fats[0].value, Decimal("16.3"))
        self.assertEqual(len(measurements), 1)
