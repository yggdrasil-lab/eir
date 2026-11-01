"""Tests for Withings and Apple Health integration webhooks in Eir."""

import datetime as dt
from decimal import Decimal

from app.schemas import _clean_numeric


def test_clean_numeric_multiline_and_strings() -> None:
    assert _clean_numeric("2 kcal\n1.5 kcal\n3.0 kcal") == Decimal("6.5")
    assert _clean_numeric(["10", "20.5"]) == Decimal("30.5")
    assert _clean_numeric({"value": "72.5 kg"}) == Decimal("72.5")
    assert _clean_numeric("invalid") is None
    assert _clean_numeric(None) is None


def test_withings_webhook_verification(client) -> None:
    response = client.get("/integrations/withings/webhook")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_withings_measurements_ingestion(client) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    response = client.post(
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
    assert response.status_code == 201
    data = response.json()
    assert data["source"] == "withings"
    assert float(data["weight_kg"]) == 87.2
    assert float(data["body_fat_pct"]) == 17.8


def test_webhook_token_authentication(client, monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "webhook_secret", "secret-test-token-123")

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "measured_at": now,
        "weight_kg": 85.0,
    }

    # Request without token should fail 401
    unauth = client.post("/integrations/withings/measurements", json=payload)
    assert unauth.status_code == 401

    # Request with wrong token should fail 401
    wrong = client.post(
        "/integrations/withings/measurements",
        json=payload,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert wrong.status_code == 401

    # Request with valid Bearer token should succeed 201
    valid_bearer = client.post(
        "/integrations/withings/measurements",
        json=payload,
        headers={"Authorization": "Bearer secret-test-token-123"},
    )
    assert valid_bearer.status_code == 201

    # Request with valid X-Webhook-Secret header should succeed 201
    valid_header = client.post(
        "/integrations/withings/measurements",
        json=payload,
        headers={"X-Webhook-Secret": "secret-test-token-123"},
    )
    assert valid_header.status_code == 201

    # Request with valid query parameter should succeed 201
    valid_query = client.post(
        "/integrations/withings/measurements?secret=secret-test-token-123",
        json=payload,
    )
    assert valid_query.status_code == 201


def test_apple_health_comprehensive_sync_two_days(client) -> None:
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    t_yest_morn = dt.datetime.combine(yesterday, dt.time(7, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
    t_today_morn = dt.datetime.combine(today, dt.time(7, 45)).replace(tzinfo=dt.timezone.utc).isoformat()

    t_yest_hr1 = dt.datetime.combine(yesterday, dt.time(12, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    t_yest_hr2 = dt.datetime.combine(yesterday, dt.time(12, 5)).replace(tzinfo=dt.timezone.utc).isoformat()
    t_today_hr1 = dt.datetime.combine(today, dt.time(10, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    t_today_hr2 = dt.datetime.combine(today, dt.time(10, 5)).replace(tzinfo=dt.timezone.utc).isoformat()

    t_yest_sleep_st = dt.datetime.combine(yesterday - dt.timedelta(days=1), dt.time(23, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
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

    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "ok"
    assert data["total_samples_ingested"] == 14
    assert data["metrics_summary"]["heart_rate"] == 4
    assert data["metrics_summary"]["weight"] == 2
    assert len(data["days_processed"]) >= 2

    # Check granular endpoint for today's heart rate
    gran_res = client.get(f"/biometrics/granular?target_date={today.isoformat()}&sample_type=heart_rate")
    assert gran_res.status_code == 200
    gran_data = gran_res.json()
    assert len(gran_data) == 2
    assert float(gran_data[0]["value"]) == 130.0
    assert float(gran_data[1]["value"]) == 135.0

    # Idempotent re-sync
    res_repeat = client.post("/integrations/apple-health/sync", json=payload)
    assert res_repeat.status_code == 201
    assert res_repeat.json()["total_samples_ingested"] == 14

    # Verify daily summary for today
    today_summary = client.get(f"/biometrics/daily/{today.isoformat()}").json()
    assert float(today_summary["latest_weight_kg"]) == 87.5
    assert today_summary["resting_hr_bpm"] == 50
    assert today_summary["active_energy_kcal"] == 300


def test_apple_health_sync_multiline_strings_support(client) -> None:
    today = dt.date.today()
    t1 = dt.datetime.combine(today, dt.time(10, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    t2 = dt.datetime.combine(today, dt.time(10, 5)).replace(tzinfo=dt.timezone.utc).isoformat()

    # iOS Shortcuts can serialize lists as newline-separated strings in dictionary values
    payload = {
        "active_energy_dates": f"{t1}\n{t2}",
        "active_energy_values": "5.5 kcal\n4.5 kcal",
    }
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201
    assert res.json()["total_samples_ingested"] == 2
    assert res.json()["metrics_summary"]["active_energy"] == 2


def test_apple_health_sync_keyless_nested_dictionary_support(client) -> None:
    today = dt.date.today()
    t1 = dt.datetime.combine(today, dt.time(12, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    t2 = dt.datetime.combine(today, dt.time(12, 10)).replace(tzinfo=dt.timezone.utc).isoformat()

    # iOS Shortcuts keyless dictionary wrapper: {"": {"active_energy_dates": ..., ...}}
    payload = {
        "": {
            "active_energy_dates": f"{t1}\n{t2}",
            "active_energy_values": "8.0 kcal\n9.5 kcal",
        }
    }
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201
    assert res.json()["total_samples_ingested"] == 2
    assert res.json()["metrics_summary"]["active_energy"] == 2


def test_apple_health_sync_root_level_metric_support(client) -> None:
    today = dt.date.today()
    t1 = dt.datetime.combine(today, dt.time(11, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    t2 = dt.datetime.combine(today, dt.time(11, 15)).replace(tzinfo=dt.timezone.utc).isoformat()

    payload = {
        "sample_type": "active_energy",
        "dates": [t1, t2],
        "values": [12.0, 18.0],
    }
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201
    assert res.json()["total_samples_ingested"] == 2
    assert res.json()["metrics_summary"]["active_energy"] == 2


def test_apple_health_sync_human_locale_date_format(client) -> None:
    payload = {
        "active_energy_dates": "29 Oct 2026 at 6:35\u202fPM\n29 Oct 2026 at 6:40\u202fPM",
        "active_energy_values": "15.0\n20.0",
    }
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201
    assert res.json()["total_samples_ingested"] == 2
    assert res.json()["metrics_summary"]["active_energy"] == 2


def test_apple_health_sync_sleep_stages_and_duration(client, db) -> None:
    from app.models import BiometricMeasurement, HealthSample
    from sqlalchemy import select

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
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201
    assert res.json()["total_samples_ingested"] == 3
    assert res.json()["metrics_summary"]["sleep"] == 3

    samples = db.scalars(
        select(HealthSample).where(HealthSample.sample_type == "sleep").order_by(HealthSample.measured_at.asc())
    ).all()
    assert len(samples) == 3
    assert samples[0].extra.get("stage") == "Deep"
    assert samples[0].value == Decimal("1800")
    assert samples[1].extra.get("stage") == "Core"
    assert samples[1].value == Decimal("3600")
    assert samples[2].extra.get("stage") == "Awake"
    assert samples[2].value == Decimal("900")

    measurement = db.scalar(select(BiometricMeasurement))
    assert measurement is not None
    assert measurement.sleep_duration_s == 5400
    assert measurement.sleep_deep_s == 1800
    assert measurement.sleep_rem_s is None


def test_apple_health_sync_duration_string_parsing(client, db) -> None:
    from app.models import HealthSample
    from sqlalchemy import select

    today = dt.date.today()
    t1_s = dt.datetime.combine(today, dt.time(3, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    t2_s = dt.datetime.combine(today, dt.time(3, 30)).replace(tzinfo=dt.timezone.utc).isoformat()

    payload = {
        "sleep_start_dates": f"{t1_s}\n{t2_s}",
        "sleep_durations": "19:25\n1:18:43",
        "sleep_values": "REM\nDeep",
    }
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201
    assert res.json()["total_samples_ingested"] == 2

    samples = db.scalars(
        select(HealthSample).where(HealthSample.sample_type == "sleep").order_by(HealthSample.measured_at.asc())
    ).all()
    assert len(samples) == 2
    assert samples[0].value == Decimal("1165")
    assert samples[0].extra.get("stage") == "REM"
    assert samples[1].value == Decimal("4723")
    assert samples[1].extra.get("stage") == "Deep"


def test_apple_health_sync_smart_step_deduplication(client, db) -> None:
    from app.models import BiometricMeasurement, HealthSample
    from sqlalchemy import select

    today = dt.date.today()
    # 2 Watch intervals (12:00-12:10 = 200 steps, 12:10-12:20 = 300 steps) -> 500 steps
    w1_s = dt.datetime.combine(today, dt.time(12, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    w1_e = dt.datetime.combine(today, dt.time(12, 10)).replace(tzinfo=dt.timezone.utc).isoformat()
    w2_s = dt.datetime.combine(today, dt.time(12, 10)).replace(tzinfo=dt.timezone.utc).isoformat()
    w2_e = dt.datetime.combine(today, dt.time(12, 20)).replace(tzinfo=dt.timezone.utc).isoformat()

    # 1 Overlapping Phone interval (12:05-12:15 = 250 steps) -> should be suppressed
    p_ov_s = dt.datetime.combine(today, dt.time(12, 5)).replace(tzinfo=dt.timezone.utc).isoformat()
    p_ov_e = dt.datetime.combine(today, dt.time(12, 15)).replace(tzinfo=dt.timezone.utc).isoformat()

    # 1 Non-overlapping Phone interval when watch was charging (08:00-08:15 = 80 steps) -> should be counted
    p_ch_s = dt.datetime.combine(today, dt.time(8, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    p_ch_e = dt.datetime.combine(today, dt.time(8, 15)).replace(tzinfo=dt.timezone.utc).isoformat()

    payload = {
        "step_start_dates": f"{w1_s}\n{w2_s}\n{p_ov_s}\n{p_ch_s}",
        "step_end_dates": f"{w1_e}\n{w2_e}\n{p_ov_e}\n{p_ch_e}",
        "step_values": "200\n300\n250\n80",
        "step_sources": "Raymund's Apple Watch\nRaymund's Apple Watch\nRaymund's iPhone\nRaymund's iPhone",
    }
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201
    assert res.json()["total_samples_ingested"] == 4

    samples = db.scalars(
        select(HealthSample).where(HealthSample.sample_type == "step_count").order_by(HealthSample.measured_at.asc())
    ).all()
    assert len(samples) == 4
    assert samples[0].extra.get("device") == "Raymund's iPhone"
    assert samples[1].extra.get("device") == "Raymund's Apple Watch"

    # Rollup should count watch steps (200 + 300) + non-overlapping phone steps (80) = 580 (ignoring 250 overlapping phone steps)
    measurement = db.scalar(select(BiometricMeasurement))
    assert measurement is not None
    assert measurement.step_count == 580


def test_withings_authorize_endpoint(client, monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "withings_client_id", "test-client-id-123")
    monkeypatch.setattr(settings, "withings_redirect_uri", "https://eir.tienzo.net/integrations/withings/callback")

    res = client.get("/integrations/withings/authorize?format=json")
    assert res.status_code == 200
    data = res.json()
    assert "https://account.withings.com/oauth2_user/authorize2" in data["authorization_url"]
    assert "client_id=test-client-id-123" in data["authorization_url"]


def test_withings_status_endpoint(client, db) -> None:
    from app.models import OAuthToken

    # Initially disconnected
    res = client.get("/integrations/withings/status")
    assert res.status_code == 200
    assert res.json()["status"] == "disconnected"
    assert res.json()["linked"] is False

    # Insert token
    token = OAuthToken(
        provider="withings",
        user_id="user-999",
        access_token="acc-123",
        refresh_token="ref-123",
        scope="user.metrics",
        expires_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3),
    )
    db.add(token)
    db.commit()

    res = client.get("/integrations/withings/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "connected"
    assert data["linked"] is True
    assert data["user_id"] == "user-999"


def test_cross_source_scale_deduplication(client, db) -> None:
    from app.models import HealthSample
    from sqlalchemy import select

    today = dt.date.today()
    # 1. Withings scale creates weight sample at 07:30:15 UTC (82.5 kg)
    t_withings = dt.datetime.combine(today, dt.time(7, 30, 15)).replace(tzinfo=dt.timezone.utc)
    s_withings = HealthSample(
        source="withings",
        sample_type="weight",
        measured_at=t_withings,
        ended_at=t_withings,
        value=Decimal("82.5"),
        unit="kg",
        external_id="withings_12345_1",
    )
    db.add(s_withings)
    db.commit()

    # 2. Apple Health later sends the same weigh-in at 07:30:00 UTC (82.5 kg) within 15 seconds
    t_apple = dt.datetime.combine(today, dt.time(7, 30, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    payload = {
        "weight_dates": t_apple,
        "weight_values": "82.5",
    }
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201

    # Should NOT have inserted a second row
    samples = db.scalars(select(HealthSample).where(HealthSample.sample_type == "weight")).all()
    assert len(samples) == 1
    assert samples[0].value == Decimal("82.5")


def test_unified_daily_rollup_merges_scale_and_wearables(client, db) -> None:
    from app.models import BiometricMeasurement, HealthSample
    from sqlalchemy import select

    today = dt.date.today()
    t_morning = dt.datetime.combine(today, dt.time(7, 0)).replace(tzinfo=dt.timezone.utc)

    # 1. Withings creates body composition samples (weight, body fat %, muscle mass, water %)
    samples = [
        HealthSample(source="withings", sample_type="weight", measured_at=t_morning, ended_at=t_morning, value=Decimal("81.4"), unit="kg"),
        HealthSample(source="withings", sample_type="body_fat_pct", measured_at=t_morning, ended_at=t_morning, value=Decimal("16.8"), unit="%"),
        HealthSample(source="withings", sample_type="muscle_mass_kg", measured_at=t_morning, ended_at=t_morning, value=Decimal("64.2"), unit="kg"),
        HealthSample(source="withings", sample_type="water_pct", measured_at=t_morning, ended_at=t_morning, value=Decimal("58.5"), unit="%"),
    ]
    for s in samples:
        db.add(s)
    db.commit()

    # 2. Apple Watch syncs sleep and steps for the same day
    t_sleep_s = dt.datetime.combine(today, dt.time(0, 30)).replace(tzinfo=dt.timezone.utc).isoformat()
    t_sleep_e = dt.datetime.combine(today, dt.time(7, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    t_step_s = dt.datetime.combine(today, dt.time(8, 0)).replace(tzinfo=dt.timezone.utc).isoformat()
    t_step_e = dt.datetime.combine(today, dt.time(8, 30)).replace(tzinfo=dt.timezone.utc).isoformat()

    payload = {
        "sleep_start_dates": t_sleep_s,
        "sleep_end_dates": t_sleep_e,
        "sleep_values": "Deep",
        "step_start_dates": t_step_s,
        "step_end_dates": t_step_e,
        "step_values": "1500",
    }
    res = client.post("/integrations/apple-health/sync", json=payload)
    assert res.status_code == 201

    # Should have strictly ONE unified daily biometric_measurement row merging both sources
    measurements = db.scalars(select(BiometricMeasurement)).all()
    assert len(measurements) == 1
    m = measurements[0]
    assert float(m.weight_kg) == 81.4
    assert float(m.body_fat_pct) == 16.8
    assert float(m.muscle_mass_kg) == 64.2
    assert float(m.water_pct) == 58.5
    assert m.sleep_duration_s == 23400  # 6.5 hours
    assert m.sleep_deep_s == 23400
    assert m.step_count == 1500

