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
