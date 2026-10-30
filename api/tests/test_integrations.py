"""Tests for Withings and Apple Health integration webhooks in Eir."""

import datetime as dt


def test_apple_health_biometrics_ingestion(client) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    response = client.post(
        "/integrations/apple-health/biometrics",
        json={
            "measured_at": now,
            "weight_kg": 88.5,
            "resting_hr_bpm": 52,
            "hrv_ms": 62.4,
            "sleep_duration_s": 28800,
            "sleep_deep_s": 5400,
            "sleep_rem_s": 6200,
            "sleep_score": 87,
            "step_count": 8500,
            "active_energy_kcal": 450,
            "notes": "Morning health sync",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["source"] == "apple_health"
    assert data["resting_hr_bpm"] == 52
    assert data["sleep_duration_s"] == 28800

    # Query latest
    latest = client.get("/biometrics/latest")
    assert latest.status_code == 200
    assert latest.json()["source"] == "apple_health"


def test_apple_health_workout_ingestion(client) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    started_at = (now - dt.timedelta(minutes=60)).isoformat()
    ended_at = now.isoformat()

    response = client.post(
        "/integrations/apple-health/workout",
        json={
            "workout_type": "Badminton",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": 3600,
            "active_energy_kcal": 550,
            "avg_heart_rate_bpm": 145,
            "max_heart_rate_bpm": 175,
            "notes": "Fast-paced match",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["source"] == "apple_health_workout"
    assert data["extra"]["workout_type"] == "Badminton"
    assert data["active_energy_kcal"] == 550


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
    assert data["weight_kg"] == 87.2
    assert data["body_fat_pct"] == 17.8


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
