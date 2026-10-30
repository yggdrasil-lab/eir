"""Tests for core Eir API endpoints."""

import datetime as dt
import uuid


def test_health_check(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "ok"


def test_create_and_get_biometric_entry(client) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    response = client.post(
        "/biometrics",
        json={
            "measured_at": now,
            "source": "manual",
            "weight_kg": 85.5,
            "body_fat_pct": 18.2,
            "resting_hr_bpm": 55,
            "hrv_ms": 68.4,
            "sleep_duration_s": 27000,
            "sleep_score": 88,
            "step_count": 9500,
            "notes": "Good morning rest",
        },
    )
    assert response.status_code == 201
    created = response.json()
    assert float(created["weight_kg"]) == 85.5
    assert created["source"] == "manual"
    assert created["resting_hr_bpm"] == 55
    measurement_id = created["id"]

    # Fetch by ID
    get_res = client.get(f"/biometrics/{measurement_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == measurement_id

    # Fetch latest
    latest_res = client.get("/biometrics/latest")
    assert latest_res.status_code == 200
    assert latest_res.json()["id"] == measurement_id

    # List
    list_res = client.get("/biometrics")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1


def test_daily_summary(client) -> None:
    today = dt.date.today()
    measured_dt = dt.datetime.combine(today, dt.time(8, 0)).replace(tzinfo=dt.timezone.utc).isoformat()

    client.post(
        "/biometrics",
        json={
            "measured_at": measured_dt,
            "source": "manual",
            "weight_kg": 86.0,
            "body_fat_pct": 18.5,
            "resting_hr_bpm": 54,
            "hrv_ms": 70.0,
            "sleep_duration_s": 28800,
            "sleep_deep_s": 5400,
            "sleep_score": 85,
            "step_count": 10000,
            "active_energy_kcal": 450,
            "notes": "Feeling energized",
        },
    )

    summary_res = client.get(f"/biometrics/daily/{today.isoformat()}")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["date"] == today.isoformat()
    assert float(summary["latest_weight_kg"]) == 86.0
    assert summary["resting_hr_bpm"] == 54
    assert summary["sleep_duration_hours"] == 8.0
    assert summary["sleep_deep_hours"] == 1.5
    assert summary["step_count"] == 10000
