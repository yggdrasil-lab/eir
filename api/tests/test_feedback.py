"""Unit tests for active health feedback endpoints."""

import datetime as dt


def test_create_and_get_generic_feedback(client) -> None:
    today_str = dt.date.today().isoformat()
    response = client.post(
        "/feedback",
        json={
            "date": today_str,
            "feedback_type": "sport",
            "rating": "brutal_match",
            "score": 9.0,
            "tags": ["court:singles", "fatigue:calves"],
            "payload": {"duration_min": 90, "games_won": 2},
            "notes": "Intense badminton rallies",
        },
    )
    assert response.status_code == 200
    created = response.json()
    assert created["feedback_type"] == "sport"
    assert created["rating"] == "brutal_match"
    assert float(created["score"]) == 9.0
    assert "court:singles" in created["tags"]

    # Verify retrieval via daily endpoint
    daily_res = client.get(f"/feedback/daily/{today_str}")
    assert daily_res.status_code == 200
    daily_data = daily_res.json()
    assert len(daily_data) >= 1
    assert any(f["feedback_type"] == "sport" for f in daily_data)


def test_fasting_feedback_convenience_endpoint(client) -> None:
    today_str = dt.date.today().isoformat()
    response = client.post(
        "/feedback/fasting",
        json={
            "date": today_str,
            "target_window": "16:8",
            "adherence": "completed",
            "energy": "high",
            "headache": "none",
            "notes": "Smooth fasting window, high focus",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["feedback_type"] == "fasting"
    assert data["rating"] == "completed"
    assert float(data["score"]) == 5.0
    assert data["payload"]["target_window"] == "16:8"
    assert "energy:high" in data["tags"]


def test_workout_feedback_convenience_endpoint(client) -> None:
    today_str = dt.date.today().isoformat()
    response = client.post(
        "/feedback/workout",
        json={
            "date": today_str,
            "routine_name": "Upper Body A",
            "duration_minutes": 48,
            "total_volume_kg": 4200.5,
            "completed_sets": 8,
            "effort_rating": "optimal",
            "tags": ["💪 Felt Strong", "⚡ Great Pump"],
            "notes": "Solid failure reached on bench press",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["feedback_type"] == "workout"
    assert data["rating"] == "optimal"
    assert float(data["score"]) == 8.0
    assert data["payload"]["routine_name"] == "Upper Body A"
    assert "💪 Felt Strong" in data["tags"]


def test_wellness_feedback_convenience_endpoint(client) -> None:
    today_str = dt.date.today().isoformat()
    response = client.post(
        "/feedback/wellness",
        json={
            "date": today_str,
            "readiness_rating": "high",
            "soreness_score": 2.0,
            "sore_areas": ["Quads", "Hamstrings"],
            "stress_level": "low",
            "notes": "Good sleep, ready to lift",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["feedback_type"] == "wellness"
    assert data["rating"] == "high"
    assert float(data["score"]) == 2.0
    assert "Quads" in data["tags"]


def test_feedback_upsert_on_same_date_and_type(client) -> None:
    today_str = dt.date.today().isoformat()
    # First check-in: broke early
    client.post(
        "/feedback/fasting",
        json={
            "date": today_str,
            "target_window": "16:8",
            "adherence": "early",
            "energy": "low",
            "headache": "mild",
        },
    )

    # Edit check-in on the same date: updated to completed
    res = client.post(
        "/feedback/fasting",
        json={
            "date": today_str,
            "target_window": "16:8",
            "adherence": "completed",
            "energy": "normal",
            "headache": "none",
            "notes": "Pushed through the hunger wave",
        },
    )
    assert res.status_code == 200
    updated = res.json()
    assert updated["rating"] == "completed"

    # Ensure only ONE record exists for today's fasting
    daily = client.get(f"/feedback/daily/{today_str}").json()
    fasting_records = [f for f in daily if f["feedback_type"] == "fasting"]
    assert len(fasting_records) == 1
    assert fasting_records[0]["notes"] == "Pushed through the hunger wave"


def test_daily_biometrics_includes_feedback(client) -> None:
    today_str = dt.date.today().isoformat()
    # Add a scale measurement
    client.post(
        "/biometrics",
        json={
            "measured_at": f"{today_str}T07:30:00Z",
            "source": "withings",
            "weight_kg": 86.5,
            "resting_hr_bpm": 62,
        },
    )

    # Add a fasting check-in
    client.post(
        "/feedback/fasting",
        json={
            "date": today_str,
            "target_window": "14:10",
            "adherence": "completed",
            "energy": "high",
        },
    )

    # Fetch daily summary
    res = client.get(f"/biometrics/daily/{today_str}")
    assert res.status_code == 200
    summary = res.json()
    assert float(summary["latest_weight_kg"]) == 86.5
    assert summary["fasting"] is not None
    assert summary["fasting"]["rating"] == "completed"
    assert summary["fasting"]["payload"]["target_window"] == "14:10"
