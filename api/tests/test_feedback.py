"""Unit tests for active health feedback endpoints."""

from __future__ import annotations

import datetime as dt

from tests._harness import EirTestCase


class TestFeedbackRouter(EirTestCase):
    """Router tests for /feedback check-in, daily, and history endpoints."""

    def test_create_and_get_generic_feedback(self) -> None:
        today_str = dt.date.today().isoformat()
        response = self.client.post(
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
        self.assertEqual(response.status_code, 200)
        created = response.json()
        self.assertEqual(created["feedback_type"], "sport")
        self.assertEqual(created["rating"], "brutal_match")
        self.assertEqual(float(created["score"]), 9.0)
        self.assertIn("court:singles", created["tags"])

        daily_res = self.client.get(f"/feedback/daily/{today_str}")
        self.assertEqual(daily_res.status_code, 200)
        daily_data = daily_res.json()
        self.assertGreaterEqual(len(daily_data), 1)
        self.assertTrue(any(f["feedback_type"] == "sport" for f in daily_data))

    def test_fasting_feedback_convenience_endpoint(self) -> None:
        today_str = dt.date.today().isoformat()
        response = self.client.post(
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["feedback_type"], "fasting")
        self.assertEqual(data["rating"], "completed")
        self.assertEqual(float(data["score"]), 5.0)
        self.assertEqual(data["payload"]["target_window"], "16:8")
        self.assertIn("energy:high", data["tags"])

    def test_workout_feedback_convenience_endpoint(self) -> None:
        today_str = dt.date.today().isoformat()
        response = self.client.post(
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["feedback_type"], "workout")
        self.assertEqual(data["rating"], "optimal")
        self.assertEqual(float(data["score"]), 8.0)
        self.assertEqual(data["payload"]["routine_name"], "Upper Body A")
        self.assertIn("💪 Felt Strong", data["tags"])

    def test_wellness_feedback_convenience_endpoint(self) -> None:
        today_str = dt.date.today().isoformat()
        response = self.client.post(
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["feedback_type"], "wellness")
        self.assertEqual(data["rating"], "high")
        self.assertEqual(float(data["score"]), 2.0)
        self.assertIn("Quads", data["tags"])

    def test_feedback_upsert_on_same_date_and_type(self) -> None:
        today_str = dt.date.today().isoformat()
        self.client.post(
            "/feedback/fasting",
            json={
                "date": today_str,
                "target_window": "16:8",
                "adherence": "early",
                "energy": "low",
                "headache": "mild",
            },
        )

        res = self.client.post(
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
        self.assertEqual(res.status_code, 200)
        updated = res.json()
        self.assertEqual(updated["rating"], "completed")

        daily = self.client.get(f"/feedback/daily/{today_str}").json()
        fasting_records = [f for f in daily if f["feedback_type"] == "fasting"]
        self.assertEqual(len(fasting_records), 1)
        self.assertEqual(fasting_records[0]["notes"], "Pushed through the hunger wave")

    def test_daily_biometrics_includes_feedback(self) -> None:
        today_str = dt.date.today().isoformat()
        self.client.post(
            "/biometrics",
            json={
                "measured_at": f"{today_str}T07:30:00Z",
                "source": "withings",
                "weight_kg": 86.5,
                "resting_hr_bpm": 62,
            },
        )
        self.client.post(
            "/feedback/fasting",
            json={
                "date": today_str,
                "target_window": "14:10",
                "adherence": "completed",
                "energy": "high",
            },
        )

        res = self.client.get(f"/biometrics/daily/{today_str}")
        self.assertEqual(res.status_code, 200)
        summary = res.json()
        self.assertEqual(float(summary["latest_weight_kg"]), 86.5)
        self.assertIsNotNone(summary["fasting"])
        self.assertEqual(summary["fasting"]["rating"], "completed")
        self.assertEqual(summary["fasting"]["payload"]["target_window"], "14:10")

    def test_feedback_history_filtering_and_pagination(self) -> None:
        d1 = dt.date(2026, 11, 20).isoformat()
        d2 = dt.date(2026, 11, 21).isoformat()
        self.client.post("/feedback/fasting", json={"date": d1, "adherence": "completed"})
        self.client.post("/feedback/fasting", json={"date": d2, "adherence": "early"})
        self.client.post("/feedback/wellness", json={"date": d2, "readiness_rating": "high"})

        filtered = self.client.get("/feedback/history?feedback_type=fasting&limit=1&offset=1")
        self.assertEqual(filtered.status_code, 200)
        rows = filtered.json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["feedback_type"], "fasting")
        self.assertEqual(rows[0]["date"], d1)
