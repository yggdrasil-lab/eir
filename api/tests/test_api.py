"""Tests for core Eir API endpoints."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
import uuid

from app.config import settings
from app.db import get_db
from app.main import app
from app.models import BiometricMeasurement, HealthSample
from tests._harness import EirTestCase


class _FailingSession:
    """Minimal Session double raising on execute() to exercise degraded /health."""

    def execute(self, *_args, **_kwargs) -> None:
        raise RuntimeError("simulated database failure")


class TestEirAPI(EirTestCase):
    """Router tests for /health and /biometrics endpoints."""

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["database"], "ok")

    def test_health_check_degraded_on_database_error(self) -> None:
        def _broken_db():
            yield _FailingSession()

        app.dependency_overrides[get_db] = _broken_db
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "degraded")
        self.assertTrue(data["database"].startswith("unhealthy:"))

    def test_create_and_get_biometric_entry(self) -> None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        response = self.client.post(
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
        self.assertEqual(response.status_code, 201)
        created = response.json()
        self.assertEqual(float(created["weight_kg"]), 85.5)
        self.assertEqual(created["source"], "manual")
        self.assertEqual(created["resting_hr_bpm"], 55)
        measurement_id = created["id"]

        get_res = self.client.get(f"/biometrics/{measurement_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["id"], measurement_id)

        latest_res = self.client.get("/biometrics/latest")
        self.assertEqual(latest_res.status_code, 200)
        self.assertEqual(latest_res.json()["id"], measurement_id)

        list_res = self.client.get("/biometrics")
        self.assertEqual(list_res.status_code, 200)
        self.assertGreaterEqual(len(list_res.json()), 1)

    def test_biometrics_not_found_source_filter_and_pagination(self) -> None:
        missing_res = self.client.get(f"/biometrics/{uuid.uuid4()}")
        self.assertEqual(missing_res.status_code, 404)

        base = dt.datetime(2026, 11, 20, 8, 0, tzinfo=dt.timezone.utc)
        for idx, src in enumerate(("withings", "manual", "withings")):
            self.db.add(
                BiometricMeasurement(
                    id=uuid.uuid4(),
                    measured_at=base + dt.timedelta(hours=idx),
                    source=src,
                    weight_kg=Decimal(f"{80 + idx}.0"),
                )
            )
        self.db.commit()

        filtered = self.client.get("/biometrics?source=withings&limit=1&offset=1")
        self.assertEqual(filtered.status_code, 200)
        rows = filtered.json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "withings")
        self.assertEqual(float(rows[0]["weight_kg"]), 80.0)

        latest_manual = self.client.get("/biometrics/latest?source=manual")
        self.assertEqual(latest_manual.status_code, 200)
        self.assertEqual(latest_manual.json()["source"], "manual")

        latest_missing = self.client.get("/biometrics/latest?source=unknown")
        self.assertEqual(latest_missing.status_code, 200)
        self.assertIsNone(latest_missing.json())

    def test_biometrics_granular_and_workout_metrics_400_errors(self) -> None:
        granular_err = self.client.get("/biometrics/granular?target_date=2026-11-24&sample_type=%20%20%20")
        self.assertEqual(granular_err.status_code, 400)

        workout_err = self.client.get(
            "/biometrics/workout-metrics?started_at=2026-11-24T10:00:00Z&ended_at=2026-11-24T09:00:00Z"
        )
        self.assertEqual(workout_err.status_code, 400)

    def test_daily_summary(self) -> None:
        today = dt.date.today()
        measured_dt = dt.datetime.combine(today, dt.time(8, 0)).replace(tzinfo=dt.timezone.utc).isoformat()

        self.client.post(
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

        summary_res = self.client.get(f"/biometrics/daily/{today.isoformat()}")
        self.assertEqual(summary_res.status_code, 200)
        summary = summary_res.json()
        self.assertEqual(summary["date"], today.isoformat())
        self.assertEqual(float(summary["latest_weight_kg"]), 86.0)
        self.assertEqual(summary["resting_hr_bpm"], 54)
        self.assertEqual(summary["sleep_duration_hours"], 8.0)
        self.assertEqual(summary["sleep_deep_hours"], 1.5)
        self.assertEqual(summary["step_count"], 10000)

    def test_daily_summary_granular_steps_do_not_clobber_deduplicated_count(self) -> None:
        """Overlapping Watch/iPhone samples must not override the deduplicated rollup step count."""
        today = dt.date.today()

        def at(hour: int, minute: int) -> dt.datetime:
            return dt.datetime.combine(today, dt.time(hour, minute)).replace(tzinfo=settings.tz)

        self.db.add(
            BiometricMeasurement(
                id=uuid.uuid4(),
                measured_at=at(8, 0),
                source="daily_rollup",
                step_count=580,
            )
        )
        granular = [
            ("Alex's Apple Watch", at(12, 0), at(12, 10), "200"),
            ("Alex's Apple Watch", at(12, 10), at(12, 20), "300"),
            ("Alex's iPhone", at(12, 5), at(12, 15), "250"),
            ("Alex's iPhone", at(9, 0), at(9, 15), "80"),
        ]
        for device, start, end, value in granular:
            self.db.add(
                HealthSample(
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=start,
                    ended_at=end,
                    value=Decimal(value),
                    unit="count",
                    extra={"device": device},
                )
            )
        self.db.commit()

        res = self.client.get(f"/biometrics/daily/{today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["step_count"], 580)
