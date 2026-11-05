"""Unit tests for biometrics exertion detection and unlogged workout inference."""

import datetime as dt
from decimal import Decimal
import unittest
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app
from app.models import HealthSample


class TestDetectedWorkouts(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        self.db = self.SessionLocal()

        def override_get_db():
            session = self.SessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        app.dependency_overrides.clear()

    def test_detected_workouts_empty(self):
        today_str = dt.date.today().isoformat()
        res = self.client.get(f"/biometrics/detected-workouts?target_date={today_str}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), [])

    def test_detected_workouts_clustering(self):
        target_d = dt.date(2026, 11, 5)
        base_time = dt.datetime(2026, 11, 5, 19, 0, tzinfo=dt.timezone.utc)

        # Ingest 90 minutes of simulated badminton: 19:00 to 20:30 (18 x 5-minute intervals)
        for m in range(0, 90, 5):
            sample_time = base_time + dt.timedelta(minutes=m)
            end_time = sample_time + dt.timedelta(minutes=5)
            # Active energy sample: 35 kcal / 5 min
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("35.0"),
                    unit="kcal",
                )
            )
            # Heart rate sample: ~145 bpm
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal(str(140 + (m % 15))),
                    unit="bpm",
                )
            )
            # Step sample
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("400"),
                    unit="count",
                )
            )

        self.db.commit()

        # 1. Test dedicated endpoint
        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertGreaterEqual(workout["duration_minutes"], 85)
        self.assertGreaterEqual(workout["active_energy_kcal"], 600)
        self.assertGreaterEqual(workout["avg_hr_bpm"], 135)
        self.assertGreaterEqual(workout["peak_hr_bpm"], 145)
        self.assertGreaterEqual(workout["step_count"], 7000)
        self.assertEqual(workout["confidence"], "high")

        # 2. Test daily summary integration
        daily_res = self.client.get(f"/biometrics/daily/{target_d.isoformat()}")
        self.assertEqual(daily_res.status_code, 200)
        daily_data = daily_res.json()
        self.assertIn("detected_workouts", daily_data)
        self.assertEqual(len(daily_data["detected_workouts"]), 1)
        self.assertGreaterEqual(daily_data["detected_workouts"][0]["duration_minutes"], 85)

    def test_detected_workouts_step_deduplication(self):
        target_d = dt.date(2026, 11, 5)
        base_time = dt.datetime(2026, 11, 5, 12, 40, tzinfo=dt.timezone.utc)

        # Ingest 30 minutes of simulated walk (6 x 5-minute intervals) with elevated HR
        for m in range(0, 30, 5):
            sample_time = base_time + dt.timedelta(minutes=m)
            end_time = sample_time + dt.timedelta(minutes=5)
            # Active energy
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("25.0"),
                    unit="kcal",
                )
            )
            # HR: 125 bpm
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("125"),
                    unit="bpm",
                )
            )
            # Apple Watch steps: 350 steps per 5 min (2,100 total)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("350"),
                    unit="count",
                    extra={"device": "Raymund's Apple Watch"},
                )
            )
            # iPhone steps overlapping in the same 5-minute interval: 400 steps per 5 min (2,400 total)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=sample_time + dt.timedelta(seconds=5),
                    ended_at=end_time - dt.timedelta(seconds=2),
                    value=Decimal("400"),
                    unit="count",
                    extra={"device": "Raymund's iPhone"},
                )
            )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 30)
        # Deduplication must prioritize Apple Watch (2,100 steps) rather than naively summing both (4,500)
        self.assertEqual(workout["step_count"], 2100)


if __name__ == "__main__":
    unittest.main()
