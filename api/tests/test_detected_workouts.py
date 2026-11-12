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
        self.assertEqual(workout["classification"], "badminton")
        self.assertEqual(workout["activity_type"], "badminton")
        self.assertEqual(workout["routine_name"], "Badminton Match Play")

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

    def test_detected_workouts_walking_cadence(self):
        target_d = dt.date(2026, 11, 9)
        base_time = dt.datetime(2026, 11, 9, 20, 30, tzinfo=dt.timezone.utc)

        # Ingest 35 minutes of brisk walk (7 x 5-minute intervals)
        # Sub-threshold for energy (<20 kcal/5m) and HR (<120 bpm), but elevated cadence (320 steps/5m, HR 102 bpm)
        for m in range(0, 35, 5):
            sample_time = base_time + dt.timedelta(minutes=m)
            end_time = sample_time + dt.timedelta(minutes=5)
            # Low active energy: 14 kcal per 5m (total ~98 kcal, < 250 kcal)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("14.0"),
                    unit="kcal",
                )
            )
            # Moderate HR: 102 bpm (< 120 bpm)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("102"),
                    unit="bpm",
                )
            )
            # Cadence: 320 steps per 5 min (2,240 total)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("320"),
                    unit="count",
                    extra={"device": "Raymund's Apple Watch"},
                )
            )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 35)
        self.assertEqual(workout["step_count"], 2240)
        self.assertEqual(workout["avg_hr_bpm"], 102)
        self.assertEqual(workout["classification"], "walk")
        self.assertEqual(workout["activity_type"], "walk")
        self.assertEqual(workout["routine_name"], "Walk / Cardio")

    def test_detected_workouts_gym_resistance(self):
        target_d = dt.date(2026, 11, 10)
        base_time = dt.datetime(2026, 11, 10, 16, 0, tzinfo=dt.timezone.utc)

        # Ingest 45 minutes of gym session (9 x 5-minute intervals)
        # Cyclic HR spikes: 135 bpm on working sets, 95 bpm during rest intervals
        # Lower step count: 40 steps per 5 min (360 total steps, cadence ~8 steps/min)
        for m in range(0, 45, 5):
            sample_time = base_time + dt.timedelta(minutes=m)
            end_time = sample_time + dt.timedelta(minutes=5)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("22.0"),
                    unit="kcal",
                )
            )
            # Cyclic HR spikes: even intervals 135 bpm, odd intervals 95 bpm
            hr_val = 135 if (m // 5) % 2 == 0 else 95
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal(str(hr_val)),
                    unit="bpm",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal("40"),
                    unit="count",
                    extra={"device": "Raymund's Apple Watch"},
                )
            )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 45)
        self.assertEqual(workout["classification"], "gym")
        self.assertEqual(workout["activity_type"], "gym")
        self.assertEqual(workout["routine_name"], "Resistance Training")
        self.assertEqual(workout["step_count"], 360)

    def test_detected_workouts_gym_with_warmup_jog(self):
        target_d = dt.date(2026, 11, 11)
        base_time = dt.datetime(2026, 11, 11, 17, 0, tzinfo=dt.timezone.utc)

        # Ingest 40 minutes: 10 min warmup jog (2 intervals @ 300 steps/5m) + 30 min lifting (6 intervals @ 30 steps/5m)
        for m in range(0, 40, 5):
            sample_time = base_time + dt.timedelta(minutes=m)
            end_time = sample_time + dt.timedelta(minutes=5)
            is_warmup = m < 10
            steps = Decimal("300") if is_warmup else Decimal("30")
            hr_val = 125 if is_warmup else (130 if (m // 5) % 2 == 0 else 100)
            energy = Decimal("25.0") if is_warmup else Decimal("18.0")

            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=energy,
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=Decimal(str(hr_val)),
                    unit="bpm",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=sample_time,
                    ended_at=end_time,
                    value=steps,
                    unit="count",
                    extra={"device": "Raymund's Apple Watch"},
                )
            )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 40)
        self.assertEqual(workout["classification"], "gym")
        self.assertEqual(workout["activity_type"], "gym")
        self.assertEqual(workout["routine_name"], "Resistance Training")

    def test_detected_workouts_rest_gap_splitting(self):
        target_d = dt.date(2026, 11, 12)
        base_time = dt.datetime(2026, 11, 12, 14, 0, tzinfo=dt.timezone.utc)

        # Block 1: 14:00 to 14:30 (6 intervals of active walking/cadence)
        for m in range(0, 30, 5):
            t = base_time + dt.timedelta(minutes=m)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("25.0"),
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("105"),
                    unit="bpm",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("350"),
                    unit="count",
                )
            )

        # Rest gap: 14:30 to 14:40 (2 intervals = 10 minutes stationary, steps < 50)
        for m in (30, 35):
            t = base_time + dt.timedelta(minutes=m)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("10"),
                    unit="count",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("75"),
                    unit="bpm",
                )
            )

        # Block 2: 14:40 to 15:10 (6 intervals of active walking/cadence)
        for m in range(40, 70, 5):
            t = base_time + dt.timedelta(minutes=m)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("25.0"),
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("105"),
                    unit="bpm",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("350"),
                    unit="count",
                )
            )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Rest gap splitting must separate the two exertion blocks into 2 sessions
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["duration_minutes"], 30)
        self.assertEqual(data[1]["duration_minutes"], 30)

    def test_detected_workouts_treadmill_warmup_and_resistance_spikes(self):
        target_d = dt.date(2026, 11, 13)
        base_time = dt.datetime(2026, 11, 13, 16, 0, tzinfo=dt.timezone.utc)

        # 45 minutes total: 10 min treadmill warmup + 35 min heavy lifting with spikes up to 148 bpm
        for m in range(0, 45, 5):
            t = base_time + dt.timedelta(minutes=m)
            is_warmup = m < 10
            steps = Decimal("320") if is_warmup else Decimal("25")
            hr_val = 125 if is_warmup else (148 if (m // 5) % 2 == 0 else 102)
            energy = Decimal("26.0") if is_warmup else Decimal("22.0")

            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=energy,
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal(str(hr_val)),
                    unit="bpm",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=steps,
                    unit="count",
                )
            )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 45)
        # Refined classification order must prioritize gym workout over badminton
        self.assertEqual(workout["classification"], "gym")
        self.assertEqual(workout["routine_name"], "Resistance Training")

    def test_detected_workouts_swimming_detection(self):
        target_d = dt.date(2026, 11, 14)
        base_time = dt.datetime(2026, 11, 14, 12, 50, tzinfo=dt.timezone.utc)

        # Ingest 35 minutes of swimming (7 x 5-minute intervals)
        # Sustained elevated HR (135 bpm), high continuous energy (~25 kcal/5m),
        # zero steps during the swim (intervals 0-5), followed by pool exit steps in interval 6 (300 steps).
        for m in range(0, 35, 5):
            t = base_time + dt.timedelta(minutes=m)
            steps = Decimal("300") if m == 30 else Decimal("0")
            energy = Decimal("26.0")
            hr_val = 135

            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=energy,
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal(str(hr_val)),
                    unit="bpm",
                )
            )
            if steps > 0:
                self.db.add(
                    HealthSample(
                        id=uuid.uuid4(),
                        source="apple_health",
                        sample_type="step_count",
                        measured_at=t,
                        ended_at=t + dt.timedelta(minutes=5),
                        value=steps,
                        unit="count",
                        extra={"device": "Raymund's Apple Watch"},
                    )
                )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 35)
        self.assertEqual(workout["classification"], "swim")
        self.assertEqual(workout["activity_type"], "swim")
        self.assertEqual(workout["routine_name"], "Swimming")
        self.assertEqual(workout["step_count"], 300)
        self.assertEqual(workout["avg_hr_bpm"], 135)
        self.assertEqual(workout["active_energy_kcal"], 182)

    def test_detected_workouts_swimming_transit_walk_separation(self):
        target_d = dt.date(2026, 11, 15)
        base_time = dt.datetime(2026, 11, 15, 12, 50, tzinfo=dt.timezone.utc)

        # Swim session: 12:50 to 13:25 (7 intervals)
        for m in range(0, 35, 5):
            t = base_time + dt.timedelta(minutes=m)
            steps = Decimal("200") if m == 30 else Decimal("0")
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("25.0"),
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("130"),
                    unit="bpm",
                )
            )
            if steps > 0:
                self.db.add(
                    HealthSample(
                        id=uuid.uuid4(),
                        source="apple_health",
                        sample_type="step_count",
                        measured_at=t,
                        ended_at=t + dt.timedelta(minutes=5),
                        value=steps,
                        unit="count",
                    )
                )

        # Gap slot with transit walk: 13:25 to 13:30 (interval 7 @ 35 min, steps = 1,000, energy = 18 kcal)
        t_gap = base_time + dt.timedelta(minutes=35)
        self.db.add(
            HealthSample(
                id=uuid.uuid4(),
                source="apple_health",
                sample_type="step_count",
                measured_at=t_gap,
                ended_at=t_gap + dt.timedelta(minutes=5),
                value=Decimal("1000"),
                unit="count",
            )
        )
        self.db.add(
            HealthSample(
                id=uuid.uuid4(),
                source="apple_health",
                sample_type="active_energy",
                measured_at=t_gap,
                ended_at=t_gap + dt.timedelta(minutes=5),
                value=Decimal("18.0"),
                unit="kcal",
            )
        )

        # Subsequent active walk: 13:30 to 13:40 (2 intervals @ 400 steps/5m, energy = 25 kcal, hr = 125 bpm)
        for m in range(40, 50, 5):
            t = base_time + dt.timedelta(minutes=m)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("25.0"),
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("125"),
                    unit="bpm",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("400"),
                    unit="count",
                )
            )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # The transit walk gap must cleanly split the swim session from the subsequent 10m walk
        # The 10m walk is noise-filtered (duration < 25m), leaving only the 35m swim session
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 35)
        self.assertEqual(workout["classification"], "swim")
        self.assertEqual(workout["routine_name"], "Swimming")


if __name__ == "__main__":
    unittest.main()
