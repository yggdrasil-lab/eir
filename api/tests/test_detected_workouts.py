"""Unit tests for biometrics exertion detection and unlogged workout inference."""

import datetime as dt
from decimal import Decimal
import unittest
import uuid

from app.models import HealthSample
from tests._harness import EirTestCase


class TestDetectedWorkouts(EirTestCase):
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
                    extra={"device": "Alex's Apple Watch"},
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
                    extra={"device": "Alex's iPhone"},
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
                    extra={"device": "Alex's Apple Watch"},
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
                    extra={"device": "Alex's Apple Watch"},
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
                    extra={"device": "Alex's Apple Watch"},
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
                        extra={"device": "Alex's Apple Watch"},
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

    def test_detected_workouts_badminton_intermittent_games_not_gym(self):
        target_d = dt.date(2026, 11, 16)
        base_time = dt.datetime(2026, 11, 16, 19, 0, tzinfo=dt.timezone.utc)

        # 65 minutes total (13 intervals): games (400 steps) alternating with court bench rest (0 steps)
        # High sustained HR (avg 157 bpm, peak 176 bpm), overall cadence ~35 steps/min
        slot_pattern_steps = [350, 0, 450, 0, 30, 420, 0, 400, 0, 400, 0, 180, 15]
        slot_pattern_hrs = [100, 125, 115, 120, 145, 170, 165, 162, 160, 172, 155, 130, 130]

        for i, (steps, hr_val) in enumerate(zip(slot_pattern_steps, slot_pattern_hrs)):
            t = base_time + dt.timedelta(minutes=i * 5)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("15.0") if steps > 100 else Decimal("2.0"),
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
                        value=Decimal(str(steps)),
                        unit="count",
                    )
                )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 65)
        self.assertEqual(workout["classification"], "badminton")
        self.assertEqual(workout["activity_type"], "badminton")
        self.assertEqual(workout["routine_name"], "Badminton Match Play")
        # Assert effort rating, RPE, TRIMP, and HR zones are populated
        self.assertIn(workout["effort_rating"], ["optimal", "hard", "too_hard"])
        self.assertIsInstance(workout["rpe"], float)
        self.assertGreaterEqual(workout["rpe"], 5.0)
        self.assertLessEqual(workout["rpe"], 10.0)
        self.assertIsInstance(workout["trimp"], float)
        self.assertGreater(workout["trimp"], 0.0)
        self.assertIsNotNone(workout["hr_zones"])
        self.assertIn("z1_recovery_pct", workout["hr_zones"])
        self.assertIn("z2_aerobic_pct", workout["hr_zones"])
        self.assertIn("z3_tempo_pct", workout["hr_zones"])
        self.assertIn("z4_threshold_pct", workout["hr_zones"])
        self.assertIn("z5_anaerobic_pct", workout["hr_zones"])

    def test_get_workout_metrics_for_custom_split_window(self):
        """GET /biometrics/workout-metrics computes full HR zones, TRIMP, RPE, and energy for an arbitrary window."""
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        base_time = dt.datetime(2026, 11, 16, 18, 35, tzinfo=nz_tz)

        for m in range(0, 35, 5):
            t = base_time + dt.timedelta(minutes=m)
            t_end = t + dt.timedelta(minutes=5)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t_end,
                    value=Decimal("22.5"),
                    unit="kcal",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t_end,
                    value=Decimal(str(130 + (m % 15))),
                    unit="bpm",
                )
            )
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=t,
                    ended_at=t_end,
                    value=Decimal("150"),
                    unit="count",
                )
            )
        self.db.commit()

        res = self.client.get(
            "/biometrics/workout-metrics",
            params={
                "started_at": base_time.isoformat(),
                "ended_at": (base_time + dt.timedelta(minutes=35)).isoformat(),
                "classification": "gym",
                "routine_name": "Lower Body A - Squat Emphasis (MEV Deload)",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["duration_minutes"], 35)
        self.assertEqual(data["classification"], "gym")
        self.assertEqual(data["routine_name"], "Lower Body A - Squat Emphasis (MEV Deload)")
        self.assertEqual(data["active_energy_kcal"], 158)
        self.assertEqual(data["step_count"], 1050)
        self.assertIsNotNone(data["hr_zones"])
        self.assertGreater(data["trimp"], 0.0)
        self.assertGreaterEqual(data["rpe"], 1.0)

    def test_detected_workouts_badminton_with_cooldown_hr_tail(self):
        target_d = dt.date(2026, 11, 19)
        base_time = dt.datetime(2026, 11, 19, 20, 15, tzinfo=dt.timezone.utc)

        # 85 minutes total (17 intervals): 50 min of court rallies (4 burst slots >= 250 steps)
        # followed by 35 min post-match elevated HR cooldown tail with minimal steps,
        # diluting overall cadence to ~27.1 steps/min (just under 28.0).
        slot_pattern_steps = [437, 0, 131, 474, 0, 571, 0, 329, 0, 203, 157, 0, 0, 159, 0, 0, 0]
        slot_pattern_hrs = [110, 160, 155, 148, 160, 166, 162, 159, 166, 153, 120, 123, 115, 123, 120, 113, 136]
        slot_pattern_energies = [0, 16, 0, 14, 0, 20, 0, 11, 0, 10, 0, 0, 0, 9, 0, 0, 0]

        for i, (steps, hr_val, energy) in enumerate(zip(slot_pattern_steps, slot_pattern_hrs, slot_pattern_energies)):
            t = base_time + dt.timedelta(minutes=i * 5)
            if energy > 0:
                self.db.add(
                    HealthSample(
                        id=uuid.uuid4(),
                        source="apple_health",
                        sample_type="active_energy",
                        measured_at=t,
                        ended_at=t + dt.timedelta(minutes=5),
                        value=Decimal(str(energy)),
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
                        value=Decimal(str(steps)),
                        unit="count",
                    )
                )

        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 85)
        self.assertEqual(workout["classification"], "badminton")
        self.assertEqual(workout["activity_type"], "badminton")
        self.assertEqual(workout["routine_name"], "Badminton Match Play")

    def test_detected_workouts_cross_slot_step_dedup_resistance_not_badminton(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        target_d = dt.date(2026, 11, 20)
        workout_start = dt.datetime(2026, 11, 20, 16, 55, tzinfo=nz_tz)

        # Replay 2026-11-20 Lower Body A (16:55-17:35 NZDT): avg_hr=115, peak_hr=140, 172 kcal
        slot_hrs = [112, 120, 102, 140, 118, 104, 124, 100]
        slot_energies = [22, 22, 22, 22, 21, 21, 21, 21]
        for i, (hr_val, energy_val) in enumerate(zip(slot_hrs, slot_energies)):
            t = workout_start + dt.timedelta(minutes=i * 5)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal(str(energy_val)),
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

        # Staggered Watch vs overlapping iPhone step intervals across adjacent 5-min slots
        watch_steps = [
            (dt.datetime(2026, 11, 20, 16, 53, 46, tzinfo=nz_tz), dt.datetime(2026, 11, 20, 17, 3, 27, tzinfo=nz_tz), 619),
            (dt.datetime(2026, 11, 20, 17, 3, 45, tzinfo=nz_tz), dt.datetime(2026, 11, 20, 17, 13, 44, tzinfo=nz_tz), 222),
            (dt.datetime(2026, 11, 20, 17, 13, 44, tzinfo=nz_tz), dt.datetime(2026, 11, 20, 17, 23, 34, tzinfo=nz_tz), 610),
            (dt.datetime(2026, 11, 20, 17, 24, 25, tzinfo=nz_tz), dt.datetime(2026, 11, 20, 17, 33, 46, tzinfo=nz_tz), 132),
        ]
        phone_steps = [
            (dt.datetime(2026, 11, 20, 16, 56, 1, tzinfo=nz_tz), dt.datetime(2026, 11, 20, 17, 5, 24, tzinfo=nz_tz), 325),
            (dt.datetime(2026, 11, 20, 17, 6, 25, tzinfo=nz_tz), dt.datetime(2026, 11, 20, 17, 16, 19, tzinfo=nz_tz), 232),
            (dt.datetime(2026, 11, 20, 17, 16, 19, tzinfo=nz_tz), dt.datetime(2026, 11, 20, 17, 25, 37, tzinfo=nz_tz), 697),
        ]
        for s_dt, e_dt, val in watch_steps:
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=s_dt,
                    ended_at=e_dt,
                    value=Decimal(str(val)),
                    unit="count",
                    extra={"device": "Alex's Apple Watch"},
                )
            )
        for s_dt, e_dt, val in phone_steps:
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=s_dt,
                    ended_at=e_dt,
                    value=Decimal(str(val)),
                    unit="count",
                    extra={"device": "Alex's iPhone"},
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
        self.assertEqual(workout["routine_name"], "Resistance Training")
        # Proportional interval bucketing attributes ~540 steps of 16:53:46->17:03:27 into 16:55-17:05
        # while deduplicating all 1,254 overlapping iPhone steps (540 + 964 = 1,504).
        self.assertEqual(workout["step_count"], 1504)
        self.assertEqual(workout["avg_hr_bpm"], 115)
        self.assertEqual(workout["peak_hr_bpm"], 140)

        # Window metrics endpoint must also deduplicate and pro-rate overlapping intervals consistently
        win_res = self.client.get(
            "/biometrics/workout-metrics",
            params={
                "started_at": workout_start.isoformat(),
                "ended_at": (workout_start + dt.timedelta(minutes=40)).isoformat(),
            },
        )
        self.assertEqual(win_res.status_code, 200)
        win_data = win_res.json()
        self.assertEqual(win_data["classification"], "gym")
        self.assertEqual(win_data["step_count"], 1504)

    def test_detected_workouts_moderate_hr_resistance_not_badminton(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        target_d = dt.date(2026, 11, 15)
        base_time = dt.datetime(2026, 11, 15, 15, 50, tzinfo=nz_tz)

        # 60 min Lower Body B pattern: avg_hr=126, peak_hr=151, cadence=48 steps/min (6 burst slots)
        slot_hrs = [151, 120, 130, 118, 128, 115, 132, 118, 130, 116, 134, 120]
        slot_steps = [280, 200, 280, 200, 280, 200, 280, 200, 280, 200, 280, 200]
        for i, (hr_val, steps_val) in enumerate(zip(slot_hrs, slot_steps)):
            t = base_time + dt.timedelta(minutes=i * 5)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("22.5"),
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
                    value=Decimal(str(steps_val)),
                    unit="count",
                    extra={"device": "Alex's Apple Watch"},
                )
            )
        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 60)
        self.assertEqual(workout["avg_hr_bpm"], 126)
        self.assertEqual(workout["peak_hr_bpm"], 151)
        self.assertEqual(workout["step_count"], 2880)
        self.assertEqual(workout["classification"], "gym")
        self.assertEqual(workout["routine_name"], "Resistance Training")

    def test_detected_workouts_brisk_walk_hr_spike_not_badminton(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        target_d = dt.date(2026, 11, 5)
        base_time = dt.datetime(2026, 11, 5, 12, 30, tzinfo=nz_tz)

        # 40 min brisk walk pattern: avg_hr=120, peak_hr=146, cadence=74 steps/min
        slot_hrs = [112, 116, 118, 146, 120, 116, 118, 114]
        for i, hr_val in enumerate(slot_hrs):
            t = base_time + dt.timedelta(minutes=i * 5)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="active_energy",
                    measured_at=t,
                    ended_at=t + dt.timedelta(minutes=5),
                    value=Decimal("23.5"),
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
                    value=Decimal("370"),
                    unit="count",
                    extra={"device": "Alex's Apple Watch"},
                )
            )
        self.db.commit()

        res = self.client.get(f"/biometrics/detected-workouts?target_date={target_d.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        workout = data[0]
        self.assertEqual(workout["duration_minutes"], 40)
        self.assertEqual(workout["avg_hr_bpm"], 120)
        self.assertEqual(workout["peak_hr_bpm"], 146)
        self.assertEqual(workout["step_count"], 2960)
        self.assertEqual(workout["classification"], "walk")
        self.assertEqual(workout["routine_name"], "Walk / Cardio")

    def _add_bin_samples(
        self,
        bin_start: dt.datetime,
        *,
        hr_values: list[int],
        energy_kcal: Decimal,
        watch_steps: int = 0,
        phone_steps: int = 0,
    ) -> None:
        """Seeds granular samples inside a single 5-minute window."""
        bin_end = bin_start + dt.timedelta(minutes=5)
        self.db.add(
            HealthSample(
                id=uuid.uuid4(),
                source="apple_health",
                sample_type="active_energy",
                measured_at=bin_start,
                ended_at=bin_end,
                value=energy_kcal,
                unit="kcal",
            )
        )
        for idx, hr in enumerate(hr_values):
            offset_s = min(295, idx * 10)
            t = bin_start + dt.timedelta(seconds=offset_s)
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="heart_rate",
                    measured_at=t,
                    ended_at=t,
                    value=Decimal(str(hr)),
                    unit="bpm",
                )
            )
        if watch_steps > 0:
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=bin_start,
                    ended_at=bin_end,
                    value=Decimal(str(watch_steps)),
                    unit="count",
                    extra={"device": "Alex's Apple Watch"},
                )
            )
        if phone_steps > 0:
            self.db.add(
                HealthSample(
                    id=uuid.uuid4(),
                    source="apple_health",
                    sample_type="step_count",
                    measured_at=bin_start + dt.timedelta(seconds=5),
                    ended_at=bin_end - dt.timedelta(seconds=5),
                    value=Decimal(str(phone_steps)),
                    unit="count",
                    extra={"device": "Alex's iPhone"},
                )
            )

    def test_get_workout_timeline_rejects_invalid_window(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        t = dt.datetime(2026, 11, 23, 17, 30, tzinfo=nz_tz)
        res = self.client.get(
            "/biometrics/workout-timeline",
            params={"started_at": t.isoformat(), "ended_at": t.isoformat()},
        )
        self.assertEqual(res.status_code, 400)

    def test_get_workout_timeline_detects_passive_to_active_transition(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        start = dt.datetime(2026, 11, 23, 17, 30, tzinfo=nz_tz)
        end = start + dt.timedelta(minutes=70)

        for i in range(6):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[140, 144],
                energy_kcal=Decimal("30"),
                watch_steps=80,
                phone_steps=110,
            )
        for i in range(6, 14):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[126 + (k % 8) for k in range(18)],
                energy_kcal=Decimal("25"),
                watch_steps=120,
            )
        self.db.commit()

        res = self.client.get(
            "/biometrics/workout-timeline",
            params={"started_at": start.isoformat(), "ended_at": end.isoformat()},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["duration_minutes"], 70)
        self.assertEqual(len(data["bins"]), 14)

        self.assertFalse(data["bins"][0]["is_active_watch_mode"])
        self.assertEqual(data["bins"][0]["hr_sample_count"], 2)
        self.assertEqual(data["bins"][0]["step_count"], 80)
        self.assertEqual(data["bins"][0]["active_energy_kcal"], 30)

        self.assertTrue(data["bins"][6]["is_active_watch_mode"])
        self.assertEqual(data["bins"][6]["hr_sample_count"], 18)
        self.assertEqual(data["bins"][6]["step_count"], 120)

        self.assertEqual(len(data["suggested_splits"]), 1)
        split = data["suggested_splits"][0]
        self.assertEqual(
            dt.datetime.fromisoformat(split["split_at"]),
            start + dt.timedelta(minutes=30),
        )
        self.assertEqual(
            dt.datetime.fromisoformat(split["resume_at"]),
            start + dt.timedelta(minutes=30),
        )
        self.assertEqual(split["confidence"], "high")

    def test_get_workout_timeline_skips_low_hr_transition_gap(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        start = dt.datetime(2026, 11, 23, 17, 30, tzinfo=nz_tz)
        end = start + dt.timedelta(minutes=75)

        # 30m passive class
        for i in range(6):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[138, 142, 145],
                energy_kcal=Decimal("28"),
                watch_steps=60,
            )
        # 10m low-HR transition gap (2 bins)
        for i in range(6, 8):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[96, 99],
                energy_kcal=Decimal("6"),
                watch_steps=40,
            )
        # 35m active Watch lifting
        for i in range(8, 15):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[125 + (k % 10) for k in range(20)],
                energy_kcal=Decimal("24"),
                watch_steps=90,
            )
        self.db.commit()

        res = self.client.get(
            "/biometrics/workout-timeline",
            params={"started_at": start.isoformat(), "ended_at": end.isoformat()},
        )
        self.assertEqual(res.status_code, 200)
        splits = res.json()["suggested_splits"]
        self.assertEqual(len(splits), 1)
        self.assertEqual(
            dt.datetime.fromisoformat(splits[0]["split_at"]),
            start + dt.timedelta(minutes=30),
        )
        self.assertEqual(
            dt.datetime.fromisoformat(splits[0]["resume_at"]),
            start + dt.timedelta(minutes=40),
        )
        self.assertEqual(splits[0]["confidence"], "high")

    def test_get_workout_timeline_active_to_gap_to_passive_and_edge_suppression(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        start = dt.datetime(2026, 11, 23, 17, 30, tzinfo=nz_tz)
        end = start + dt.timedelta(minutes=70)

        # First 5m passive edge (must NOT trigger split at +5m)
        self._add_bin_samples(start, hr_values=[130, 134], energy_kcal=Decimal("22"))
        # Next 30m active Watch mode
        for i in range(1, 7):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[142 + (k % 6) for k in range(16)],
                energy_kcal=Decimal("32"),
            )
        # 5m low-HR gap
        self._add_bin_samples(
            start + dt.timedelta(minutes=35),
            hr_values=[96],
            energy_kcal=Decimal("5"),
        )
        # Final 30m passive exertion
        for i in range(8, 14):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[124, 128],
                energy_kcal=Decimal("22"),
            )
        self.db.commit()

        res = self.client.get(
            "/biometrics/workout-timeline",
            params={"started_at": start.isoformat(), "ended_at": end.isoformat()},
        )
        self.assertEqual(res.status_code, 200)
        splits = res.json()["suggested_splits"]
        self.assertEqual(len(splits), 1)
        self.assertEqual(
            dt.datetime.fromisoformat(splits[0]["split_at"]),
            start + dt.timedelta(minutes=35),
        )
        self.assertEqual(
            dt.datetime.fromisoformat(splits[0]["resume_at"]),
            start + dt.timedelta(minutes=40),
        )
        self.assertEqual(splits[0]["confidence"], "high")

    def test_get_workout_timeline_active_gap_active_emits_single_split(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        start = dt.datetime(2026, 11, 23, 17, 30, tzinfo=nz_tz)
        end = start + dt.timedelta(minutes=70)

        for i in range(6):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[140 + (k % 6) for k in range(20)],
                energy_kcal=Decimal("30"),
            )
        self._add_bin_samples(
            start + dt.timedelta(minutes=30),
            hr_values=[95],
            energy_kcal=Decimal("5"),
        )
        for i in range(7, 14):
            self._add_bin_samples(
                start + dt.timedelta(minutes=i * 5),
                hr_values=[130 + (k % 6) for k in range(20)],
                energy_kcal=Decimal("25"),
            )
        self.db.commit()

        res = self.client.get(
            "/biometrics/workout-timeline",
            params={"started_at": start.isoformat(), "ended_at": end.isoformat()},
        )
        self.assertEqual(res.status_code, 200)
        splits = res.json()["suggested_splits"]
        self.assertEqual(len(splits), 1)
        self.assertEqual(
            dt.datetime.fromisoformat(splits[0]["split_at"]),
            start + dt.timedelta(minutes=30),
        )
        self.assertEqual(
            dt.datetime.fromisoformat(splits[0]["resume_at"]),
            start + dt.timedelta(minutes=35),
        )
        self.assertEqual(splits[0]["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
