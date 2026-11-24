"""Seam tests for the app.exertion deep module, callers, and architectural boundaries."""

import ast
import asyncio
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
from zoneinfo import ZoneInfo

from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.exertion as exertion_pkg
from app.db import Base
from app.exertion import ExertionAnalyzer, ExertionProfile, InvalidWindowError
from app.exertion._slots import bucket_samples_into_5min_slots, is_exertion_slot, sample_local_dt
from app.exertion._types import SampleSlot
from app.models import HealthSample
from app.routers.mcp import build_mcp_server


class TestExertionDeepModule(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        self.db = self.SessionLocal()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)

    def _add_sample(
        self,
        *,
        sample_type: str,
        measured_at: datetime,
        value: Decimal,
        ended_at: datetime | None = None,
        unit: str = "count",
    ) -> None:
        self.db.add(
            HealthSample(
                id=uuid.uuid4(),
                source="apple_health",
                sample_type=sample_type,
                measured_at=measured_at,
                ended_at=ended_at or (measured_at + timedelta(minutes=5)),
                value=value,
                unit=unit,
            )
        )

    def test_public_exports_are_strictly_bounded(self):
        self.assertEqual(
            set(exertion_pkg.__all__),
            {"ExertionAnalyzer", "ExertionProfile", "InvalidWindowError"},
        )

    def test_invalid_window_error_for_equal_inverted_and_mixed_tz_bounds(self):
        nz_tz = ZoneInfo("Pacific/Auckland")
        analyzer = ExertionAnalyzer(self.db, ExertionProfile(tz=nz_tz, max_hr=186))

        aware_start = datetime(2026, 11, 20, 18, 0, tzinfo=nz_tz)
        aware_earlier = datetime(2026, 11, 20, 17, 30, tzinfo=nz_tz)
        naive_equal = datetime(2026, 11, 20, 18, 0)
        naive_earlier = datetime(2026, 11, 20, 17, 45)

        invalid_pairs = [
            (aware_start, aware_start),
            (aware_start, aware_earlier),
            (aware_start, naive_equal),
            (aware_start, naive_earlier),
            (naive_equal, aware_earlier),
        ]
        for s_dt, e_dt in invalid_pairs:
            with self.subTest(started_at=s_dt, ended_at=e_dt):
                with self.assertRaises(InvalidWindowError):
                    analyzer.analyze_window(s_dt, e_dt)
                with self.assertRaises(InvalidWindowError):
                    analyzer.build_timeline(s_dt, e_dt)

    def test_custom_profile_max_hr_derives_zone_cutoffs(self):
        nz_tz = ZoneInfo("Pacific/Auckland")
        start = datetime(2026, 11, 21, 10, 0, tzinfo=nz_tz)
        end = start + timedelta(minutes=30)

        # Boundary HR values:
        # - 115: Z2 at max_hr=186 (112..130), Z1 at max_hr=200 (<120)
        # - 135: Z3 at max_hr=186 (130..149), Z2 at max_hr=200 (120..140)
        # - 155: Z4 at max_hr=186 (149..167), Z3 at max_hr=200 (140..160)
        # - 175: Z5 at max_hr=186 (>=167), Z4 at max_hr=200 (160..180)
        # - 182: Z5 at both max_hr=186 and max_hr=200 (>=180)
        hrs = [115, 135, 155, 175, 182]
        for idx, hr in enumerate(hrs):
            self._add_sample(
                sample_type="heart_rate",
                measured_at=start + timedelta(minutes=idx * 5),
                value=Decimal(str(hr)),
                unit="bpm",
            )
            self._add_sample(
                sample_type="active_energy",
                measured_at=start + timedelta(minutes=idx * 5),
                value=Decimal("30"),
                unit="kcal",
            )
        self.db.commit()

        default_res = ExertionAnalyzer(self.db).analyze_window(start, end)
        custom_res = ExertionAnalyzer(
            self.db, ExertionProfile(tz=nz_tz, max_hr=200)
        ).analyze_window(start, end)

        self.assertEqual(default_res.hr_zones["z1_recovery_pct"], 0.0)
        self.assertEqual(default_res.hr_zones["z2_aerobic_pct"], 20.0)
        self.assertEqual(default_res.hr_zones["z3_tempo_pct"], 20.0)
        self.assertEqual(default_res.hr_zones["z4_threshold_pct"], 20.0)
        self.assertEqual(default_res.hr_zones["z5_anaerobic_pct"], 40.0)

        self.assertEqual(custom_res.hr_zones["z1_recovery_pct"], 20.0)
        self.assertEqual(custom_res.hr_zones["z2_aerobic_pct"], 20.0)
        self.assertEqual(custom_res.hr_zones["z3_tempo_pct"], 20.0)
        self.assertEqual(custom_res.hr_zones["z4_threshold_pct"], 20.0)
        self.assertEqual(custom_res.hr_zones["z5_anaerobic_pct"], 20.0)

    def test_timezone_normalization_and_unified_is_exertion_slot(self):
        tokyo_tz = ZoneInfo("Asia/Tokyo")
        target_day = date(2026, 11, 22)
        start_jst = datetime(2026, 11, 22, 18, 0, tzinfo=tokyo_tz)
        for i in range(7):
            t_local = start_jst + timedelta(minutes=i * 5)
            self._add_sample(sample_type="active_energy", measured_at=t_local, value=Decimal("40"), unit="kcal")
            self._add_sample(sample_type="heart_rate", measured_at=t_local, value=Decimal("140"), unit="bpm")
            self._add_sample(sample_type="step_count", measured_at=t_local, value=Decimal("320"), unit="count")
        self.db.commit()

        analyzer = ExertionAnalyzer(self.db, ExertionProfile(tz=tokyo_tz, max_hr=186))
        detected = analyzer.detect_day(target_day)
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0].started_at.tzinfo, tokyo_tz)
        self.assertEqual(detected[0].started_at, start_jst)

        # Mixed UTC + naive window bounds normalize into analyzer profile tz
        utc_start = start_jst.astimezone(timezone.utc)
        naive_end = datetime(2026, 11, 22, 18, 35)
        window_res = analyzer.analyze_window(utc_start, naive_end)
        self.assertEqual(window_res.started_at, start_jst)
        self.assertEqual(window_res.ended_at, start_jst + timedelta(minutes=35))
        self.assertEqual(window_res.duration_minutes, 35)
        self.assertEqual(window_res.active_energy_kcal, 280)

        # Aware sample timestamp conversion into profile tz
        aware_sample = HealthSample(
            id=uuid.uuid4(),
            source="apple_health",
            sample_type="heart_rate",
            measured_at=utc_start,
            value=Decimal("140"),
            unit="bpm",
        )
        self.assertEqual(sample_local_dt(aware_sample, tokyo_tz), start_jst)

        # Unified 4-branch is_exertion_slot contract
        t0 = datetime(2026, 11, 22, 18, 0, tzinfo=tokyo_tz)
        t1 = t0 + timedelta(minutes=5)
        self.assertTrue(is_exertion_slot(SampleSlot(t0, t1, Decimal("20"), (), Decimal("0"))))
        self.assertTrue(is_exertion_slot(SampleSlot(t0, t1, Decimal("0"), (120.0,), Decimal("0"))))
        self.assertTrue(is_exertion_slot(SampleSlot(t0, t1, Decimal("0"), (95.0,), Decimal("250"))))
        self.assertTrue(is_exertion_slot(SampleSlot(t0, t1, Decimal("15"), (100.0,), Decimal("0"))))
        self.assertFalse(is_exertion_slot(SampleSlot(t0, t1, Decimal("14"), (99.0,), Decimal("249"))))

    def test_mcp_get_workout_metrics_translates_invalid_window_and_succeeds(self):
        nz_tz = ZoneInfo("Pacific/Auckland")
        start = datetime(2026, 11, 23, 19, 0, tzinfo=nz_tz)
        end = start + timedelta(minutes=30)
        for i in range(6):
            t = start + timedelta(minutes=i * 5)
            self._add_sample(sample_type="active_energy", measured_at=t, value=Decimal("30"), unit="kcal")
            self._add_sample(sample_type="heart_rate", measured_at=t, value=Decimal("135"), unit="bpm")
        self.db.commit()

        @contextmanager
        def fake_mcp_session():
            session = self.SessionLocal()
            try:
                yield session
            finally:
                session.close()

        server = build_mcp_server()
        with patch("app.routers.mcp.mcp_session", fake_mcp_session):
            # Mixed naive + aware where aware end <= naive start (in Pacific/Auckland)
            with self.assertRaises(ToolError):
                asyncio.run(
                    server.call_tool(
                        "eir_get_workout_metrics",
                        {
                            "started_at": "2026-11-23T19:00:00",
                            "ended_at": "2026-11-23T18:30:00+13:00",
                        },
                    )
                )

            result = asyncio.run(
                server.call_tool(
                    "eir_get_workout_metrics",
                    {
                        "started_at": start.isoformat(),
                        "ended_at": end.isoformat(),
                        "classification": "gym",
                    },
                )
            )
        content_blocks = result[0] if isinstance(result, tuple) else result.content
        payload = json.loads(content_blocks[0].text)
        self.assertEqual(payload["duration_minutes"], 30)
        self.assertEqual(payload["classification"], "gym")
        self.assertEqual(payload["active_energy_kcal"], 180)
        self.assertEqual(payload["avg_hr_bpm"], 135)

    def test_zero_imports_from_app_routers_biometrics_in_app_package(self):
        app_root = Path(__file__).resolve().parents[1] / "app"
        offenders: list[str] = []
        for py_file in sorted(app_root.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "app.routers.biometrics":
                    offenders.append(str(py_file.relative_to(app_root)))
                elif isinstance(node, ast.Import):
                    if any(alias.name == "app.routers.biometrics" for alias in node.names):
                        offenders.append(str(py_file.relative_to(app_root)))
        self.assertEqual(offenders, [], f"Found forbidden imports from app.routers.biometrics: {offenders}")

    def _seed_hr_burst(self, slot_start: datetime, count: int, hr_values: list[int]) -> None:
        step_s = max(1, 295 // max(1, count))
        for idx in range(count):
            t = slot_start + timedelta(seconds=min(295, idx * step_s))
            self._add_sample(
                sample_type="heart_rate",
                measured_at=t,
                ended_at=t,
                value=Decimal(str(hr_values[idx % len(hr_values)])),
                unit="bpm",
            )

    def test_noon_swim_and_elevated_hr_post_swim_walk_split(self):
        nz_tz = ZoneInfo("Pacific/Auckland")
        target_day = date(2026, 11, 24)
        swim_start = datetime(2026, 11, 24, 11, 30, tzinfo=nz_tz)

        swim_hrs = [126, 130, 136, 144, 143, 150, 150]
        swim_counts = [59, 60, 60, 60, 60, 61, 55]
        swim_kcals = ["34", "27", "33", "37", "31", "33", "23.5"]
        for i in range(7):
            slot_t = swim_start + timedelta(minutes=i * 5)
            self._add_sample(sample_type="active_energy", measured_at=slot_t, value=Decimal(swim_kcals[i]), unit="kcal")
            hrs = [swim_hrs[i], min(164, swim_hrs[i] + 14)] if i == 6 else [swim_hrs[i]]
            self._seed_hr_burst(slot_t, swim_counts[i], hrs)

        walk_specs = [
            (0, 6, [149, 151, 147], "9.3"),
            (5, 12, [131, 138, 126], "5.9"),
            (10, 19, [125, 130, 122], "13.8"),
            (15, 15, [127, 130, 124], "36.6"),
        ]
        walk_start = datetime(2026, 11, 24, 12, 5, tzinfo=nz_tz)
        for offset_m, cnt, hrs, kcal in walk_specs:
            slot_t = walk_start + timedelta(minutes=offset_m)
            self._add_sample(sample_type="active_energy", measured_at=slot_t, value=Decimal(kcal), unit="kcal")
            self._seed_hr_burst(slot_t, cnt, hrs)

        step_intervals = [
            (datetime(2026, 11, 24, 12, 1, 57, tzinfo=nz_tz), datetime(2026, 11, 24, 12, 11, 36, tzinfo=nz_tz), "318"),
            (datetime(2026, 11, 24, 12, 12, 32, tzinfo=nz_tz), datetime(2026, 11, 24, 12, 13, 33, tzinfo=nz_tz), "24"),
            (datetime(2026, 11, 24, 12, 13, 51, tzinfo=nz_tz), datetime(2026, 11, 24, 12, 14, 29, tzinfo=nz_tz), "36"),
            (datetime(2026, 11, 24, 12, 14, 29, tzinfo=nz_tz), datetime(2026, 11, 24, 12, 24, 26, tzinfo=nz_tz), "947"),
            (datetime(2026, 11, 24, 12, 24, 29, tzinfo=nz_tz), datetime(2026, 11, 24, 12, 26, 49, tzinfo=nz_tz), "253"),
        ]
        for s_dt, e_dt, val in step_intervals:
            self._add_sample(sample_type="step_count", measured_at=s_dt, ended_at=e_dt, value=Decimal(val), unit="count")
        self.db.commit()

        workouts = ExertionAnalyzer(self.db, ExertionProfile(tz=nz_tz, max_hr=186)).detect_day(target_day)
        self.assertEqual(len(workouts), 2)
        self.assertEqual(workouts[0].classification, "swim")
        self.assertEqual(workouts[0].started_at, swim_start)
        self.assertEqual(workouts[0].ended_at, walk_start)
        self.assertEqual(workouts[0].duration_minutes, 35)
        self.assertEqual(workouts[1].classification, "walk")
        self.assertEqual(workouts[1].started_at, walk_start)
        self.assertEqual(workouts[1].ended_at, datetime(2026, 11, 24, 12, 25, tzinfo=nz_tz))
        self.assertEqual(workouts[1].duration_minutes, 20)
        self.assertGreaterEqual(workouts[1].step_count or 0, 1250)

    def test_step_interval_proration_conserves_decimal_and_handles_boundaries(self):
        nz_tz = ZoneInfo("Pacific/Auckland")
        win_start = datetime(2026, 11, 24, 12, 0, tzinfo=nz_tz)
        win_end = datetime(2026, 11, 24, 12, 20, tzinfo=nz_tz)

        # 10-min interval fully inside window (12:01:57 -> 12:11:36 UTC-converted)
        s1_start = datetime(2026, 11, 24, 12, 1, 57, tzinfo=nz_tz).astimezone(timezone.utc)
        s1_end = datetime(2026, 11, 24, 12, 11, 36, tzinfo=nz_tz).astimezone(timezone.utc)
        s1 = HealthSample(
            id=uuid.uuid4(), source="apple_health", sample_type="step_count",
            measured_at=s1_start, ended_at=s1_end, value=Decimal("318"), unit="count",
        )
        # Single-slot fallback when ended_at <= measured_at
        s2_t = datetime(2026, 11, 24, 12, 15, tzinfo=nz_tz)
        s2 = HealthSample(
            id=uuid.uuid4(), source="apple_health", sample_type="step_count",
            measured_at=s2_t, ended_at=s2_t, value=Decimal("50"), unit="count",
        )
        bucketed = bucket_samples_into_5min_slots([s1, s2], win_start, win_end, nz_tz)
        self.assertEqual(sum(slot.steps for slot in bucketed.slots), Decimal("368"))
        self.assertAlmostEqual(float(bucketed.slots[0].steps), 318 * 183 / 579, places=4)
        self.assertAlmostEqual(float(bucketed.slots[1].steps), 318 * 300 / 579, places=4)
        self.assertAlmostEqual(float(bucketed.slots[2].steps), 318 * 96 / 579, places=4)
        self.assertEqual(bucketed.slots[3].steps, Decimal("50"))

        # Sample crossing window end is clamped proportionally without inflating into final slot
        clamped = bucket_samples_into_5min_slots([s1], win_start, win_start + timedelta(minutes=5), nz_tz)
        self.assertAlmostEqual(float(clamped.slots[0].steps), 318 * 183 / 579, places=4)

    def test_15_min_walk_capture_and_subthreshold_filtering(self):
        nz_tz = ZoneInfo("Pacific/Auckland")
        analyzer = ExertionAnalyzer(self.db, ExertionProfile(tz=nz_tz, max_hr=186))

        # 15-min brisk walk (960 steps >= 900) -> detected as walk
        d_pass = date(2026, 11, 17)
        t_pass = datetime(2026, 11, 17, 19, 50, tzinfo=nz_tz)
        for i in range(3):
            t = t_pass + timedelta(minutes=i * 5)
            self._add_sample(sample_type="active_energy", measured_at=t, value=Decimal("16"), unit="kcal")
            self._add_sample(sample_type="heart_rate", measured_at=t, value=Decimal("104"), unit="bpm")
            self._add_sample(sample_type="step_count", measured_at=t, value=Decimal("320"), unit="count")

        # 10-min brief movement (700 steps, < 15 min) and 15-min low-step (< 900 steps) -> filtered
        d_fail = date(2026, 11, 18)
        t_short = datetime(2026, 11, 18, 9, 0, tzinfo=nz_tz)
        for i in range(2):
            t = t_short + timedelta(minutes=i * 5)
            self._add_sample(sample_type="active_energy", measured_at=t, value=Decimal("20"), unit="kcal")
            self._add_sample(sample_type="heart_rate", measured_at=t, value=Decimal("110"), unit="bpm")
            self._add_sample(sample_type="step_count", measured_at=t, value=Decimal("350"), unit="count")

        t_low_steps = datetime(2026, 11, 18, 15, 0, tzinfo=nz_tz)
        for i in range(3):
            t = t_low_steps + timedelta(minutes=i * 5)
            self._add_sample(sample_type="active_energy", measured_at=t, value=Decimal("16"), unit="kcal")
            self._add_sample(sample_type="heart_rate", measured_at=t, value=Decimal("104"), unit="bpm")
            self._add_sample(sample_type="step_count", measured_at=t, value=Decimal("280"), unit="count")
        self.db.commit()

        pass_workouts = analyzer.detect_day(d_pass)
        self.assertEqual(len(pass_workouts), 1)
        self.assertEqual(pass_workouts[0].classification, "walk")
        self.assertEqual(pass_workouts[0].duration_minutes, 15)
        self.assertEqual(pass_workouts[0].step_count, 960)
        self.assertEqual(analyzer.detect_day(d_fail), [])

    def test_elevated_hr_steady_walk_vs_prorated_badminton(self):
        nz_tz = ZoneInfo("Pacific/Auckland")
        analyzer = ExertionAnalyzer(self.db, ExertionProfile(tz=nz_tz, max_hr=186))

        # Steady elevated-HR walk (20m, 1,284 steps = 64.2 steps/min, avg HR 134, peak HR 151, declining HR)
        d_walk = date(2026, 11, 21)
        t_walk = datetime(2026, 11, 21, 12, 5, tzinfo=nz_tz)
        for i, hr in enumerate([151, 134, 126, 126]):
            t = t_walk + timedelta(minutes=i * 5)
            self._add_sample(sample_type="active_energy", measured_at=t, value=Decimal("20"), unit="kcal")
            self._add_sample(sample_type="heart_rate", measured_at=t, value=Decimal(str(hr)), unit="bpm")
            self._add_sample(sample_type="step_count", measured_at=t, value=Decimal("321"), unit="count")

        # True badminton with 10-min step intervals pro-rated to ~200 steps/5m (avg HR 156, peak 178)
        d_badmin = date(2026, 11, 22)
        t_badmin = datetime(2026, 11, 22, 19, 15, tzinfo=nz_tz)
        for i in range(8):
            t = t_badmin + timedelta(minutes=i * 5)
            self._add_sample(sample_type="active_energy", measured_at=t, value=Decimal("25"), unit="kcal")
            self._add_sample(sample_type="heart_rate", measured_at=t, value=Decimal(str(150 if i % 2 == 0 else 164)), unit="bpm")
        spike_t = t_badmin + timedelta(minutes=15, seconds=30)
        self._add_sample(sample_type="heart_rate", measured_at=spike_t, ended_at=spike_t, value=Decimal("178"), unit="bpm")
        for i in range(4):
            s_t = t_badmin + timedelta(minutes=i * 10)
            self._add_sample(
                sample_type="step_count", measured_at=s_t, ended_at=s_t + timedelta(minutes=10),
                value=Decimal("370"), unit="count",
            )
        self.db.commit()

        walk_res = analyzer.detect_day(d_walk)
        self.assertEqual(len(walk_res), 1)
        self.assertEqual(walk_res[0].classification, "walk")
        self.assertEqual(walk_res[0].routine_name, "Walk / Cardio")

        badmin_res = analyzer.detect_day(d_badmin)
        self.assertEqual(len(badmin_res), 1)
        self.assertEqual(badmin_res[0].classification, "badminton")
        self.assertEqual(badmin_res[0].routine_name, "Badminton Match Play")


if __name__ == "__main__":
    unittest.main()
