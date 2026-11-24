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
from app.exertion._slots import is_exertion_slot, sample_local_dt
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


if __name__ == "__main__":
    unittest.main()
