"""Tests for the app.daily_biometrics deep module, step relocation, and MCP/router parity."""

import ast
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import unittest
import uuid
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app import schemas
from app.daily_biometrics import (
    METRIC_REGISTRY,
    BiometricsInputError,
    DailyBiometrics,
    RollupBatch,
)
from app.exertion import deduplicate_step_samples, filter_deduplicated_step_samples
from app.models import BiometricMeasurement, HealthFeedback, HealthSample
from app.schemas import BiometricCreate, DetectedWorkoutRead
from tests._harness import EirTestCase, call_mcp_tool


class _SpyExertion:
    """Test double recording detect_day calls and returning canned workouts."""

    def __init__(self, workouts_by_day: dict[date, list[DetectedWorkoutRead]] | None = None) -> None:
        self.calls: list[date] = []
        self._workouts_by_day = workouts_by_day or {}

    def detect_day(self, target_date: date) -> list[DetectedWorkoutRead]:
        self.calls.append(target_date)
        return list(self._workouts_by_day.get(target_date, []))


class TestDailyBiometricsDeepModule(EirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.nz_tz = ZoneInfo("Pacific/Auckland")

    def _add_sample(
        self,
        *,
        sample_type: str,
        measured_at: datetime,
        ended_at: datetime | None = None,
        value: Decimal,
        unit: str | None = None,
        source: str = "apple_health",
        extra: dict | None = None,
        deleted_at: datetime | None = None,
    ) -> HealthSample:
        row = HealthSample(
            id=uuid.uuid4(),
            source=source,
            sample_type=sample_type,
            measured_at=measured_at,
            ended_at=ended_at or measured_at,
            value=value,
            unit=unit,
            extra=extra or {},
            deleted_at=deleted_at,
        )
        self.db.add(row)
        return row

    def test_registry_canonicalization_aliases_and_schema_backing(self) -> None:
        self.assertEqual(METRIC_REGISTRY.canonicalize("active_energy_kcal"), "active_energy")
        self.assertEqual(METRIC_REGISTRY.canonicalize("walking_ asymmetry"), "walking_asymmetry")
        self.assertEqual(METRIC_REGISTRY.canonicalize("custom_metric"), "custom_metric")

        self.assertEqual(
            METRIC_REGISTRY.aliases_for("active_energy"),
            ("active_energy", "active_energy_kcal", "energy"),
        )
        self.assertEqual(
            METRIC_REGISTRY.aliases_for("active_energy_kcal"),
            ("active_energy", "active_energy_kcal", "energy"),
        )
        self.assertEqual(
            METRIC_REGISTRY.aliases_for("sleep"),
            ("sleep", "sleep_analysis", "sleep_duration", "sleep_duration_s"),
        )

        asymmetry_aliases = METRIC_REGISTRY.aliases_for("walking_asymmetry")
        self.assertIn("walking_asymmetry", asymmetry_aliases)
        self.assertIn("walking asymmetry", asymmetry_aliases)
        self.assertNotIn("walking_ asymmetry", asymmetry_aliases)

        self.assertEqual(schemas.METRIC_ALIASES, METRIC_REGISTRY.input_aliases())
        self.assertEqual(schemas.DEFAULT_METRIC_UNITS, METRIC_REGISTRY.default_units())
        self.assertEqual(schemas.METRIC_ALIASES["walking_ asymmetry"], "walking_asymmetry")
        self.assertEqual(METRIC_REGISTRY.default_unit_for("sleep_duration"), "s")
        self.assertEqual(METRIC_REGISTRY.default_unit_for("active_energy"), "kcal")

    def test_resolve_local_days_utc_boundary_and_validation(self) -> None:
        utc_evening = datetime(2026, 11, 24, 11, 30, 0, tzinfo=timezone.utc)
        naive_evening = datetime(2026, 11, 24, 11, 30, 0)
        early_day = date(2026, 11, 23)
        service = DailyBiometrics(self.db, tz=self.nz_tz, exertion=_SpyExertion())

        batch = service.materialize_rollups([utc_evening, naive_evening, early_day, date(2026, 11, 25)])
        self.assertEqual(batch.days, (date(2026, 11, 23), date(2026, 11, 25)))
        self.assertEqual(service.materialize_rollups([]).days, ())

        for invalid in (True, "2026-11-24", 123):
            with self.assertRaises(BiometricsInputError):
                service.materialize_rollups([invalid])  # type: ignore[list-item]

    def test_materialize_rollups_single_pass_energy_floor_and_flush_only(self) -> None:
        target_day = date(2026, 11, 24)
        sample_dt = datetime(2026, 11, 24, 9, 0, tzinfo=self.nz_tz)
        self._add_sample(sample_type="active_energy", measured_at=sample_dt, value=Decimal("150"), unit="kcal")
        self._add_sample(
            sample_type="step_count",
            measured_at=sample_dt,
            ended_at=sample_dt + timedelta(minutes=10),
            value=Decimal("400"),
            unit="count",
            extra={"device": "Alex Apple Watch"},
        )
        self.db.commit()

        workout = DetectedWorkoutRead(
            started_at=sample_dt,
            ended_at=sample_dt + timedelta(minutes=45),
            duration_minutes=45,
            active_energy_kcal=420,
            avg_hr_bpm=148,
            peak_hr_bpm=172,
            classification="badminton",
        )
        spy = _SpyExertion({target_day: [workout]})
        service = DailyBiometrics(self.db, tz=self.nz_tz, exertion=spy)

        batch = service.materialize_rollups(
            [sample_dt, sample_dt + timedelta(hours=2), target_day]
        )
        self.assertIsInstance(batch, RollupBatch)
        self.assertEqual(batch.days, (target_day,))
        self.assertEqual(spy.calls, [target_day])
        self.assertEqual(batch.detected_workouts, (workout,))
        self.assertEqual(batch.results[0].measurement.active_energy_kcal, 420)
        self.assertEqual(batch.results[0].measurement.step_count, 400)

        # Verify flush-only semantics: rollback removes the uncommitted rollup row
        self.db.rollback()
        self.assertIsNone(self.db.scalar(select(BiometricMeasurement)))

        # Re-run and commit persists the rollup row
        spy_commit = _SpyExertion({target_day: [workout]})
        DailyBiometrics(self.db, tz=self.nz_tz, exertion=spy_commit).materialize_rollups([target_day])
        self.db.commit()
        persisted = self.db.scalar(select(BiometricMeasurement))
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.active_energy_kcal, 420)

    def test_daily_summary_sleep_steps_vo2_feedback_notes_and_workouts(self) -> None:
        prior_day = date(2026, 11, 20)
        target_day = date(2026, 11, 24)

        # Rolling VO2 baseline from an earlier day
        self._add_sample(
            sample_type="vo2_max",
            measured_at=datetime(2026, 11, 20, 8, 0, tzinfo=self.nz_tz),
            value=Decimal("47.4"),
            unit="ml/kg/min",
        )
        # Sleep stages: Awake and In Bed must be excluded
        sleep_t = datetime(2026, 11, 24, 1, 0, tzinfo=self.nz_tz)
        for idx, (stage, secs) in enumerate(
            [("Deep", 3600), ("REM", 1800), ("Core", 5400), ("Awake", 900), ("In Bed", 1200)]
        ):
            self._add_sample(
                sample_type="sleep",
                measured_at=sleep_t + timedelta(hours=idx),
                value=Decimal(str(secs)),
                unit="s",
                extra={"stage": stage},
            )
        # Watch-priority steps: overlapping iPhone dropped, non-overlapping iPhone kept
        step_t = datetime(2026, 11, 24, 12, 0, tzinfo=self.nz_tz)
        self._add_sample(
            sample_type="step_count",
            measured_at=step_t,
            ended_at=step_t + timedelta(minutes=10),
            value=Decimal("300"),
            unit="count",
            extra={"device": "Alex Apple Watch"},
        )
        self._add_sample(
            sample_type="step_count",
            measured_at=step_t + timedelta(minutes=2),
            ended_at=step_t + timedelta(minutes=8),
            value=Decimal("250"),
            unit="count",
            extra={"device": "Alex iPhone"},
        )
        self._add_sample(
            sample_type="step_count",
            measured_at=datetime(2026, 11, 24, 8, 0, tzinfo=self.nz_tz),
            ended_at=datetime(2026, 11, 24, 8, 15, tzinfo=self.nz_tz),
            value=Decimal("120"),
            unit="count",
            extra={"device": "Alex iPhone"},
        )
        # Readiness note from measurement row (without step_count/sleep so granular arbitration runs)
        self.db.add(
            BiometricMeasurement(
                id=uuid.uuid4(),
                measured_at=datetime(2026, 11, 24, 7, 30, tzinfo=self.nz_tz),
                source="withings",
                weight_kg=Decimal("78.2"),
                notes="Rested well",
            )
        )
        # Active HealthFeedback rows
        self.db.add(
            HealthFeedback(
                id=uuid.uuid4(),
                date=target_day,
                feedback_type="fasting",
                rating="completed",
                notes="16:8 clean",
            )
        )
        self.db.add(
            HealthFeedback(
                id=uuid.uuid4(),
                date=target_day,
                feedback_type="wellness",
                rating="high",
                score=Decimal("8.5"),
            )
        )
        self.db.commit()

        workout = DetectedWorkoutRead(
            started_at=datetime(2026, 11, 24, 18, 0, tzinfo=self.nz_tz),
            ended_at=datetime(2026, 11, 24, 18, 40, tzinfo=self.nz_tz),
            duration_minutes=40,
            active_energy_kcal=360,
            classification="run",
        )
        spy = _SpyExertion({target_day: [workout]})
        summary = DailyBiometrics(self.db, tz=self.nz_tz, exertion=spy).daily_summary(target_day)

        self.assertEqual(summary.date, target_day)
        self.assertEqual(summary.latest_weight_kg, Decimal("78.2"))
        self.assertEqual(summary.sleep_duration_hours, 3.0)
        self.assertEqual(summary.sleep_deep_hours, 1.0)
        self.assertEqual(summary.sleep_rem_hours, 0.5)
        self.assertEqual(summary.step_count, 420)
        self.assertEqual(summary.vo2_max, 47.4)
        self.assertEqual(summary.readiness_notes, ["Rested well"])
        self.assertIsNotNone(summary.fasting)
        self.assertEqual(summary.fasting.rating, "completed")
        self.assertIsNotNone(summary.wellness)
        self.assertEqual(summary.wellness.rating, "high")
        self.assertIsNone(summary.workout)
        self.assertEqual(summary.detected_workouts, [workout])
        self.assertEqual(summary.active_energy_kcal, 360)
        self.assertEqual(spy.calls, [target_day])
        self.assertNotEqual(prior_day, target_day)

    def test_granular_samples_aliases_soft_delete_and_input_validation(self) -> None:
        target_day = date(2026, 11, 24)
        t1 = datetime(2026, 11, 24, 10, 0, tzinfo=self.nz_tz)
        t2 = datetime(2026, 11, 24, 10, 5, tzinfo=self.nz_tz)
        t3 = datetime(2026, 11, 24, 10, 10, tzinfo=self.nz_tz)

        self._add_sample(sample_type="active_energy", measured_at=t1, value=Decimal("12.0"), unit="kcal")
        self._add_sample(sample_type="energy", measured_at=t2, value=Decimal("8.0"), unit="kcal")
        self._add_sample(
            sample_type="active_energy_kcal",
            measured_at=t3,
            value=Decimal("99.0"),
            unit="kcal",
            deleted_at=datetime(2026, 11, 24, 10, 15, tzinfo=self.nz_tz),
        )
        self._add_sample(sample_type="sleep", measured_at=t1, value=Decimal("1800"), unit="s")
        self.db.commit()

        service = DailyBiometrics(self.db, tz=self.nz_tz)
        energy_rows = service.granular_samples(target_day, "active_energy_kcal")
        self.assertEqual([r.value for r in energy_rows], [Decimal("12.0000"), Decimal("8.0000")])

        sleep_rows = service.granular_samples(target_day, "sleep_analysis")
        self.assertEqual(len(sleep_rows), 1)
        self.assertEqual(sleep_rows[0].value, Decimal("1800.0000"))

        with self.assertRaises(BiometricsInputError):
            service.granular_samples(target_day, "   ")
        with self.assertRaises(BiometricsInputError):
            service.granular_samples(target_day, "active_energy", limit=0)
        with self.assertRaises(BiometricsInputError):
            service.granular_samples(target_day, "active_energy", limit=5001)
        with self.assertRaises(BiometricsInputError):
            service.granular_samples(target_day, "active_energy", limit=True)

    def _seed_mcp_parity_samples(self, sample_dt: datetime) -> None:
        for sample_type, value, unit in (
            ("wrist_temperature", "36.55", "C"),
            ("respiratory_rate", "14.2", "breaths/min"),
            ("oxygen_saturation", "98.1", "%"),
            ("walking_asymmetry", "1.25", "%"),
            ("vo2_max", "48.3", "ml/kg/min"),
            ("active_energy_kcal", "220", "kcal"),
        ):
            self._add_sample(
                sample_type=sample_type,
                measured_at=sample_dt,
                value=Decimal(value),
                unit=unit,
            )
        self.db.commit()

    def test_mcp_tools_parity_with_daily_biometrics(self) -> None:
        target_day = date(2026, 11, 24)
        self._seed_mcp_parity_samples(datetime(2026, 11, 24, 7, 15, tzinfo=self.nz_tz))

        token = "parity-mcp-token"
        self.patch_settings(mcp_token=token)
        with self.open_lifespan_client() as client:
            is_err, empty_latest = call_mcp_tool(client, "eir_get_latest_biometrics", {}, token=token)
            self.assertFalse(is_err)
            self.assertEqual(empty_latest["status"], "empty")

            is_err, logged = call_mcp_tool(
                client,
                "eir_log_measurement",
                {
                    "weight_kg": 80.4,
                    "body_fat_pct": 16.2,
                    "resting_hr_bpm": 52,
                    "hrv_ms": 71.0,
                    "sleep_duration_hours": 7.5,
                    "sleep_score": 89,
                    "step_count": 6400,
                    "notes": "Alex Tan check-in",
                },
                token=token,
            )
            self.assertFalse(is_err)
            self.assertEqual(logged["source"], "agent")
            self.assertEqual(logged["weight_kg"], 80.4)

            is_err, latest = call_mcp_tool(
                client, "eir_get_latest_biometrics", {"source": "agent"}, token=token
            )
            self.assertFalse(is_err)
            self.assertEqual(latest["id"], logged["id"])
            latest_rec = DailyBiometrics(self.db, tz=self.nz_tz).latest_measurement(source="agent")
            self.assertIsNotNone(latest_rec)
            self.assertEqual(str(latest_rec.id), logged["id"])

            is_err, granular = call_mcp_tool(
                client,
                "eir_get_granular_samples",
                {"target_date": target_day.isoformat(), "sample_type": "active_energy"},
                token=token,
            )
            self.assertFalse(is_err)
            self.assertEqual(granular["count"], 1)
            self.assertEqual(granular["samples"][0]["value"], 220.0)

            is_err, mcp_summary = call_mcp_tool(
                client, "eir_get_daily_summary", {"target_date": target_day.isoformat()}, token=token
            )
            self.assertFalse(is_err)
            expected_summary = DailyBiometrics(self.db).daily_summary(target_day).model_dump(mode="json")
            self.assertEqual(mcp_summary, expected_summary)
            self.assertNotIn("readings_count", mcp_summary)
            for field in (
                "sleep_deep_hours",
                "sleep_rem_hours",
                "wrist_temperature_c",
                "respiratory_rate_bpm",
                "oxygen_saturation_pct",
                "walking_asymmetry_pct",
                "vo2_max",
                "readiness_notes",
                "fasting",
                "workout",
                "wellness",
                "detected_workouts",
            ):
                self.assertIn(field, mcp_summary)

    def test_step_dedup_reexported_and_acyclic_boundaries(self) -> None:
        t0 = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
        s1 = HealthSample(
            source="apple_health",
            sample_type="step_count",
            measured_at=t0,
            ended_at=t0 + timedelta(minutes=10),
            value=Decimal("200"),
            extra={"device": "Alex Watch"},
        )
        s2 = HealthSample(
            source="apple_health",
            sample_type="step_count",
            measured_at=t0 + timedelta(minutes=2),
            ended_at=t0 + timedelta(minutes=8),
            value=Decimal("180"),
            extra={"device": "Alex iPhone"},
        )
        self.assertEqual(filter_deduplicated_step_samples([s1, s2]), [s1])
        self.assertEqual(deduplicate_step_samples([s1, s2]), Decimal("200"))

        app_root = Path(__file__).resolve().parents[1] / "app"
        self.assertFalse((app_root / "services" / "biometrics.py").exists())

        for py_path in sorted((app_root / "exertion").glob("*.py")):
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith("app.daily_biometrics"))
                    self.assertFalse(node.module.startswith("app.services.biometrics"))

        mcp_tree = ast.parse((app_root / "routers" / "mcp.py").read_text(encoding="utf-8"))
        for node in ast.walk(mcp_tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith("app.models"))
                self.assertNotEqual(node.module, "sqlalchemy")


if __name__ == "__main__":
    unittest.main()
