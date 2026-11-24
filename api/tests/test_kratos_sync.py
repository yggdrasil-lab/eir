"""Unit and contract tests for the Kratos reconciliation deep module."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from types import MappingProxyType
import unittest
from unittest.mock import patch
import uuid
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.kratos_sync import (
    HttpKratosSessionStore,
    InMemoryKratosSessionStore,
    KratosReconciler,
    KratosSession,
    KratosSessionWrite,
    KratosStoreError,
    SyncBatchResult,
    SyncOutcome,
    generate_session_id,
)
from app.main import app
from app.models import HealthSample
from app.schemas import DetectedWorkoutRead


class _StubWindowMetrics:
    """Deterministic stub implementing WindowMetricsPort."""

    def __init__(self, workout: DetectedWorkoutRead) -> None:
        self.workout = workout
        self.calls: list[tuple[datetime, datetime]] = []

    def analyze_window(self, start_dt: datetime, end_dt: datetime) -> DetectedWorkoutRead:
        self.calls.append((start_dt, end_dt))
        return self.workout


def _make_workout(
    start_dt: datetime | None = None,
    duration_mins: int = 60,
    *,
    kcal: int = 450,
    classification: str = "badminton",
    activity_type: str = "badminton",
    routine_name: str = "Badminton Match Play",
    effort_rating: str | None = "hard",
    rpe: float | None = 8.5,
    trimp: float | None = 220.0,
    hr_zones: dict[str, float] | None = None,
    hr_recovery_60s: int | None = 25,
    avg_hr_bpm: int | None = 142,
    peak_hr_bpm: int | None = 155,
    step_count: int | None = 5200,
    confidence: str = "high",
) -> DetectedWorkoutRead:
    base_start = start_dt or datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
    return DetectedWorkoutRead(
        started_at=base_start,
        ended_at=base_start + timedelta(minutes=duration_mins),
        duration_minutes=duration_mins,
        active_energy_kcal=kcal,
        avg_hr_bpm=avg_hr_bpm,
        peak_hr_bpm=peak_hr_bpm,
        step_count=step_count,
        confidence=confidence,
        classification=classification,
        activity_type=activity_type,
        routine_name=routine_name,
        effort_rating=effort_rating,
        rpe=rpe,
        trimp=trimp,
        hr_zones=hr_zones if hr_zones is not None else {"z4_threshold_pct": 35.0},
        hr_recovery_60s=hr_recovery_60s,
    )


class TestKratosReconciler(unittest.TestCase):
    def setUp(self) -> None:
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

    def tearDown(self) -> None:
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        app.dependency_overrides.clear()

    def test_generate_session_id_deterministic(self) -> None:
        s_iso = "2026-11-11T09:00:00+13:00"
        e_iso = "2026-11-11T10:00:00+13:00"
        id1 = generate_session_id(s_iso, e_iso)
        id2 = generate_session_id(s_iso, e_iso)
        self.assertEqual(id1, id2)
        self.assertIsInstance(id1, uuid.UUID)
        self.assertNotEqual(id1, generate_session_id(s_iso, "2026-11-11T10:30:00+13:00"))

    def test_blank_kratos_url_skips_with_zero_io(self) -> None:
        result = KratosReconciler(kratos_url="   ").sync([_make_workout()])
        self.assertEqual(result, SyncBatchResult(configured=False, outcomes=()))

    def test_empty_workouts_returns_configured_with_zero_io(self) -> None:
        store = InMemoryKratosSessionStore()
        result = KratosReconciler(store=store).sync([])
        self.assertEqual(result, SyncBatchResult(configured=True, outcomes=()))
        self.assertEqual(store.upserts, [])
        self.assertEqual(store.deleted_ids, [])

    def test_sync_creates_new_inferred_session(self) -> None:
        workout = _make_workout()
        store = InMemoryKratosSessionStore()

        result = KratosReconciler(store=store).sync([workout])

        self.assertTrue(result.configured)
        self.assertEqual(len(result.outcomes), 1)
        outcome = result.outcomes[0]
        self.assertEqual(outcome.status, "synced")
        self.assertEqual(outcome.routine_name, "Badminton Match Play")
        self.assertEqual(len(store.upserts), 1)

        payload = store.upserts[0].to_api_dict()
        self.assertEqual(payload["id"], outcome.session_id)
        self.assertIsNone(payload["routine_id"])
        self.assertEqual(payload["items"], [])
        self.assertTrue(payload["extra"]["inferred"])
        self.assertEqual(payload["extra"]["classification"], "badminton")
        self.assertEqual(payload["extra"]["active_energy_kcal"], 450)
        self.assertEqual(payload["extra"]["effort_rating"], "hard")
        self.assertEqual(payload["extra"]["rpe"], 8.5)
        self.assertEqual(payload["extra"]["trimp"], 220.0)
        self.assertEqual(payload["extra"]["hr_recovery_60s"], 25)

    def test_sync_updates_existing_inferred_session(self) -> None:
        workout = _make_workout(kcal=510)
        session_id = str(generate_session_id(workout.started_at.isoformat(), workout.ended_at.isoformat()))
        store = InMemoryKratosSessionStore(
            initial=[
                {
                    "id": session_id,
                    "started_at": workout.started_at.isoformat(),
                    "ended_at": workout.ended_at.isoformat(),
                    "extra": {"inferred": True, "active_energy_kcal": 300, "custom_inferred_tag": "keep-me"},
                    "items": [],
                }
            ]
        )

        result = KratosReconciler(store=store).sync([workout])
        self.assertEqual(result.outcomes[0].status, "synced")
        self.assertEqual(result.outcomes[0].session_id, session_id)
        self.assertEqual(len(store.upserts), 1)
        payload = store.upserts[0].to_api_dict()
        self.assertEqual(payload["extra"]["active_energy_kcal"], 510)
        self.assertEqual(payload["extra"]["custom_inferred_tag"], "keep-me")

    def test_sync_skips_already_reconciled_confirmed_session(self) -> None:
        workout = _make_workout()
        session_id = str(generate_session_id(workout.started_at.isoformat(), workout.ended_at.isoformat()))
        store = InMemoryKratosSessionStore(
            initial=[
                {
                    "id": session_id,
                    "started_at": workout.started_at.isoformat(),
                    "ended_at": workout.ended_at.isoformat(),
                    "extra": {
                        "confirmed": True,
                        "inferred": False,
                        "telemetry_reconciled": True,
                        "hr_zones": {"z4_threshold_pct": 35.0},
                        "rpe": 8.5,
                        "active_energy_kcal": 450,
                    },
                }
            ]
        )

        result = KratosReconciler(store=store).sync([workout])
        self.assertEqual(result.outcomes[0].status, "skipped_confirmed")
        self.assertEqual(store.upserts, [])

    def test_sync_reconciles_confirmed_session_and_preserves_metadata_and_extra(self) -> None:
        workout = _make_workout()
        session_id = str(generate_session_id(workout.started_at.isoformat(), workout.ended_at.isoformat()))
        routine_uuid = str(uuid.uuid4())
        item_uuid = str(uuid.uuid4())
        confirmed_start = "2026-11-11T08:55:00+00:00"
        confirmed_end = "2026-11-11T10:02:00+00:00"
        store = InMemoryKratosSessionStore(
            initial=[
                {
                    "id": session_id,
                    "routine_id": routine_uuid,
                    "started_at": confirmed_start,
                    "ended_at": confirmed_end,
                    "notes": "User logged session",
                    "items": [{"id": item_uuid, "sets": []}],
                    "extra": {
                        "confirmed": True,
                        "inferred": False,
                        "classification": "court_sport",
                        "routine_name": "Club Doubles",
                        "effort_rating": "optimal",
                        "custom_coach_note": "Keep footwork light",
                        "venue_code": "COURT-4",
                    },
                }
            ]
        )

        result = KratosReconciler(store=store).sync([workout])

        self.assertEqual(result.outcomes[0].status, "reconciled")
        self.assertEqual(result.outcomes[0].routine_name, "Club Doubles")
        self.assertEqual(len(store.upserts), 1)
        payload = store.upserts[0].to_api_dict()
        self.assertEqual(payload["routine_id"], routine_uuid)
        self.assertEqual(payload["started_at"], confirmed_start)
        self.assertEqual(payload["ended_at"], confirmed_end)
        self.assertEqual(payload["notes"], "User logged session")
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["extra"]["classification"], "court_sport")
        self.assertEqual(payload["extra"]["routine_name"], "Club Doubles")
        self.assertEqual(payload["extra"]["effort_rating"], "optimal")
        self.assertEqual(payload["extra"]["custom_coach_note"], "Keep footwork light")
        self.assertEqual(payload["extra"]["venue_code"], "COURT-4")
        self.assertEqual(payload["extra"]["active_energy_kcal"], 450)
        self.assertEqual(payload["extra"]["avg_hr"], 142)
        self.assertTrue(payload["extra"]["telemetry_reconciled"])

    def test_sync_populates_null_effort_rating(self) -> None:
        workout = _make_workout(
            start_dt=datetime(2026, 11, 20, 11, 0, tzinfo=timezone.utc),
            kcal=420,
            avg_hr_bpm=140,
            peak_hr_bpm=158,
            step_count=4500,
            rpe=None,
            trimp=None,
            hr_zones=None,
            hr_recovery_60s=None,
        )
        session_id = str(generate_session_id(workout.started_at.isoformat(), workout.ended_at.isoformat()))
        store = InMemoryKratosSessionStore(
            initial=[
                {
                    "id": session_id,
                    "started_at": workout.started_at.isoformat(),
                    "ended_at": workout.ended_at.isoformat(),
                    "extra": {"confirmed": True, "inferred": False, "effort_rating": None},
                    "notes": None,
                    "items": [],
                }
            ]
        )

        result = KratosReconciler(store=store).sync([workout])
        self.assertEqual(result.outcomes[0].status, "reconciled")
        self.assertEqual(store.upserts[0].extra["effort_rating"], "hard")

    def test_sync_prefers_confirmed_over_stale_inferred_and_handles_delete_failures(self) -> None:
        workout = _make_workout(
            start_dt=datetime(2026, 11, 20, 16, 55, tzinfo=timezone.utc),
            duration_mins=40,
            kcal=172,
            avg_hr_bpm=115,
            peak_hr_bpm=140,
            step_count=964,
            classification="gym",
            activity_type="gym",
            routine_name="Resistance Training",
            confidence="medium",
        )
        inferred_id = str(generate_session_id(workout.started_at.isoformat(), workout.ended_at.isoformat()))
        failing_dup_id = str(uuid.uuid4())
        confirmed_id = str(uuid.uuid4())
        store = InMemoryKratosSessionStore(
            initial=[
                {
                    "id": inferred_id,
                    "started_at": workout.started_at.isoformat(),
                    "ended_at": workout.ended_at.isoformat(),
                    "extra": {"inferred": True, "classification": "badminton"},
                    "items": [],
                },
                {
                    "id": failing_dup_id,
                    "started_at": workout.started_at.isoformat(),
                    "ended_at": (workout.ended_at - timedelta(minutes=5)).isoformat(),
                    "extra": {"inferred": True},
                    "items": [],
                },
                {
                    "id": confirmed_id,
                    "started_at": "2026-11-20T17:04:00+00:00",
                    "ended_at": "2026-11-20T18:03:00+00:00",
                    "notes": "Lower Body A",
                    "routine_id": str(uuid.uuid4()),
                    "extra": {"confirmed": True, "inferred": False, "routine_name": "Lower Body A"},
                    "items": [{"exercise_id": str(uuid.uuid4())}],
                },
            ],
            fail_delete_ids=[failing_dup_id],
        )

        result = KratosReconciler(store=store).sync([workout])

        outcome = result.outcomes[0]
        self.assertEqual(outcome.status, "reconciled")
        self.assertEqual(outcome.session_id, confirmed_id)
        self.assertEqual(outcome.deleted_duplicate_ids, (inferred_id,))
        self.assertEqual(store.deleted_ids, [inferred_id])
        self.assertEqual(len(store.upserts), 1)
        self.assertEqual(store.upserts[0].id, confirmed_id)

    def test_sync_direct_only_confirmed_fallback(self) -> None:
        workout = _make_workout()
        session_id = str(generate_session_id(workout.started_at.isoformat(), workout.ended_at.isoformat()))
        store = InMemoryKratosSessionStore(
            initial=(),
            direct_only=[
                {
                    "id": session_id,
                    "extra": {"confirmed": True, "inferred": False, "custom_tag": "direct"},
                    "notes": "Direct lookup confirmed",
                    "items": [{"id": str(uuid.uuid4()), "sets": []}],
                }
            ],
        )

        result = KratosReconciler(store=store).sync([workout])
        self.assertEqual(result.outcomes[0].status, "reconciled")
        self.assertEqual(result.outcomes[0].session_id, session_id)
        self.assertEqual(len(store.upserts), 1)
        self.assertEqual(store.upserts[0].extra["custom_tag"], "direct")

    def test_sync_direct_only_inferred_fallback(self) -> None:
        workout = _make_workout(kcal=480)
        session_id = str(generate_session_id(workout.started_at.isoformat(), workout.ended_at.isoformat()))
        store = InMemoryKratosSessionStore(
            initial=(),
            direct_only=[
                {
                    "id": session_id,
                    "started_at": workout.started_at.isoformat(),
                    "ended_at": workout.ended_at.isoformat(),
                    "extra": {"inferred": True, "active_energy_kcal": 200},
                    "items": [],
                }
            ],
        )

        result = KratosReconciler(store=store).sync([workout])
        self.assertEqual(result.outcomes[0].status, "synced")
        self.assertEqual(store.upserts[0].extra["active_energy_kcal"], 480)

    def test_sync_skips_dismissed_and_split_tombstones_under_drift(self) -> None:
        base = datetime(2026, 11, 23, 4, 30, tzinfo=timezone.utc)
        workout = _make_workout(base + timedelta(minutes=5), 95, kcal=650)

        for tombstone_extra in (
            {"dismissed": True, "inferred": True},
            {"split_into_ids": [str(uuid.uuid4()), str(uuid.uuid4())]},
        ):
            store = InMemoryKratosSessionStore(
                initial=[
                    {
                        "id": str(uuid.uuid4()),
                        "started_at": base.isoformat(),
                        "ended_at": (base + timedelta(minutes=95)).isoformat(),
                        "deleted_at": "2026-11-23T06:30:00+00:00",
                        "extra": tombstone_extra,
                        "items": [],
                    }
                ]
            )
            result = KratosReconciler(store=store).sync([workout])
            self.assertEqual(result.outcomes[0].status, "skipped_dismissed")
            self.assertEqual(store.upserts, [])
            self.assertEqual(store.deleted_ids, [])

    def test_sync_active_confirmed_overrides_smaller_tombstone_overlap(self) -> None:
        base = datetime(2026, 11, 23, 4, 30, tzinfo=timezone.utc)
        workout = _make_workout(base, 95, kcal=650)
        confirmed_id = str(uuid.uuid4())
        store = InMemoryKratosSessionStore(
            initial=[
                {
                    "id": str(uuid.uuid4()),
                    "started_at": base.isoformat(),
                    "ended_at": (base + timedelta(minutes=50)).isoformat(),
                    "deleted_at": "2026-11-23T06:30:00+00:00",
                    "extra": {"dismissed": True, "inferred": True},
                    "items": [],
                },
                {
                    "id": confirmed_id,
                    "started_at": (base + timedelta(minutes=15)).isoformat(),
                    "ended_at": (base + timedelta(minutes=95)).isoformat(),
                    "routine_id": str(uuid.uuid4()),
                    "extra": {"confirmed": True, "inferred": False},
                    "items": [],
                },
            ]
        )

        result = KratosReconciler(store=store).sync([workout])
        self.assertEqual(result.outcomes[0].status, "reconciled")
        self.assertEqual(result.outcomes[0].session_id, confirmed_id)
        self.assertEqual(len(store.upserts), 1)

    def test_sync_skips_multi_overlap_split_sessions_and_cleans_inferred(self) -> None:
        base = datetime(2026, 11, 23, 4, 30, tzinfo=timezone.utc)
        workout = _make_workout(base, 95, kcal=650)
        stale_inferred_id = str(uuid.uuid4())
        store = InMemoryKratosSessionStore(
            initial=[
                {
                    "id": str(uuid.uuid4()),
                    "started_at": base.isoformat(),
                    "ended_at": (base + timedelta(minutes=60)).isoformat(),
                    "extra": {"confirmed": True, "inferred": False, "routine_name": "Group Core"},
                    "items": [],
                },
                {
                    "id": str(uuid.uuid4()),
                    "started_at": (base + timedelta(minutes=60)).isoformat(),
                    "ended_at": (base + timedelta(minutes=95)).isoformat(),
                    "routine_id": str(uuid.uuid4()),
                    "extra": {"confirmed": True, "inferred": False, "routine_name": "Lower Body A"},
                    "items": [],
                },
                {
                    "id": stale_inferred_id,
                    "started_at": base.isoformat(),
                    "ended_at": (base + timedelta(minutes=95)).isoformat(),
                    "extra": {"inferred": True},
                    "items": [],
                },
            ]
        )

        result = KratosReconciler(store=store).sync([workout])
        self.assertEqual(result.outcomes[0].status, "skipped_split_overlap")
        self.assertEqual(result.outcomes[0].deleted_duplicate_ids, (stale_inferred_id,))
        self.assertEqual(store.upserts, [])
        self.assertEqual(store.deleted_ids, [stale_inferred_id])

    def test_sync_skips_locked_or_manual_split_sessions(self) -> None:
        base = datetime(2026, 11, 23, 4, 30, tzinfo=timezone.utc)
        workout = _make_workout(base, 95, kcal=650)

        for lock_extra in (
            {"confirmed": True, "inferred": False, "telemetry_locked": True, "active_energy_kcal": 250},
            {"confirmed": True, "inferred": False, "manual_split": True, "active_energy_kcal": 250},
            {"confirmed": True, "inferred": False, "split_from_id": str(uuid.uuid4()), "active_energy_kcal": 250},
        ):
            store = InMemoryKratosSessionStore(
                initial=[
                    {
                        "id": str(uuid.uuid4()),
                        "started_at": base.isoformat(),
                        "ended_at": (base + timedelta(minutes=60)).isoformat(),
                        "extra": lock_extra,
                        "items": [],
                    }
                ]
            )
            result = KratosReconciler(store=store).sync([workout])
            self.assertEqual(result.outcomes[0].status, "skipped_confirmed")
            self.assertEqual(store.upserts, [])

    def test_sync_narrower_window_uses_window_metrics_port_and_db_fallback_guard(self) -> None:
        nz_tz = ZoneInfo("Pacific/Auckland")
        base = datetime(2026, 11, 23, 17, 30, tzinfo=nz_tz)
        lift_start = base + timedelta(minutes=60)
        lift_end = base + timedelta(minutes=95)
        workout = _make_workout(base, 95, kcal=650, classification="gym", activity_type="gym", routine_name="Resistance Training")
        narrow_metrics = _StubWindowMetrics(
            _make_workout(
                lift_start,
                35,
                kcal=210,
                avg_hr_bpm=126,
                peak_hr_bpm=141,
                step_count=700,
                classification="gym",
                activity_type="gym",
                routine_name="Lower Body A",
                rpe=7.4,
            )
        )

        lift_id = str(uuid.uuid4())
        initial_confirmed = {
            "id": lift_id,
            "started_at": lift_start.astimezone(timezone.utc).isoformat(),
            "ended_at": lift_end.astimezone(timezone.utc).isoformat(),
            "routine_id": str(uuid.uuid4()),
            "extra": {"confirmed": True, "inferred": False, "routine_name": "Lower Body A"},
            "items": [],
        }

        # 1. With WindowMetricsPort injected: recomputes 35m window instead of clobbering with 650 kcal
        store = InMemoryKratosSessionStore(initial=[initial_confirmed])
        res1 = KratosReconciler(store=store, window_metrics=narrow_metrics).sync([workout])
        self.assertEqual(res1.outcomes[0].status, "reconciled")
        self.assertEqual(len(narrow_metrics.calls), 1)
        put_extra = store.upserts[0].extra
        self.assertEqual(put_extra["active_energy_kcal"], 210)
        self.assertEqual(put_extra["step_count"], 700)
        self.assertEqual(put_extra["avg_hr"], 126)

        # 2. Re-sync against updated store skips even though 210 < 650
        store.upserts.clear()
        res2 = KratosReconciler(store=store, window_metrics=narrow_metrics).sync([workout])
        self.assertEqual(res2.outcomes[0].status, "skipped_confirmed")
        self.assertEqual(store.upserts, [])

        # 3. Without db or window_metrics for narrower unreconciled session: never clobbers
        store_no_port = InMemoryKratosSessionStore(initial=[initial_confirmed])
        res3 = KratosReconciler(store=store_no_port).sync([workout])
        self.assertEqual(res3.outcomes[0].status, "skipped_confirmed")
        self.assertEqual(store_no_port.upserts, [])

    def test_sync_recomputes_narrower_window_via_real_exertion_analyzer(self) -> None:
        nz_tz = ZoneInfo("Pacific/Auckland")
        base = datetime(2026, 11, 23, 17, 30, tzinfo=nz_tz)
        lift_start = base + timedelta(minutes=60)
        lift_end = base + timedelta(minutes=95)
        workout = _make_workout(base, 95, kcal=650, classification="gym", activity_type="gym", routine_name="Resistance Training")

        for m in range(0, 60, 5):
            t = base + timedelta(minutes=m)
            self.db.add(HealthSample(id=uuid.uuid4(), source="apple_health", sample_type="active_energy", measured_at=t, ended_at=t + timedelta(minutes=5), value=Decimal("36.7"), unit="kcal"))
            self.db.add(HealthSample(id=uuid.uuid4(), source="apple_health", sample_type="heart_rate", measured_at=t, ended_at=t + timedelta(minutes=5), value=Decimal("152"), unit="bpm"))
        for m in range(60, 95, 5):
            t = base + timedelta(minutes=m)
            self.db.add(HealthSample(id=uuid.uuid4(), source="apple_health", sample_type="active_energy", measured_at=t, ended_at=t + timedelta(minutes=5), value=Decimal("30"), unit="kcal"))
            self.db.add(HealthSample(id=uuid.uuid4(), source="apple_health", sample_type="heart_rate", measured_at=t, ended_at=t + timedelta(minutes=5), value=Decimal("126"), unit="bpm"))
            self.db.add(HealthSample(id=uuid.uuid4(), source="apple_health", sample_type="step_count", measured_at=t, ended_at=t + timedelta(minutes=5), value=Decimal("100"), unit="count"))
        self.db.commit()

        lift_id = str(uuid.uuid4())
        store = InMemoryKratosSessionStore(
            initial=[
                {
                    "id": lift_id,
                    "started_at": lift_start.astimezone(timezone.utc).isoformat(),
                    "ended_at": lift_end.astimezone(timezone.utc).isoformat(),
                    "routine_id": str(uuid.uuid4()),
                    "extra": {"confirmed": True, "inferred": False, "routine_name": "Lower Body A"},
                    "items": [],
                }
            ]
        )

        result = KratosReconciler(self.db, store=store).sync([workout])
        self.assertEqual(result.outcomes[0].status, "reconciled")
        put_extra = store.upserts[0].extra
        self.assertEqual(put_extra["active_energy_kcal"], 210)
        self.assertEqual(put_extra["step_count"], 700)
        self.assertEqual(put_extra["avg_hr"], 126)

    def test_sync_updates_working_sessions_within_same_batch(self) -> None:
        t0 = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
        w1 = _make_workout(t0, 45, kcal=320)
        w2 = _make_workout(t0, 45, kcal=390)
        store = InMemoryKratosSessionStore()

        result = KratosReconciler(store=store).sync([w1, w2])
        self.assertEqual(len(result.outcomes), 2)
        self.assertEqual(result.outcomes[0].session_id, result.outcomes[1].session_id)
        self.assertEqual(len(store.upserts), 2)
        self.assertEqual(store.upserts[1].extra["active_energy_kcal"], 390)

    def test_apple_health_ingest_invokes_kratos_reconciler(self) -> None:
        base_dt = datetime(2026, 11, 11, 8, 0, tzinfo=timezone.utc)
        active_energy_dates, active_energy_values = [], []
        step_dates, step_end_dates, step_values = [], [], []
        hr_dates, hr_values = [], []
        for m in range(0, 35, 5):
            t1 = (base_dt + timedelta(minutes=m)).isoformat()
            t2 = (base_dt + timedelta(minutes=m + 5)).isoformat()
            active_energy_dates.append(t1)
            active_energy_values.append(25.0)
            step_dates.append(t1)
            step_end_dates.append(t2)
            step_values.append(350)
            hr_dates.append(t1)
            hr_values.append(105)

        payload = {
            "source": "apple_health",
            "active_energy_dates": active_energy_dates,
            "active_energy_values": active_energy_values,
            "step_dates": step_dates,
            "step_end_dates": step_end_dates,
            "step_values": step_values,
            "hr_dates": hr_dates,
            "hr_values": hr_values,
        }

        with patch.object(KratosReconciler, "sync", autospec=True) as mock_sync:
            mock_sync.return_value = SyncBatchResult(
                configured=True,
                outcomes=(SyncOutcome(session_id="test-id", status="synced", routine_name="Walk / Cardio"),),
            )
            res = self.client.post("/integrations/apple-health/sync", json=payload)
            self.assertEqual(res.status_code, 201)
            data = res.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(len(data["detected_workouts"]), 1)
            self.assertEqual(data["detected_workouts"][0]["routine_name"], "Walk / Cardio")

            mock_sync.assert_called_once()
            passed_workouts = mock_sync.call_args.args[1]
            self.assertEqual(len(passed_workouts), 1)
            self.assertEqual(passed_workouts[0].routine_name, "Walk / Cardio")


class TestHttpKratosSessionStore(unittest.TestCase):
    def test_list_sessions_contracts(self) -> None:
        sid = str(uuid.uuid4())

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/sessions")
            self.assertEqual(request.url.params["limit"], "200")
            self.assertEqual(request.url.params["include_deleted"], "true")
            return httpx.Response(
                200,
                json=[{"id": sid, "extra": {"confirmed": True, "tag": "alpha"}, "items": []}],
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            store = HttpKratosSessionStore("http://kratos-api:8000/", client=client)
            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertIsInstance(sessions[0], KratosSession)
            self.assertIsInstance(sessions[0].extra, MappingProxyType)
            self.assertTrue(sessions[0].is_confirmed)
            self.assertEqual(sessions[0].extra["tag"], "alpha")

        with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404))) as client:
            self.assertEqual(HttpKratosSessionStore("http://kratos-api:8000", client=client).list_sessions(), ())

        for status_code, body in ((500, {"detail": "err"}), (200, {"not": "a-list"})):
            with httpx.Client(transport=httpx.MockTransport(lambda _, sc=status_code, b=body: httpx.Response(sc, json=b))) as client:
                with self.assertRaises(KratosStoreError) as ctx:
                    HttpKratosSessionStore("http://kratos-api:8000", client=client).list_sessions()
                self.assertEqual(ctx.exception.operation, "list_sessions")
                self.assertEqual(ctx.exception.status_code, status_code)

        def fail_transport(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with httpx.Client(transport=httpx.MockTransport(fail_transport)) as client:
            with self.assertRaises(KratosStoreError) as ctx:
                HttpKratosSessionStore("http://kratos-api:8000", client=client).list_sessions()
            self.assertEqual(ctx.exception.operation, "list_sessions")
            self.assertIsNone(ctx.exception.status_code)

    def test_get_upsert_and_delete_contracts(self) -> None:
        sid = str(uuid.uuid4())
        captured_put: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == f"/sessions/{sid}":
                return httpx.Response(200, json={"id": sid, "extra": {"inferred": True}, "items": []})
            if request.method == "GET" and request.url.path == "/sessions/missing":
                return httpx.Response(404)
            if request.method == "GET" and request.url.path == "/sessions/bad-shape":
                return httpx.Response(200, json=[])
            if request.method == "GET" and request.url.path == "/sessions/boom":
                return httpx.Response(500)
            if request.method == "PUT" and request.url.path == f"/sessions/{sid}":
                captured_put["body"] = json.loads(request.content.decode("utf-8"))
                return httpx.Response(200, json={"id": sid})
            if request.method == "PUT" and request.url.path == "/sessions/boom":
                return httpx.Response(500)
            if request.method == "DELETE" and request.url.path in {"/sessions/d200", "/sessions/d204", "/sessions/d404"}:
                return httpx.Response(int(request.url.path[-3:]))
            if request.method == "DELETE" and request.url.path == "/sessions/d500":
                return httpx.Response(500)
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with HttpKratosSessionStore("http://kratos-api:8000", client=client) as store:
                fetched = store.get_session(sid)
                self.assertIsNotNone(fetched)
                self.assertEqual(fetched.id, sid)
                self.assertIsNone(store.get_session("missing"))

                with self.assertRaises(KratosStoreError):
                    store.get_session("bad-shape")
                with self.assertRaises(KratosStoreError):
                    store.get_session("boom")

                write = KratosSessionWrite(
                    id=sid,
                    routine_id=None,
                    started_at="2026-11-11T09:00:00+00:00",
                    ended_at="2026-11-11T10:00:00+00:00",
                    notes="Synced session",
                    extra={"inferred": True, "active_energy_kcal": 400},
                    items=(),
                )
                store.upsert_session(write)
                self.assertEqual(captured_put["body"], write.to_api_dict())

                with self.assertRaises(KratosStoreError):
                    store.upsert_session(KratosSessionWrite(id="boom", notes="fail"))

                for ok_id in ("d200", "d204", "d404"):
                    self.assertTrue(store.delete_session(ok_id))
                with self.assertRaises(KratosStoreError):
                    store.delete_session("d500")

            self.assertFalse(client.is_closed)

        self.assertTrue(client.is_closed)

        with HttpKratosSessionStore("http://kratos-api:8000") as owned_store:
            self.assertFalse(owned_store._client.is_closed)
        self.assertTrue(owned_store._client.is_closed)


class TestKratosSyncArchitectureBoundaries(unittest.TestCase):
    def test_acyclic_boundaries_and_legacy_services_removed(self) -> None:
        api_root = Path(__file__).resolve().parents[1]
        app_root = api_root / "app"
        self.assertFalse((app_root / "services").exists())

        for py_path in sorted((app_root / "kratos_sync").glob("*.py")):
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith("app.daily_biometrics"))
                    self.assertFalse(node.module.startswith("app.routers"))
                    self.assertFalse(node.module.startswith("app.services"))

        for py_path in sorted(app_root.rglob("*.py")):
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith("app.services"))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith("app.services"))

        test_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        for node in ast.walk(test_tree):
            if isinstance(node, ast.Name):
                self.assertNotEqual(node.id, "MagicMock")
            elif isinstance(node, ast.Attribute):
                self.assertNotEqual(node.attr, "MagicMock")


if __name__ == "__main__":
    unittest.main()
