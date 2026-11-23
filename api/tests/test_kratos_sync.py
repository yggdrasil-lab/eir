"""Unit tests for Kratos workout synchronization bridge and Apple Health ingest hook."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import unittest
from unittest.mock import MagicMock, patch
import uuid

from fastapi.testclient import TestClient
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app
from app.models import HealthSample
from app.schemas import DetectedWorkoutRead
from app.services.kratos_sync import generate_session_id, sync_detected_workouts_to_kratos


class TestKratosSync(unittest.TestCase):
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

    def test_generate_session_id_deterministic(self):
        s_iso = "2026-11-11T09:00:00+13:00"
        e_iso = "2026-11-11T10:00:00+13:00"
        id1 = generate_session_id(s_iso, e_iso)
        id2 = generate_session_id(s_iso, e_iso)
        self.assertEqual(id1, id2)
        self.assertIsInstance(id1, uuid.UUID)

        # Different timestamp produces different UUID
        id3 = generate_session_id(s_iso, "2026-11-11T10:30:00+13:00")
        self.assertNotEqual(id1, id3)

    def test_sync_detected_workouts_empty(self):
        res = sync_detected_workouts_to_kratos([])
        self.assertEqual(res, [])

    def test_sync_detected_workouts_success(self):
        w = DetectedWorkoutRead(
            started_at=datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc),
            duration_minutes=60,
            active_energy_kcal=450,
            avg_hr_bpm=142,
            peak_hr_bpm=155,
            step_count=5200,
            confidence="high",
            classification="badminton",
            activity_type="badminton",
            routine_name="Badminton Match Play",
            effort_rating="hard",
            rpe=8.5,
            trimp=220.0,
            hr_zones={"z4_threshold_pct": 35.0},
            hr_recovery_60s=25,
        )

        mock_http = MagicMock(spec=httpx.Client)
        # GET /sessions/{id} returns 404 (new session)
        mock_check = MagicMock()
        mock_check.status_code = 404
        mock_http.get.return_value = mock_check

        # PUT /sessions/{id} returns 200
        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_http.put.return_value = mock_put

        results = sync_detected_workouts_to_kratos([w], kratos_url="http://kratos-api:8000", client=mock_http)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "synced")
        self.assertEqual(results[0]["routine_name"], "Badminton Match Play")

        # Verify PUT body has required structure
        self.assertEqual(mock_http.put.call_count, 1)
        put_args, put_kwargs = mock_http.put.call_args
        payload = put_kwargs["json"]
        self.assertEqual(payload["id"], results[0]["session_id"])
        self.assertEqual(payload["routine_id"], None)
        self.assertEqual(payload["items"], [])
        self.assertTrue(payload["extra"]["inferred"])
        self.assertEqual(payload["extra"]["classification"], "badminton")
        self.assertEqual(payload["extra"]["active_energy_kcal"], 450)
        self.assertEqual(payload["extra"]["effort_rating"], "hard")
        self.assertEqual(payload["extra"]["rpe"], 8.5)
        self.assertEqual(payload["extra"]["trimp"], 220.0)
        self.assertEqual(payload["extra"]["hr_recovery_60s"], 25)

    def test_sync_detected_workouts_skips_confirmed_session(self):
        w = DetectedWorkoutRead(
            started_at=datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc),
            duration_minutes=60,
            active_energy_kcal=450,
            avg_hr_bpm=142,
            peak_hr_bpm=155,
            step_count=5200,
            confidence="high",
            classification="badminton",
            activity_type="badminton",
            routine_name="Badminton Match Play",
            effort_rating="hard",
            rpe=8.5,
            trimp=220.0,
            hr_zones={"z4_threshold_pct": 35.0},
            hr_recovery_60s=25,
        )

        mock_http = MagicMock(spec=httpx.Client)
        # GET /sessions/{id} returns 200 with confirmed and already reconciled session
        mock_check = MagicMock()
        mock_check.status_code = 200
        mock_check.json.return_value = {
            "id": str(generate_session_id(w.started_at.isoformat(), w.ended_at.isoformat())),
            "extra": {
                "confirmed": True,
                "inferred": False,
                "telemetry_reconciled": True,
                "hr_zones": {"z4_threshold_pct": 35.0},
                "rpe": 8.5,
                "active_energy_kcal": 450,
            },
        }
        mock_http.get.return_value = mock_check

        results = sync_detected_workouts_to_kratos([w], kratos_url="http://kratos-api:8000", client=mock_http)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "skipped_confirmed")
        # PUT should never be called when session is already confirmed and reconciled
        self.assertEqual(mock_http.put.call_count, 0)

    def test_sync_detected_workouts_reconciles_confirmed_session(self):
        w = DetectedWorkoutRead(
            started_at=datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc),
            duration_minutes=60,
            active_energy_kcal=450,
            avg_hr_bpm=142,
            peak_hr_bpm=155,
            step_count=5200,
            confidence="high",
            classification="badminton",
            activity_type="badminton",
            routine_name="Badminton Match Play",
            effort_rating="hard",
            rpe=8.5,
            trimp=220.0,
            hr_zones={"z4_threshold_pct": 35.0},
            hr_recovery_60s=25,
        )

        mock_http = MagicMock(spec=httpx.Client)
        mock_check = MagicMock()
        mock_check.status_code = 200
        mock_check.json.return_value = {
            "id": str(generate_session_id(w.started_at.isoformat(), w.ended_at.isoformat())),
            "extra": {"confirmed": True, "inferred": False},
            "notes": "User logged session",
            "items": [{"id": str(uuid.uuid4()), "sets": []}],
        }
        mock_http.get.return_value = mock_check

        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_http.put.return_value = mock_put

        results = sync_detected_workouts_to_kratos([w], kratos_url="http://kratos-api:8000", client=mock_http)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "reconciled")
        self.assertEqual(mock_http.put.call_count, 1)
        put_kwargs = mock_http.put.call_args[1]["json"]
        self.assertEqual(put_kwargs["extra"]["active_energy_kcal"], 450)
        self.assertEqual(put_kwargs["extra"]["avg_hr"], 142)
        self.assertEqual(put_kwargs["extra"]["effort_rating"], "hard")
        self.assertEqual(put_kwargs["extra"]["rpe"], 8.5)
        self.assertEqual(put_kwargs["extra"]["trimp"], 220.0)
        self.assertEqual(put_kwargs["extra"]["hr_recovery_60s"], 25)
        self.assertTrue(put_kwargs["extra"]["telemetry_reconciled"])
        self.assertEqual(len(put_kwargs["items"]), 1)

    def test_sync_detected_workouts_preserves_existing_effort_rating(self):
        w = DetectedWorkoutRead(
            started_at=datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc),
            duration_minutes=60,
            active_energy_kcal=310,
            avg_hr_bpm=118,
            peak_hr_bpm=134,
            step_count=1800,
            confidence="high",
            classification="gym",
            activity_type="gym",
            routine_name="Upper Body A",
            effort_rating="too_easy",
            rpe=6.0,
            trimp=95.0,
        )

        session_id = str(generate_session_id(w.started_at.isoformat(), w.ended_at.isoformat()))
        mock_http = MagicMock(spec=httpx.Client)
        mock_list = MagicMock()
        mock_list.status_code = 200
        mock_list.json.return_value = [
            {
                "id": session_id,
                "started_at": w.started_at.isoformat(),
                "ended_at": w.ended_at.isoformat(),
                "routine_id": str(uuid.uuid4()),
                "extra": {"confirmed": True, "inferred": False, "effort_rating": "optimal"},
                "notes": "Manual PWA debrief rating",
                "items": [{"id": str(uuid.uuid4()), "sets": []}],
            }
        ]
        mock_http.get.return_value = mock_list

        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_http.put.return_value = mock_put

        results = sync_detected_workouts_to_kratos([w], kratos_url="http://kratos-api:8000", client=mock_http)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "reconciled")
        self.assertEqual(mock_http.put.call_count, 1)
        put_kwargs = mock_http.put.call_args[1]["json"]
        self.assertEqual(put_kwargs["extra"]["effort_rating"], "optimal")
        self.assertEqual(put_kwargs["extra"]["active_energy_kcal"], 310)
        self.assertTrue(put_kwargs["extra"]["telemetry_reconciled"])

    def test_sync_detected_workouts_populates_null_effort_rating(self):
        w = DetectedWorkoutRead(
            started_at=datetime(2026, 11, 20, 11, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc),
            duration_minutes=60,
            active_energy_kcal=420,
            avg_hr_bpm=140,
            peak_hr_bpm=158,
            step_count=4500,
            confidence="high",
            classification="badminton",
            activity_type="badminton",
            routine_name="Badminton Match Play",
            effort_rating="hard",
        )

        session_id = str(generate_session_id(w.started_at.isoformat(), w.ended_at.isoformat()))
        mock_http = MagicMock(spec=httpx.Client)
        mock_list = MagicMock()
        mock_list.status_code = 200
        mock_list.json.return_value = [
            {
                "id": session_id,
                "started_at": w.started_at.isoformat(),
                "ended_at": w.ended_at.isoformat(),
                "routine_id": None,
                "extra": {"confirmed": True, "inferred": False, "effort_rating": None},
                "notes": None,
                "items": [],
            }
        ]
        mock_http.get.return_value = mock_list

        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_http.put.return_value = mock_put

        results = sync_detected_workouts_to_kratos([w], kratos_url="http://kratos-api:8000", client=mock_http)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "reconciled")
        self.assertEqual(mock_http.put.call_count, 1)
        put_kwargs = mock_http.put.call_args[1]["json"]
        self.assertEqual(put_kwargs["extra"]["effort_rating"], "hard")

    def test_sync_detected_workouts_overlap_guard_deduplication(self):
        w = DetectedWorkoutRead(
            started_at=datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc),
            duration_minutes=60,
            active_energy_kcal=500,
            avg_hr_bpm=135,
            peak_hr_bpm=150,
            step_count=2000,
            confidence="high",
            classification="gym",
            activity_type="gym",
            routine_name="Resistance Training",
        )

        existing_session_id = str(uuid.uuid4())
        mock_http = MagicMock(spec=httpx.Client)

        def mock_get(url, **kwargs):
            resp = MagicMock()
            if url.endswith("/sessions"):
                resp.status_code = 200
                resp.json.return_value = [{
                    "id": existing_session_id,
                    "started_at": "2026-11-11T08:50:00+00:00",
                    "ended_at": "2026-11-11T10:05:00+00:00",
                    "notes": "Chest day",
                    "routine_id": str(uuid.uuid4()),
                    "extra": {"confirmed": True, "inferred": False},
                    "items": [{"exercise_id": str(uuid.uuid4())}],
                }]
            else:
                resp.status_code = 404
            return resp

        mock_http.get.side_effect = mock_get
        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_http.put.return_value = mock_put

        results = sync_detected_workouts_to_kratos([w], kratos_url="http://kratos-api:8000", client=mock_http)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], existing_session_id)
        self.assertEqual(results[0]["status"], "reconciled")
        # Verified PUT target was the existing overlapping session, preventing duplicate suggestion
        self.assertEqual(mock_http.put.call_count, 1)
        put_url = mock_http.put.call_args[0][0]
        self.assertTrue(put_url.endswith(f"/sessions/{existing_session_id}"))
        put_body = mock_http.put.call_args[1]["json"]
        self.assertEqual(put_body["extra"]["active_energy_kcal"], 500)
        self.assertEqual(put_body["extra"]["step_count"], 2000)
        self.assertTrue(put_body["extra"]["telemetry_reconciled"])

    def test_sync_detected_workouts_prefers_confirmed_over_stale_inferred_duplicate(self):
        w = DetectedWorkoutRead(
            started_at=datetime(2026, 11, 20, 16, 55, tzinfo=timezone.utc),
            ended_at=datetime(2026, 11, 20, 17, 35, tzinfo=timezone.utc),
            duration_minutes=40,
            active_energy_kcal=172,
            avg_hr_bpm=115,
            peak_hr_bpm=140,
            step_count=964,
            confidence="medium",
            classification="gym",
            activity_type="gym",
            routine_name="Resistance Training",
        )

        inferred_id = str(generate_session_id(w.started_at.isoformat(), w.ended_at.isoformat()))
        confirmed_id = str(uuid.uuid4())
        mock_http = MagicMock(spec=httpx.Client)

        def mock_get(url, **kwargs):
            resp = MagicMock()
            if url.endswith("/sessions"):
                resp.status_code = 200
                resp.json.return_value = [
                    {
                        "id": inferred_id,
                        "started_at": w.started_at.isoformat(),
                        "ended_at": w.ended_at.isoformat(),
                        "routine_id": None,
                        "extra": {
                            "inferred": True,
                            "classification": "badminton",
                            "routine_name": "Badminton Match Play",
                        },
                        "items": [],
                    },
                    {
                        "id": confirmed_id,
                        "started_at": "2026-11-20T17:04:00+00:00",
                        "ended_at": "2026-11-20T18:03:00+00:00",
                        "notes": "Lower Body A",
                        "routine_id": str(uuid.uuid4()),
                        "extra": {
                            "confirmed": True,
                            "inferred": False,
                            "routine_name": "Lower Body A",
                        },
                        "items": [{"exercise_id": str(uuid.uuid4())}],
                    },
                ]
            else:
                resp.status_code = 404
            return resp

        mock_http.get.side_effect = mock_get
        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_http.put.return_value = mock_put
        mock_del = MagicMock()
        mock_del.status_code = 204
        mock_http.delete.return_value = mock_del

        results = sync_detected_workouts_to_kratos([w], kratos_url="http://kratos-api:8000", client=mock_http)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "reconciled")
        self.assertEqual(results[0]["session_id"], confirmed_id)

        self.assertEqual(mock_http.put.call_count, 1)
        put_url = mock_http.put.call_args[0][0]
        self.assertTrue(put_url.endswith(f"/sessions/{confirmed_id}"))
        put_body = mock_http.put.call_args[1]["json"]
        self.assertEqual(put_body["extra"]["active_energy_kcal"], 172)
        self.assertEqual(put_body["extra"]["step_count"], 964)
        self.assertTrue(put_body["extra"]["telemetry_reconciled"])

        self.assertEqual(mock_http.delete.call_count, 1)
        del_url = mock_http.delete.call_args[0][0]
        self.assertTrue(del_url.endswith(f"/sessions/{inferred_id}"))

    def test_apple_health_ingest_runs_detection_and_sync(self):
        target_date = date(2026, 11, 11)
        base_dt = datetime(2026, 11, 11, 8, 0, tzinfo=timezone.utc)

        # Construct 35 minutes of walk data
        active_energy_dates = []
        active_energy_values = []
        step_dates = []
        step_end_dates = []
        step_values = []
        hr_dates = []
        hr_values = []

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

        with patch("app.routers.integrations.sync_detected_workouts_to_kratos") as mock_sync:
            mock_sync.return_value = [{"session_id": "test-id", "status": "synced"}]
            res = self.client.post("/integrations/apple-health/sync", json=payload)
            self.assertEqual(res.status_code, 201)
            data = res.json()
            self.assertEqual(data["status"], "ok")
            self.assertIn("detected_workouts", data)
            self.assertEqual(len(data["detected_workouts"]), 1)
            detected = data["detected_workouts"][0]
            self.assertEqual(detected["classification"], "walk")
            self.assertEqual(detected["routine_name"], "Walk / Cardio")
            self.assertEqual(detected["duration_minutes"], 35)

            # Verify sync_detected_workouts_to_kratos was invoked with db session
            mock_sync.assert_called_once()
            passed_workouts = mock_sync.call_args[0][0]
            self.assertEqual(len(passed_workouts), 1)
            self.assertEqual(passed_workouts[0].routine_name, "Walk / Cardio")
            self.assertIsNotNone(mock_sync.call_args.kwargs.get("db"))

    def _make_workout(
        self,
        start_dt: datetime,
        duration_mins: int,
        kcal: int = 650,
    ) -> DetectedWorkoutRead:
        return DetectedWorkoutRead(
            started_at=start_dt,
            ended_at=start_dt + timedelta(minutes=duration_mins),
            duration_minutes=duration_mins,
            active_energy_kcal=kcal,
            avg_hr_bpm=142,
            peak_hr_bpm=165,
            step_count=3200,
            confidence="high",
            classification="gym",
            activity_type="gym",
            routine_name="Resistance Training",
            effort_rating="hard",
            rpe=8.2,
            trimp=240.0,
            hr_zones={"z3_tempo_pct": 40.0, "z4_threshold_pct": 35.0},
            hr_recovery_60s=22,
        )

    def test_sync_skips_dismissed_and_split_tombstones_under_drift(self):
        base = datetime(2026, 11, 23, 4, 30, tzinfo=timezone.utc)
        w = self._make_workout(base + timedelta(minutes=5), 95, kcal=650)

        for tombstone_extra in (
            {"dismissed": True, "inferred": True},
            {"split_into_ids": [str(uuid.uuid4()), str(uuid.uuid4())]},
        ):
            mock_http = MagicMock(spec=httpx.Client)
            mock_list = MagicMock()
            mock_list.status_code = 200
            mock_list.json.return_value = [
                {
                    "id": str(uuid.uuid4()),
                    "started_at": base.isoformat(),
                    "ended_at": (base + timedelta(minutes=95)).isoformat(),
                    "deleted_at": "2026-11-23T06:30:00+00:00",
                    "extra": tombstone_extra,
                    "items": [],
                }
            ]
            mock_http.get.return_value = mock_list

            results = sync_detected_workouts_to_kratos(
                [w], kratos_url="http://kratos-api:8000", client=mock_http, db=self.db
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "skipped_dismissed")
            self.assertEqual(mock_http.put.call_count, 0)
            self.assertEqual(mock_http.delete.call_count, 0)
            self.assertEqual(
                mock_http.get.call_args[1]["params"],
                {"limit": 200, "include_deleted": "true"},
            )

    def test_sync_active_confirmed_overrides_smaller_tombstone_overlap(self):
        base = datetime(2026, 11, 23, 4, 30, tzinfo=timezone.utc)
        w = self._make_workout(base, 95, kcal=650)
        confirmed_id = str(uuid.uuid4())

        mock_http = MagicMock(spec=httpx.Client)
        mock_list = MagicMock()
        mock_list.status_code = 200
        mock_list.json.return_value = [
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
                "deleted_at": None,
                "routine_id": str(uuid.uuid4()),
                "extra": {"confirmed": True, "inferred": False},
                "items": [],
            },
        ]
        mock_http.get.return_value = mock_list
        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_http.put.return_value = mock_put

        results = sync_detected_workouts_to_kratos(
            [w], kratos_url="http://kratos-api:8000", client=mock_http, db=self.db
        )
        self.assertEqual(results[0]["status"], "reconciled")
        self.assertEqual(results[0]["session_id"], confirmed_id)
        self.assertEqual(mock_http.put.call_count, 1)

    def test_sync_skips_multi_overlap_split_sessions_and_cleans_inferred(self):
        base = datetime(2026, 11, 23, 4, 30, tzinfo=timezone.utc)
        w = self._make_workout(base, 95, kcal=650)
        stale_inferred_id = str(uuid.uuid4())

        mock_http = MagicMock(spec=httpx.Client)
        mock_list = MagicMock()
        mock_list.status_code = 200
        mock_list.json.return_value = [
            {
                "id": str(uuid.uuid4()),
                "started_at": base.isoformat(),
                "ended_at": (base + timedelta(minutes=60)).isoformat(),
                "deleted_at": None,
                "extra": {"confirmed": True, "inferred": False, "routine_name": "Group Core"},
                "items": [],
            },
            {
                "id": str(uuid.uuid4()),
                "started_at": (base + timedelta(minutes=60)).isoformat(),
                "ended_at": (base + timedelta(minutes=95)).isoformat(),
                "deleted_at": None,
                "routine_id": str(uuid.uuid4()),
                "extra": {"confirmed": True, "inferred": False, "routine_name": "Lower Body A"},
                "items": [],
            },
            {
                "id": stale_inferred_id,
                "started_at": base.isoformat(),
                "ended_at": (base + timedelta(minutes=95)).isoformat(),
                "deleted_at": None,
                "extra": {"inferred": True},
                "items": [],
            },
        ]
        mock_http.get.return_value = mock_list
        mock_del = MagicMock()
        mock_del.status_code = 204
        mock_http.delete.return_value = mock_del

        results = sync_detected_workouts_to_kratos(
            [w], kratos_url="http://kratos-api:8000", client=mock_http, db=self.db
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "skipped_split_overlap")
        self.assertEqual(mock_http.put.call_count, 0)
        self.assertEqual(mock_http.delete.call_count, 1)
        self.assertTrue(mock_http.delete.call_args[0][0].endswith(f"/sessions/{stale_inferred_id}"))

    def test_sync_skips_locked_or_manual_split_sessions(self):
        base = datetime(2026, 11, 23, 4, 30, tzinfo=timezone.utc)
        w = self._make_workout(base, 95, kcal=650)

        for lock_extra in (
            {"confirmed": True, "inferred": False, "telemetry_locked": True, "active_energy_kcal": 250},
            {"confirmed": True, "inferred": False, "manual_split": True, "active_energy_kcal": 250},
            {"confirmed": True, "inferred": False, "split_from_id": str(uuid.uuid4()), "active_energy_kcal": 250},
        ):
            mock_http = MagicMock(spec=httpx.Client)
            mock_list = MagicMock()
            mock_list.status_code = 200
            mock_list.json.return_value = [
                {
                    "id": str(uuid.uuid4()),
                    "started_at": base.isoformat(),
                    "ended_at": (base + timedelta(minutes=60)).isoformat(),
                    "deleted_at": None,
                    "extra": lock_extra,
                    "items": [],
                }
            ]
            mock_http.get.return_value = mock_list

            results = sync_detected_workouts_to_kratos(
                [w], kratos_url="http://kratos-api:8000", client=mock_http, db=self.db
            )
            self.assertEqual(results[0]["status"], "skipped_confirmed")
            self.assertEqual(mock_http.put.call_count, 0)

    def test_sync_recomputes_narrower_window_instead_of_clobbering_with_merged_block(self):
        from zoneinfo import ZoneInfo

        nz_tz = ZoneInfo("Pacific/Auckland")
        base = datetime(2026, 11, 23, 17, 30, tzinfo=nz_tz)
        lift_start = base + timedelta(minutes=60)
        lift_end = base + timedelta(minutes=95)
        w = self._make_workout(base, 95, kcal=650)

        # Seed samples in settings.tz (SQLite strips tzinfo on write):
        # first 60m class (~440 kcal, high HR), last 35m lift (210 kcal, avg HR 126, 700 steps)
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
        mock_http = MagicMock(spec=httpx.Client)
        mock_list = MagicMock()
        mock_list.status_code = 200
        mock_list.json.return_value = [
            {
                "id": lift_id,
                "started_at": lift_start.astimezone(timezone.utc).isoformat(),
                "ended_at": lift_end.astimezone(timezone.utc).isoformat(),
                "deleted_at": None,
                "routine_id": str(uuid.uuid4()),
                "extra": {"confirmed": True, "inferred": False, "routine_name": "Lower Body A"},
                "items": [],
            }
        ]
        mock_http.get.return_value = mock_list
        mock_put = MagicMock()
        mock_put.status_code = 200
        mock_http.put.return_value = mock_put

        # 1. With db provided: recomputes exact 35m window metrics (210 kcal, not 650)
        results = sync_detected_workouts_to_kratos(
            [w], kratos_url="http://kratos-api:8000", client=mock_http, db=self.db
        )
        self.assertEqual(results[0]["status"], "reconciled")
        self.assertEqual(mock_http.put.call_count, 1)
        put_extra = mock_http.put.call_args[1]["json"]["extra"]
        self.assertEqual(put_extra["active_energy_kcal"], 210)
        self.assertEqual(put_extra["step_count"], 700)
        self.assertEqual(put_extra["avg_hr"], 126)
        self.assertTrue(put_extra["telemetry_reconciled"])

        # 2. Next sync once reconciled with < 0.80 * w_duration: skips even though 210 < 650
        mock_http.reset_mock()
        mock_list.json.return_value = [
            {
                "id": lift_id,
                "started_at": lift_start.isoformat(),
                "ended_at": lift_end.isoformat(),
                "deleted_at": None,
                "routine_id": str(uuid.uuid4()),
                "extra": put_extra,
                "items": [],
            }
        ]
        mock_http.get.return_value = mock_list
        results_2 = sync_detected_workouts_to_kratos(
            [w], kratos_url="http://kratos-api:8000", client=mock_http, db=self.db
        )
        self.assertEqual(results_2[0]["status"], "skipped_confirmed")
        self.assertEqual(mock_http.put.call_count, 0)

        # 3. Without db provided for narrower unlocked unreconciled session: never clobbers with whole block
        mock_http.reset_mock()
        mock_list.json.return_value = [
            {
                "id": lift_id,
                "started_at": lift_start.isoformat(),
                "ended_at": lift_end.isoformat(),
                "deleted_at": None,
                "routine_id": str(uuid.uuid4()),
                "extra": {"confirmed": True, "inferred": False, "routine_name": "Lower Body A"},
                "items": [],
            }
        ]
        mock_http.get.return_value = mock_list
        results_3 = sync_detected_workouts_to_kratos(
            [w], kratos_url="http://kratos-api:8000", client=mock_http, db=None
        )
        self.assertEqual(results_3[0]["status"], "skipped_confirmed")
        self.assertEqual(mock_http.put.call_count, 0)


if __name__ == "__main__":
    unittest.main()
