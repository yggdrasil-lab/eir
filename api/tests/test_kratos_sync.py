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
        )

        mock_http = MagicMock(spec=httpx.Client)
        # GET /sessions/{id} returns 200 with confirmed session
        mock_check = MagicMock()
        mock_check.status_code = 200
        mock_check.json.return_value = {
            "id": str(generate_session_id(w.started_at.isoformat(), w.ended_at.isoformat())),
            "extra": {"confirmed": True, "inferred": False},
        }
        mock_http.get.return_value = mock_check

        results = sync_detected_workouts_to_kratos([w], kratos_url="http://kratos-api:8000", client=mock_http)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "skipped_confirmed")
        # PUT should never be called when session is already confirmed
        self.assertEqual(mock_http.put.call_count, 0)

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

            # Verify sync_detected_workouts_to_kratos was invoked
            mock_sync.assert_called_once()
            passed_workouts = mock_sync.call_args[0][0]
            self.assertEqual(len(passed_workouts), 1)
            self.assertEqual(passed_workouts[0].routine_name, "Walk / Cardio")


if __name__ == "__main__":
    unittest.main()
