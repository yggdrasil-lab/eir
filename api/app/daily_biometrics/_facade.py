"""Cohesive DailyBiometrics facade for reading, summarizing, and rolling up daily vitals."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING
import uuid
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.config import settings
from app.daily_biometrics._aggregation import build_daily_summary, upsert_daily_rollup
from app.daily_biometrics._queries import fetch_granular_samples, fetch_latest_measurement
from app.daily_biometrics._registry import METRIC_REGISTRY
from app.daily_biometrics._types import BiometricsInputError, DailyRollupResult, ExertionPort, RollupBatch
from app.daily_biometrics._window import day_bounds, resolve_local_days, validate_target_date
from app.models import BiometricMeasurement, HealthSample

if TYPE_CHECKING:
    from app.schemas import BiometricCreate, DailyBiometricSummary


class DailyBiometrics:
    """Deep module encapsulating daily biometric summaries, rollups, and sample reads."""

    def __init__(
        self,
        db: Session,
        *,
        tz: ZoneInfo | None = None,
        exertion: ExertionPort | None = None,
    ) -> None:
        self._db = db
        self._tz = tz if tz is not None else settings.tz
        if exertion is not None:
            self._exertion: ExertionPort = exertion
        else:
            from app.exertion import ExertionAnalyzer, ExertionProfile

            self._exertion = ExertionAnalyzer(db, ExertionProfile(tz=self._tz, max_hr=settings.user_max_hr))

    def daily_summary(self, target_date: date) -> DailyBiometricSummary:
        """Returns the full aggregated vitals summary for a single local calendar day."""
        clean_date = validate_target_date(target_date)
        return build_daily_summary(self._db, clean_date, self._tz, self._exertion)

    def materialize_rollups(self, targets: Sequence[date | datetime]) -> RollupBatch:
        """Materializes daily rollups across unique local days, invoking exertion 1x per day."""
        days = resolve_local_days(targets, self._tz)
        results: list[DailyRollupResult] = []
        for day in days:
            results.append(upsert_daily_rollup(self._db, day, self._tz, self._exertion))
        all_detected = tuple(w for res in results for w in res.detected_workouts)
        return RollupBatch(days=days, results=tuple(results), detected_workouts=all_detected)

    def granular_samples(
        self,
        target_date: date,
        sample_type: str,
        *,
        limit: int = 2000,
    ) -> list[HealthSample]:
        """Returns ordered granular HealthSample rows for a day, expanding clean aliases."""
        clean_date = validate_target_date(target_date)
        if not isinstance(sample_type, str) or not sample_type.strip():
            raise BiometricsInputError("sample_type must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 5000):
            raise BiometricsInputError("limit must be an integer between 1 and 5000")

        start_dt, end_dt = day_bounds(clean_date, self._tz)
        aliases = METRIC_REGISTRY.aliases_for(sample_type.strip())
        return fetch_granular_samples(self._db, aliases, start_dt, end_dt, limit=limit)

    def latest_measurement(self, *, source: str | None = None) -> BiometricMeasurement | None:
        """Returns the most recent active BiometricMeasurement, optionally filtered by source."""
        clean_source = source.strip() if isinstance(source, str) and source.strip() else None
        return fetch_latest_measurement(self._db, source=clean_source)

    def record_measurement(self, payload: BiometricCreate) -> BiometricMeasurement:
        """Creates and flushes a BiometricMeasurement row without committing."""
        measurement = BiometricMeasurement(
            id=uuid.uuid4(),
            measured_at=payload.measured_at,
            source=payload.source,
            weight_kg=payload.weight_kg,
            body_fat_pct=payload.body_fat_pct,
            muscle_mass_kg=payload.muscle_mass_kg,
            water_pct=payload.water_pct,
            bone_mass_kg=payload.bone_mass_kg,
            resting_hr_bpm=payload.resting_hr_bpm,
            hrv_ms=payload.hrv_ms,
            sleep_duration_s=payload.sleep_duration_s,
            sleep_deep_s=payload.sleep_deep_s,
            sleep_rem_s=payload.sleep_rem_s,
            sleep_score=payload.sleep_score,
            step_count=payload.step_count,
            active_energy_kcal=payload.active_energy_kcal,
            notes=payload.notes,
            extra=payload.extra,
        )
        self._db.add(measurement)
        self._db.flush()
        return measurement
