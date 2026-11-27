"""Pydantic schemas for Eir API request and response validation."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.daily_biometrics import METRIC_REGISTRY
from app.ingestion import clean_datetime, clean_numeric


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HealthRead(BaseModel):
    status: str
    version: str
    database: str


# ------------------------------------------------------------- Integrations
class WithingsMeasurementPayload(BaseModel):
    """Direct body scale measurement from Withings Body Smart scale."""

    measured_at: datetime
    weight_kg: Decimal
    body_fat_pct: Decimal | None = None
    muscle_mass_kg: Decimal | None = None
    water_pct: Decimal | None = None
    bone_mass_kg: Decimal | None = None
    resting_hr_bpm: int | None = None
    notes: str | None = None
    extra: dict = Field(default_factory=dict)


class WithingsAuthStatusResponse(BaseModel):
    """Status of Withings OAuth2 connection."""

    status: str
    linked: bool
    provider: str = "withings"
    user_id: str | None = None
    expires_at: datetime | None = None
    scope: str | None = None


# ----------------------------------------------------------------- Biometrics
class BiometricCreate(BaseModel):
    """Manual biometric entry."""

    measured_at: datetime = Field(default_factory=datetime.now)
    source: str = "manual"
    weight_kg: Decimal | None = None
    body_fat_pct: Decimal | None = None
    muscle_mass_kg: Decimal | None = None
    water_pct: Decimal | None = None
    bone_mass_kg: Decimal | None = None
    resting_hr_bpm: int | None = None
    hrv_ms: Decimal | None = None
    sleep_duration_s: int | None = None
    sleep_deep_s: int | None = None
    sleep_rem_s: int | None = None
    sleep_score: int | None = None
    step_count: int | None = None
    active_energy_kcal: int | None = None
    notes: str | None = None
    extra: dict = Field(default_factory=dict)


class BiometricRead(ORMModel):
    id: uuid.UUID
    measured_at: datetime
    source: str
    weight_kg: Decimal | None
    body_fat_pct: Decimal | None
    muscle_mass_kg: Decimal | None
    water_pct: Decimal | None
    bone_mass_kg: Decimal | None
    resting_hr_bpm: int | None
    hrv_ms: Decimal | None
    sleep_duration_s: int | None
    sleep_deep_s: int | None
    sleep_rem_s: int | None
    sleep_score: int | None
    step_count: int | None
    active_energy_kcal: int | None
    notes: str | None
    extra: dict
    created_at: datetime


# ------------------------------------------------------------- Active Health Feedback
class HealthFeedbackBase(BaseModel):
    date: date
    feedback_type: str  # 'fasting', 'workout', 'wellness', 'sport'
    rating: str | None = None  # 'completed', 'optimal', 'too_hard', etc.
    score: Decimal | None = None  # 1.0 - 10.0 scale
    tags: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    logged_at: datetime = Field(default_factory=datetime.now)


class HealthFeedbackCreate(HealthFeedbackBase):
    pass


class HealthFeedbackRead(ORMModel, HealthFeedbackBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class FastingFeedbackCreate(BaseModel):
    date: date
    target_window: str = "16:8"
    adherence: str  # 'completed', 'early', 'off_plan'
    energy: str = "normal"  # 'high', 'normal', 'low'
    headache: str = "none"  # 'none', 'mild', 'sugar_spike'
    notes: str | None = None
    logged_at: datetime = Field(default_factory=datetime.now)


class WorkoutFeedbackCreate(BaseModel):
    date: date
    routine_name: str | None = None
    duration_minutes: int | None = None
    total_volume_kg: Decimal | None = None
    completed_sets: int | None = None
    effort_rating: str  # 'too_easy', 'optimal', 'hard', 'too_hard'
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    logged_at: datetime = Field(default_factory=datetime.now)


class WellnessFeedbackCreate(BaseModel):
    date: date
    readiness_rating: str | None = None  # 'high', 'moderate', 'low'
    soreness_score: Decimal | None = None  # 1.0 - 5.0
    sore_areas: list[str] = Field(default_factory=list)  # e.g. ['Legs', 'Back']
    stress_level: str | None = None  # 'low', 'normal', 'high'
    notes: str | None = None
    logged_at: datetime = Field(default_factory=datetime.now)


class DetectedWorkoutRead(BaseModel):
    """Sustained physical exertion block inferred from heart rate and active energy telemetry."""

    started_at: datetime
    ended_at: datetime
    duration_minutes: int
    active_energy_kcal: int
    avg_hr_bpm: int | None = None
    peak_hr_bpm: int | None = None
    step_count: int | None = None
    confidence: str = "medium"  # 'high', 'medium'
    activity_type: str | None = None
    classification: str | None = None
    routine_name: str | None = None
    effort_rating: str | None = None  # 'too_easy', 'optimal', 'hard', 'too_hard'
    rpe: float | None = None  # 1.0 - 10.0 scale
    trimp: float | None = None  # Edwards TRIMP score
    hr_zones: dict[str, Any] | None = None  # Z1-Z5 percentage and duration breakdown
    hr_recovery_60s: int | None = None  # Post-exercise 60s heart rate drop in bpm


class TimelineBinRead(BaseModel):
    """5-minute biometric bin used to inspect Watch mode transitions inside a workout window."""

    started_at: datetime
    ended_at: datetime
    avg_hr_bpm: int | None = None
    peak_hr_bpm: int | None = None
    hr_sample_count: int = 0
    step_count: int = 0
    active_energy_kcal: int = 0
    is_active_watch_mode: bool = False


class SuggestedSplitRead(BaseModel):
    """Suggested split boundary around passive<->active Apple Watch mode transitions."""

    split_at: datetime
    resume_at: datetime
    reason: str
    confidence: str = "high"


class WorkoutTimelineRead(BaseModel):
    """5-minute resolution workout timeline and suggested split boundaries."""

    started_at: datetime
    ended_at: datetime
    duration_minutes: int
    bins: list[TimelineBinRead] = Field(default_factory=list)
    suggested_splits: list[SuggestedSplitRead] = Field(default_factory=list)


class SleepNapRead(BaseModel):
    """A distinct daytime nap episode segmented from granular sleep telemetry."""

    start_at: datetime
    end_at: datetime
    duration_hours: float


class DailyBiometricSummary(BaseModel):
    """Daily aggregated vitals rollup for morning briefings and retrospectives."""

    date: date
    latest_weight_kg: Decimal | None = None
    latest_body_fat_pct: Decimal | None = None
    resting_hr_bpm: int | None = None
    hrv_ms: Decimal | None = None
    overnight_hrv_ms: Decimal | None = None
    sleep_duration_hours: float | None = None
    sleep_deep_hours: float | None = None
    sleep_rem_hours: float | None = None
    nap_duration_hours: float | None = None
    nap_count: int = 0
    total_sleep_hours: float | None = None
    naps: list[SleepNapRead] = Field(default_factory=list)
    sleep_score: int | None = None
    wrist_temperature_c: float | None = None
    respiratory_rate_bpm: float | None = None
    oxygen_saturation_pct: float | None = None
    walking_asymmetry_pct: float | None = None
    vo2_max: float | None = None
    step_count: int | None = None
    active_energy_kcal: int | None = None
    readiness_notes: list[str] = Field(default_factory=list)
    fasting: HealthFeedbackRead | None = None
    workout: HealthFeedbackRead | None = None
    wellness: HealthFeedbackRead | None = None
    detected_workouts: list[DetectedWorkoutRead] = Field(default_factory=list)


# ------------------------------------------------------------- Granular Samples
class HealthSampleItem(BaseModel):
    """A single granular time-series sample (e.g. active energy slice)."""

    measured_at: datetime
    ended_at: datetime | None = None
    value: Decimal
    sample_type: str | None = None
    unit: str | None = None
    source: str | None = None
    external_id: str | None = None
    extra: dict = Field(default_factory=dict)

    @field_validator("measured_at", "ended_at", mode="before")
    @classmethod
    def parse_datetimes(cls, v: Any) -> Any:
        return clean_datetime(v)

    @field_validator("value", mode="before")
    @classmethod
    def parse_value(cls, v: Any) -> Decimal:
        parsed = clean_numeric(v)
        if parsed is None:
            raise ValueError(f"Cannot parse numeric value from {v}")
        return parsed


class HealthSampleRead(ORMModel):
    id: uuid.UUID
    source: str
    sample_type: str
    measured_at: datetime
    ended_at: datetime
    value: Decimal
    unit: str | None = None
    created_at: datetime


# ------------------------------------------------------------- Comprehensive Sync
DEFAULT_METRIC_UNITS: dict[str, str] = METRIC_REGISTRY.default_units()

METRIC_ALIASES: dict[str, str] = METRIC_REGISTRY.input_aliases()


class AppleHealthComprehensiveSyncPayload(BaseModel):
    """Universal payload for high-resolution multi-metric Apple Health synchronization."""

    model_config = ConfigDict(extra="allow")

    source: str = "apple_health"
    samples: list[HealthSampleItem] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class AppleHealthSyncResponse(BaseModel):
    status: str = "ok"
    total_samples_ingested: int
    metrics_summary: dict[str, int] = Field(default_factory=dict)
    days_processed: list[date] = Field(default_factory=list)
    detected_workouts: list[DetectedWorkoutRead] = Field(default_factory=list)
