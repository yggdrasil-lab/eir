"""Pydantic schemas for Eir API request and response validation."""

from datetime import date, datetime
from decimal import Decimal
import uuid

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HealthRead(BaseModel):
    status: str
    version: str
    database: str


# ------------------------------------------------------------- Integrations
class AppleWorkoutPayload(BaseModel):
    """Workout telemetry pushed from Apple Watch via iOS Shortcuts."""

    workout_type: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: int | None = None
    active_energy_kcal: Decimal | None = None
    avg_heart_rate_bpm: int | None = None
    max_heart_rate_bpm: int | None = None
    distance_meters: Decimal | None = None
    pool_length_meters: Decimal | None = None
    strokes: int | None = None
    notes: str | None = None
    forward_to_kratos: bool = False
    extra: dict = Field(default_factory=dict)


class AppleBiometricsPayload(BaseModel):
    """Daily or morning biometrics from Apple Health via iOS Shortcuts."""

    measured_at: datetime = Field(default_factory=datetime.now)
    weight_kg: Decimal | None = None
    body_fat_pct: Decimal | None = None
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


class WithingsNotificationPayload(BaseModel):
    """Webhook callback from Withings notification subscription."""

    userid: str | None = None
    appli: int | None = None  # 1 = weight, 44 = sleep, 16 = activity
    startdate: int | None = None
    enddate: int | None = None


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


class DailyBiometricSummary(BaseModel):
    """Daily aggregated vitals rollup for morning briefings and retrospectives."""

    date: date
    latest_weight_kg: Decimal | None = None
    latest_body_fat_pct: Decimal | None = None
    resting_hr_bpm: int | None = None
    hrv_ms: Decimal | None = None
    sleep_duration_hours: float | None = None
    sleep_deep_hours: float | None = None
    sleep_score: int | None = None
    step_count: int | None = None
    active_energy_kcal: int | None = None
    readiness_notes: list[str] = Field(default_factory=list)
