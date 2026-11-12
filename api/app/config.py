"""Application configuration loaded from environment variables."""

import zoneinfo

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EIR_",
        env_file=".env",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://eir:eir@localhost:5432/eir"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"
    mcp_token: str | None = None
    webhook_secret: str | None = None
    timezone: str = "Pacific/Auckland"
    withings_client_id: str | None = None
    withings_client_secret: str | None = None
    withings_redirect_uri: str | None = None
    kratos_api_url: str = "http://kratos-api:8000"

    # User physiological profile settings
    user_dob: str = "1986-07-16"
    user_sex: str = "male"
    user_height_cm: float = 169.0
    user_max_hr: int = 186
    user_resting_hr_baseline: int = 63

    @property
    def user_age(self) -> int:
        try:
            from datetime import date
            dob = date.fromisoformat(self.user_dob)
            today = date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except Exception:
            return 40

    @property
    def tz(self) -> zoneinfo.ZoneInfo:
        return zoneinfo.ZoneInfo(self.timezone)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
