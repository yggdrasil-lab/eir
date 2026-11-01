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

    @property
    def tz(self) -> zoneinfo.ZoneInfo:
        return zoneinfo.ZoneInfo(self.timezone)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
