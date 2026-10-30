"""Application configuration loaded from environment variables."""

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
    kratos_api_url: str = "http://kratos-api:8000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
