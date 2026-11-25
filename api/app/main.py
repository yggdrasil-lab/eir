"""Main FastAPI application for Eir Health & Biometrics."""

from fastapi import FastAPI

from app import __version__
from app.routers import biometrics, feedback, health, integrations
from app.routers.mcp import install_mcp

app = FastAPI(
    title="Eir Health API",
    description="Health, biometrics, and wearable telemetry service for Yggdrasil.",
    version=__version__,
)

app.include_router(health.router)
app.include_router(biometrics.router)
app.include_router(feedback.router)
app.include_router(integrations.router)

install_mcp(app)
