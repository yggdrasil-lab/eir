"""Main FastAPI application for Eir Health & Biometrics."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.routers import biometrics, health, integrations
from app.routers.mcp import install_mcp

app = FastAPI(
    title="Eir Health API",
    description="Health, biometrics, and wearable telemetry service for Yggdrasil.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(biometrics.router)
app.include_router(integrations.router)

install_mcp(app)
