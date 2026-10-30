"""Health check and liveness probe router."""

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__
from app.db import get_db
from app.schemas import HealthRead

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthRead, status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db)) -> HealthRead:
    """Returns application health and database connection status."""
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    return HealthRead(
        status="ok" if db_status == "ok" else "degraded",
        version=__version__,
        database=db_status,
    )
