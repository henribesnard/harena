from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from core.metrics_collector import metrics_collector
from . import websocket


router = APIRouter()
router.include_router(websocket.router)


@router.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for scraping."""
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)


@router.get("/health", tags=["health"])
async def health() -> dict:
    """Return basic service health information."""
    return {"status": "ok", "metrics": metrics_collector.get_metrics()}
