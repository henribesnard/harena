from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from conversation_service.api.routes import router as conversation_router
from conversation_service.core.metrics_collector import metrics_collector


router = APIRouter()
router.include_router(conversation_router, prefix="/conversation")


@router.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for scraping."""
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)


@router.get("/health", tags=["health"])
async def health() -> dict:
    """Return basic service health information."""
    return {"status": "ok", "metrics": metrics_collector.summary()}
