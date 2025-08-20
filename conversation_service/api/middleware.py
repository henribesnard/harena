"""Application middleware utilities."""

import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import get_metrics_collector

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """Configure CORS and global error handling on the application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        metrics = get_metrics_collector()
        start = time.time()
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - handled by global handler
            metrics.record_error("request", str(exc))
            raise
        duration_ms = (time.time() - start) * 1000
        metrics.record_request(request.url.path, 0)
        metrics.record_response_time(request.url.path, duration_ms)
        return response

    @app.exception_handler(Exception)
    async def _exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
