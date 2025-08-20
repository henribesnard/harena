"""Application middleware utilities."""

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_metrics_collector


def setup_middleware(app: FastAPI) -> None:
    """Configure CORS and request metrics middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        collector = await get_metrics_collector()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - handled by global handler
            collector.record_error("request", str(exc))
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        collector.record_request(request.url.path, 0)
        collector.record_response_time(request.url.path, duration_ms)
        return response
