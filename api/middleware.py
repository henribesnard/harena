"""Application middleware setup for logging and metrics."""

from typing import Dict

from fastapi import FastAPI

from utils.logging import StructuredLoggingMiddleware
from core.metrics_collector import metrics_collector


def setup_middleware(app: FastAPI) -> None:
    """Configure structured logging and expose metrics endpoint."""
    app.add_middleware(StructuredLoggingMiddleware)

    @app.get("/metrics", tags=["monitoring"])
    async def metrics() -> Dict[str, Dict[str, float]]:
        """Return collected application metrics."""
        return metrics_collector.get_metrics()

