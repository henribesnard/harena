import json
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.metrics_collector import metrics_collector

logger = logging.getLogger("harena.middleware")

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware de journalisation structurée des requêtes HTTP."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start_time = time.time()
        response: Response = await call_next(request)
        process_time = time.time() - start_time

        log_payload = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_ms": round(process_time * 1000, 2),
        }
        logger.info(json.dumps(log_payload))

        metrics_collector.record_request(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            process_time=process_time,
        )

        return response
