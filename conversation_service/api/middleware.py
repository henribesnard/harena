"""Global middleware for exception handling in Conversation Service.

This middleware converts exceptions into JSON responses and logs them with
user and conversation context when available.
"""

import json
import logging
from typing import Any, Tuple

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.exceptions import HarenaException
"""Application middleware utilities."""

import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import get_metrics_collector

logger = logging.getLogger(__name__)


def _extract_context(request: Request, body: Any) -> Tuple[Any, Any]:
    """Extract user and conversation identifiers from request/headers."""
    user_id = getattr(request.state, "user_id", None) or request.headers.get("X-User-Id")
    conversation_id = getattr(request.state, "conversation_id", None) or request.headers.get("X-Conversation-Id")

    if isinstance(body, dict):
        user_id = user_id or body.get("user_id")
        conversation_id = conversation_id or body.get("conversation_id")

    return user_id, conversation_id


class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    """Middleware that handles uncaught exceptions and returns JSON responses."""

    async def dispatch(self, request: Request, call_next):
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes.decode()) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}

        async def receive() -> dict:
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request._receive = receive
        request.state.body = body

        try:
            response = await call_next(request)
            return response
        except HarenaException as exc:
            user_id, conversation_id = _extract_context(request, body)
            logger.error(
                f"{exc.message} | user_id={user_id} conversation_id={conversation_id}"
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.message},
            )
        except HTTPException as exc:
            user_id, conversation_id = _extract_context(request, body)
            logger.warning(
                f"HTTP {exc.status_code} {exc.detail} | user_id={user_id} conversation_id={conversation_id}"
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": True,
                    "status_code": exc.status_code,
                    "message": exc.detail,
                },
            )
        except RequestValidationError as exc:
            user_id, conversation_id = _extract_context(request, body)
            logger.warning(
                f"Validation error {exc} | user_id={user_id} conversation_id={conversation_id}"
            )
            return JSONResponse(
                status_code=422,
                content={
                    "error": True,
                    "status_code": 422,
                    "message": "Request validation failed",
                    "details": exc.errors(),
                },
            )
        except Exception as exc:
            user_id, conversation_id = _extract_context(request, body)
            logger.exception(
                f"Unhandled error: {exc} | user_id={user_id} conversation_id={conversation_id}"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "status_code": 500,
                    "message": "Internal server error",
                },
            )
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
