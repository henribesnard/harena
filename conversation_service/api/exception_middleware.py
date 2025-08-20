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

logger = logging.getLogger(__name__)


def _extract_context(request: Request, body: Any) -> Tuple[Any, Any]:
    """Extract user and conversation identifiers from request/headers."""
    user_id = getattr(request.state, "user_id", None) or request.headers.get("X-User-Id")
    conversation_id = getattr(request.state, "conversation_id", None) or request.headers.get(
        "X-Conversation-Id"
    )

    if isinstance(body, dict):
        user_id = user_id or body.get("user_id")
        conversation_id = conversation_id or body.get("conversation_id")

    return user_id, conversation_id


class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    """Middleware that handles uncaught exceptions and returns JSON responses."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes.decode()) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}

        async def receive() -> dict:
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request._receive = receive  # type: ignore[attr-defined]
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
        except Exception as exc:  # pragma: no cover - unexpected errors
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
