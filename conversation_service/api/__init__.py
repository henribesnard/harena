"""API package for the conversation service.

from . import routes, dependencies, websocket, middleware
from .routes import router
This package exposes the main :class:`APIRouter` along with helper
dependencies used by the tests.  The implementation is intentionally minimal
and is not intended to represent the production code base.
"""

from .dependencies import (
    get_team_manager,
    get_current_user,
    validate_conversation_request,
    get_conversation_manager,
    validate_request_rate_limit,
    get_metrics_collector,
    get_conversation_repository,
    get_conversation_service,
    get_conversation_read_service,
)
from .routes import router

__all__ = [
    "router",
    "get_team_manager",
    "get_current_user",
    "validate_conversation_request",
    "get_conversation_manager",
    "validate_request_rate_limit",
    "get_metrics_collector",
    "get_conversation_repository",
    "get_conversation_service",
    "get_conversation_read_service",
]

