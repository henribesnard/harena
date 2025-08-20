"""API package for the conversation service."""

from . import routes, dependencies, websocket, middleware
from .routes import router

from .dependencies import (
    get_team_manager,
    get_current_user,
    validate_conversation_request,
    get_conversation_manager,
    validate_request_rate_limit,
    get_metrics_collector
)

from .routes import (
    chat_router,
    health_router,
    router as main_router,
    ws_router
)

# API Models for external use
from ..models.conversation_models import (
    ConversationRequest,
    ConversationResponse,
    ConversationTurn,
    ConversationContext,
)

from ..models.agent_models import (
    AgentResponse,
    TeamWorkflow,
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Conversation Service Team"

# Export all API components
__all__ = [
    # Dependencies
    "get_team_manager",
    "get_current_user", 
    "validate_conversation_request",
    "get_conversation_manager",
    "validate_request_rate_limit",
    "get_metrics_collector",
    
    # Routers
    "chat_router",
    "health_router",
    "main_router",
    "ws_router",
    
    # API Models
    "ConversationRequest",
    "ConversationResponse",
    "ConversationTurn",
    "ConversationContext",
    "AgentResponse",
    "TeamWorkflow"
]

# API Configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Supported endpoints summary for documentation
ENDPOINTS_SUMMARY = {
    "POST /chat": "Main conversation endpoint with AutoGen multi-agent processing",
    "GET /health": "Service health check with component status",
    "GET /metrics": "Performance metrics and agent statistics",
    "GET /docs": "Interactive API documentation",
    "GET /redoc": "Alternative API documentation",
    "WS /ws/chat": "WebSocket chat endpoint with incremental agent messages"
}

# Rate limiting configuration
RATE_LIMITS = {
    "chat": "30/minute",  # 30 conversations per minute per user
    "health": "100/minute",  # 100 health checks per minute
    "metrics": "20/minute"   # 20 metrics requests per minute
}

# CORS origins for production (to be configured via settings)
CORS_ORIGINS = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8080",  # Vue dev server
    "https://app.harena.ai",  # Production frontend
]

# Request timeout configuration
REQUEST_TIMEOUTS = {
    "chat": 30.0,      # 30 seconds for conversation processing
    "health": 5.0,     # 5 seconds for health checks
    "metrics": 10.0    # 10 seconds for metrics collection
}

# Logging configuration for API layer
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "api": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [API] %(message)s"
        }
    },
    "handlers": {
        "api_handler": {
            "class": "logging.StreamHandler", 
            "formatter": "api",
            "level": "INFO"
        }
    },
    "loggers": {
        "conversation_service.api": {
            "handlers": ["api_handler"],
            "level": "INFO",
            "propagate": False
        }
    }
}
__all__ = ["routes", "dependencies", "websocket", "middleware", "router"]
