"""
API package for Conversation Service MVP.

This module provides the main API components for the AutoGen-based conversation
service, including FastAPI routes, dependencies, and endpoint configurations.

The API layer handles HTTP/WebSocket communication, request validation,
authentication, and integrates with the AutoGen MVPTeamManager for
multi-agent conversation processing.

Exports:
    Dependencies:
        - get_team_manager: Dependency for MVPTeamManager instance
        - get_current_user: Authentication dependency with JWT decoding fallback
        - validate_conversation_request: Request validation
    
    Router:
        - chat_router: Main conversation endpoints router
        - health_router: Health check endpoints
    
    Models:
        - ConversationRequest: API request model
        - ConversationResponse: API response model
        - HealthResponse: Health check response
        - MetricsResponse: Metrics response

Architecture:
    - FastAPI with async/await for high performance
    - AutoGen v0.4 integration for multi-agent workflows
    - Pydantic V2 for request/response validation
    - Structured logging and metrics collection

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - FastAPI API Layer
"""

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
    router as main_router
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
    "GET /redoc": "Alternative API documentation"
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