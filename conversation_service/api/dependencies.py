"""
FastAPI dependencies for Conversation Service MVP.

This module provides dependency injection functions for FastAPI endpoints,
managing AutoGen team managers, conversation context, authentication,
and request validation.

Dependencies:
    - get_team_manager: Provides singleton MVPTeamManager instance
    - get_current_user: Authentication and user context (placeholder)
    - validate_conversation_request: Request validation and enrichment
    - get_conversation_manager: Conversation context management
    - validate_request_rate_limit: Rate limiting validation
    - get_metrics_collector: Metrics collection dependency

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - FastAPI Dependencies
"""

import logging
import time
from typing import Dict, Optional, Any, Annotated
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..core.mvp_team_manager import MVPTeamManager
from ..core.conversation_manager import ConversationManager
from ..models.conversation_models import ConversationRequest, ConversationResponse
from ..utils.metrics import MetricsCollector
import os

# Configure logging
logger = logging.getLogger(__name__)

# Security scheme for authentication
security = HTTPBearer(auto_error=False)

# Global instances (singleton pattern)
_team_manager: Optional[MVPTeamManager] = None
_conversation_manager: Optional[ConversationManager] = None  
_metrics_collector: Optional[MetricsCollector] = None

# Rate limiting storage (in-memory for MVP, should use Redis in production)
_rate_limit_storage: Dict[str, Dict[str, Any]] = {}


async def get_team_manager() -> MVPTeamManager:
    """
    Dependency to get the singleton MVPTeamManager instance.
    
    Returns:
        MVPTeamManager: Configured team manager with AutoGen agents
        
    Raises:
        HTTPException: If team manager initialization fails
    """
    global _team_manager
    
    if _team_manager is None:
        try:
            logger.info("Initializing MVPTeamManager singleton")
            _team_manager = MVPTeamManager()
            await _team_manager.initialize_agents()
            logger.info("MVPTeamManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MVPTeamManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation service temporarily unavailable"
            )
    
    return _team_manager


async def get_conversation_manager() -> ConversationManager:
    """
    Dependency to get the singleton ConversationManager instance.
    
    Returns:
        ConversationManager: Configured conversation context manager
        
    Raises:
        HTTPException: If conversation manager initialization fails
    """
    global _conversation_manager
    
    if _conversation_manager is None:
        try:
            logger.info("Initializing ConversationManager singleton")
            _conversation_manager = ConversationManager(storage_backend="memory")
            await _conversation_manager.initialize()
            logger.info("ConversationManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConversationManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation context service unavailable"
            )
    
    return _conversation_manager


async def get_metrics_collector() -> MetricsCollector:
    """
    Dependency to get the singleton MetricsCollector instance.
    
    Returns:
        MetricsCollector: Metrics collection and analysis instance
    """
    global _metrics_collector
    
    if _metrics_collector is None:
        logger.info("Initializing MetricsCollector singleton")
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]
) -> Dict[str, Any]:
    """
    Authentication dependency (placeholder implementation).
    
    In MVP, returns a mock user. In production, this should validate JWT tokens,
    check user permissions, and return user context from database.
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        Dict containing user information
        
    Raises:
        HTTPException: If authentication fails (disabled in MVP)
    """
    # MVP: Return mock user without authentication
    # TODO: Implement real JWT validation for production
    enable_auth = os.getenv("ENABLE_AUTHENTICATION", "false").lower() == "true"
    
    if enable_auth:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # TODO: Validate JWT token here
        # For now, accept any token
        token = credentials.credentials
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
    
    # Return mock user for MVP
    mock_user = {
        "user_id": 12345,
        "username": "mvp_user",
        "email": "user@harena.ai",
        "is_active": True,
        "permissions": ["chat", "view_metrics"],
        "subscription_type": "premium",
        "rate_limit_tier": "standard"
    }
    
    logger.debug(f"Authenticated user: {mock_user['user_id']}")
    return mock_user


def validate_conversation_request(request: ConversationRequest) -> ConversationRequest:
    """
    Validates and enriches conversation requests.
    
    Args:
        request: Raw conversation request from client
        
    Returns:
        ConversationRequest: Validated and enriched request
        
    Raises:
        HTTPException: If request validation fails
    """
    try:
        # Basic validation (Pydantic handles most of this)
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Message cannot be empty"
            )
        
        # Message length validation
        max_message_length = 10000  # 10KB max message
        if len(request.message) > max_message_length:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Message too long (max {max_message_length} characters)"
            )
        
        # Clean and normalize message
        request.message = request.message.strip()
        
        # Set default values if not provided
        if not request.conversation_id:
            import uuid
            request.conversation_id = str(uuid.uuid4())
            logger.info(f"Generated new conversation_id: {request.conversation_id}")
        
        # Enrich request with metadata
        if not hasattr(request, 'timestamp'):
            from datetime import datetime
            request.timestamp = datetime.utcnow()
        
        logger.debug(f"Validated request for conversation: {request.conversation_id}")
        return request
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid request format"
        )


async def validate_request_rate_limit(
    request: Request,
    user: Annotated[Dict[str, Any], Depends(get_current_user)]
) -> None:
    """
    Validates request rate limits per user.
    
    Args:
        request: FastAPI request object
        user: Current authenticated user
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    user_id = user["user_id"]
    rate_limit_tier = user.get("rate_limit_tier", "standard")
    
    # Rate limits by tier (requests per minute)
    rate_limits = {
        "standard": 30,
        "premium": 100,
        "enterprise": 500
    }
    
    limit = rate_limits.get(rate_limit_tier, 30)
    current_time = time.time()
    window_size = 60  # 1 minute window
    
    # Clean old entries
    if user_id in _rate_limit_storage:
        _rate_limit_storage[user_id]["requests"] = [
            req_time for req_time in _rate_limit_storage[user_id]["requests"]
            if current_time - req_time < window_size
        ]
    else:
        _rate_limit_storage[user_id] = {"requests": []}
    
    # Check current rate
    current_requests = len(_rate_limit_storage[user_id]["requests"])
    
    if current_requests >= limit:
        logger.warning(f"Rate limit exceeded for user {user_id}: {current_requests}/{limit}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per minute",
            headers={"Retry-After": "60"}
        )
    
    # Record this request
    _rate_limit_storage[user_id]["requests"].append(current_time)
    logger.debug(f"Rate limit check passed for user {user_id}: {current_requests + 1}/{limit}")


async def cleanup_dependencies():
    """
    Cleanup function for graceful shutdown of dependencies.
    
    Should be called during application shutdown to properly close
    connections and cleanup resources.
    """
    global _team_manager, _conversation_manager, _metrics_collector
    
    logger.info("Cleaning up API dependencies")
    
    if _team_manager:
        try:
            await _team_manager.shutdown()
            logger.info("MVPTeamManager shutdown completed")
        except Exception as e:
            logger.error(f"Error shutting down MVPTeamManager: {e}")
    
    if _conversation_manager:
        try:
            await _conversation_manager.close()
            logger.info("ConversationManager shutdown completed")
        except Exception as e:
            logger.error(f"Error shutting down ConversationManager: {e}")
    
    if _metrics_collector:
        try:
            _metrics_collector.export_metrics()
            logger.info("MetricsCollector export completed")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    # Clear global instances
    _team_manager = None
    _conversation_manager = None
    _metrics_collector = None
    
    logger.info("API dependencies cleanup completed")