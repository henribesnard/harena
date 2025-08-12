"""
FastAPI dependencies for Conversation Service MVP.

This module provides dependency injection functions for FastAPI endpoints,
managing AutoGen team managers, conversation context, authentication,
and request validation.

Dependencies:
    - get_team_manager: Provides singleton MVPTeamManager instance
    - get_current_user: Authentication and user context (local JWT decoding with user service fallback)
    - validate_conversation_request: Request validation and enrichment
    - get_conversation_manager: Conversation context management
    - validate_request_rate_limit: Rate limiting validation
    - get_metrics_collector: Metrics collection dependency
    - get_conversation_service: Database conversation service

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - FastAPI Dependencies
"""

import logging
import time
import asyncio
from collections import deque
from typing import Dict, Optional, Any, Annotated, Deque, TYPE_CHECKING, Generator
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import httpx
from jose import JWTError, jwt

from db_service.session import SessionLocal
from ..core import load_team_manager
from ..core.conversation_manager import ConversationManager
from ..models import ConversationRequest, ConversationResponse
from ..utils.metrics import MetricsCollector
from ..utils.logging import log_unauthorized_access
from ..services.conversation_db import (
    ConversationService as ConversationWriteService,
)
from ..services.conversation_service import (
    ConversationService as ConversationReadService,
)
from config_service.config import settings

if TYPE_CHECKING:
    from ..core.mvp_team_manager import MVPTeamManager

# Configure logging
logger = logging.getLogger(__name__)

# Global instances (singleton pattern)
_team_manager: Optional["MVPTeamManager"] = None
_conversation_manager: Optional[ConversationManager] = None
_metrics_collector: Optional[MetricsCollector] = None

# Rate limiting storage (in-memory for MVP; use Redis or another shared backend in production)
_rate_limit_storage: Dict[str, Deque[float]] = {}
_rate_limit_lock = asyncio.Lock()
# Evict users who haven't made a request within this TTL (seconds)
_RATE_LIMIT_TTL = 300

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")
ALGORITHM = "HS256"


def get_db() -> Generator[Session, None, None]:
    """Provide a database session with automatic commit/rollback.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def get_team_manager() -> "MVPTeamManager":
    """Dependency to get the singleton MVPTeamManager instance.

    Returns:
        MVPTeamManager: Configured team manager with AutoGen agents

    Raises:
        HTTPException: If team manager initialization or health check fails
    """
    global _team_manager

    if _team_manager is None:
        try:
            logger.info("Initializing MVPTeamManager singleton")
            MVPTeamManager, _ = load_team_manager()
            if MVPTeamManager is None:
                raise ImportError("MVPTeamManager not available")
            _team_manager = MVPTeamManager()
            await _team_manager.initialize_agents()
            team_health = getattr(_team_manager, "team_health", None)
            if team_health is None:
                logger.info(
                    "Skipping health check: team health not yet verified"
                )
            elif not team_health.overall_healthy:
                raise RuntimeError("MVPTeamManager health check failed")
            logger.info("MVPTeamManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MVPTeamManager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation service temporarily unavailable",
            )
    else:
        team_health = getattr(_team_manager, "team_health", None)
        if team_health is None:
            logger.info(
                "Skipping health check: team health not yet verified"
            )
        elif not team_health.overall_healthy:
            logger.error("MVPTeamManager is unhealthy")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation service temporarily unavailable",
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


def get_conversation_service(
    db: Annotated[Session, Depends(get_db)]
) -> ConversationWriteService:
    """
    Dependency to provide ConversationWriteService instance bound to a database
    session.

    Args:
        db: Database session from FastAPI dependency injection

    Returns:
        ConversationWriteService: Service instance for conversation write
        operations
    """
    return ConversationWriteService(db)


def get_conversation_read_service(
    db: Annotated[Session, Depends(get_db)]
) -> ConversationReadService:
    """Dependency to provide ConversationReadService for read-only operations."""
    return ConversationReadService(db)


async def get_current_user(
    request: Request,
    token: Annotated[str, Depends(oauth2_scheme)]
) -> Dict[str, Any]:
    """Attempt to authenticate the request using the Bearer token.

    The function first tries to decode the JWT locally using ``SECRET_KEY`` to
    extract the user identifier and any embedded claims. If decoding succeeds,
    default values for ``permissions`` and ``rate_limit_tier`` are applied when
    absent. Only when these fields are missing and a user service URL is
    configured does the function contact the user service to retrieve the full
    profile. If the user service call fails (503 or network error), the locally
    extracted information is returned.

    Args:
        token: OAuth2 Bearer token extracted from the request.

    Returns:
        Dict containing user information derived from the token or user service.

    Raises:
        HTTPException: If authentication fails and no usable token information is
            available.
    """

    user_data: Dict[str, Any] = {}
    needs_profile = False

    # First attempt: decode the JWT locally
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise JWTError("Missing subject")

        user_data = dict(payload)
        user_data["user_id"] = int(user_id) if str(user_id).isdigit() else user_id

        missing_permissions = "permissions" not in user_data
        missing_rate_limit = "rate_limit_tier" not in user_data

        if missing_permissions:
            user_data["permissions"] = []
        if missing_rate_limit:
            user_data["rate_limit_tier"] = "standard"

        needs_profile = missing_permissions or missing_rate_limit
        if not needs_profile:
            logger.debug(
                f"Authenticated user from token: {user_data.get('user_id')}"
            )
            return user_data
    except JWTError as exc:
        logger.debug(f"JWT decode failed, falling back to user service: {exc}")
        needs_profile = True

    token_perms = user_data.get("permissions", [])

    # Fallback: contact the user service for full profile information
    if settings.USER_SERVICE_URL and needs_profile:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{settings.USER_SERVICE_URL}{settings.API_V1_STR}/users/me",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == status.HTTP_401_UNAUTHORIZED:
                log_unauthorized_access(reason="invalid token")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if exc.response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
                logger.error(f"User service unavailable: {exc}")
                return user_data
            logger.error(f"User service returned error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="User service error",
            )
        except httpx.RequestError as exc:
            logger.error(f"Failed to contact user service: {exc}")
            return user_data

        resp = response.json()
        user_data.update(resp)
        user_data.setdefault("user_id", user_data.get("id"))
        user_data.setdefault("permissions", token_perms)
        user_data.setdefault("rate_limit_tier", "standard")
        if "chat:write" in token_perms and "chat:write" not in user_data["permissions"]:
            user_data["permissions"].append("chat:write")
        logger.debug(
            f"Authenticated user via user service: {user_data.get('user_id')}"
        )
        return user_data

    if user_data:
        return user_data

    log_unauthorized_access(reason="invalid token")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

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

    async with _rate_limit_lock:
        # TTL cleanup for users who haven't made requests recently
        expired_users = [
            uid for uid, timestamps in list(_rate_limit_storage.items())
            if timestamps and current_time - timestamps[-1] > _RATE_LIMIT_TTL
        ]
        for uid in expired_users:
            del _rate_limit_storage[uid]

        requests = _rate_limit_storage.setdefault(user_id, deque())

        # Remove entries outside the current window
        while requests and current_time - requests[0] >= window_size:
            requests.popleft()

        current_requests = len(requests)

        if current_requests >= limit:
            logger.warning(
                f"Rate limit exceeded for user {user_id}: {current_requests}/{limit}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per minute",
                headers={"Retry-After": "60"}
            )

        # Record this request
        requests.append(current_time)

    logger.debug(
        f"Rate limit check passed for user {user_id}: {current_requests + 1}/{limit}"
    )


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
