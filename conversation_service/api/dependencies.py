"""FastAPI dependencies for the Conversation Service MVP.

Minimal dependency definitions for the conversation service.

This module provides dependency injection helpers for FastAPI endpoints,
    managing AutoGen team managers, conversation context, authentication and
    request validation.

Dependencies:
    - get_team_manager: Provides singleton ``MVPTeamManager`` instance
    - get_current_user: Authentication and user context
    - validate_conversation_request: Request validation and enrichment
    - get_conversation_manager: Conversation context management
    - validate_request_rate_limit: Rate limiting validation
    - get_metrics_collector: Metrics collection dependency

The real project contains a rich set of dependencies for authentication,
metrics, database access and agent management. For the purposes of the kata we
only implement lightweight placeholders so that the API can be exercised in
isolation and tests can override these dependencies.
"""

import logging
from typing import Any, Dict, Generator, Optional, TYPE_CHECKING, Annotated

from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session

from db_service.session import SessionLocal
from ..core import load_team_manager
from ..core.conversation_manager import ConversationManager
from ..models.conversation_models import ConversationRequest, ConversationResponse
from ..core.metrics_collector import MetricsCollector
from ..core.cache_manager import CacheManager
from ..repositories.conversation_repository import ConversationRepository
from ..utils.logging import log_unauthorized_access
from config_service.config import settings

if TYPE_CHECKING:
    from ..core.mvp_team_manager import MVPTeamManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# Global instances (singleton pattern)
_team_manager: Optional["MVPTeamManager"] = None
_conversation_manager: Optional[ConversationManager] = None
_metrics_collector: Optional[MetricsCollector] = None
_cache_manager: Optional[CacheManager] = None


# ---------------------------------------------------------------------------
# Database session management
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed afterwards."""
    db = SessionLocal()
    try:
        yield db
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
            metrics = await get_metrics_collector()
            cache = await get_cache_manager()
            _team_manager = MVPTeamManager(
                metrics_collector=metrics,
                cache_manager=cache,
            )
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


async def get_cache_manager() -> CacheManager:
    """Dependency to get the singleton CacheManager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_conversation_repository(
    db: Annotated[Session, Depends(get_db)]
) -> ConversationRepository:
    """Dependency providing a ConversationRepository bound to a DB session."""
    return ConversationRepository(db)


def get_conversation_service(
    db: Annotated[Session, Depends(get_db)]
) -> ConversationRepository:
    """Backward compatible alias for :class:`ConversationRepository`."""
    return ConversationRepository(db)


def get_conversation_read_service(
    db: Annotated[Session, Depends(get_db)]
) -> ConversationRepository:
    """Backward compatible alias for :class:`ConversationRepository`."""
    return ConversationRepository(db)


# ---------------------------------------------------------------------------
# Simplified authentication and request validation
# ---------------------------------------------------------------------------


async def get_current_user() -> Dict[str, Any]:
    """Placeholder authentication dependency used in tests."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


async def validate_conversation_request(
    request: ConversationRequest,
) -> ConversationRequest:
    """Return the request unchanged (placeholder)."""
    return request


async def validate_request_rate_limit() -> None:
    """No-op rate limit validator used in tests."""
    return None

async def cleanup_dependencies():
    """
    Cleanup function for graceful shutdown of dependencies.
    
    Should be called during application shutdown to properly close
    connections and cleanup resources.
    """
    global _team_manager, _conversation_manager, _metrics_collector, _cache_manager
    
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

    if _cache_manager:
        try:
            _cache_manager.clear()
            logger.info("CacheManager cleared")
        except Exception as e:
            logger.error(f"Error clearing CacheManager: {e}")
    
    # Clear global instances
    _team_manager = None
    _conversation_manager = None
    _metrics_collector = None
    _cache_manager = None

