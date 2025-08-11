"""
FastAPI routes for Conversation Service MVP.

This module defines the main API endpoints for the AutoGen-based conversation
service, including chat processing, health checks, and metrics collection.

Routes:
    POST /chat - Main conversation endpoint with multi-agent processing
    GET /health - Service health check with component status
    GET /metrics - Performance metrics and agent statistics
    GET /status - Service status and configuration

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - FastAPI Routes
"""

import asyncio
import logging
import time
from typing import Annotated, Any, Dict, List, Optional, Protocol

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .dependencies import (
    get_team_manager,
    get_current_user,
    validate_conversation_request,
    get_conversation_manager,
    validate_request_rate_limit,
    get_metrics_collector,
    get_conversation_service,
)
from ..core.conversation_manager import ConversationManager
from ..models import (
    ConversationRequest,
    ConversationResponse,
    ConversationOut,
    ConversationTurn,
)
import os

from ..utils.logging import log_unauthorized_access
from ..services.conversation_db import ConversationService as ConversationDBService
from ..utils.metrics import MetricsCollector
from ..services.conversation_db import ConversationService
from db_service.session import get_db

try:
    from ..core.mvp_team_manager import MVPTeamManager
except ImportError:
    class MVPTeamManager(Protocol):
        async def process_user_message(
            self, user_message: str, user_id: int, conversation_id: str
        ) -> Any:
            ...

        async def get_health_status(self) -> Dict[str, Any]:
            ...

# Configure logging
logger = logging.getLogger(__name__)

# Create main router
router = APIRouter()

# Create specialized routers
chat_router = APIRouter(prefix="/chat", tags=["conversation"])
health_router = APIRouter(prefix="/health", tags=["monitoring"])
conversations_router = APIRouter(prefix="/conversations", tags=["conversation"])


@chat_router.post(
    "",
    response_model=ConversationResponse,
    summary="Process conversation with AutoGen multi-agents",
    description="Main conversation endpoint that processes user messages through AutoGen multi-agent team",
    responses={
        200: {"description": "Successful conversation processing"},
        422: {"description": "Invalid request format"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service temporarily unavailable"}
    }
)
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
    conversation_manager: Annotated[ConversationManager, Depends(get_conversation_manager)],
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics_collector)],
    conversation_service: Annotated[
        ConversationDBService, Depends(get_conversation_service)
    ],
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[None, Depends(validate_request_rate_limit)],
    validated_request: Annotated[ConversationRequest, Depends(validate_conversation_request)]
) -> ConversationResponse:
    """
    Process a conversation message through AutoGen multi-agent team.
    
    This endpoint orchestrates the complete conversation workflow:
    1. Validates user input and context
    2. Retrieves conversation history
    3. Processes message through AutoGen agents
    4. Stores conversation turn
    5. Returns AI response with metadata
    
    Args:
        background_tasks: FastAPI background tasks
        team_manager: AutoGen team manager dependency
        conversation_manager: Conversation context dependency
        user: Authenticated user context
        metrics: Metrics collector dependency
        validated_request: Validated conversation request
        
    Returns:
        ConversationResponse: AI response with metadata
    """
    start_time = time.time()
    conversation_id = validated_request.conversation_id
    user_id = user["user_id"]
    
    logger.info(f"Processing conversation for user {user_id}, conversation {conversation_id}")

    try:
        conversation = conversation_service.get_or_create_conversation(
            user_id, conversation_id
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conversation access denied",
        )
    except Exception as e:
        logger.error(f"Conversation retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database access error",
        )

    conversation_id = conversation.conversation_id
    logger.info(
        f"Processing conversation for user {user_id}, conversation {conversation_id}"
    )
    
    try:
        # Record request metrics
        metrics.record_request("chat", user_id)

        # Permission check
        if "chat:write" not in user.get("permissions", []):
            log_unauthorized_access(user_id=user_id, conversation_id=conversation_id, reason="insufficient permissions")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        # Get conversation context
        context = await conversation_manager.store.get_context(conversation_id)
        if context is None:
            log_unauthorized_access(user_id=user_id, conversation_id=conversation_id, reason="conversation not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        if getattr(context, "user_id", user_id) != user_id:
            log_unauthorized_access(user_id=user_id, conversation_id=conversation_id, reason="forbidden conversation access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        logger.debug(f"Retrieved context with {len(context.turns)} previous turns")

        # Process message through AutoGen team
        try:
            team_response = await team_manager.process_user_message(
                user_message=validated_request.message,
                user_id=user_id,
                conversation_id=conversation_id
            )
            if not team_response.success:
                metrics.record_error(
                    "team_processing", team_response.error_message or "unknown_error"
                )
            logger.info("AutoGen team processed message")

        except Exception as e:
            logger.error(f"AutoGen team processing failed: {e}")
            metrics.record_error("team_processing", str(e))

            # Fallback response
            fallback_response = ConversationResponse(
                message="Je suis désolé, je rencontre une difficulté technique. Pouvez-vous reformuler votre question ?",
                conversation_id=conversation_id,
                success=False,
                error_code="TEAM_PROCESSING_ERROR",
                processing_time_ms=int((time.time() - start_time) * 1000),
                agent_used="fallback",
                confidence=0.1
            )

            return fallback_response
        
        # Store conversation turn in background
        background_tasks.add_task(
            store_conversation_turn,
            conversation_manager,
            conversation_id,
            user_id,
            validated_request.message,
            team_response.content,
            int((time.time() - start_time) * 1000),
            metrics,
            intent_detected=team_response.metadata.get("intent_detected"),
            entities_extracted=team_response.metadata.get("entities_extracted"),
            agent_chain=team_response.metadata.get("agent_chain"),
            search_results_count=team_response.metadata.get("search_results_count"),
            confidence_score=team_response.confidence_score,
        )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)

        try:
            conversation_service.add_turn(
                conversation_id=conversation.conversation_id,
                user_id=user_id,
                user_message=validated_request.message,
                assistant_response=team_response.content,
                processing_time_ms=processing_time,
                intent_detected=team_response.metadata.get("intent_detected"),
                entities_extracted=team_response.metadata.get("entities_extracted"),
                confidence_score=team_response.confidence_score,
                agent_chain=team_response.metadata.get("agent_chain"),
                search_results_count=team_response.metadata.get(
                    "search_results_count", 0
                ),
                search_execution_time_ms=team_response.metadata.get(
                    "search_execution_time_ms"
                ),
            )
        except Exception as e:
            logger.error(f"Failed to store conversation turn: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store conversation turn",
            )

        # Create response
        response = ConversationResponse(
            message=team_response.content,
            conversation_id=conversation_id,
            success=team_response.success,
            processing_time_ms=processing_time,
            agent_used="orchestrator_agent",
            confidence=team_response.confidence_score or 0.0,
            metadata=team_response.metadata,
            error_code=None if team_response.success else "TEAM_PROCESSING_ERROR",
        )
        
        # Record metrics based on success
        if team_response.success:
            metrics.record_response_time("chat", processing_time)
            metrics.record_success("chat")
            logger.info(f"Conversation processed successfully in {processing_time}ms")
        else:
            metrics.record_error("chat", team_response.error_message or "unknown_error")
            logger.warning(f"Conversation processed with errors in {processing_time}ms")

        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in chat endpoint: {e}")
        metrics.record_error("chat_endpoint", str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal conversation processing error"
        )


@health_router.get(
    "",
    summary="Service health check",
    description="Comprehensive health check including all service components",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    }
)
async def health_check(
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
    conversation_manager: Annotated[ConversationManager, Depends(get_conversation_manager)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics_collector)],
    user: Annotated[Dict[str, Any], Depends(get_current_user)]
) -> Dict[str, Any]:
    """
    Comprehensive health check for all service components.

    Returns:
        Dict containing health status of all components
    """
    start_time = time.time()
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "conversation_service_mvp",
        "version": "1.0.0",
        "components": {},
        "errors": []
    }
    component_timeout = 2

    try:
        # Check AutoGen team manager with timeout
        try:
            team_health = await asyncio.wait_for(
                team_manager.health_check(), timeout=component_timeout
            )
            details = team_health.get("details", {})
            health_status["components"]["team_manager"] = {
                "status": "healthy" if team_health.get("healthy") else "unhealthy",
                "agents_loaded": len(details.get("agent_statuses", {})),
                "last_activity": details.get("last_health_check")
            }
            if not team_health.get("healthy"):
                health_status["status"] = "degraded"
                health_status["errors"].append({"component": "team_manager", "error": "unhealthy"})
        except Exception as e:
            logger.error(f"Team manager health check failed: {e}")
            health_status["status"] = "degraded"
            health_status["components"]["team_manager"] = {
                "status": "unavailable",
                "error": str(e)
            }
            health_status["errors"].append({"component": "team_manager", "error": str(e)})
            health_status["response_time_ms"] = int((time.time() - start_time) * 1000)
            return JSONResponse(status_code=status.HTTP_200_OK, content=health_status)

        # Check conversation manager
        try:
            conv_stats = await conversation_manager.get_stats()
            health_status["components"]["conversation_manager"] = {
                "status": "healthy",
                "active_conversations": conv_stats.get("active_conversations", 0),
                "total_turns": conv_stats.get("total_turns", 0)
            }
        except Exception as e:
            logger.error(f"Conversation manager health check failed: {e}")
            health_status["status"] = "degraded"
            health_status["components"]["conversation_manager"] = {
                "status": "unavailable",
                "error": str(e)
            }
            health_status["errors"].append({"component": "conversation_manager", "error": str(e)})
            health_status["response_time_ms"] = int((time.time() - start_time) * 1000)
            return JSONResponse(status_code=status.HTTP_200_OK, content=health_status)

        # Check metrics collector
        try:
            metrics_summary = metrics.get_summary()
            health_status["components"]["metrics"] = {
                "status": "healthy",
                "requests_processed": metrics_summary.get("total_requests", 0),
                "average_response_time": metrics_summary.get("avg_response_time", 0)
            }
        except Exception as e:
            logger.warning(f"Metrics health check failed: {e}")
            health_status["status"] = "degraded"
            health_status["components"]["metrics"] = {
                "status": "unavailable",
                "error": str(e)
            }
            health_status["errors"].append({"component": "metrics", "error": str(e)})
            health_status["response_time_ms"] = int((time.time() - start_time) * 1000)
            return JSONResponse(status_code=status.HTTP_200_OK, content=health_status)

        health_status["response_time_ms"] = int((time.time() - start_time) * 1000)
        return JSONResponse(status_code=status.HTTP_200_OK, content=health_status)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        error_response = {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": "Health check system failure",
            "response_time_ms": int((time.time() - start_time) * 1000)
        }
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response
        )
@health_router.get(
    "/metrics",
    summary="Service performance metrics",
    description="Detailed performance metrics and agent statistics",
    responses={
        200: {"description": "Metrics retrieved successfully"},
        503: {"description": "Metrics service unavailable"}
    }
)
async def get_metrics(
    metrics: Annotated[MetricsCollector, Depends(get_metrics_collector)],
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
    user: Annotated[Dict[str, Any], Depends(get_current_user)]
) -> Dict[str, Any]:
    """
    Get detailed performance metrics and statistics.
    
    Returns:
        Dict containing comprehensive service metrics
    """
    try:
        # Check if user has metrics permission
        if "view_metrics" not in user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view metrics"
            )
        
        # Get metrics summary
        metrics_summary = metrics.get_summary()
        
        # Get team performance
        team_performance = team_manager.get_team_performance()
        
        # Compile comprehensive metrics
        comprehensive_metrics = {
            "timestamp": time.time(),
            "service_metrics": metrics_summary,
            "agent_metrics": team_performance,
            "system_info": {
                "uptime_seconds": time.time() - metrics.start_time,
                "memory_usage": metrics.get_memory_usage(),
                "cpu_usage": metrics.get_cpu_usage()
            }
        }
        
        return comprehensive_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics service temporarily unavailable"
        )


@conversations_router.get(
    "",
    response_model=List[ConversationOut],
    summary="List user conversations",
    description="Return conversations for the authenticated user",
    responses={
        200: {"description": "Conversations retrieved successfully"},
        401: {"description": "Unauthorized"},
    },
)
async def list_conversations(
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    service: Annotated[ConversationDBService, Depends(get_conversation_service)],
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> List[ConversationOut]:
    """List conversations belonging to the current user."""
    conversations = service.get_conversations(user["user_id"], limit=limit, offset=offset)
    return [
        ConversationOut(
            conversation_id=c.conversation_id,
            title=c.title,
            status=c.status,
            total_turns=c.total_turns,
            last_activity_at=c.last_activity_at,
        )
        for c in conversations
    ]


@conversations_router.get(
    "/{conversation_id}/turns",
    response_model=List[ConversationTurn],
    summary="Get conversation turns",
    description="Return turns of a conversation if it belongs to the user",
    responses={
        200: {"description": "Conversation turns retrieved"},
        404: {"description": "Conversation not found"},
        401: {"description": "Unauthorized"},
    },
)
async def get_conversation_turns(
    conversation_id: str,
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    service: Annotated[ConversationDBService, Depends(get_conversation_service)],
) -> List[ConversationTurn]:
    """Return the turns for a specific conversation."""
    conversation = service.get_conversation(conversation_id)
    if conversation is None or conversation.user_id != user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    turns: List[ConversationTurn] = []
    for t in conversation.turns:
        turns.append(
            ConversationTurn(
                turn_id=t.turn_id,
                user_message=t.user_message,
                assistant_response=t.assistant_response,
                timestamp=t.created_at,
                metadata=t.turn_metadata or {},
                turn_number=t.turn_number,
                processing_time_ms=t.processing_time_ms or 0.0,
                intent_detected=t.intent_detected,
                entities_extracted=t.entities_extracted,
                confidence_score=t.confidence_score,
                error_occurred=t.error_occurred,
                agent_chain=t.agent_chain,
            )
        )
    return turns


@router.get(
    "/status",
    summary="Service status information",
    description="Basic service status and configuration",
    tags=["monitoring"]
)
async def get_status(
    user: Annotated[Dict[str, Any], Depends(get_current_user)]
) -> Dict[str, Any]:
    """
    Get basic service status and configuration information.
    
    Returns:
        Dict containing service status
    """
    environment = os.getenv("ENVIRONMENT", "development")
    enable_auth = True
    
    return {
        "service": "conversation_service_mvp",
        "version": "1.0.0",
        "status": "running",
        "timestamp": time.time(),
        "environment": environment,
        "autogen_version": "0.4.0",
        "api_version": "v1",
        "features": {
            "authentication": enable_auth,
            "rate_limiting": True,
            "metrics_collection": True,
            "conversation_memory": True
        }
    }


# Background task functions
async def store_conversation_turn(
    conversation_manager: ConversationManager,
    conversation_id: str,
    user_id: int,
    user_message: str,
    assistant_message: str,
    processing_time_ms: int,
    metrics: MetricsCollector,
    intent_detected: Optional[str] = None,
    entities_extracted: Optional[List[Dict[str, Any]]] = None,
    agent_chain: Optional[List[str]] = None,
    search_results_count: Optional[int] = None,
    confidence_score: Optional[float] = None,
) -> None:
    """
    Background task to store conversation turn.
    
    Args:
        conversation_manager: Conversation manager instance
        conversation_id: Conversation identifier
        user_id: User identifier
        user_message: User's message
        assistant_message: Assistant's response
        processing_time_ms: Processing time in milliseconds
        metrics: Metrics collector
        intent_detected: Detected user intent
        entities_extracted: Entities extracted from the user message
        agent_chain: Chain of agents involved in processing
        search_results_count: Number of results returned by search
        confidence_score: Confidence score of the response
    """
    try:
        await conversation_manager.add_turn(
            conversation_id=conversation_id,
            user_id=user_id,
            user_msg=user_message,
            assistant_msg=assistant_message,
            processing_time_ms=float(processing_time_ms),
            intent_detected=intent_detected,
            entities_extracted=entities_extracted,
            agent_chain=agent_chain,
            search_results_count=search_results_count,
            confidence_score=confidence_score,
        )
        logger.debug(f"Stored conversation turn for {conversation_id}")
        
    except Exception as e:
        logger.error(f"Failed to store conversation turn: {e}")
        metrics.record_error("conversation_storage", str(e))


# Include routers in main router
router.include_router(chat_router)
router.include_router(health_router)
router.include_router(conversations_router)