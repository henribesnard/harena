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

import logging
import time
import asyncio
from typing import Dict, Any, Annotated, TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
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
from ..models import ConversationRequest, ConversationResponse
from ..utils.metrics import MetricsCollector
from ..services.conversation_db import ConversationService
from db_service.session import get_db
import os

if TYPE_CHECKING:
    from ..core.mvp_team_manager import MVPTeamManager
else:
    MVPTeamManager = Any

# Configure logging
logger = logging.getLogger(__name__)

# Create main router
router = APIRouter()

# Create specialized routers
chat_router = APIRouter(prefix="/chat", tags=["conversation"])
health_router = APIRouter(prefix="/health", tags=["monitoring"])


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
        ConversationService, Depends(get_conversation_service)
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
        
        # Get conversation context
        try:
            context = await conversation_manager.get_context(conversation_id)
            logger.debug(f"Retrieved context with {len(context.turns)} previous turns")
        except Exception as e:
            logger.warning(f"Failed to retrieve context for {conversation_id}: {e}")
            # Continue with empty context
            from ..models.conversation_models import ConversationContext
            context = ConversationContext(conversation_id=conversation_id, turns=[])
        
        # Process message through AutoGen team
        try:
            team_response = await team_manager.process_user_message(
                user_message=validated_request.message,
                user_id=user_id,
                conversation_id=conversation_id
            )

            logger.info(f"AutoGen team processed message successfully")
            
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
            team_response,
            int((time.time() - start_time) * 1000),
            metrics
        )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)

        try:
            conversation_service.add_turn(
                conversation,
                validated_request.message,
                team_response,
                processing_time,
            )
        except Exception as e:
            logger.error(f"Failed to store conversation turn: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store conversation turn",
            )

        # Create response
        response = ConversationResponse(
            message=team_response,
            conversation_id=conversation_id,
            success=True,
            processing_time_ms=processing_time,
            agent_used="orchestrator_agent",
            confidence=1.0,
            metadata={}
        )
        
        # Record success metrics
        metrics.record_response_time("chat", processing_time)
        metrics.record_success("chat")
        
        logger.info(f"Conversation processed successfully in {processing_time}ms")
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
                team_manager.get_health_status(), timeout=component_timeout
            )
            health_status["components"]["team_manager"] = {
                "status": "healthy",
                "agents_loaded": team_health.get("agents_count", 0),
                "last_activity": team_health.get("last_activity")
            }
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
    metrics: MetricsCollector
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
    """
    try:
        await conversation_manager.add_turn(
            conversation_id=conversation_id,
            user_id=user_id,
            user_msg=user_message,
            assistant_msg=assistant_message,
            processing_time_ms=float(processing_time_ms)
        )
        logger.debug(f"Stored conversation turn for {conversation_id}")
        
    except Exception as e:
        logger.error(f"Failed to store conversation turn: {e}")
        metrics.record_error("conversation_storage", str(e))


# Include routers in main router
router.include_router(chat_router)
router.include_router(health_router)