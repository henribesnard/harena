"""HTTP API routes for the conversation service.

Only a very small subset of the original project is implemented here.  The
endpoints are intentionally lightweight so that they can be exercised in tests
without requiring the full production stack.
"""

import asyncio
import logging
import time
import hashlib
from typing import Annotated, Any, Dict, List, Optional, Protocol
from datetime import datetime
from typing import Any, Dict, List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, HTTPException

from .dependencies import (
    get_team_manager,
    get_conversation_manager,

    get_current_user,
    get_metrics_collector,
    get_conversation_repository,
    get_conversation_service,
    get_conversation_read_service,
    validate_conversation_request,
    validate_request_rate_limit,
)
from .websocket import ws_router
from ..core.conversation_manager import ConversationManager

from ..models.conversation_models import (
    ConversationRequest,
    ConversationResponse,
    ConversationOut,
)
from ..models.financial_models import IntentResult
import os

from ..utils.logging import log_unauthorized_access
from ..repositories.conversation_repository import ConversationRepository
from ..core.metrics_collector import MetricsCollector
from ..utils.metrics import MetricsCollector
from db_service.session import get_db
from ..utils.cache_client import cache_client
from ..utils.decorators import rate_limit

try:
    from ..core.mvp_team_manager import MVPTeamManager
except ImportError:
    class MVPTeamManager(Protocol):
        async def process_user_message(
            self, user_message: str, user_id: int, conversation_id: str
        ) -> Any:
            ...

router = APIRouter()



@chat_router.websocket("/ws")
async def chat_websocket(
    websocket: WebSocket,
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
):
    """Simple WebSocket endpoint forwarding messages to the team manager."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = await team_manager.process_user_message_with_metadata(
                user_message=data, user_id=0, conversation_id="websocket"
            )
            await websocket.send_text(result["content"])
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")


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
@rate_limit(calls=5, period=60)
@router.post("/chat", response_model=ConversationResponse)
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
    conversation_manager: Annotated[ConversationManager, Depends(get_conversation_manager)],
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics_collector)],
    conversation_repo: Annotated[
        ConversationRepository, Depends(get_conversation_repository)
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
        user: Authenticated user context derived from JWT claims
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
        conversation = conversation_repo.get_or_create_conversation(
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

    cache_key = f"chat:{user_id}:{hashlib.md5(validated_request.message.encode()).hexdigest()}"
    cached_response = await cache_client.get(cache_key)
    if cached_response:
        return ConversationResponse(**cached_response)

    await cache_client.set(f"prompt:{user_id}", validated_request.message, ttl=300)
    
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
        context = await conversation_manager.get_context(conversation_id, user_id)
        context_user_id = getattr(context, "user_id", None)
        if context_user_id not in (None, user_id):
            log_unauthorized_access(
                user_id=user_id,
                conversation_id=conversation_id,
                reason="forbidden conversation access",
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        if context_user_id != user_id:
            context.user_id = user_id
            await conversation_manager.store.save_context(context)
        logger.debug(f"Retrieved context with {len(context.turns)} previous turns")

        # Process message through AutoGen team
        try:
            team_result = await team_manager.process_user_message_with_metadata(
                user_message=validated_request.message,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if not team_result["success"]:
                metrics.record_error(
                    "team_processing", team_result.get("error_message") or "unknown_error"
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
            team_result["content"],
            int((time.time() - start_time) * 1000),
            metrics,
            intent_result=team_result["metadata"].get("intent_result"),
            agent_chain=team_result["metadata"].get("agent_chain"),
            search_results_count=team_result["metadata"].get("search_results_count"),
            confidence_score=team_result.get("confidence_score"),
        )

        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)

        try:
            conversation_repo.add_turn(
                conversation_id=conversation.conversation_id,
                user_id=user_id,
                user_message=validated_request.message,
                assistant_response=team_result["content"],
                processing_time_ms=processing_time,
                intent_result=team_result["metadata"].get("intent_result"),
                confidence_score=team_result.get("confidence_score"),
                agent_chain=team_result["metadata"].get("agent_chain"),
                search_results_count=team_result["metadata"].get(
                    "search_results_count", 0
                ),
                search_execution_time_ms=team_result["metadata"].get(
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
            message=team_result["content"],
            conversation_id=conversation_id,
            success=team_result["success"],
            processing_time_ms=processing_time,
            agent_used="orchestrator_agent",
            confidence=team_result.get("confidence_score") or 0.0,
            metadata=team_result["metadata"],
            error_code=None if team_result["success"] else "TEAM_PROCESSING_ERROR",
        )

        await cache_client.set(cache_key, response.dict(), ttl=300)

        # Record metrics based on success
        if team_result["success"]:
            metrics.record_response_time("chat", processing_time)
            metrics.record_success("chat")
            logger.info(f"Conversation processed successfully in {processing_time}ms")
        else:
            metrics.record_error("chat", team_result.get("error_message") or "unknown_error")
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
@rate_limit(calls=10, period=60)
async def health_check(
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
    conversation_manager: Annotated[ConversationManager, Depends(get_conversation_manager)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics_collector)]
) -> Dict[str, Any]:
    """
    Comprehensive health check for all service components.
    validated_request: ConversationRequest = Depends(validate_conversation_request),
    team_manager: Any = Depends(get_team_manager),
    conversation_manager: Any = Depends(get_conversation_manager),
    user: Dict[str, Any] = Depends(get_current_user),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    conversation_service: Any = Depends(get_conversation_service),
    _: None = Depends(validate_request_rate_limit),
) -> ConversationResponse:
    """Process a chat message through the team manager and store the result."""
    metrics.record_request("chat", user.get("user_id", 0))
    start = datetime.utcnow()
    result = await team_manager.process_user_message_with_metadata(
        validated_request.message,
        user.get("user_id", 0),
        validated_request.conversation_id,
    )
    processing_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    metrics.record_response_time("chat", processing_ms)
    metrics.record_success("chat")

    # Ensure a conversation exists
    conv = conversation_service.get_or_create_conversation(
        user.get("user_id", 0), validated_request.conversation_id
    )
    conversation_id = getattr(conv, "conversation_id", validated_request.conversation_id)

    # Persist turn via conversation manager and service (best effort)
    try:
        await conversation_manager.add_turn(
            conversation_id=conversation_id,
            user_id=user.get("user_id", 0),
            user_msg=validated_request.message,
            assistant_msg=result.get("content", ""),
            processing_time_ms=processing_ms,
        )
        conversation_service.add_turn(
            conversation_id=conversation_id,
            user_id=user.get("user_id", 0),
            user_message=validated_request.message,
            assistant_response=result.get("content", ""),
            processing_time_ms=processing_ms,
        )
    except Exception:
        # Failure to persist should not break the response
        metrics.record_error("chat_storage", "persist_failed")

    return ConversationResponse(
        message=result.get("content", ""),
        conversation_id=conversation_id,
        processing_time_ms=processing_ms,
        agent_used=result.get("agent_used"),
        confidence=result.get("confidence_score"),
        metadata=result.get("metadata", {}),
        success=result.get("success", True),
    )


@router.get("/conversations", response_model=List[ConversationOut])
async def list_conversations(
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> List[ConversationOut]:
    """List conversations belonging to the current user."""
    conversations = repo.list_conversations(user["user_id"])[offset : offset + limit]
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
    response_model=ConversationTurnsResponse,
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
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    limit: int = Query(10, ge=1, le=50),
) -> ConversationTurnsResponse:
    """Return the turns for a specific conversation."""
    conversation = repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    if conversation.user_id != user["user_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

    turns_raw = repo.get_conversation_turns(conversation_id)

    turns: List[ConversationTurn] = []
    for t in turns_raw[:limit]:
        turns.append(
            ConversationTurn(
                turn_id=t.turn_id,
                user_message=t.user_message,
                assistant_response=t.assistant_response,
                timestamp=t.created_at,
                metadata=t.turn_metadata or {},
                turn_number=t.turn_number,
                processing_time_ms=t.processing_time_ms or 0.0,
                intent_result=t.intent_result,
                confidence_score=t.confidence_score,
                error_occurred=t.error_occurred,
                agent_chain=t.agent_chain,

    user: Dict[str, Any] = Depends(get_current_user),
    service: Any = Depends(get_conversation_read_service),
) -> List[ConversationOut]:
    """Return conversations for the current user."""
    conversations = service.get_conversations(user.get("user_id", 0))
    results: List[ConversationOut] = []
    for c in conversations:
        last_activity = getattr(c, "last_activity_at", datetime.utcnow())
        if isinstance(last_activity, str):
            try:
                last_activity = datetime.fromisoformat(last_activity)
            except ValueError:
                last_activity = datetime.utcnow()
        results.append(
            ConversationOut(
                conversation_id=getattr(c, "conversation_id", ""),
                title=getattr(c, "title", None),
                status=getattr(c, "status", "active"),
                total_turns=getattr(c, "total_turns", 0),
                last_activity_at=last_activity,
            )
        )
    return results


@router.get(
    "/status",
    summary="Service status information",
    description="Basic service status and configuration",
    tags=["monitoring"]
)
@rate_limit(calls=10, period=60)
async def get_status() -> Dict[str, Any]:
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
    intent_result: Optional[Dict[str, Any]] = None,
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
        intent_result: Result of intent detection
        agent_chain: Chain of agents involved in processing
        search_results_count: Number of results returned by search
        confidence_score: Confidence score of the response
    """
    try:
        intent_obj = IntentResult(**intent_result) if intent_result else None
        await conversation_manager.add_turn(
            conversation_id=conversation_id,
            user_id=user_id,
            user_msg=user_message,
            assistant_msg=assistant_message,
            processing_time_ms=float(processing_time_ms),
            intent_result=intent_obj,
            agent_chain=agent_chain,
            search_results_count=search_results_count,
            confidence_score=confidence_score,
        )
        logger.debug(f"Stored conversation turn for {conversation_id}")
        
    except Exception as e:
        logger.error(f"Failed to store conversation turn: {e}")
        metrics.record_error("conversation_storage", str(e))
@router.get("/health")
async def healthcheck() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok"}


# Include routers in main router
router.include_router(chat_router)
router.include_router(health_router)
router.include_router(conversations_router)
router.include_router(ws_router)
@router.get("/metrics")
async def metrics_endpoint(metrics: MetricsCollector = Depends(get_metrics_collector)) -> Dict[str, Any]:
    """Expose collected metrics."""
    return metrics.get_summary()
