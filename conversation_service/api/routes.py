"""HTTP API routes for the conversation service.

Only a very small subset of the original project is implemented here.  The
endpoints are intentionally lightweight so that they can be exercised in tests
without requiring the full production stack.
"""

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
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

router = APIRouter()


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
