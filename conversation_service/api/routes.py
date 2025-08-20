"""HTTP API routes for the conversation service.

Only a very small subset of the original project is implemented here.  The
endpoints are intentionally lightweight so that they can be exercised in tests
without requiring the full production stack.
"""

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import (
    get_team_manager,
    get_conversation_manager,
    get_current_user,
    get_metrics_collector,
    get_conversation_service,
    get_conversation_read_service,
    validate_conversation_request,
    validate_request_rate_limit,
)
from ..models.conversation_models import (
    ConversationRequest,
    ConversationResponse,
    ConversationOut,
)
from ..utils.metrics import MetricsCollector

router = APIRouter()


@router.post("/chat", response_model=ConversationResponse)
async def chat_endpoint(
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


@router.get("/metrics")
async def metrics_endpoint(metrics: MetricsCollector = Depends(get_metrics_collector)) -> Dict[str, Any]:
    """Expose collected metrics."""
    return metrics.get_summary()
