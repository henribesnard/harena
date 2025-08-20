"""API routes for the conversation service.

This module exposes a single :class:`APIRouter` instance that handles both
REST and WebSocket endpoints.  The implementation is intentionally lightweight
so that it can be exercised in tests without requiring the full production
stack.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Protocol

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
from sqlalchemy.orm import Session

from .dependencies import (
    get_team_manager,
    get_conversation_manager,
    get_current_user,
    get_metrics_collector,
    get_conversation_repository,
    get_conversation_read_service,
    validate_conversation_request,
    validate_request_rate_limit,
)
from ..core.conversation_manager import ConversationManager
from ..repositories.conversation_repository import ConversationRepository
from ..models.conversation_models import (
    ConversationRequest,
    ConversationResponse,
    ConversationOut,
)
from ..models.financial_models import IntentResult
from ..core.metrics_collector import MetricsCollector
from ..utils.logging import log_unauthorized_access
from ..utils.cache_client import cache_client
from ..utils.decorators import rate_limit
from db_service.session import get_db

logger = logging.getLogger(__name__)

try:  # pragma: no cover - used only when the real implementation is present
    from ..core.mvp_team_manager import MVPTeamManager
except ImportError:  # pragma: no cover - fallback protocol for tests
    class MVPTeamManager(Protocol):
        async def process_user_message_with_metadata(
            self, user_message: str, user_id: int, conversation_id: str
        ) -> Dict[str, Any]:
            ...


router = APIRouter()


@router.websocket("/ws")
async def chat_websocket(
    websocket: WebSocket,
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
) -> None:
    """Simple WebSocket endpoint forwarding messages to the team manager."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = await team_manager.process_user_message_with_metadata(
                user_message=data, user_id=0, conversation_id="websocket"
            )
            await websocket.send_text(result.get("content", ""))
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")


@router.post("/chat", response_model=ConversationResponse)
@rate_limit(calls=5, period=60)
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
    conversation_manager: Annotated[
        ConversationManager, Depends(get_conversation_manager)
    ],
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics_collector)],
    conversation_repo: Annotated[
        ConversationRepository, Depends(get_conversation_repository)
    ],
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[None, Depends(validate_request_rate_limit)],
    validated_request: Annotated[
        ConversationRequest, Depends(validate_conversation_request)
    ],
) -> ConversationResponse:
    """Process a conversation message through AutoGen multi-agent team."""
    start_time = time.time()
    user_id = user["user_id"]
    conversation_id = validated_request.conversation_id

    logger.info(
        f"Processing conversation for user {user_id}, conversation {conversation_id}"
    )

    try:
        conversation = conversation_repo.get_or_create_conversation(
            user_id, conversation_id
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conversation access denied",
        )
    except Exception as e:  # pragma: no cover - safety net
        logger.error(f"Conversation retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database access error",
        )

    conversation_id = conversation.conversation_id
    cache_key = f"chat:{user_id}:{hashlib.md5(validated_request.message.encode()).hexdigest()}"
    cached_response = await cache_client.get(cache_key)
    if cached_response:
        return ConversationResponse(**cached_response)

    await cache_client.set(f"prompt:{user_id}", validated_request.message, ttl=300)

    metrics.record_request("chat", user_id)

    if "chat:write" not in user.get("permissions", []):
        log_unauthorized_access(
            user_id=user_id,
            conversation_id=conversation_id,
            reason="insufficient permissions",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

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

    try:
        result = await team_manager.process_user_message_with_metadata(
            validated_request.message, user_id, conversation_id
        )
        assistant_message = result.get("content", "")
        metadata = result.get("metadata", {})
        processing_ms = int((time.time() - start_time) * 1000)

        background_tasks.add_task(
            store_conversation_turn,
            conversation_manager,
            conversation_id,
            user_id,
            validated_request.message,
            assistant_message,
            metadata.get("intent_result"),
            processing_ms,
            metadata.get("agent_chain"),
            metadata.get("search_results_count"),
            result.get("confidence_score"),
        )

        metrics.record_response_time("chat", processing_ms)
        metrics.record_success("chat")

        response = ConversationResponse(
            message=assistant_message,
            conversation_id=conversation_id,
            processing_time_ms=processing_ms,
            agent_used=metadata.get("agent_used"),
            confidence=result.get("confidence_score"),
            metadata=metadata,
            success=result.get("success", True),
        )
        await cache_client.set(cache_key, response.model_dump(), ttl=60)
        return response
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - safety net
        processing_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error in chat endpoint: {e}")
        metrics.record_error("chat_endpoint", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal conversation processing error",
        )


async def store_conversation_turn(
    conversation_manager: ConversationManager,
    conversation_id: str,
    user_id: int,
    user_message: str,
    assistant_message: str,
    intent_result: Optional[Dict[str, Any]],
    processing_time_ms: int,
    agent_chain: Optional[List[str]],
    search_results_count: Optional[int],
    confidence_score: Optional[float],
) -> None:
    """Persist a conversation turn asynchronously."""
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
    except Exception as e:  # pragma: no cover - logging only
        logger.error(f"Failed to store conversation turn: {e}")


@router.get("/conversations", response_model=List[ConversationOut])
async def list_conversations(
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
    service: Annotated[Any, Depends(get_conversation_read_service)],
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
async def metrics_endpoint(
    metrics: Annotated[MetricsCollector, Depends(get_metrics_collector)]
) -> Dict[str, Any]:
    """Expose collected metrics."""
    return metrics.get_summary()

