"""REST endpoints for conversation service."""

from __future__ import annotations


import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db_service.models.user import User
from db_service.session import get_db
from user_service.api.deps import get_current_active_user

from ..models.conversation_models import (
    AgentQueryRequest,
    AgentQueryResponse,
    ConversationHistoryResponse,
    ConversationStartResponse,
)
from conversation_service.service import ConversationService
from conversation_service.repository import ConversationRepository
from teams.team_orchestrator import TeamOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversation"])

orchestrator = TeamOrchestrator()
logger = logging.getLogger(__name__)


@router.post("/start", response_model=ConversationStartResponse)
async def start_conversation(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> ConversationStartResponse:
    """Create a new conversation session for the authenticated user.

    Args:
        current_user: Authenticated user associated with the request.
        db: Database session dependency.

    Returns:
        ConversationStartResponse containing the new conversation identifier.
    """
    conv_id = orchestrator.start_conversation(current_user.id, db)
    return ConversationStartResponse(
        conversation_id=conv_id, created_at=datetime.utcnow()
    )


@router.get("/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_history(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> ConversationHistoryResponse:
    """Return the message history for a conversation.

    Args:
        conversation_id: Identifier of the conversation to retrieve history for.
        current_user: Authenticated user associated with the request.
        db: Database session dependency.

    Returns:
        ConversationHistoryResponse containing the conversation history.

    Raises:
        HTTPException: If the conversation does not exist for the user.
    """
    repo = ConversationRepository(db)
    conv = repo.get_by_conversation_id(conversation_id)
    if conv is None or conv.user_id != current_user.id:
        logger.error(
            "Conversation not found",
            extra={"conversation_id": conversation_id, "user_id": current_user.id},
        )
    """Return the message history for a conversation."""
    service = ConversationService(db)
    if service.get_for_user(conversation_id, current_user.id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    history = orchestrator.get_history(conversation_id, db)
    if history is None:
        logger.error(
            "Conversation history not found",
            extra={"conversation_id": conversation_id, "user_id": current_user.id},
        )
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryResponse(
        conversation_id=conversation_id, messages=history
    )


@router.post("/{conversation_id}/query", response_model=AgentQueryResponse)
async def query_agents(
    conversation_id: str,
    payload: AgentQueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> AgentQueryResponse:
    """Send a message to the agent team and return their response.

    Args:
        conversation_id: Identifier of the conversation being queried.
        payload: Request payload containing the message to send.
        current_user: Authenticated user associated with the request.
        db: Database session dependency.

    Returns:
        AgentQueryResponse with the agent team's reply.

    Raises:
        HTTPException: If the conversation does not exist for the user.
    """
    repo = ConversationRepository(db)
    conv = repo.get_by_conversation_id(conversation_id)
    if conv is None or conv.user_id != current_user.id:
        logger.error(
            "Conversation not found",
            extra={"conversation_id": conversation_id, "user_id": current_user.id},
        )
    """Send a message to the agent team and return their response."""
    service = ConversationService(db)
    conv = service.get_for_user(conversation_id, current_user.id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        reply = await orchestrator.query_agents(
            conversation_id, payload.message, current_user.id, db
        )
        service.save_conversation_turn(
            conversation=conv,
            user_message=payload.message,
            assistant_response=reply,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.exception("Failed to process conversation turn", exc_info=exc)
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from exc
    return AgentQueryResponse(conversation_id=conversation_id, reply=reply)
