from __future__ import annotations

"""REST endpoints for conversation service."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
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
from conversation_service.repository import ConversationRepository
from teams.team_orchestrator import TeamOrchestrator

router = APIRouter(tags=["conversation"])

orchestrator = TeamOrchestrator()


@router.post("/start", response_model=ConversationStartResponse)
async def start_conversation(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> ConversationStartResponse:
    """Create a new conversation session for the authenticated user."""
    conv_id = orchestrator.start_conversation(current_user.id, db)
    return ConversationStartResponse(
        conversation_id=conv_id, created_at=datetime.utcnow()
    )


@router.get(
    "/{conversation_id}/history", response_model=ConversationHistoryResponse
)
async def get_history(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> ConversationHistoryResponse:
    """Return the message history for a conversation."""
    repo = ConversationRepository(db)
    conv = repo.get_by_conversation_id(conversation_id)
    if conv is None or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    history = orchestrator.get_history(conversation_id, db)
    if history is None:
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
    """Send a message to the agent team and return their response."""
    repo = ConversationRepository(db)
    conv = repo.get_by_conversation_id(conversation_id)
    if conv is None or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    reply = await orchestrator.query_agents(
        conversation_id, payload.message, current_user.id, db
    )
    return AgentQueryResponse(conversation_id=conversation_id, reply=reply)
