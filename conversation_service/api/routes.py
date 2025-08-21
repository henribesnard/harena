from __future__ import annotations

"""REST endpoints for conversation service."""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from ..models.conversation_models import (
    AgentQueryRequest,
    AgentQueryResponse,
    ConversationHistoryResponse,
    ConversationStartRequest,
    ConversationStartResponse,
)
from teams.team_orchestrator import TeamOrchestrator

router = APIRouter(tags=["conversation"])

orchestrator = TeamOrchestrator()


@router.post("/start", response_model=ConversationStartResponse)
async def start_conversation(payload: ConversationStartRequest) -> ConversationStartResponse:
    """Create a new conversation session."""
    conv_id = orchestrator.start_conversation(payload.user_id)
    return ConversationStartResponse(conversation_id=conv_id, created_at=datetime.utcnow())


@router.get("/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_history(conversation_id: str) -> ConversationHistoryResponse:
    """Return the message history for a conversation."""
    history = orchestrator.get_history(conversation_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryResponse(conversation_id=conversation_id, messages=history)


@router.post("/{conversation_id}/query", response_model=AgentQueryResponse)
async def query_agents(conversation_id: str, payload: AgentQueryRequest) -> AgentQueryResponse:
    """Send a message to the agent team and return their response."""
    reply = await orchestrator.query_agents(conversation_id, payload.message)
    return AgentQueryResponse(conversation_id=conversation_id, reply=reply)
