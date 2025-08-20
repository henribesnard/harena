"""Service contract models composed of lower level models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .agent_models import AgentConfig, AgentResponse
from .conversation_models import ConversationRequest, ConversationResponse, ConversationTurn
from .financial_models import Transaction


class AgentTransaction(BaseModel):
    """Contract linking an agent configuration to a financial transaction."""

    agent: AgentConfig
    transaction: Transaction


class ConversationSession(BaseModel):
    """Contract representing a completed conversation turn."""

    turn: ConversationTurn
    agent_response: Optional[AgentResponse] = Field(
        default=None,
        description="Raw response from the agent if available",
    )


class ServiceRequest(BaseModel):
    """General contract for external services interacting with the system."""

    request: ConversationRequest
    config: AgentConfig


class ServiceResponse(BaseModel):
    """General response wrapper containing conversation and transaction info."""

    response: ConversationResponse
    transaction: Optional[Transaction] = None


__all__ = [
    "AgentTransaction",
    "ConversationSession",
    "ServiceRequest",
    "ServiceResponse",
]
