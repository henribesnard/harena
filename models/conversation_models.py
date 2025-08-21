from __future__ import annotations

"""Pydantic models for conversation API."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationStartRequest(BaseModel):
    """Request payload to start a conversation."""

    user_id: Optional[int] = Field(default=None, ge=0, description="Identifier of the user")


class ConversationStartResponse(BaseModel):
    """Response returned when a conversation is created."""

    conversation_id: str
    created_at: datetime


class AgentQueryRequest(BaseModel):
    """Request for querying the agent team."""

    message: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class AgentQueryResponse(BaseModel):
    """Response returned by the agent team."""

    conversation_id: str
    reply: str


class ConversationMessage(BaseModel):
    """Single message in the conversation history."""

    role: Literal["user", "assistant"]
    content: str


class ConversationHistoryResponse(BaseModel):
    """History of a conversation."""

    conversation_id: str
    messages: List[ConversationMessage]
