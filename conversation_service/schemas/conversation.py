"""Pydantic models for conversation service.

These schemas mirror the ORM models defined in
``db_service.models.conversation`` and are used by the repository to
serialize database rows.

Example:
    >>> from conversation_service.schemas.conversation import ConversationCreate
    >>> ConversationCreate(user_id=123, title="My chat")
    ConversationCreate(user_id=123, title='My chat', language='fr', domain='financial', conversation_metadata={}, user_preferences={}, session_metadata={})
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict


class ConversationTurnCreate(BaseModel):
    """Schema for creating a conversation turn."""

    user_message: str
    assistant_response: str
    turn_metadata: Dict[str, object] = Field(default_factory=dict)


class ConversationTurn(ConversationTurnCreate):
    """Conversation turn stored in the database."""

    id: int
    turn_id: str
    conversation_id: int
    turn_number: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationCreate(BaseModel):
    """Schema used when creating a new conversation."""

    user_id: int
    title: Optional[str] = None
    language: str = "fr"
    domain: str = "financial"
    conversation_metadata: Dict[str, object] = Field(default_factory=dict)
    user_preferences: Dict[str, object] = Field(default_factory=dict)
    session_metadata: Dict[str, object] = Field(default_factory=dict)


class Conversation(ConversationCreate):
    """Conversation representation returned by the repository."""

    id: int
    conversation_id: str
    status: str
    total_turns: int
    max_turns: int
    last_activity_at: datetime
    created_at: datetime
    updated_at: datetime
    turns: List[ConversationTurn] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
