"""Pydantic models for conversation requests and responses."""

from __future__ import annotations

from uuid import UUID
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import IntentType
from .agent_models import DynamicFinancialEntity


class ConversationContext(BaseModel):
    """Context information provided with a request or response."""

    conversation_id: UUID | None = None
    turn_number: int = Field(ge=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "turn_number": 1,
            }
        }
    )


class ConversationRequest(BaseModel):
    """User request sent to the conversation service."""

    message: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=2)
    context: ConversationContext
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "message": "Bonjour",
                "language": "fr",
                "context": {
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "turn_number": 1,
                },
                "user_preferences": {"tone": "friendly"},
            }
        },
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if len(v) != 2 or not v.isalpha():
            raise ValueError("language must be a 2-letter ISO code")
        return v.lower()


class ConversationResponse(BaseModel):
    """Response returned by the conversation service."""

    original_message: str = Field(min_length=1, description="Original user message")
    response: str = Field(min_length=1, description="Generated response")
    intent: IntentType = Field(..., description="Detected intent for the message")
    entities: List[DynamicFinancialEntity] = Field(
        default_factory=list, description="Extracted entities"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the intent"
    )
    language: str = Field(min_length=2, max_length=2)
    context: ConversationContext
    suggested_actions: List[str] = Field(
        default_factory=list, description="Recommended follow-up actions"
    )
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "original_message": "Bonjour",
                "response": "Bonjour! Comment puis-je vous aider?",
                "intent": "GREETING",
                "entities": [],
                "confidence_score": 0.95,
                "language": "fr",
                "context": {
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "turn_number": 1,
                },
                "suggested_actions": ["check_balance"],
                "user_preferences": {"tone": "friendly"},
            }
        },
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if len(v) != 2 or not v.isalpha():
            raise ValueError("language must be a 2-letter ISO code")
        return v.lower()
