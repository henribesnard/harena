"""Pydantic models for conversation requests and responses."""

from __future__ import annotations

from uuid import UUID
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import IntentType
from .agent_models import DynamicFinancialEntity


class ConversationMetadata(BaseModel):
    """Optional metadata associated with a conversation."""

    intent: IntentType | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    agents_used: List[str] = Field(default_factory=list)
    search_performed: bool = False
    cache_stats: Dict[str, int] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    processing_times: Dict[str, float] = Field(default_factory=dict)
    extraction_mode: str = Field(..., min_length=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent": "greeting",
                "confidence_score": 0.95,
                "agents_used": ["intent_classifier", "entity_extractor"],
                "search_performed": False,
                "cache_stats": {"hits": 3, "misses": 1},
                "token_usage": {"prompt": 100, "completion": 20},
                "processing_times": {"classification": 12.5},
                "extraction_mode": "rule_based",
            }
        }
    )


    @field_validator("agents_used")
    @classmethod
    def _agents_not_empty(cls, v: List[str]) -> List[str]:
        if any(not agent for agent in v):
            raise ValueError("agent names must not be empty")
        return v

    @field_validator("cache_stats", "token_usage")
    @classmethod
    def _non_negative_ints(cls, v: Dict[str, int]) -> Dict[str, int]:
        if any(value < 0 for value in v.values()):
            raise ValueError("values must be non-negative")
        return v

    @field_validator("processing_times")
    @classmethod
    def _non_negative_floats(cls, v: Dict[str, float]) -> Dict[str, float]:
        if any(value < 0 for value in v.values()):
            raise ValueError("processing times must be non-negative")
        return v


class ConversationContext(BaseModel):
    """Context information provided with a request or response."""

    turn_number: int = Field(ge=1)
    recent_intents: List[IntentType] = Field(default_factory=list)
    previous_entities: List[DynamicFinancialEntity] = Field(default_factory=list)
    user_display_prefs: Dict[str, Any] = Field(default_factory=dict)
    session_state: Literal["new", "returning"] = "new"
    auto_summary: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "turn_number": 1,
                "recent_intents": ["greeting"],
                "previous_entities": [],
                "user_display_prefs": {"theme": "dark"},
                "session_state": "new",
                "auto_summary": None,
            }
        }
        json_schema_extra={"example": {"turn_number": 1}}
    )

    @field_validator("auto_summary")
    @classmethod
    def _auto_summary_not_empty(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("auto_summary must not be empty")
        return v


class ConversationRequest(BaseModel):
    """User request sent to the conversation service."""

    message: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=2)
    extraction_mode: Literal["strict", "flexible"] = "strict"
    conversation_id: UUID | None = None
    context: ConversationContext
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "message": "Bonjour",
                "language": "fr",
                "extraction_mode": "strict",
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "context": {"turn_number": 1},
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

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: UUID | None) -> UUID | None:
        if v is None:
            return v
        try:
            return UUID(str(v))
        except ValueError as e:
            raise ValueError("conversation_id must be a valid UUID") from e


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
                "context": {"turn_number": 1},
                "metadata": {
                    "intent": "greeting",
                    "confidence_score": 0.95,
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
