from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ConversationMetadata(BaseModel):
    """Metadata about the conversation such as intent and confidence."""

    intent: str | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "intent": "greeting",
            "confidence_score": 0.95,
        }
    })

    @field_validator("user_id")
    @classmethod
    def user_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("user_id must be positive")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "Conversation":
        if self.total_turns > self.max_turns:
            raise ValueError("total_turns cannot exceed max_turns")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be after created_at")
        if self.last_activity_at < self.created_at:
            raise ValueError("last_activity_at must be after created_at")
        return self


class ConversationContext(BaseModel):
    """Contextual information for the conversation."""

    conversation_id: UUID | None = None
    turn_number: int = Field(ge=1)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
            "turn_number": 1,
        }
    })

    @field_validator("conversation_id", "start_turn", "end_turn")
    @classmethod
    def positive_numbers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @model_validator(mode="after")
    def validate_turns(self) -> "ConversationSummary":
        if self.end_turn < self.start_turn:
            raise ValueError("end_turn must be >= start_turn")
        return self


class ConversationRequest(BaseModel):
    """User request sent to the conversation service."""

    message: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=2)
    context: ConversationContext

    model_config = ConfigDict(str_strip_whitespace=True, json_schema_extra={
        "example": {
            "message": "Bonjour",
            "language": "fr",
            "context": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "turn_number": 1,
            },
        }
    })

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if len(v) != 2 or not v.isalpha():
            raise ValueError("language must be a 2-letter ISO code")
        return v.lower()


class ConversationResponse(BaseModel):
    """Assistant response returned by the conversation service."""

    response: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=2)
    context: ConversationContext
    metadata: ConversationMetadata | None = None

    model_config = ConfigDict(str_strip_whitespace=True, json_schema_extra={
        "example": {
            "response": "Bonjour! Comment puis-je vous aider?",
            "language": "fr",
            "context": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "turn_number": 1,
            },
            "metadata": {
                "intent": "greeting",
                "confidence_score": 0.95,
            },
        }
    })

    @field_validator("conversation_id", "turn_number", "search_results_count")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("processing_time_ms", "search_execution_time_ms")
    @classmethod
    def positive_floats(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "ConversationTurn":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be after created_at")
        return self
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if len(v) != 2 or not v.isalpha():
            raise ValueError("language must be a 2-letter ISO code")
        return v.lower()
