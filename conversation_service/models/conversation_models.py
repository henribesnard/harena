"""Pydantic models for the conversation service API (phase 2)."""

"""Pydantic models for conversation requests and responses."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationMetadata(BaseModel):
    """Optional metadata associated with a conversation."""

    intent: str | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent": "greeting",
                "confidence_score": 0.95,
            }
        }
    )
    model_config = ConfigDict(json_schema_extra={
        "example": {"intent": "greeting", "confidence_score": 0.95}
    })


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
    })



class ConversationRequest(BaseModel):
    """User request sent to the conversation service."""

    message: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=2)
    context: ConversationContext

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

    response: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=2)
    context: ConversationContext
    metadata: ConversationMetadata | None = None

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
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
        },
    )

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

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if len(v) != 2 or not v.isalpha():
            raise ValueError("language must be a 2-letter ISO code")
        return v.lower()

