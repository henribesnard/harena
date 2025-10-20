"""Request models for conversation endpoints."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ConversationContext(BaseModel):
    """Context for a conversation request."""

    conversation_id: Optional[str] = Field(
        None,
        description="UUID of existing conversation to continue"
    )
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="User preferences for response"
    )

    @field_validator('preferences', mode='before')
    @classmethod
    def set_default_preferences(cls, v):
        """Set default preferences if not provided."""
        if v is None:
            return {
                "language": "fr",
                "include_visualization": True
            }
        return v


class ConversationRequest(BaseModel):
    """Request body for conversation endpoint."""

    query: Optional[str] = Field(
        None,
        min_length=3,
        max_length=500,
        description="User question in natural language",
        examples=["Combien j'ai dépensé en restaurants ce mois-ci ?"]
    )
    message: Optional[str] = Field(
        None,
        min_length=3,
        max_length=500,
        description="User message (alias for query for backward compatibility)"
    )
    message_type: Optional[str] = Field(None, description="Type of message")
    priority: Optional[str] = Field(None, description="Message priority")
    client_info: Optional[Dict[str, Any]] = Field(None, description="Client information")
    context: Optional[ConversationContext] = Field(
        default_factory=ConversationContext,
        description="Optional conversation context"
    )

    @field_validator('query', mode='before')
    @classmethod
    def validate_query(cls, v: Optional[str], info) -> Optional[str]:
        """Validate and clean the query, using message if query not provided."""
        # If query is not provided but message is, use message as query
        if v is None and info.data.get('message'):
            v = info.data.get('message')

        if v is None:
            raise ValueError("Either 'query' or 'message' field is required")

        # Strip whitespace
        v = v.strip()

        # Check minimum length
        if len(v) < 3:
            raise ValueError("Query must be at least 3 characters long")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Combien j'ai dépensé en restaurants ce mois-ci ?",
                "context": {
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "preferences": {
                        "language": "fr",
                        "include_visualization": True
                    }
                }
            }
        }
