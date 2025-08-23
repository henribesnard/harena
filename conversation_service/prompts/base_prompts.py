"""Standard system prompts and response schemas for the conversation service."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BASE_SYSTEM_MESSAGES: List[str] = [
    "Adopt a warm, professional and concise tone.",
    "Always prioritise user safety and security.",
    (
        "When a JSON response is requested, produce valid JSON but allow "
        "optional fields to be omitted."
    ),
    (
        "If an error occurs or the request cannot be fulfilled, respond "
        "with a JSON object containing an 'error' field and optional 'code'."
    ),
]

ADVANCED_BEHAVIOUR: List[str] = [
    "Never provide financial advice.",
    (
        "Escalate the conversation to a human operator when the request "
        "cannot be resolved or falls outside your capabilities."
    ),
    "Reply in the user's language whenever possible.",
]


def build_system_prompt() -> str:
    """Return the complete system prompt combining all instructions."""
    return "\n".join(BASE_SYSTEM_MESSAGES + ADVANCED_BEHAVIOUR)


# ---------------------------------------------------------------------------
# Response patterns
# ---------------------------------------------------------------------------


class BaseResponse(BaseModel):
    """Generic successful response returned by an agent."""

    message: str = Field(..., description="Primary response text")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the response"
    )
    language: Optional[str] = Field(
        None, description="ISO language code used in the response"
    )
    extra: Optional[Dict[str, str]] = Field(
        default=None, description="Optional additional key/value pairs"
    )


class ErrorResponse(BaseModel):
    """Error response pattern returned when a request fails."""

    error: str = Field(..., description="Error description")
    code: Optional[str] = Field(
        None, description="Optional machine-readable error code"
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the error categorisation",
    )


__all__ = [
    "BASE_SYSTEM_MESSAGES",
    "ADVANCED_BEHAVIOUR",
    "build_system_prompt",
    "BaseResponse",
    "ErrorResponse",
]

