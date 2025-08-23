"""Pydantic models describing agent steps and responses."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator, ValidationError

from .enums import EntityType, IntentType


class AgentStep(BaseModel):
    """Single step executed by an agent with execution metadata."""

    agent_name: str = Field(..., min_length=1, description="Name of the agent")
    success: bool = Field(..., description="Whether the step executed successfully")
    error_message: str | None = Field(
        default=None, description="Error message if the step failed"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metrics such as duration, tokens or cost",
    )
    from_cache: bool = Field(
        False, description="Indicates if the response originated from cache"
    )
    reasoning_trace: str | None = Field(
        default=None, description="Optional reasoning trace returned by the agent"
    )

    @field_validator("agent_name")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value

    def __init__(self, **data: Any) -> None:  # type: ignore[override]
        super().__init__(**data)
        if not getattr(self, "agent_name", None):
            raise ValidationError("agent_name must not be empty")


class AgentTrace(BaseModel):
    """Trace of agent steps with total execution time."""

    steps: List[AgentStep] = Field(
        ..., min_length=1, description="Steps executed by the agent"
    )
    total_time_ms: float = Field(
        ..., ge=0, description="Total execution time of all steps in milliseconds"
    )


class AgentConfig(BaseModel):
    """Configuration for a conversational agent."""

    name: str = Field(..., min_length=1, description="Name of the agent")
    system_prompt: str = Field(
        ..., min_length=1, description="System prompt guiding the agent"
    )
    model: str = Field(..., description="Name of the model")
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1, lt=4000)
    timeout: int = Field(30, ge=1, le=60, description="Maximum generation time in seconds")


class IntentResult(BaseModel):
    """Result of the intent classification."""

    intent_type: IntentType = Field(..., description="Detected intent")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the intent"
    )


class DynamicFinancialEntity(BaseModel):
    """Financial entity extracted from a message."""

    entity_type: EntityType = Field(..., description="Type of the entity")
    value: str = Field(..., description="Value associated with the entity")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the entity"
    )


class AgentResponse(BaseModel):
    """Full response returned by the chain of agents."""

    response: str = Field(..., description="Generated response text")
    intent: IntentResult = Field(..., description="Detected intent")
    entities: List[DynamicFinancialEntity] = Field(
        default_factory=list, description="Extracted financial entities"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score for the response"
    )

