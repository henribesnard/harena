"""Pydantic models describing agent steps and responses."""

from __future__ import annotations

from typing import Any, Dict, List
from typing import List, Dict, Any
from typing import List, Literal

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
    model: Literal["gpt-4o-mini", "gpt-4o"] = Field(
        ..., description="Name of the model"
    )
    few_shot_examples: List[List[str]] = Field(
        default_factory=list,
        description="Few-shot examples; each item is a pair [prompt, completion]",
    )
    cache_ttl: int = Field(
        60, gt=0, description="Time-to-live for cached responses in seconds"
    )
    cache_strategy: Literal["memory", "redis"] = Field(
        "memory", description="Caching backend strategy"
    )

    @field_validator("few_shot_examples")
    @classmethod
    def _validate_examples(cls, v: List[List[str]]) -> List[List[str]]:
        for example in v:
            if not isinstance(example, list) or len(example) != 2:
                raise ValueError("each example must contain exactly two strings")
            if not all(isinstance(item, str) and item for item in example):
                raise ValueError("few-shot example items must be non-empty strings")
        return v
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
    raw_value: str = Field(
        ..., description="Original value extracted from the text"
    )
    normalized_value: str | None = Field(
        default=None, description="Normalized value, if available"
    )
    context: str | None = Field(
        default=None, description="Source sentence for the entity"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the extraction"
    )
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
    reasoning: str | None = Field(
        None, description="Reasoning provided by the agent for the response"
    )
    latency_ms: float | None = Field(
        None, ge=0, description="Time taken to produce the response in milliseconds"
    )
    suggested_actions: List[str] = Field(
        default_factory=list, description="Suggested follow-up actions"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score for the response"
    )

