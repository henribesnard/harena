"""Pydantic models describing agent steps and responses."""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field, ValidationError


class AgentStep(BaseModel):
    """Single step executed by an agent."""

    agent: str = Field(..., min_length=1, description="Name of the agent")
    status: str = Field(..., min_length=1, description="Resulting status of the step")

    def __init__(self, **data: Any) -> None:  # noqa: D401
        errors = []
        if not data.get("agent"):
            errors.append({"loc": ("agent",), "msg": "must not be empty", "type": "value_error"})
        if not data.get("status"):
            errors.append({"loc": ("status",), "msg": "must not be empty", "type": "value_error"})
        if errors:
            raise ValidationError(errors, type(self))
        super().__init__(**data)


class AgentTrace(BaseModel):
    """Trace of agent steps with total execution time."""

    steps: List[AgentStep] = Field(..., description="Steps executed by the agent")
    total_time_ms: float = Field(
        ..., ge=0, description="Total execution time of all steps in milliseconds"
    )

    def __init__(self, **data: Any) -> None:  # noqa: D401
        errors = []
        steps_raw = data.get("steps") or []
        converted_steps: List[AgentStep] = []
        for s in steps_raw:
            try:
                converted_steps.append(s if isinstance(s, AgentStep) else AgentStep(**s))
            except ValidationError as e:  # pragma: no cover - forwarded errors
                errors.extend(getattr(e, "errors", [e]))
        if not converted_steps:
            errors.append({"loc": ("steps",), "msg": "steps cannot be empty", "type": "value_error"})
        data["steps"] = converted_steps
        total = data.get("total_time_ms")
        if total is not None and total < 0:
            errors.append({
                "loc": ("total_time_ms",),
                "msg": "total_time_ms must be non-negative",
                "type": "value_error",
            })
        if errors:
            raise ValidationError(errors, type(self))
        super().__init__(**data)


class AgentConfig(BaseModel):
    """Configuration for a conversational agent."""

    model: str = Field(..., description="Name of the model")
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1, lt=4000)
    timeout: int = Field(30, ge=1, le=60, description="Maximum generation time in seconds")


class IntentResult(BaseModel):
    """Result of the intent classification."""

    intent_type: str = Field(..., description="Detected intent")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the intent"
    )


class DynamicFinancialEntity(BaseModel):
    """Financial entity extracted from a message."""

    entity_type: str = Field(..., description="Type of the entity")
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

    def __init__(self, **data: Any) -> None:  # noqa: D401
        errors = []
        score = data.get("confidence_score")
        if score is not None and not 0.0 <= score <= 1.0:
            errors.append({
                "loc": ("confidence_score",),
                "msg": "confidence_score must be between 0.0 and 1.0",
                "type": "value_error",
            })
        if errors:
            raise ValidationError(errors, type(self))
        super().__init__(**data)

