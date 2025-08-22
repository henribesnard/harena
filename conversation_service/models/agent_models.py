"""Pydantic models related to agent configuration and execution."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class AgentStep(BaseModel):
    """Single step executed by an agent."""

    agent: str = Field(..., min_length=1, description="Name of the agent")
    status: str = Field(..., min_length=1, description="Resulting status of the step")

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"agent": "retriever", "status": "ok"}},
    )

    @field_validator("agent", "status")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v

    def __init__(self, **data: Any) -> None:
        errors = []
        if not data.get("agent"):
            errors.append({"loc": ("agent",), "msg": "must not be empty", "type": "value_error"})
        if not data.get("status"):
            errors.append({"loc": ("status",), "msg": "must not be empty", "type": "value_error"})
        if errors:
            raise ValidationError(errors, type(self))
        super().__init__(**data)


class AgentTrace(BaseModel):
    """Trace of agent steps with execution time."""

    steps: list[AgentStep] = Field(..., description="Steps executed by the agent")
    total_time_ms: float = Field(
        ..., ge=0, description="Total execution time of all steps in milliseconds"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "steps": [{"agent": "retriever", "status": "ok"}],
                "total_time_ms": 12.5,
            }
        }
    )

    @model_validator(mode="after")
    def validate_steps(self) -> "AgentTrace":
        if not self.steps:
            raise ValueError("steps cannot be empty")
        return self

    def __init__(self, **data: Any) -> None:
        errors = []
        steps_raw = data.get("steps") or []
        converted_steps: list[AgentStep] = []
        for s in steps_raw:
            try:
                converted_steps.append(s if isinstance(s, AgentStep) else AgentStep(**s))
            except ValidationError as e:
                errors.extend(getattr(e, "args", [e]))
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

    model: str = Field(..., description="Name of the OpenAI model")
    temperature: float = Field(
        0.7, description="Model temperature", ge=0.0, le=1.0
    )
    max_tokens: int = Field(
        512, description="Maximum number of generated tokens", ge=1, lt=4000
    )
    timeout: int = Field(
        30, description="Maximum generation time in seconds", ge=1, le=60
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "gpt-4o-mini",
                "temperature": 0.5,
                "max_tokens": 1024,
                "timeout": 30,
            }
        }
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if not 1 <= v < 4000:
            raise ValueError("max_tokens must be between 1 and 3999")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if not 1 <= v <= 60:
            raise ValueError("timeout must be between 1 and 60 seconds")
        return v


class IntentResult(BaseModel):
    """Result of the intent classification."""

    intent_type: str = Field(..., description="Detected intent")
    confidence_score: float = Field(
        ..., description="Confidence score for the intent", ge=0.0, le=1.0
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent_type": "CHECK_BALANCE",
                "confidence_score": 0.94,
            }
        }
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return v


class DynamicFinancialEntity(BaseModel):
    """Financial entity extracted from a message."""

    entity_type: str = Field(..., description="Type of the entity")
    value: str = Field(..., description="Value associated with the entity")
    confidence_score: float = Field(
        ..., description="Confidence score for the entity", ge=0.0, le=1.0
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_type": "account_number",
                "value": "FR7612345678901234567890189",
                "confidence_score": 0.87,
            }
        }
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return v


class AgentResponse(BaseModel):
    """Full response returned by the chain of agents."""

    response: str = Field(..., description="Generated response text")
    intent: IntentResult = Field(..., description="Detected intent")
    entities: list[DynamicFinancialEntity] = Field(
        default_factory=list, description="Extracted financial entities"
    )
    confidence_score: float = Field(
        ..., description="Overall confidence score for the response", ge=0.0, le=1.0
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "Votre solde est de 50€.",
                "intent": {
                    "intent_type": "CHECK_BALANCE",
                    "confidence_score": 0.94,
                },
                "entities": [
                    {
                        "entity_type": "account_number",
                        "value": "FR7612345678901234567890189",
                        "confidence_score": 0.87,
                    }
                ],
                "confidence_score": 0.92,
            }
        }
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return v

