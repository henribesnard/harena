"""Pydantic models describing agent configuration, steps, and responses."""

from __future__ import annotations

from typing import Any, List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class AgentStep(BaseModel):
    """Single step executed by an agent."""

    agent: str
    status: str

    @model_validator(mode="after")
    def validate_fields(self) -> "AgentStep":
        if not self.agent:
            raise ValueError("agent must not be empty")
        if not self.status:
            raise ValueError("status must not be empty")
        return self


class AgentTrace(BaseModel):
    """Trace of agent steps with total execution time."""

    steps: List[AgentStep] = Field(default_factory=list)
    total_time_ms: float

    @field_validator("total_time_ms")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("total_time_ms must be non-negative")
        return v

    @model_validator(mode="before")
    def convert_steps(cls, values: Any) -> Any:
        steps = values.get("steps", [])
        values["steps"] = [s if isinstance(s, AgentStep) else AgentStep(**s) for s in steps]
        return values

    @model_validator(mode="after")
    def validate_steps(self) -> "AgentTrace":
        if not self.steps:
            raise ValueError("steps cannot be empty")
        return self


class AgentConfig(BaseModel):
    """Configuration for an agent model."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 30

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
    """Result of intent classification."""

    intent_type: str
    confidence_score: float

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
    """Financial entity extracted dynamically from a message."""

    entity_type: str
    value: str
    confidence_score: float

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
    """Full response returned by the agent chain."""

    response: str
    intent: IntentResult
    entities: List[DynamicFinancialEntity] = Field(default_factory=list)
    confidence_score: float

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
