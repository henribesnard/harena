"""Models describing agent execution traces."""

from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentStep(BaseModel):
    """Single step executed by an agent."""

    agent: str
    status: str

    @field_validator("agent", "status")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v


class AgentTrace(BaseModel):
    """Trace of agent steps with execution time."""

    steps: List[AgentStep] = Field(default_factory=list)
    total_time_ms: float

    @field_validator("total_time_ms")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("total_time_ms must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_steps(self) -> "AgentTrace":
        if not self.steps:
            raise ValueError("steps cannot be empty")
        return self

