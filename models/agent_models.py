"""Agent related data models."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .enums import AgentType
from core.validators import non_empty_str, positive_number, percentage


class AgentConfig(BaseModel):
    """Configuration for a conversational agent."""

    name: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., description="LLM model identifier")
    agent_type: AgentType = Field(default=AgentType.ASSISTANT)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(512, gt=0)

    @field_validator("name", "model")
    @classmethod
    def _ensure_non_empty(cls, v: str) -> str:
        return non_empty_str(v)

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        return percentage(v, maximum=2.0)

    @field_validator("max_tokens")
    @classmethod
    def _check_tokens(cls, v: int) -> int:
        return int(positive_number(v))


class AgentResponse(BaseModel):
    """Standard agent response envelope."""

    agent_name: str = Field(...)
    content: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(default=True)
    error_message: Optional[str] = None

    @field_validator("agent_name", "content")
    @classmethod
    def _ensure_content(cls, v: str) -> str:
        return non_empty_str(v)


class TeamWorkflow(BaseModel):
    """Ordered list of agents participating in a workflow."""

    agents: List[AgentConfig] = Field(..., min_length=1)
    description: Optional[str] = Field(default=None, max_length=500)

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return non_empty_str(v)
        return v


__all__ = ["AgentConfig", "AgentResponse", "TeamWorkflow"]
