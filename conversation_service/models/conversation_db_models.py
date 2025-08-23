"""Pydantic models representing conversation data persisted in the database."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class Conversation(BaseModel):
    """Represents a persisted conversation."""

    id: int
    conversation_id: str
    user_id: int
    title: str | None = None
    status: str
    language: str
    domain: str
    total_turns: int = Field(ge=0)
    max_turns: int = Field(ge=0)
    last_activity_at: datetime
    conversation_metadata: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    intents: List[Dict[str, Any]] | None = None
    entities: List[Dict[str, Any]] | None = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    financial_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences_ai: Dict[str, Any] = Field(default_factory=dict)
    key_entities_history: List[Dict[str, Any]] = Field(default_factory=list)
    intent_classification: Dict[str, Any] = Field(default_factory=dict)
    entities_extracted: List[Dict[str, Any]] = Field(default_factory=list)
    intent_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    total_tokens_used: int = Field(default=0, ge=0)
    openai_usage_stats: Dict[str, Any] = Field(default_factory=dict)
    openai_cost_usd: float = Field(default=0.0, ge=0.0)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "conversation_id": "conv_123",
                "user_id": 42,
                "title": "Budget 2024",
                "status": "active",
                "language": "fr",
                "domain": "finance",
                "total_turns": 2,
                "max_turns": 50,
                "last_activity_at": "2024-06-01T12:00:00Z",
                "conversation_metadata": {"topic": "budget"},
                "user_preferences": {"tone": "friendly"},
                "session_metadata": {"browser": "firefox"},
                "financial_context": {"balance": 1000},
                "user_preferences_ai": {"style": "formal"},
                "key_entities_history": [{"name": "Compte", "type": "bank_account"}],
                "intent_classification": {"intent": "check_balance"},
                "entities_extracted": [{"name": "solde", "type": "financial"}],
                "intent_confidence": 0.95,
                "total_tokens_used": 200,
                "openai_usage_stats": {"prompt_tokens": 150, "completion_tokens": 50},
                "openai_cost_usd": 0.04,
                "created_at": "2024-06-01T12:00:00Z",
                "updated_at": "2024-06-01T12:05:00Z",
            }
        }
    )


    @field_validator("id", "user_id")
    @classmethod
    def id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @field_validator("total_tokens_used")
    @classmethod
    def non_negative_tokens(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("intent_confidence", "openai_cost_usd")
    @classmethod
    def non_negative_floats(
        cls, v: Optional[float]
    ) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "Conversation":
        if self.total_turns > self.max_turns:
            raise ValueError("total_turns cannot exceed max_turns")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be after created_at")
        if self.last_activity_at < self.created_at:
            raise ValueError("last_activity_at must be after created_at")
        return self


class ConversationSummary(BaseModel):
    """Partial summary of a conversation."""

    id: int
    conversation_id: int
    start_turn: int = Field(ge=1)
    end_turn: int = Field(ge=1)
    summary_text: str
    key_topics: List[str] = Field(default_factory=list)
    important_entities: List[Dict[str, Any]] = Field(default_factory=list)
    summary_method: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "conversation_id": 1,
                "start_turn": 1,
                "end_turn": 4,
                "summary_text": "Résumé des échanges...",
                "key_topics": ["budget", "économie"],
                "important_entities": [{"name": "Paris", "type": "city"}],
                "summary_method": "llm",
                "created_at": "2024-06-01T12:00:00Z",
                "updated_at": "2024-06-01T12:05:00Z",
            }
        }
    )


    @field_validator("id", "conversation_id", "start_turn", "end_turn")
    @classmethod
    def positive_numbers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @model_validator(mode="after")
    def validate_turns(self) -> "ConversationSummary":
        if self.end_turn < self.start_turn:
            raise ValueError("end_turn must be >= start_turn")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be after created_at")
        return self


class ConversationTurn(BaseModel):
    """Single turn exchanged in a conversation."""

    id: int
    turn_id: str
    conversation_id: int
    turn_number: int = Field(ge=1)
    user_message: str = Field(min_length=1)
    assistant_response: str = Field(min_length=1)
    processing_time_ms: Optional[float] = Field(default=None, ge=0.0)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    error_occurred: bool
    error_message: Optional[str] = None
    intent_result: Optional[Dict[str, Any]] = None
    agent_chain: List[Dict[str, Any]] = Field(default_factory=list)
    intent: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    financial_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences_ai: Dict[str, Any] = Field(default_factory=dict)
    key_entities_history: List[Dict[str, Any]] = Field(default_factory=list)
    intent_classification: Dict[str, Any] = Field(default_factory=dict)
    entities_extracted: List[Dict[str, Any]] = Field(default_factory=list)
    intent_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    total_tokens_used: int = Field(default=0, ge=0)
    openai_usage_stats: Dict[str, Any] = Field(default_factory=dict)
    openai_cost_usd: float = Field(default=0.0, ge=0.0)
    search_query_used: Optional[str] = None
    search_results_count: int = Field(ge=0)
    search_execution_time_ms: Optional[float] = Field(default=None, ge=0.0)
    turn_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 10,
                "turn_id": "turn_0010",
                "conversation_id": 1,
                "turn_number": 3,
                "user_message": "Quel est mon solde ?",
                "assistant_response": "Votre solde est de 50€.",
                "processing_time_ms": 120.5,
                "confidence_score": 0.98,
                "error_occurred": False,
                "error_message": None,
                "intent_result": {"name": "check_balance"},
                "agent_chain": [
                    {
                        "agent_name": "retrieval",
                        "success": True,
                        "error_message": None,
                        "metrics": {},
                        "from_cache": False,
                        "reasoning_trace": None,
                    }
                ],
                "financial_context": {"balance": 50},
                "user_preferences_ai": {"style": "casual"},
                "key_entities_history": [{"name": "Compte", "type": "bank_account"}],
                "intent_classification": {"intent": "check_balance"},
                "entities_extracted": [{"name": "solde", "type": "financial"}],
                "intent_confidence": 0.93,
                "total_tokens_used": 30,
                "openai_usage_stats": {"prompt_tokens": 20, "completion_tokens": 10},
                "openai_cost_usd": 0.003,
                "search_query_used": "balance account",
                "search_results_count": 3,
                "search_execution_time_ms": 50.2,
                "turn_metadata": {"debug": True},
                "created_at": "2024-06-01T12:00:00Z",
                "updated_at": "2024-06-01T12:00:01Z",
            }
        }
    )


    @field_validator("id", "conversation_id")
    @classmethod
    def positive_ids(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v
    @field_validator("search_results_count", "total_tokens_used")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator(
        "processing_time_ms",
        "search_execution_time_ms",
        "confidence_score",
        "intent_confidence",
        "openai_cost_usd",
    )
    @classmethod
    def positive_floats(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "ConversationTurn":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be after created_at")
        return self

