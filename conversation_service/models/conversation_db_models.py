from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Conversation(BaseModel):
    """Représente une conversation utilisateur.

    Example:
        {
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
            "created_at": "2024-06-01T12:00:00Z",
            "updated_at": "2024-06-01T12:05:00Z"
        }
    """

    id: int
    conversation_id: str
    user_id: int
    title: Optional[str] = None
    status: str
    language: str
    domain: str
    total_turns: int = Field(ge=0)
    max_turns: int = Field(ge=0)
    last_activity_at: datetime
    conversation_metadata: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(json_schema_extra={
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
            "created_at": "2024-06-01T12:00:00Z",
            "updated_at": "2024-06-01T12:05:00Z"
        }
    })

    @field_validator("user_id")
    @classmethod
    def user_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("user_id must be positive")
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
    """Résumé partiel d'une conversation.

    Example:
        {
            "id": 1,
            "conversation_id": 1,
            "start_turn": 1,
            "end_turn": 4,
            "summary_text": "...",
            "key_topics": ["budget", "économie"],
            "important_entities": [{"name": "Paris", "type": "city"}],
            "summary_method": "llm",
            "created_at": "2024-06-01T12:00:00Z",
            "updated_at": "2024-06-01T12:05:00Z"
        }
    """

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

    model_config = ConfigDict(json_schema_extra={
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
            "updated_at": "2024-06-01T12:05:00Z"
        }
    })

    @field_validator("conversation_id", "start_turn", "end_turn")
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
    """Tour de conversation entre l'utilisateur et l'assistant.

    Example:
        {
            "id": 10,
            "turn_id": "turn_0010",
            "conversation_id": 1,
            "turn_number": 3,
            "user_message": "Quel est mon solde ?",
            "assistant_response": "Votre solde est de 50€.",
            "processing_time_ms": 120.5,
            "confidence_score": 0.98,
            "error_occurred": false,
            "error_message": null,
            "intent_result": {"name": "check_balance"},
            "agent_chain": [{"agent": "retrieval", "status": "ok"}],
            "search_query_used": "balance account",
            "search_results_count": 3,
            "search_execution_time_ms": 50.2,
            "turn_metadata": {"debug": true},
            "created_at": "2024-06-01T12:00:00Z",
            "updated_at": "2024-06-01T12:00:01Z"
        }
    """

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
    search_query_used: Optional[str] = None
    search_results_count: int = Field(ge=0)
    search_execution_time_ms: Optional[float] = Field(default=None, ge=0.0)
    turn_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(json_schema_extra={
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
            "agent_chain": [{"agent": "retrieval", "status": "ok"}],
            "search_query_used": "balance account",
            "search_results_count": 3,
            "search_execution_time_ms": 50.2,
            "turn_metadata": {"debug": True},
            "created_at": "2024-06-01T12:00:00Z",
            "updated_at": "2024-06-01T12:00:01Z"
        }
    })

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("conversation_id must be positive")
        return v

    @model_validator(mode="after")
    def validate_times(self) -> "ConversationTurn":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be after created_at")
        return self
