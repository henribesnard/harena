"""
Conversation models for multi-turn conversation management in Conversation Service MVP.

This module defines the data models for managing conversation context, turns,
and multi-turn dialogue state. Optimized for financial conversation scenarios
with AutoGen agents.

Classes:
    - ConversationRequest: API request model for conversation messages
    - ConversationResponse: API response model with assistant reply
    - ConversationTurn: Individual conversation turn between user and assistant
    - ConversationContext: Complete conversation context with all turns

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Pydantic V2
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from uuid import uuid4

__all__ = [
    "ConversationRequest",
    "ConversationResponse",
    "ConversationTurn",
    "ConversationContext",
    "ConversationOut",
    "ConversationTurnsResponse",
]


class ConversationRequest(BaseModel):
    """
    Request model for processing user conversation messages.

    Attributes:
        message: The user's input message.
        conversation_id: Optional identifier of an existing conversation.
        language: Optional language code for the message.
        metadata: Additional request metadata.
        timestamp: When the message was created.
    """

    message: str = Field(
        ...,
        description="The user's input message",
        min_length=1,
        max_length=10000,
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Existing conversation identifier if any"
    )
    language: Optional[str] = Field(
        default=None, description="Language code of the incoming message"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the request"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Message timestamp"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Ensure message is not empty."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class ConversationResponse(BaseModel):
    """
    Response model returned after processing a conversation message.

    Attributes:
        message: Assistant response message.
        conversation_id: Identifier of the conversation.
        success: Indicates if processing was successful.
        processing_time_ms: Time taken to process the request.
        agent_used: Primary agent that generated the response.
        confidence: Confidence score of the response.
        metadata: Additional response metadata.
        error_code: Optional error code when processing fails.
        timestamp: When the response was generated.
    """

    message: str = Field(
        ...,
        description="Assistant response message",
        min_length=1,
        max_length=4000,
    )
    conversation_id: str = Field(
        ..., description="Identifier of the conversation this response belongs to"
    )
    success: bool = Field(
        True, description="Whether the request was processed successfully"
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )
    agent_used: Optional[str] = Field(
        default=None, description="Primary agent that generated the response"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the response",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
    error_code: Optional[str] = Field(
        default=None, description="Error code if the request failed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    @field_validator("message")
    @classmethod
    def validate_response_message(cls, v: str) -> str:
        """Ensure response message is not empty."""
        if not v.strip():
            raise ValueError("Response message cannot be empty")
        return v.strip()


class ConversationTurn(BaseModel):
    """
    Individual conversation turn between user and assistant.
    
    This model represents a single exchange in a conversation, including
    user input, assistant response, and associated metadata for tracking
    and analysis purposes.
    
    Attributes:
        turn_id: Unique identifier for this conversation turn
        user_message: The user's input message
        assistant_response: The assistant's response message
        timestamp: When this turn occurred
        metadata: Additional metadata about the turn
        turn_number: Sequential number of this turn in conversation
        processing_time_ms: Time taken to process and respond
        intent_detected: Detected intent from user message
        entities_extracted: Entities extracted from user message
        confidence_score: Confidence in the response quality
        error_occurred: Whether any errors occurred during processing
        agent_chain: Chain of agents involved in processing this turn
    """
    
    turn_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this conversation turn"
    )
    
    user_message: str = Field(
        ...,
        description="The user's input message",
        min_length=1,
        max_length=2000
    )
    
    assistant_response: str = Field(
        ...,
        description="The assistant's response message",
        min_length=1,
        max_length=4000
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this turn occurred"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the turn"
    )
    
    turn_number: int = Field(
        ...,
        description="Sequential number of this turn in conversation",
        ge=1
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Time taken to process and respond",
        ge=0.0
    )
    
    intent_detected: Optional[str] = Field(
        default=None,
        description="Detected intent from user message"
    )
    
    entities_extracted: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Entities extracted from user message"
    )
    
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score for the response quality",
        ge=0.0,
        le=1.0
    )
    
    error_occurred: bool = Field(
        default=False,
        description="Whether any errors occurred during processing"
    )
    
    agent_chain: Optional[List[str]] = Field(
        default=None,
        description="Chain of agents involved in processing this turn"
    )

    @field_validator("user_message", "assistant_response")
    @classmethod
    def validate_message_content(cls, v: str) -> str:
        """Validate message content is not empty."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        },
        "json_schema_extra": {
            "example": {
                "turn_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_message": "Montre-moi mes transactions de janvier 2024",
                "assistant_response": "Voici vos transactions de janvier 2024...",
                "turn_number": 1,
                "processing_time_ms": 1250.5,
                "intent_detected": "FINANCIAL_QUERY",
                "entities_extracted": [
                    {
                        "entity_type": "DATE_RANGE",
                        "raw_value": "janvier 2024",
                        "normalized_value": "2024-01",
                        "confidence": 0.95
                    }
                ],
                "confidence_score": 0.92,
                "error_occurred": False,
                "agent_chain": [
                    "orchestrator_agent",
                    "intent_classifier_agent",
                    "search_query_agent",
                    "response_agent"
                ],
                "metadata": {
                    "language": "fr",
                    "user_type": "premium",
                    "search_results_count": 15
                }
            }
        }
    }


class ConversationOut(BaseModel):
    """Lightweight conversation representation for API responses."""

    conversation_id: str = Field(
        ..., description="Unique identifier for the conversation"
    )
    title: Optional[str] = Field(
        default=None, description="Optional conversation title"
    )
    status: str = Field(
        ..., description="Current status of the conversation"
    )
    total_turns: int = Field(
        ..., description="Total number of turns in the conversation", ge=0
    )
    last_activity_at: datetime = Field(
        ..., description="Timestamp of the last activity"
    )


class ConversationTurnsResponse(BaseModel):
    """Response model for conversation turns retrieval."""

    conversation_id: str = Field(
        ..., description="Identifier of the conversation"
    )
    turns: List[ConversationTurn] = Field(
        ..., description="List of conversation turns"
    )


class ConversationContext(BaseModel):
    """
    Complete conversation context with all turns and state management.
    
    This model maintains the full state of a conversation across multiple
    turns, enabling context-aware responses and conversation continuity.
    
    Attributes:
        conversation_id: Unique identifier for the conversation
        user_id: Identifier for the user participating
        turns: List of all conversation turns
        current_turn: Current turn number
        created_at: When the conversation was created
        updated_at: When the conversation was last updated
        status: Current status of the conversation
        context_summary: Summary of conversation context for efficiency
        user_preferences: User preferences affecting conversation
        session_metadata: Session-level metadata
        max_turns: Maximum allowed turns in this conversation
        language: Language of the conversation
        domain: Domain context (e.g., "financial", "general")
        last_intent: Last detected intent for continuity
        active_entities: Currently active entities in context
    """
    
    conversation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the conversation"
    )
    
    user_id: int = Field(
        ...,
        description="Identifier for the user participating",
        gt=0
    )
    
    turns: List[ConversationTurn] = Field(
        default_factory=list,
        description="List of all conversation turns"
    )
    
    current_turn: int = Field(
        default=0,
        description="Current turn number",
        ge=0
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the conversation was created"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the conversation was last updated"
    )
    
    status: Literal["active", "completed", "error", "timeout"] = Field(
        default="active",
        description="Current status of the conversation"
    )
    
    context_summary: Optional[str] = Field(
        default=None,
        description="Summary of conversation context for efficiency",
        max_length=1000
    )
    
    user_preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="User preferences affecting conversation"
    )
    
    session_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session-level metadata"
    )
    
    max_turns: int = Field(
        default=20,
        description="Maximum allowed turns in this conversation",
        gt=0,
        le=50
    )
    
    language: str = Field(
        default="fr",
        description="Language of the conversation",
        pattern=r"^[a-z]{2}$"
    )
    
    domain: str = Field(
        default="financial",
        description="Domain context",
        max_length=50
    )
    
    last_intent: Optional[str] = Field(
        default=None,
        description="Last detected intent for continuity"
    )
    
    active_entities: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Currently active entities in context"
    )

    @model_validator(mode='after')
    def validate_turns_consistency(self) -> 'ConversationContext':
        """Validate turns consistency with current_turn."""
        if len(self.turns) != self.current_turn:
            raise ValueError("Number of turns must match current_turn")
        
        # Validate turn numbers are sequential
        for i, turn in enumerate(self.turns, 1):
            if turn.turn_number != i:
                raise ValueError(f"Turn {i} has incorrect turn_number {turn.turn_number}")
        
        return self
    
    @model_validator(mode='after')
    def validate_updated_at_after_created_at(self) -> 'ConversationContext':
        """Validate updated_at is not before created_at."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        return self
    
    @model_validator(mode='after')
    def validate_current_turn_within_limit(self) -> 'ConversationContext':
        """Validate current_turn doesn't exceed max_turns."""
        if self.current_turn > self.max_turns:
            raise ValueError(f"current_turn {self.current_turn} exceeds max_turns {self.max_turns}")
        return self

    def add_turn(self, user_message: str, assistant_response: str, **kwargs) -> ConversationTurn:
        """
        Add a new turn to the conversation.
        
        Args:
            user_message: User's input message
            assistant_response: Assistant's response
            **kwargs: Additional turn metadata
            
        Returns:
            The created ConversationTurn
            
        Raises:
            ValueError: If max_turns would be exceeded
        """
        if self.current_turn >= self.max_turns:
            raise ValueError(f"Cannot add turn: max_turns {self.max_turns} reached")
        
        turn_number = self.current_turn + 1
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            turn_number=turn_number,
            **kwargs
        )
        
        self.turns.append(turn)
        self.current_turn = turn_number
        self.updated_at = datetime.utcnow()
        
        return turn
    
    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the last conversation turn."""
        return self.turns[-1] if self.turns else None
    
    def get_context_for_llm(self, max_turns: int = 5) -> str:
        """
        Get formatted context for LLM consumption.
        
        Args:
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Formatted conversation context string
        """
        if not self.turns:
            return ""
        
        recent_turns = self.turns[-max_turns:]
        context_parts = []
        
        if self.context_summary:
            context_parts.append(f"Context Summary: {self.context_summary}")
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Assistant: {turn.assistant_response}")
        
        return "\n".join(context_parts)

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        },
        "json_schema_extra": {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
                "user_id": 12345,
                "current_turn": 2,
                "status": "active",
                "max_turns": 20,
                "language": "fr",
                "domain": "financial",
                "last_intent": "FINANCIAL_QUERY",
                "active_entities": [
                    {
                        "entity_type": "DATE_RANGE",
                        "value": "2024-01",
                        "confidence": 0.95
                    }
                ],
                "user_preferences": {
                    "currency": "EUR",
                    "date_format": "DD/MM/YYYY",
                    "preferred_response_length": "medium"
                },
                "session_metadata": {
                    "client_version": "1.0.0",
                    "platform": "web",
                    "user_agent": "Mozilla/5.0..."
                }
            }
        }
    }
