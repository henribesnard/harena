from typing import Dict, List, Optional, Any, Union, Literal, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError
from datetime import datetime
import json
import re

from .enums import IntentType, EntityType


class ConversationStartResponse(BaseModel):
    """Response returned when a conversation is created."""

    conversation_id: str = Field(
        ..., description="Unique identifier of the conversation"
    )
    created_at: datetime = Field(
        ..., description="Creation timestamp of the conversation"
    )

    model_config = ConfigDict(extra="forbid")


class AgentQueryRequest(BaseModel):
    """Request payload for querying the agent team."""

    message: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class AgentQueryResponse(BaseModel):
    """Response returned by the agent team."""

    conversation_id: str = Field(
        ..., description="Unique identifier of the conversation"
    )
    reply: str = Field(..., description="Agent team's reply")

    model_config = ConfigDict(extra="forbid")


class MessageCreate(BaseModel):
    """Input model for creating a conversation message."""

    role: str = Field(..., description="Role of the message author")
    content: str = Field(..., description="Message content")

    model_config = ConfigDict(extra="forbid")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("role must not be empty")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class ConversationMessage(BaseModel):
    """Single message persisted in the conversation history."""

    user_id: int = Field(..., description="Identifier of the user")
    conversation_id: str = Field(..., description="Conversation identifier")
    role: str = Field(..., description="Role of the message author")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")

    model_config = ConfigDict(extra="forbid")


class ConversationHistoryResponse(BaseModel):
    """History of a conversation."""

    conversation_id: str = Field(..., description="Conversation identifier")
    messages: List[ConversationMessage] = Field(
        ..., description="Chronological list of conversation messages"
    )

    model_config = ConfigDict(extra="forbid")


class IntentResult(BaseModel):
    """
    Result of intent classification with Harena scope validation.
    
    Includes confidence scoring, reasoning, and automatic validation
    of confidence thresholds based on Harena's operational scope.
    
    Examples:
        >>> result = IntentResult(
        ...     intent=IntentType.BALANCE_INQUIRY,
        ...     confidence=0.95,
        ...     reasoning="Clear request for account balance information"
        ... )
        >>> result.is_supported_by_harena()
        True
    """
    
    model_config = ConfigDict(
        extra="forbid", 
        validate_default=True,
        json_schema_extra={
            "example": {
                "intent": "BALANCE_INQUIRY",
                "confidence": 0.95,
                "reasoning": "Clear request for account balance information",
                "alternative_intents": [
                    {"intent": "ACCOUNT_OVERVIEW", "confidence": 0.75}
                ],
                "domain_confidence": 0.98
            }
        }
    )
    
    intent: IntentType = Field(..., description="Detected intent")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Classification confidence score"
    )
    reasoning: str = Field(
        ..., 
        min_length=10, 
        max_length=500, 
        description="Explanation of classification decision"
    )
    
    # Optional metadata
    alternative_intents: Optional[List[Dict[str, Union[str, float]]]] = Field(
        None, 
        max_length=3, 
        description="Alternative intent candidates with scores"
    )
    domain_confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence that request is in banking domain"
    )
    processing_metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional processing information"
    )
    
    @field_validator('alternative_intents')
    @classmethod
    def validate_alternatives(cls, v: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """Validate alternative intents structure."""
        if v is None:
            return v
        
        for alt in v:
            if not isinstance(alt.get('intent'), str):
                raise ValueError("Alternative intent must be string")
            if not isinstance(alt.get('confidence'), (int, float)):
                raise ValueError("Alternative confidence must be number")
            if not 0.0 <= alt.get('confidence', 0) <= 1.0:
                raise ValueError("Alternative confidence must be 0-1")
            
            # Validate intent exists in taxonomy
            try:
                IntentType(alt['intent'])
            except ValueError:
                raise ValueError(f"Invalid alternative intent: {alt['intent']}")
        
        return v
    
    @model_validator(mode='after')
    def validate_confidence_thresholds(self):
        """Validate confidence thresholds based on Harena scope."""
        
        # Unsupported actions require high confidence for proper redirection
        unsupported_intents = {
            IntentType.TRANSFER_REQUEST,
            IntentType.PAYMENT_REQUEST, 
            IntentType.CARD_OPERATIONS,
            IntentType.LOAN_REQUEST,
            IntentType.ACCOUNT_MODIFICATION,
            IntentType.INVESTMENT_OPERATIONS,
            IntentType.UNSUPPORTED_ACTION
        }
        
        if self.intent in unsupported_intents and self.confidence < 0.8:
            raise ValueError(
                f"Unsupported action {self.intent} requires high confidence >= 0.8 "
                f"for proper redirection, got {self.confidence}"
            )
        
        # Supported intents can have more flexible confidence
        supported_intents = {
            IntentType.BALANCE_INQUIRY,
            IntentType.ACCOUNT_OVERVIEW,
            IntentType.TRANSACTION_SEARCH,
            IntentType.SPENDING_ANALYSIS,
            IntentType.CATEGORY_ANALYSIS,
            IntentType.MERCHANT_ANALYSIS,
            IntentType.TEMPORAL_ANALYSIS
        }
        
        if self.intent in supported_intents and self.confidence < 0.5:
            raise ValueError(
                f"Supported intent {self.intent} should have reasonable confidence >= 0.5, "
                f"got {self.confidence}"
            )
        
        # Unknown/ambiguous intents must have low confidence
        unknown_intents = {
            IntentType.UNKNOWN, IntentType.AMBIGUOUS, IntentType.OUT_OF_SCOPE, 
            IntentType.INSUFFICIENT_CONTEXT
        }
        if self.intent in unknown_intents and self.confidence > 0.5:
            raise ValueError(
                f"Unknown intent {self.intent} should have low confidence <= 0.5, "
                f"got {self.confidence}"
            )
        
        return self
    
    def is_supported_by_harena(self) -> bool:
        """Check if intent is supported by Harena's consultation scope."""
        supported_categories = {
            IntentType.BALANCE_INQUIRY, IntentType.ACCOUNT_OVERVIEW, 
            IntentType.TRANSACTION_SEARCH, IntentType.TRANSACTION_DETAILS,
            IntentType.STATEMENT_REQUEST,
            IntentType.SPENDING_ANALYSIS, IntentType.CATEGORY_ANALYSIS,
            IntentType.MERCHANT_ANALYSIS, IntentType.TEMPORAL_ANALYSIS,
            IntentType.BUDGET_ANALYSIS, IntentType.TREND_ANALYSIS,
            IntentType.INCOME_ANALYSIS, IntentType.COMPARISON_ANALYSIS,
            IntentType.ACCOUNT_INFORMATION, IntentType.PRODUCT_INFORMATION,
            IntentType.FEE_INQUIRY, IntentType.GENERAL_INQUIRY, IntentType.HELP_REQUEST,
            IntentType.GREETING, IntentType.GOODBYE, IntentType.THANKS,
            IntentType.CLARIFICATION_REQUEST, IntentType.POLITENESS
        }
        return self.intent in supported_categories
    
    def requires_redirection(self) -> bool:
        """Check if intent requires redirection to other services."""
        unsupported_actions = {
            IntentType.TRANSFER_REQUEST, IntentType.PAYMENT_REQUEST,
            IntentType.CARD_OPERATIONS, IntentType.LOAN_REQUEST,
            IntentType.ACCOUNT_MODIFICATION, IntentType.INVESTMENT_OPERATIONS,
            IntentType.UNSUPPORTED_ACTION
        }
        return self.intent in unsupported_actions

# ================================
# QUERY RESULT MODEL
# ================================

class QueryResult(BaseModel):
    """
    Elasticsearch query generation result with validation.
    
    Ensures generated queries are valid, optimized, and aligned with
    Harena's consultation scope. Includes performance hints and estimates.
    """
    
    model_config = ConfigDict(
        extra="forbid", 
        validate_default=True,
        json_schema_extra={
            "example": {
                "query_type": "filtered_search",
                "elasticsearch_query": {
                    "query": {
                        "bool": {
                            "must": [{"match": {"merchant_name": "Amazon"}}],
                            "filter": [{"range": {"date": {"gte": "2024-01-01"}}}]
                        }
                    },
                    "size": 100
                },
                "estimated_results": 45,
                "harena_scope": True
            }
        }
    )
    
    query_type: Literal[
        "simple_search", "filtered_search", 
        "aggregated_search", "temporal_search"
    ] = Field(..., description="Type of Elasticsearch query")
    
    elasticsearch_query: Dict[str, Any] = Field(
        ..., 
        description="Valid Elasticsearch query object"
    )
    estimated_results: int = Field(
        ..., 
        ge=0, 
        le=10000, 
        description="Estimated number of results"
    )
    
    # Harena-specific fields
    harena_scope: bool = Field(
        default=True, 
        description="Query respects Harena consultation scope"
    )
    timeout_ms: int = Field(
        default=3000, 
        ge=1000, 
        le=10000, 
        description="Query timeout in milliseconds"
    )
    max_results: int = Field(
        default=500, 
        ge=1, 
        le=1000, 
        description="Maximum results to return"
    )
    
    # Performance metadata
    confidence: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Query generation confidence"
    )
    optimization_hints: Optional[List[str]] = Field(
        None, 
        description="Performance optimization suggestions"
    )
    
    @field_validator('elasticsearch_query')
    @classmethod
    def validate_elasticsearch_query(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Elasticsearch query structure and safety."""
        if not isinstance(v, dict):
            raise ValueError("Elasticsearch query must be a dictionary")
        
        # Must contain a valid query clause
        valid_query_types = [
            'query', 'match_all', 'bool', 'match', 'term', 
            'range', 'terms', 'multi_match'
        ]
        has_query = any(key in v for key in valid_query_types)
        if not has_query:
            raise ValueError("Elasticsearch query must contain a valid query clause")
        
        # Validate size parameter
        if 'size' in v:
            size = v['size']
            if not isinstance(size, int) or size < 0 or size > 1000:
                raise ValueError("Elasticsearch 'size' must be integer between 0-1000")
        
        # Security: prevent script execution
        query_str = json.dumps(v)
        if 'script' in query_str.lower():
            raise ValueError("Script execution not allowed in queries")
        
        # Performance: limit aggregation complexity
        if 'aggs' in v or 'aggregations' in v:
            agg_str = json.dumps(v.get('aggs', v.get('aggregations', {})))
            if agg_str.count('{') > 10:  # Simple complexity check
                raise ValueError("Aggregation too complex - limit nested levels")
        
        return v
    
    @model_validator(mode='after')
    def validate_query_consistency(self):
        """Validate consistency between query type and structure."""
        query_str = json.dumps(self.elasticsearch_query)
        
        type_validations = {
            "simple_search": lambda q: any(x in q for x in ["match", "query_string", "simple_query_string"]),
            "filtered_search": lambda q: "bool" in q and ("filter" in q or "must" in q),
            "aggregated_search": lambda q: "aggs" in q or "aggregations" in q,
            "temporal_search": lambda q: "range" in q and any(date_field in q.lower() for date_field in ["date", "timestamp", "created"])
        }
        
        validator = type_validations.get(self.query_type)
        if validator and not validator(query_str):
            raise ValueError(f"Query structure inconsistent with type {self.query_type}")
        
        return self
    
    def to_cache_key(self) -> str:
        """Generate cache key for query result."""
        import hashlib
        query_hash = hashlib.md5(json.dumps(self.elasticsearch_query, sort_keys=True).encode()).hexdigest()
        return f"query:{self.query_type}:{query_hash}"

# ================================
# RESPONSE RESULT MODEL
# ================================

class ResponseResult(BaseModel):
    """
    Generated response with Harena scope validation and content safety.
    
    Ensures responses are appropriate, helpful, and compliant with
    Harena's consultation-only scope. Includes suggestions and tone validation.
    """
    
    model_config = ConfigDict(
        extra="forbid", 
        validate_default=True,
        json_schema_extra={
            "example": {
                "response": "Voici votre solde actuel : 2,847.30€. Votre dernier virement date du 15 janvier pour 500€.",
                "response_type": "informative",
                "suggestions": [
                    "Voir vos dernières transactions",
                    "Analyser vos dépenses par catégorie"
                ],
                "confidence": 0.95,
                "harena_limitation": False,
                "tone": "professional"
            }
        }
    )
    
    response: str = Field(
        ..., 
        min_length=10, 
        max_length=2000, 
        description="Generated response text"
    )
    response_type: Literal[
        "informative", "transactional", "analytical", 
        "error", "redirection"
    ] = Field(..., description="Type of response")
    
    # Harena-specific fields
    harena_limitation: bool = Field(
        default=False, 
        description="Response explains Harena scope limitation"
    )
    suggestions: Optional[List[str]] = Field(
        None, 
        max_length=3, 
        description="Helpful suggestions within Harena scope"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Response generation confidence"
    )
    
    # Linguistic metadata
    tone: Literal["professional", "casual", "helpful", "empathetic"] = Field(
        default="professional", 
        description="Response tone"
    )
    language: Literal["fr", "en"] = Field(
        default="fr", 
        description="Response language"
    )
    requires_followup: bool = Field(
        default=False, 
        description="Indicates if follow-up interaction expected"
    )
    
    @field_validator('suggestions')
    @classmethod
    def validate_suggestions(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate suggestions are helpful and within scope."""
        if v is None:
            return v
        
        for suggestion in v:
            if not isinstance(suggestion, str):
                raise ValueError("Suggestions must be strings")
            if len(suggestion) < 5 or len(suggestion) > 200:
                raise ValueError("Suggestion length must be 5-200 characters")
            
            # Check for action-related suggestions (should be avoided for Harena)
            action_keywords = [
                "transférer", "virer", "payer", "bloquer", "modifier", 
                "changer", "activer", "désactiver"
            ]
            if any(keyword in suggestion.lower() for keyword in action_keywords):
                raise ValueError(f"Suggestion contains unsupported action: {suggestion}")
        
        return v
    
    @model_validator(mode='after')
    def validate_response_content(self):
        """Validate response content safety and appropriateness."""
        
        # Security: Check for sensitive information patterns
        sensitive_patterns = [
            r'\b\d{16}\b',  # Credit card numbers
            r'\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b',  # Formatted card numbers
            r'\b\d{4}\b.*(?:pin|code)',  # PIN codes
            r'[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}',  # IBAN patterns
        ]
        
        response_text = self.response.lower()
        for pattern in sensitive_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                raise ValueError(f"Response contains sensitive information pattern: {pattern}")
        
        # Harena scope: Check for unsupported action promises
        if not self.harena_limitation:
            action_promises = [
                "je vais transférer", "je vais payer", "je vais bloquer",
                "je vais modifier", "je peux effectuer", "je vais faire"
            ]
            for promise in action_promises:
                if promise in response_text:
                    raise ValueError(f"Response promises unsupported action: {promise}")
        
        # Tone consistency validation
        if self.response_type == "redirection" and self.tone == "casual":
            raise ValueError("Redirection responses should use professional tone")
        
        if self.response_type == "error" and self.tone == "casual":
            raise ValueError("Error responses should be professional or empathetic")
        
        return self
    
    def contains_limitation_explanation(self) -> bool:
        """Check if response explains Harena limitations appropriately."""
        limitation_indicators = [
            "ne peux pas", "non disponible", "non supporté",
            "consultatif", "uniquement", "redirection",
            "application bancaire", "conseiller"
        ]
        return any(indicator in self.response.lower() for indicator in limitation_indicators)
    
    def requires_clarification(self) -> bool:
        """Check if response requests clarification from user."""
        clarification_indicators = [
            "pouvez-vous préciser", "quelle période", "quel compte",
            "voulez-vous dire", "clarifiez", "plus de détails"
        ]
        return any(indicator in self.response.lower() for indicator in clarification_indicators)

# ================================
# SEARCH SERVICE INTEGRATION MODELS
# ================================

class ConversationState(BaseModel):
    """
    Shared conversation state across all agents in the Harena workflow.
    
    Tracks conversation progress, agent results, validation outcomes,
    and quality metrics for comprehensive conversation management.
    """
    
    model_config = ConfigDict(
        extra="allow",  # Allow additional context fields
        validate_default=True,
        json_schema_extra={
            "example": {
                "conversation_id": "user123_1706745600",
                "user_id": 123,
                "current_turn": 1,
                "workflow_stage": "response",
                "stage_results": {
                    "intent": {"intent": "BALANCE_INQUIRY", "confidence": 0.95},
                    "entities": {"entities": []},
                    "query": {"query_type": "simple_search"},
                    "response": {"response": "Votre solde est de 2,847.30€"}
                },
                "quality_score": 0.92
            }
        }
    )
    
    # Core identifiers
    conversation_id: str = Field(..., description="Unique conversation identifier")
    user_id: int = Field(..., description="User identifier")
    current_turn: int = Field(default=1, ge=1, description="Current conversation turn")
    
    # Workflow tracking
    workflow_stage: Literal[
        "intent", "entity", "query", "response", "complete", "error"
    ] = Field(default="intent", description="Current workflow stage")
    
    stage_results: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Results from each workflow stage"
    )
    
    # Context and history
    user_context: Dict[str, Any] = Field(
        default_factory=dict, 
        description="User-specific context and preferences"
    )
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Previous conversation turns"
    )
    
    # Timestamps
    start_time: datetime = Field(
        default_factory=datetime.now, 
        description="Conversation start timestamp"
    )
    last_update: datetime = Field(
        default_factory=datetime.now, 
        description="Last update timestamp"
    )
    total_processing_time_ms: int = Field(
        default=0, 
        ge=0, 
        description="Total processing time in milliseconds"
    )
    
    # Quality and validation
    validation_results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Validation results from each stage"
    )
    quality_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Overall conversation quality score"
    )
    error_count: int = Field(
        default=0, 
        ge=0, 
        description="Number of errors encountered"
    )
    
    def update_stage(self, stage: str, results: Dict[str, Any]):
        """Update workflow stage with results."""
        self.workflow_stage = stage
        self.stage_results[stage] = results
        self.last_update = datetime.now()
    
    def get_stage_result(self, stage: str) -> Optional[Dict[str, Any]]:
        """Get results from a specific stage."""
        return self.stage_results.get(stage)
    
    def is_stage_complete(self, stage: str) -> bool:
        """Check if a workflow stage is complete."""
        return stage in self.stage_results
    
    def add_validation_result(self, stage: str, validation: Dict[str, Any]):
        """Add validation result for a stage."""
        self.validation_results.append({
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            **validation
        })
    
    def calculate_quality_score(self) -> float:
        """
        Calculate overall conversation quality score.
        
        Considers workflow completeness, confidence scores, processing time,
        and validation results to provide a comprehensive quality metric.
        """
        scores = []
        
        # Workflow completeness score
        expected_stages = ["intent", "entity", "query", "response"]
        completed_stages = sum(1 for stage in expected_stages if self.is_stage_complete(stage))
        completeness_score = completed_stages / len(expected_stages)
        scores.append(completeness_score * 0.3)  # 30% weight
        
        # Confidence scores from stages
        if self.is_stage_complete("intent"):
            intent_confidence = self.stage_results["intent"].get("confidence", 0)
            scores.append(intent_confidence * 0.25)  # 25% weight
        
        if self.is_stage_complete("response"):
            response_confidence = self.stage_results["response"].get("confidence", 0)
            scores.append(response_confidence * 0.25)  # 25% weight
        
        # Processing time score (penalty for slow responses)
        if self.total_processing_time_ms > 0:
            # Optimal time: 0-500ms = 1.0, 500-2000ms = 0.8, >2000ms = 0.5
            if self.total_processing_time_ms <= 500:
                time_score = 1.0
            elif self.total_processing_time_ms <= 2000:
                time_score = 0.8
            else:
                time_score = 0.5
            scores.append(time_score * 0.1)  # 10% weight
        
        # Error penalty
        error_penalty = max(0, 1 - (self.error_count * 0.2))  # -0.2 per error
        scores.append(error_penalty * 0.1)  # 10% weight
        
        # Calculate weighted average
        if scores:
            self.quality_score = sum(scores) / (len(scores) if len(scores) <= 4 else 1)
        else:
            self.quality_score = 0.0
        
        return self.quality_score
    
    def to_cache_summary(self) -> Dict[str, Any]:
        """Generate summary for caching purposes."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "workflow_stage": self.workflow_stage,
            "quality_score": self.quality_score,
            "processing_time_ms": self.total_processing_time_ms,
            "completed_stages": list(self.stage_results.keys())
        }

# ================================
# AGENT RESPONSE MODEL
# ================================

__all__ = [
    "ConversationStartResponse",
    "AgentQueryRequest",
    "AgentQueryResponse",
    "ConversationMessage",
    "ConversationHistoryResponse",
    "IntentResult",
    "QueryResult",
    "ResponseResult",
    "ConversationState",
]
