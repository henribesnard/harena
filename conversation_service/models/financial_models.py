"""
Financial domain models for entity extraction and intent classification in Conversation Service MVP.

This module defines specialized data models for financial entities, intent classification
results, and domain-specific data structures optimized for French financial conversations.

Classes:
    - FinancialEntity: Extracted financial entities from user messages
    - IntentResult: Results from intent classification process

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Pydantic V2
"""

from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

__all__ = [
    "EntityType",
    "IntentCategory",
    "DetectionMethod",
    "FinancialEntity",
    "IntentResult",
]


class EntityType(str, Enum):
    """Enumeration of supported financial entity types."""
    
    # Monetary entities
    AMOUNT = "AMOUNT"
    CURRENCY = "CURRENCY"
    PERCENTAGE = "PERCENTAGE"
    
    # Temporal entities
    DATE = "DATE"
    DATE_RANGE = "DATE_RANGE"
    RELATIVE_DATE = "RELATIVE_DATE"
    
    # Account entities
    ACCOUNT_TYPE = "ACCOUNT_TYPE"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    IBAN = "IBAN"
    
    # Transaction entities
    TRANSACTION_TYPE = "TRANSACTION_TYPE"
    MERCHANT = "MERCHANT"
    CATEGORY = "CATEGORY"
    DESCRIPTION = "DESCRIPTION"
    
    # Person/Organization entities
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    BANK = "BANK"
    
    # Location entities
    LOCATION = "LOCATION"
    COUNTRY = "COUNTRY"
    
    # Financial instruments
    CARD_TYPE = "CARD_TYPE"
    PAYMENT_METHOD = "PAYMENT_METHOD"
    
    # Other
    REFERENCE_NUMBER = "REFERENCE_NUMBER"
    OTHER = "OTHER"


class IntentCategory(str, Enum):
    """Enumeration of intent categories for financial conversations."""
    
    # Query intents
    FINANCIAL_QUERY = "FINANCIAL_QUERY"
    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    BALANCE_INQUIRY = "BALANCE_INQUIRY"
    ACCOUNT_INFORMATION = "ACCOUNT_INFORMATION"
    
    # Analysis intents
    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
    BUDGET_ANALYSIS = "BUDGET_ANALYSIS"
    TREND_ANALYSIS = "TREND_ANALYSIS"
    
    # Conversational intents
    GREETING = "GREETING"
    CLARIFICATION = "CLARIFICATION"
    CONFIRMATION = "CONFIRMATION"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    
    # Action intents
    EXPORT_REQUEST = "EXPORT_REQUEST"
    FILTER_REQUEST = "FILTER_REQUEST"
    SORT_REQUEST = "SORT_REQUEST"
    
    # Error handling
    UNCLEAR_INTENT = "UNCLEAR_INTENT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class DetectionMethod(str, Enum):
    """Method used for entity detection or intent classification."""

    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"
    HYBRID = "hybrid"
    PATTERN_MATCHING = "pattern_matching"
    NER_MODEL = "ner_model"
    FALLBACK = "fallback"
    AI_FALLBACK = "ai_fallback"
    AI_ERROR_FALLBACK = "ai_error_fallback"
    AI_PARSE_FALLBACK = "ai_parse_fallback"
    EXACT_RULE = "exact_rule"
    PATTERN_RULE = "pattern_rule"
    AI_DETECTION = "ai_detection"


class FinancialEntity(BaseModel):
    """
    Extracted financial entity from user messages.
    
    This model represents a financial entity extracted from user input,
    including both the raw text and normalized structured value, with
    confidence scoring and metadata for tracking extraction quality.
    
    Attributes:
        entity_type: Type of financial entity
        raw_value: Original text as extracted from user message
        normalized_value: Structured/normalized value for processing
        confidence: Confidence score for the extraction (0.0-1.0)
        start_position: Start position in original text
        end_position: End position in original text
        detection_method: Method used for entity detection
        metadata: Additional metadata about the entity
        validation_status: Status of entity validation
        alternative_values: Alternative normalized values if ambiguous
    """
    
    entity_type: EntityType = Field(
        ...,
        description="Type of financial entity"
    )
    
    raw_value: str = Field(
        ...,
        description="Original text as extracted from user message",
        min_length=1,
        max_length=500
    )
    
    normalized_value: Any = Field(
        ...,
        description="Structured/normalized value for processing"
    )
    
    confidence: float = Field(
        ...,
        description="Confidence score for the extraction",
        ge=0.0,
        le=1.0
    )
    
    start_position: Optional[int] = Field(
        default=None,
        description="Start position in original text",
        ge=0
    )
    
    end_position: Optional[int] = Field(
        default=None,
        description="End position in original text",
        ge=0
    )
    
    detection_method: DetectionMethod = Field(
        default=DetectionMethod.HYBRID,
        description="Method used for entity detection"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the entity"
    )
    
    validation_status: Literal["valid", "invalid", "uncertain", "not_validated"] = Field(
        default="not_validated",
        description="Status of entity validation"
    )
    
    alternative_values: Optional[List[Any]] = Field(
        default=None,
        description="Alternative normalized values if ambiguous"
    )

    @model_validator(mode='after')
    def validate_positions(self) -> 'FinancialEntity':
        """Validate that end_position is after start_position."""
        if (self.start_position is not None and 
            self.end_position is not None and 
            self.end_position <= self.start_position):
            raise ValueError("end_position must be greater than start_position")
        return self
    
    @field_validator("normalized_value")
    @classmethod
    def validate_normalized_value(cls, v: Any, info) -> Any:
        """Validate normalized value based on entity type."""
        # Get entity_type from the field context
        if hasattr(info, 'data') and 'entity_type' in info.data:
            entity_type = info.data['entity_type']
            
            if entity_type == EntityType.AMOUNT:
                if not isinstance(v, (int, float, Decimal, str)):
                    raise ValueError("AMOUNT normalized_value must be numeric or string")
            elif entity_type == EntityType.DATE:
                if not isinstance(v, (str, date, datetime)):
                    raise ValueError("DATE normalized_value must be date, datetime, or ISO string")
            elif entity_type == EntityType.PERCENTAGE:
                if isinstance(v, (int, float)) and not (0 <= v <= 100):
                    raise ValueError("PERCENTAGE normalized_value should be between 0 and 100")
        
        return v

    def to_search_filter(self) -> Optional[Dict[str, Any]]:
        """
        Convert entity to search filter format.
        
        Returns:
            Dictionary suitable for search service filtering
        """
        if self.validation_status == "invalid":
            return None
            
        filter_map = {
            EntityType.AMOUNT: {"field": "amount", "value": self.normalized_value},
            EntityType.DATE: {"field": "date", "value": self.normalized_value},
            EntityType.DATE_RANGE: {"field": "date_range", "value": self.normalized_value},
            EntityType.CATEGORY: {"field": "category", "value": self.normalized_value},
            EntityType.MERCHANT: {"field": "merchant", "value": self.normalized_value},
            EntityType.TRANSACTION_TYPE: {"field": "transaction_type", "value": self.normalized_value}
        }
        
        return filter_map.get(self.entity_type)

    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "example": {
                "entity_type": "AMOUNT",
                "raw_value": "500 euros",
                "normalized_value": 500.0,
                "confidence": 0.95,
                "start_position": 15,
                "end_position": 25,
                "detection_method": "hybrid",
                "validation_status": "valid",
                "metadata": {
                    "currency": "EUR",
                    "original_currency": "euros",
                    "extraction_rule": "amount_with_currency"
                }
            }
        }
    }


class IntentResult(BaseModel):
    """
    Results from intent classification process.
    
    This model contains the complete results of intent classification,
    including the detected intent, extracted entities, confidence scores,
    and processing metadata for monitoring and optimization.
    
    Attributes:
        intent_type: Specific intent type detected
        intent_category: High-level category of the intent
        confidence: Overall confidence score for classification
        entities: List of extracted financial entities
        method: Method used for intent classification
        processing_time_ms: Time taken for classification and extraction
        alternative_intents: Alternative intents with lower confidence
        context_influence: How conversation context influenced classification
        validation_errors: Any validation errors encountered
        requires_clarification: Whether user clarification is needed
        suggested_actions: Suggested actions based on intent
        raw_user_message: Original user message for reference
        normalized_query: Normalized version of user query
    """
    
    intent_type: str = Field(
        ...,
        description="Specific intent type detected",
        min_length=1,
        max_length=100
    )
    
    intent_category: IntentCategory = Field(
        ...,
        description="High-level category of the intent"
    )
    
    confidence: float = Field(
        ...,
        description="Overall confidence score for classification",
        ge=0.0,
        le=1.0
    )
    
    entities: List[FinancialEntity] = Field(
        default_factory=list,
        description="List of extracted financial entities"
    )
    
    method: DetectionMethod = Field(
        ...,
        description="Method used for intent classification"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Time taken for classification and extraction",
        ge=0.0
    )
    
    alternative_intents: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Alternative intents with lower confidence"
    )
    
    context_influence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="How conversation context influenced classification"
    )
    
    validation_errors: Optional[List[str]] = Field(
        default=None,
        description="Any validation errors encountered"
    )
    
    requires_clarification: bool = Field(
        default=False,
        description="Whether user clarification is needed"
    )
    
    suggested_actions: Optional[List[str]] = Field(
        default=None,
        description="Suggested actions based on intent"
    )
    
    raw_user_message: Optional[str] = Field(
        default=None,
        description="Original user message for reference"
    )
    
    normalized_query: Optional[str] = Field(
        default=None,
        description="Normalized version of user query"
    )

    search_required: bool = Field(
        default=True,
        description="Whether downstream search is required for this intent",
    )

    @field_validator("alternative_intents")
    @classmethod
    def validate_alternative_intents(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Validate alternative intents structure."""
        if v is not None:
            for alt_intent in v:
                required_keys = ["intent_type", "confidence"]
                for key in required_keys:
                    if key not in alt_intent:
                        raise ValueError(f"Alternative intent missing required key: {key}")
                if not (0.0 <= alt_intent["confidence"] <= 1.0):
                    raise ValueError("Alternative intent confidence must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode='after')
    def validate_entities_consistency(self) -> 'IntentResult':
        """Validate entities are consistent with intent type."""
        # Check for required entities based on intent type
        if "TRANSACTION_SEARCH" in self.intent_type:
            entity_types = [e.entity_type for e in self.entities]
            if not any(et in [EntityType.DATE, EntityType.DATE_RANGE, EntityType.AMOUNT, 
                             EntityType.CATEGORY, EntityType.MERCHANT] for et in entity_types):
                # This is a warning, not an error - we can still process the request
                pass
        
        return self

    def get_entities_by_type(self, entity_type: EntityType) -> List[FinancialEntity]:
        """
        Get entities filtered by type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List of entities matching the specified type
        """
        return [entity for entity in self.entities if entity.entity_type == entity_type]
    
    def has_entity_type(self, entity_type: EntityType) -> bool:
        """
        Check if result contains entities of specified type.
        
        Args:
            entity_type: Type to check for
            
        Returns:
            True if entities of this type exist
        """
        return any(entity.entity_type == entity_type for entity in self.entities)
    
    def to_search_query_params(self) -> Dict[str, Any]:
        """
        Convert intent result to search query parameters.
        
        Returns:
            Dictionary suitable for search service query
        """
        params = {
            "intent_type": self.intent_type,
            "intent_category": self.intent_category.value,
            "filters": []
        }
        
        # Convert entities to search filters
        for entity in self.entities:
            if entity.validation_status != "invalid":
                filter_dict = entity.to_search_filter()
                if filter_dict:
                    params["filters"].append(filter_dict)
        
        return params

    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "example": {
                "intent_type": "TRANSACTION_SEARCH_BY_AMOUNT_AND_DATE",
                "intent_category": "TRANSACTION_SEARCH",
                "confidence": 0.92,
                "method": "hybrid",
                "processing_time_ms": 245.7,
                "entities": [
                    {
                        "entity_type": "AMOUNT",
                        "raw_value": "500 euros",
                        "normalized_value": 500.0,
                        "confidence": 0.95,
                        "detection_method": "rule_based"
                    },
                    {
                        "entity_type": "DATE_RANGE",
                        "raw_value": "janvier 2024",
                        "normalized_value": {"start": "2024-01-01", "end": "2024-01-31"},
                        "confidence": 0.89,
                        "detection_method": "pattern_matching"
                    }
                ],
                "alternative_intents": [
                    {
                        "intent_type": "SPENDING_ANALYSIS",
                        "confidence": 0.15
                    }
                ],
                "requires_clarification": False,
                "suggested_actions": [
                    "search_transactions",
                    "apply_amount_filter",
                    "apply_date_filter"
                ],
                "normalized_query": "Find transactions with amount=500 EUR in date_range=2024-01",
                "search_required": True,
            }
        }
    }
