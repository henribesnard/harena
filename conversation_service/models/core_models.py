"""
Core Pydantic v2 models for Harena Conversation Service.

This module defines the foundational data models for the Harena conversation service,
focusing on consultation and analysis capabilities only. All models include strict
validation rules aligned with Harena's scope limitations.

Key Features:
- Pydantic v2 with @field_validator and @model_validator
- Harena scope compliance validation
- Strict type safety and business rules
- Automatic normalization for financial entities
- Comprehensive error messages for debugging

Author: Harena Conversation Team
Created: 2025-01-31
Version: 1.0.0 - Pydantic V2 Harena Scope
"""

from typing import Dict, List, Optional, Any, Union, Literal, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError
from datetime import datetime
from enum import Enum
import re
import json
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "IntentType", "EntityType", "FinancialEntity", "IntentResult", 
    "QueryResult", "ResponseResult", "ConversationState", "AgentResponse",
    "SearchServiceFilter", "SearchServiceQuery", "SearchServiceResponse", "HarenaValidators"
]

# ================================
# ENUMS - HARENA SCOPE DEFINITIONS
# ================================

class IntentType(str, Enum):
    """
    Intent types for Harena - Consultation and Analysis only.
    
    Harena is a consultative banking assistant that helps users understand
    and analyze their financial data. It does NOT perform any transactional
    operations like transfers, payments, or account modifications.
    """
    
    # === CONSULTATION (Harena Supported) ===
    BALANCE_INQUIRY = "BALANCE_INQUIRY"              # "What's my balance?"
    ACCOUNT_OVERVIEW = "ACCOUNT_OVERVIEW"            # "Show my accounts overview"
    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"        # "My Amazon purchases"
    TRANSACTION_DETAILS = "TRANSACTION_DETAILS"      # "Details of this €50 transaction"
    STATEMENT_REQUEST = "STATEMENT_REQUEST"          # "My last month's statement"
    
    # === ANALYSIS (Harena Supported) ===
    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"          # "My spending this month"
    CATEGORY_ANALYSIS = "CATEGORY_ANALYSIS"          # "My restaurant expenses"
    MERCHANT_ANALYSIS = "MERCHANT_ANALYSIS"          # "How much at Carrefour?"
    TEMPORAL_ANALYSIS = "TEMPORAL_ANALYSIS"          # "Spending trend this year vs last year"
    BUDGET_ANALYSIS = "BUDGET_ANALYSIS"              # "Budget vs actual"
    TREND_ANALYSIS = "TREND_ANALYSIS"                # "Spending trends"
    INCOME_ANALYSIS = "INCOME_ANALYSIS"              # "Income analysis"
    COMPARISON_ANALYSIS = "COMPARISON_ANALYSIS"       # "Compare two periods"
    
    # === INFORMATION & SUPPORT (Harena Supported) ===
    ACCOUNT_INFORMATION = "ACCOUNT_INFORMATION"      # "How does my account work?"
    PRODUCT_INFORMATION = "PRODUCT_INFORMATION"      # "What is a savings account?"
    FEE_INQUIRY = "FEE_INQUIRY"                     # "What are the fees?"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"             # "How does this work?"
    HELP_REQUEST = "HELP_REQUEST"                   # "I need help"
    
    # === CONVERSATIONAL ===
    GREETING = "GREETING"                           # "Hello"
    GOODBYE = "GOODBYE"                             # "Goodbye"
    THANKS = "THANKS"                               # "Thank you"
    CLARIFICATION_REQUEST = "CLARIFICATION_REQUEST" # "Can you repeat?"
    POLITENESS = "POLITENESS"                       # "How are you?"
    
    # === UNSUPPORTED ACTIONS (Redirection Required) ===
    TRANSFER_REQUEST = "TRANSFER_REQUEST"            # "Make a transfer" → UNSUPPORTED
    PAYMENT_REQUEST = "PAYMENT_REQUEST"              # "Pay a bill" → UNSUPPORTED  
    CARD_OPERATIONS = "CARD_OPERATIONS"              # "Block my card" → UNSUPPORTED
    LOAN_REQUEST = "LOAN_REQUEST"                    # "Apply for loan" → UNSUPPORTED
    ACCOUNT_MODIFICATION = "ACCOUNT_MODIFICATION"    # "Change my address" → UNSUPPORTED
    INVESTMENT_OPERATIONS = "INVESTMENT_OPERATIONS"  # "Buy stocks" → UNSUPPORTED
    
    # === MANAGEMENT CATEGORIES ===
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"        # Actions not authorized by Harena
    OUT_OF_SCOPE = "OUT_OF_SCOPE"                   # Requests outside banking domain
    UNKNOWN = "UNKNOWN"                             # Unidentifiable intent
    AMBIGUOUS = "AMBIGUOUS"                         # Ambiguous request requiring clarification
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"   # Need more details to understand

class EntityType(str, Enum):
    """
    Financial entity types with hierarchical taxonomy for Harena scope.
    
    These entities support consultation and analysis operations.
    Action-related entities are detected but marked for redirection.
    """
    
    # === MONETARY ===
    AMOUNT = "AMOUNT"                    # Monetary amounts for search/analysis
    CURRENCY = "CURRENCY"                # Currency codes (EUR, USD, etc.)
    PERCENTAGE = "PERCENTAGE"            # Percentages for analysis
    
    # === TEMPORAL ===
    DATE_RANGE = "DATE_RANGE"            # Date ranges for temporal analysis
    FREQUENCY = "FREQUENCY"              # Payment/transaction frequencies
    
    # === ACCOUNTS ===
    ACCOUNT_TYPE = "ACCOUNT_TYPE"        # Account types for consultation
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"    # Account identifiers (masked)
    
    # === TRANSACTIONS ===
    MERCHANT = "MERCHANT"                # Merchant names for analysis
    TRANSACTION_TYPE = "TRANSACTION_TYPE" # Transaction types
    CATEGORY = "CATEGORY"                # Expense/income categories
    
    # === GEOGRAPHIC ===
    LOCATION = "LOCATION"                # Geographic locations
    ATM_LOCATION = "ATM_LOCATION"        # ATM/branch locations
    
    # === CONTACT (Often for unsupported actions) ===
    BENEFICIARY = "BENEFICIARY"          # Transfer beneficiaries (action entity)
    COMMUNICATION = "COMMUNICATION"      # Email, phone for modifications (action entity)
    
    # === PRODUCTS ===
    CARD_TYPE = "CARD_TYPE"              # Credit/debit card types
    LOAN_TYPE = "LOAN_TYPE"              # Loan product types
    
    # === SECURITY (Often for unsupported actions) ===
    AUTHENTICATION = "AUTHENTICATION"    # PIN, tokens (action entity)
    SECURITY_LEVEL = "SECURITY_LEVEL"    # Security method types

# ================================
# FINANCIAL ENTITY MODEL
# ================================

class FinancialEntity(BaseModel):
    """
    Financial entity with comprehensive validation and normalization.
    
    Supports all entity types in Harena's taxonomy with automatic normalization
    and confidence scoring. Action-related entities are flagged for redirection.
    
    Examples:
        >>> entity = FinancialEntity(
        ...     entity_type=EntityType.AMOUNT,
        ...     raw_value="500 euros",
        ...     normalized_value="500.00",
        ...     confidence=0.95
        ... )
        >>> entity.is_action_related()
        False
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "entity_type": "AMOUNT",
                "raw_value": "500 euros",
                "normalized_value": "500.00",
                "confidence": 0.95,
                "start_position": 15,
                "end_position": 25,
                "context": "spending analysis"
            }
        }
    )
    
    entity_type: EntityType = Field(
        ..., 
        description="Type of financial entity according to Harena taxonomy"
    )
    raw_value: str = Field(
        ..., 
        min_length=1, 
        max_length=500, 
        description="Original extracted value from user input"
    )
    normalized_value: str = Field(
        ..., 
        min_length=1, 
        max_length=200, 
        description="Normalized value following business rules"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Extraction confidence score (0.0-1.0)"
    )
    
    # Optional metadata
    start_position: Optional[int] = Field(
        None, 
        ge=0, 
        description="Start position in source text"
    )
    end_position: Optional[int] = Field(
        None, 
        ge=0, 
        description="End position in source text"
    )
    alternative_values: Optional[List[str]] = Field(
        None, 
        max_length=5, 
        description="Alternative extraction candidates"
    )
    context: Optional[str] = Field(
        None, 
        max_length=200, 
        description="Extraction context for disambiguation"
    )
    
    @field_validator('normalized_value')
    @classmethod
    def validate_normalized_value(cls, v: str, info) -> str:
        """Validate and normalize value according to entity type."""
        entity_type = info.data.get('entity_type')
        
        if entity_type == EntityType.AMOUNT:
            # Dynamic amount normalization (LLM will handle edge cases)
            try:
                # Basic cleaning - remove non-numeric except dots and commas
                clean_amount = re.sub(r'[^\d.,\-+]', '', v)
                # Convert comma to dot for decimal separator
                clean_amount = clean_amount.replace(',', '.')
                
                # Validate it's a valid number
                amount_float = float(clean_amount)
                if abs(amount_float) > 10000000:  # 10M limit for safety
                    raise ValueError(f"Amount exceeds maximum limit: {v}")
                
                # Return formatted to 2 decimal places
                return f"{amount_float:.2f}"
            except ValueError as e:
                raise ValueError(f"Invalid amount format '{v}': {str(e)}")
        
        elif entity_type == EntityType.CURRENCY:
            # Dynamic currency normalization (LLM handles all variants)
            normalized = v.upper().strip()
            
            # Basic length validation - currency codes are typically 3 chars
            if len(normalized) > 5:
                raise ValueError(f"Currency code too long: {v}")
            
            return normalized
        
        elif entity_type == EntityType.PERCENTAGE:
            # Dynamic percentage normalization
            try:
                # Extract numeric part
                numeric_part = re.sub(r'[^\d.,\-]', '', v)
                numeric_part = numeric_part.replace(',', '.')
                
                pct_value = float(numeric_part)
                if abs(pct_value) > 1000:  # Reasonable limit
                    raise ValueError(f"Percentage value too extreme: {v}")
                
                return f"{pct_value:.2f}%"
            except ValueError:
                raise ValueError(f"Invalid percentage format: {v}")
        
        elif entity_type == EntityType.CATEGORY:
            # Dynamic category normalization (LLM handles all variants)
            normalized = v.lower().strip()
            
            # Basic validation
            if len(normalized) < 2 or len(normalized) > 50:
                raise ValueError(f"Category name invalid length: {v}")
            
            return normalized
        
        elif entity_type == EntityType.MERCHANT:
            # Dynamic merchant normalization
            if len(v.strip()) < 1:
                raise ValueError("Merchant name cannot be empty")
            
            # Clean and validate
            clean_name = re.sub(r'[<>{}"]', '', v.strip())  # Remove dangerous chars
            if len(clean_name) > 200:
                clean_name = clean_name[:200]
            
            return clean_name
        
        elif entity_type == EntityType.DATE_RANGE:
            # Dynamic date range normalization (LLM handles complex formats)
            normalized = v.lower().strip()
            
            # Basic validation - date ranges shouldn't be too long
            if len(normalized) > 100:
                raise ValueError(f"Date range description too long: {v}")
            
            return normalized
        
        # Default: basic cleaning and length validation
        cleaned = v.strip()
        if len(cleaned) > 500:  # General maximum
            cleaned = cleaned[:500]
        
        return cleaned
    
    @model_validator(mode='after')
    def validate_positions(self):
        """Validate position consistency."""
        if self.start_position is not None and self.end_position is not None:
            if self.start_position >= self.end_position:
                raise ValueError("start_position must be less than end_position")
        return self
    
    def is_action_related(self) -> bool:
        """Check if entity is related to unsupported actions."""
        action_entities = {
            EntityType.BENEFICIARY,
            EntityType.AUTHENTICATION,
            EntityType.COMMUNICATION
        }
        return self.entity_type in action_entities
    
    def to_search_filter(self) -> Dict[str, Any]:
        """Convert entity to Elasticsearch filter format based on search_service structure."""
        if self.entity_type == EntityType.AMOUNT:
            # Use amount_abs for absolute value filtering with tolerance
            amount_value = float(self.normalized_value)
            return {
                "range": {
                    "amount_abs": {
                        "gte": abs(amount_value) * 0.9,  # 10% tolerance
                        "lte": abs(amount_value) * 1.1
                    }
                }
            }
        elif self.entity_type == EntityType.MERCHANT:
            # Use both textual search and exact keyword filter
            return {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "merchant_name": {
                                    "query": self.normalized_value,
                                    "fuzziness": "AUTO",
                                    "boost": 2.0
                                }
                            }
                        },
                        {
                            "term": {
                                "merchant_name.keyword": self.normalized_value
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        elif self.entity_type == EntityType.CATEGORY:
            # Use exact category keyword filter
            return {
                "term": {
                    "category_name.keyword": self.normalized_value
                }
            }
        elif self.entity_type == EntityType.DATE_RANGE:
            # Dynamic date range - let LLM handle complex parsing
            # This is a placeholder - actual date parsing happens in query generation
            return {
                "exists": {
                    "field": "date"
                }
            }
        elif self.entity_type == EntityType.CURRENCY:
            return {
                "term": {
                    "currency_code": self.normalized_value
                }
            }
        
        # Default: use searchable_text for flexible matching
        return {
            "match": {
                "searchable_text": {
                    "query": self.normalized_value,
                    "fuzziness": "AUTO"
                }
            }
        }

# ================================
# INTENT RESULT MODEL
# ================================

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

class SearchServiceFilter(BaseModel):
    """
    Filter for Search Service queries based on Elasticsearch structure.
    
    Aligns with the search_service Elasticsearch index structure for
    optimal query performance and accurate filtering.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "example": {
                "field": "category_name.keyword",
                "operator": "term",
                "value": "restaurant",
                "boost": 1.0
            }
        }
    )
    
    field: str = Field(..., description="Elasticsearch field name")
    operator: Literal[
        "term", "terms", "match", "range", "exists", 
        "prefix", "wildcard", "fuzzy"
    ] = Field(..., description="Filter operator type")
    value: Union[str, int, float, List[str], Dict[str, Any]] = Field(
        ..., description="Filter value(s)"
    )
    boost: Optional[float] = Field(None, ge=0.1, le=10.0, description="Query boost factor")
    
    @field_validator('field')
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field exists in Elasticsearch structure."""
        # Based on search_service Elasticsearch structure
        valid_fields = {
            # Text search fields
            "searchable_text", "primary_description", "merchant_name", "category_name",
            
            # Keyword filter fields  
            "category_name.keyword", "merchant_name.keyword", "transaction_type",
            "currency_code", "operation_type",
            
            # Numeric fields
            "amount", "amount_abs", "user_id",
            
            # Date fields
            "date", "month_year", "weekday",
            
            # Boolean fields
            "is_future", "is_deleted"
        }
        
        if v not in valid_fields:
            # Allow dynamic fields but log warning
            logger.warning(f"Using non-standard Elasticsearch field: {v}")
        
        return v
    
    @model_validator(mode='after')
    def validate_operator_value_consistency(self):
        """Validate operator and value consistency."""
        if self.operator == "range":
            if not isinstance(self.value, dict):
                raise ValueError("Range operator requires dict value with gte/lte/gt/lt")
            
            required_keys = {"gte", "lte", "gt", "lt"}
            if not any(key in self.value for key in required_keys):
                raise ValueError("Range value must contain at least one of: gte, lte, gt, lt")
        
        elif self.operator == "terms":
            if not isinstance(self.value, list):
                raise ValueError("Terms operator requires list value")
        
        elif self.operator in ["term", "match", "prefix", "wildcard"]:
            if isinstance(self.value, (list, dict)):
                raise ValueError(f"{self.operator} operator requires simple value")
        
        return self

class SearchServiceQuery(BaseModel):
    """
    Complete query contract for Search Service integration.
    
    Provides structured interface for conversation_service to search_service
    communication with proper validation and optimization hints.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "example": {
                "user_id": 12345,
                "query_text": "restaurants paris",
                "filters": [
                    {
                        "field": "category_name.keyword", 
                        "operator": "term", 
                        "value": "restaurant"
                    }
                ],
                "query_type": "lexical_search",
                "limit": 50,
                "offset": 0,
                "sort_by": "date",
                "sort_order": "desc"
            }
        }
    )
    
    # Required fields
    user_id: int = Field(..., description="User ID for security isolation")
    
    # Query parameters
    query_text: Optional[str] = Field(
        None, 
        max_length=500, 
        description="Free text search query"
    )
    filters: List[SearchServiceFilter] = Field(
        default_factory=list, 
        max_length=20, 
        description="Additional filters"
    )
    
    # Query type and performance
    query_type: Literal[
        "lexical_search", "semantic_search", "hybrid_search", 
        "aggregation_only", "filter_only"
    ] = Field(default="lexical_search", description="Type of search to perform")
    
    # Pagination and sorting
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results offset")
    sort_by: Optional[str] = Field(
        None, 
        description="Field to sort by (date, amount, relevance)"
    )
    sort_order: Literal["asc", "desc"] = Field(
        default="desc", 
        description="Sort order"
    )
    
    # Advanced options
    include_aggregations: bool = Field(
        default=False, 
        description="Include aggregation results"
    )
    aggregation_types: Optional[List[str]] = Field(
        None, 
        description="Types of aggregations to compute"
    )
    timeout_ms: int = Field(
        default=3000, 
        ge=100, 
        le=10000, 
        description="Query timeout in milliseconds"
    )
    
    # Debugging and optimization
    explain_query: bool = Field(
        default=False, 
        description="Include query explanation in response"
    )
    boost_recent: bool = Field(
        default=True, 
        description="Boost recent transactions in scoring"
    )
    
    @model_validator(mode='after')
    def validate_query_completeness(self):
        """Validate query has sufficient parameters."""
        if not self.query_text and not self.filters:
            raise ValueError("Query must have either query_text or filters")
        
        if self.include_aggregations and not self.aggregation_types:
            # Provide default aggregations
            self.aggregation_types = ["category_breakdown", "monthly_summary"]
        
        return self
    
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convert to Elasticsearch query format."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": self.user_id}}  # Security isolation
                    ]
                }
            },
            "size": self.limit,
            "from": self.offset,
            "_source": [
                "searchable_text", "primary_description", "merchant_name",
                "amount", "amount_abs", "currency_code", "date", 
                "category_name", "transaction_type"
            ]
        }
        
        # Add text query if provided
        if self.query_text:
            text_query = {
                "multi_match": {
                    "query": self.query_text,
                    "fields": [
                        "searchable_text^2.0",
                        "primary_description^1.5", 
                        "merchant_name^1.8",
                        "category_name^1.0"
                    ],
                    "fuzziness": "AUTO",
                    "type": "best_fields"
                }
            }
            query["query"]["bool"]["must"].append(text_query)
        
        # Add filters
        if self.filters:
            filter_clauses = []
            for filter_item in self.filters:
                if filter_item.operator == "term":
                    filter_clauses.append({
                        "term": {filter_item.field: filter_item.value}
                    })
                elif filter_item.operator == "range":
                    filter_clauses.append({
                        "range": {filter_item.field: filter_item.value}
                    })
                elif filter_item.operator == "match":
                    clause = {
                        "match": {filter_item.field: {"query": filter_item.value}}
                    }
                    if filter_item.boost:
                        clause["match"][filter_item.field]["boost"] = filter_item.boost
                    filter_clauses.append(clause)
            
            if filter_clauses:
                query["query"]["bool"]["filter"] = filter_clauses
        
        # Add sorting
        if self.sort_by:
            query["sort"] = [{self.sort_by: {"order": self.sort_order}}]
        elif self.boost_recent:
            # Default: sort by date descending for recent boost
            query["sort"] = [{"date": {"order": "desc"}}]
        
        # Add aggregations if requested
        if self.include_aggregations and self.aggregation_types:
            aggs = {}
            for agg_type in self.aggregation_types:
                if agg_type == "category_breakdown":
                    aggs["category_breakdown"] = {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": 20
                        },
                        "aggs": {
                            "total_amount": {"sum": {"field": "amount_abs"}},
                            "avg_amount": {"avg": {"field": "amount_abs"}}
                        }
                    }
                elif agg_type == "monthly_summary":
                    aggs["monthly_summary"] = {
                        "date_histogram": {
                            "field": "date",
                            "calendar_interval": "month"
                        },
                        "aggs": {
                            "total_spent": {"sum": {"field": "amount_abs"}},
                            "transaction_count": {"value_count": {"field": "amount"}}
                        }
                    }
            
            if aggs:
                query["aggs"] = aggs
        
        return query

class SearchServiceResponse(BaseModel):
    """
    Structured response from Search Service.
    
    Standardized format for search results with metadata,
    aggregations, and performance information.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "example": {
                "success": True,
                "results": [
                    {
                        "transaction_id": "tx_123456",
                        "searchable_text": "AMAZON.FR Paris",
                        "amount": -45.99,
                        "date": "2024-01-15",
                        "category_name": "shopping"
                    }
                ],
                "total_hits": 156,
                "returned_hits": 50,
                "processing_time_ms": 45,
                "query_id": "q_987654321"
            }
        }
    )
    
    # Response status
    success: bool = Field(..., description="Whether search was successful")
    error_message: Optional[str] = Field(
        None, 
        description="Error message if search failed"
    )
    
    # Search results
    results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of matching transactions"
    )
    
    # Result metadata
    total_hits: int = Field(
        default=0, 
        ge=0, 
        description="Total number of matching documents"
    )
    returned_hits: int = Field(
        default=0, 
        ge=0, 
        description="Number of documents returned"
    )
    max_score: Optional[float] = Field(
        None, 
        description="Highest relevance score"
    )
    
    # Aggregation results
    aggregations: Optional[Dict[str, Any]] = Field(
        None, 
        description="Aggregation results if requested"
    )
    
    # Performance metadata
    processing_time_ms: int = Field(
        default=0, 
        ge=0, 
        description="Search processing time"
    )
    query_id: str = Field(
        ..., 
        description="Unique query identifier for debugging"
    )
    elasticsearch_took_ms: Optional[int] = Field(
        None, 
        description="Elasticsearch execution time"
    )
    
    # Query debugging (if requested)
    query_explanation: Optional[Dict[str, Any]] = Field(
        None, 
        description="Query explanation for debugging"
    )
    
    @model_validator(mode='after')
    def validate_response_consistency(self):
        """Validate response consistency."""
        if self.success:
            if self.returned_hits != len(self.results):
                raise ValueError("returned_hits must match results length")
            
            if self.returned_hits > self.total_hits:
                raise ValueError("returned_hits cannot exceed total_hits")
        else:
            if not self.error_message:
                raise ValueError("Error message required for failed responses")
        
        return self
    
    def get_category_breakdown(self) -> Optional[Dict[str, Any]]:
        """Extract category breakdown from aggregations."""
        if not self.aggregations:
            return None
        
        category_agg = self.aggregations.get("category_breakdown")
        if not category_agg:
            return None
        
        breakdown = {}
        for bucket in category_agg.get("buckets", []):
            category = bucket.get("key")
            breakdown[category] = {
                "transaction_count": bucket.get("doc_count", 0),
                "total_amount": bucket.get("total_amount", {}).get("value", 0),
                "avg_amount": bucket.get("avg_amount", {}).get("value", 0)
            }
        
        return breakdown
    
    def get_top_merchants(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Extract top merchants from results."""
        merchant_stats = {}
        
        for result in self.results:
            merchant = result.get("merchant_name", "Unknown")
            amount = abs(result.get("amount", 0))
            
            if merchant not in merchant_stats:
                merchant_stats[merchant] = {
                    "merchant_name": merchant,
                    "transaction_count": 0,
                    "total_amount": 0
                }
            
            merchant_stats[merchant]["transaction_count"] += 1
            merchant_stats[merchant]["total_amount"] += amount
        
        # Sort by total amount descending
        top_merchants = sorted(
            merchant_stats.values(),
            key=lambda x: x["total_amount"],
            reverse=True
        )
        
        return top_merchants[:limit]

# ================================
# CONVERSATION STATE MODEL
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

class AgentResponse(BaseModel):
    """
    Standardized response format from AutoGen agents.
    
    Provides consistent structure for all agent responses with
    metadata, timing, and error handling information.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "example": {
                "agent_name": "intent_classifier",
                "success": True,
                "result": {
                    "intent": "BALANCE_INQUIRY",
                    "confidence": 0.95,
                    "reasoning": "Clear balance request"
                },
                "processing_time_ms": 245,
                "tokens_used": 125,
                "cached": False
            }
        }
    )
    
    agent_name: str = Field(..., description="Name of the agent that generated response")
    success: bool = Field(..., description="Whether the agent processing succeeded")
    
    # Result data (None if failed)
    result: Optional[Dict[str, Any]] = Field(
        None, 
        description="Agent processing result data"
    )
    error_message: Optional[str] = Field(
        None, 
        description="Error message if processing failed"
    )
    
    # Performance metadata
    processing_time_ms: int = Field(
        ..., 
        ge=0, 
        description="Processing time in milliseconds"
    )
    tokens_used: Optional[int] = Field(
        None, 
        ge=0, 
        description="Number of LLM tokens consumed"
    )
    cached: bool = Field(
        default=False, 
        description="Whether result was served from cache"
    )
    
    # Additional metadata
    timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="Response generation timestamp"
    )
    model_used: Optional[str] = Field(
        None, 
        description="LLM model used for processing"
    )
    
    @model_validator(mode='after')
    def validate_success_consistency(self):
        """Validate consistency between success flag and result/error."""
        if self.success and self.result is None:
            raise ValueError("Successful response must include result data")
        
        if not self.success and not self.error_message:
            raise ValueError("Failed response must include error message")
        
        if not self.success and self.result is not None:
            # Allow partial results in failed responses for debugging
            pass
        
        return self
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics for monitoring."""
        return {
            "agent_name": self.agent_name,
            "processing_time_ms": self.processing_time_ms,
            "tokens_used": self.tokens_used or 0,
            "cached": self.cached,
            "success": self.success,
            "timestamp": self.timestamp.isoformat()
        }

# ================================
# VALIDATION UTILITIES
# ================================

class HarenaValidators:
    """
    Custom validators for Harena-specific business rules.
    
    Provides reusable validation functions for financial data,
    scope compliance, and business rule enforcement with dynamic
    LLM-based validation support.
    """
    
    @staticmethod
    def validate_harena_scope(intent: str) -> bool:
        """Validate intent is within Harena's consultation scope."""
        try:
            intent_enum = IntentType(intent)
            result = IntentResult(
                intent=intent_enum,
                confidence=0.8,  # Dummy confidence for validation
                reasoning="Scope validation check"
            )
            return result.is_supported_by_harena()
        except (ValueError, ValidationError):
            return False
    
    @staticmethod
    def validate_french_iban(iban: str) -> bool:
        """Validate French IBAN format (basic check)."""
        if not iban:
            return False
        
        # Clean IBAN
        clean_iban = re.sub(r'[^A-Z0-9]', '', iban.upper())
        
        # Check French IBAN format: FR + 2 check digits + 23 chars
        if not re.match(r'^FR\d{25}$', clean_iban):
            return False
        
        return len(clean_iban) == 27
    
    @staticmethod
    def normalize_dynamic_amount(amount_str: str) -> Optional[str]:
        """
        Normalize monetary amount with dynamic parsing.
        
        Designed to work with LLM preprocessing for complex formats.
        """
        try:
            # Dynamic cleaning - remove all non-numeric except decimal separators and signs
            clean_amount = re.sub(r'[^\d.,-+]', '', amount_str.strip())
            
            # Handle various decimal separators dynamically
            if ',' in clean_amount and '.' in clean_amount:
                # Assume last separator is decimal point
                if clean_amount.rfind(',') > clean_amount.rfind('.'):
                    clean_amount = clean_amount.replace('.', '').replace(',', '.')
                else:
                    clean_amount = clean_amount.replace(',', '')
            elif ',' in clean_amount:
                # Could be thousands or decimal - let LLM handle this in preprocessing
                clean_amount = clean_amount.replace(',', '.')
            
            # Validate and format
            amount_float = float(clean_amount)
            if abs(amount_float) > 10000000:  # 10M safety limit
                return None
            
            return f"{amount_float:.2f}"
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def requires_clarification(
        user_message: str, 
        intent: str, 
        entities: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Determine if user message requires clarification.
        
        Returns:
            - needs_clarification: Whether clarification is needed
            - clarification_questions: Suggested questions to ask
        """
        clarification_needed = False
        questions = []
        
        # Check message length - too short might be ambiguous
        if len(user_message.strip()) < 3:
            clarification_needed = True
            questions.append("Pouvez-vous préciser votre demande ?")
            return clarification_needed, questions
        
        # Check for vague pronouns without context
        vague_words = ['ça', 'ce', 'cette', 'celui', 'celle', 'le truc', 'la chose']
        if any(word in user_message.lower() for word in vague_words):
            clarification_needed = True
            questions.append("À quoi faites-vous référence exactement ?")
        
        # Check intent-specific clarification needs
        if intent == IntentType.CATEGORY_ANALYSIS.value:
            if not any(e.get("type") == "CATEGORY" for e in entities):
                clarification_needed = True
                questions.append("Quelle catégorie de dépenses souhaitez-vous analyser ?")
        
        elif intent == IntentType.MERCHANT_ANALYSIS.value:
            if not any(e.get("type") == "MERCHANT" for e in entities):
                clarification_needed = True
                questions.append("Chez quel marchand souhaitez-vous voir vos dépenses ?")
        
        elif intent == IntentType.TEMPORAL_ANALYSIS.value:
            if not any(e.get("type") == "DATE_RANGE" for e in entities):
                clarification_needed = True
                questions.append("Sur quelle période souhaitez-vous cette analyse ?")
        
        elif intent == IntentType.TRANSACTION_SEARCH.value:
            if not entities:  # No filtering criteria
                clarification_needed = True
                questions.extend([
                    "Souhaitez-vous filtrer par marchand, catégorie ou montant ?",
                    "Sur quelle période cherchez-vous ?"
                ])
        
        # Check for potentially missing context
        context_words = ['par rapport à', 'comparé à', 'comme hier', 'comme avant']
        if any(phrase in user_message.lower() for phrase in context_words):
            clarification_needed = True
            questions.append("Par rapport à quelle période ou référence ?")
        
        return clarification_needed, questions
    
    @staticmethod
    def validate_conversation_flow(
        intent: str, 
        entities: List[Dict[str, Any]], 
        stage: str
    ) -> List[str]:
        """
        Validate conversation flow consistency.
        
        Returns list of validation warnings/errors.
        """
        warnings = []
        
        # Check entity-intent consistency
        entity_types = [e.get("type") for e in entities]
        
        # Analysis intents should have relevant entities
        if intent == "CATEGORY_ANALYSIS" and "CATEGORY" not in entity_types:
            warnings.append("Category analysis should have category entity")
        
        if intent == "MERCHANT_ANALYSIS" and "MERCHANT" not in entity_types:
            warnings.append("Merchant analysis should have merchant entity")
        
        if intent == "TEMPORAL_ANALYSIS" and "DATE_RANGE" not in entity_types:
            warnings.append("Temporal analysis should have date range entity")
        
        # Check for action entities in consultation intents
        action_entities = {"BENEFICIARY", "AUTHENTICATION", "PAYMENT_REFERENCE"}
        consultation_intents = {
            "BALANCE_INQUIRY", "SPENDING_ANALYSIS", "CATEGORY_ANALYSIS", 
            "MERCHANT_ANALYSIS", "TRANSACTION_SEARCH"
        }
        
        if intent in consultation_intents:
            found_action_entities = [e for e in entity_types if e in action_entities]
            if found_action_entities:
                warnings.append(
                    f"Action entities in consultation intent: {found_action_entities}"
                )
        
        # Check for unsupported actions at response stage
        unsupported_actions = {
            "TRANSFER_REQUEST", "PAYMENT_REQUEST", "CARD_OPERATIONS",
            "LOAN_REQUEST", "ACCOUNT_MODIFICATION", "INVESTMENT_OPERATIONS"
        }
        
        if intent in unsupported_actions and stage != "response":
            warnings.append(
                f"Unsupported action {intent} should be handled at response stage"
            )
        
        return warnings
    
    @staticmethod
    def suggest_clarification_questions(intent: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Generate helpful clarification questions based on intent and entities."""
        questions = []
        
        if intent == IntentType.INSUFFICIENT_CONTEXT.value:
            questions.extend([
                "Que souhaitez-vous savoir sur vos finances ?",
                "Voulez-vous consulter vos comptes, analyser vos dépenses, ou autre chose ?",
                "Pouvez-vous préciser votre demande ?"
            ])
        
        elif intent == IntentType.AMBIGUOUS.value:
            questions.extend([
                "Pouvez-vous reformuler votre demande ?",
                "Souhaitez-vous voir des transactions ou faire une analyse ?",
                "Quelle information cherchez-vous exactement ?"
            ])
        
        elif intent == IntentType.UNKNOWN.value:
            questions.extend([
                "Je peux vous aider avec vos finances. Que voulez-vous savoir ?",
                "Souhaitez-vous voir votre solde, vos transactions, ou analyser vos dépenses ?",
                "Avez-vous une question sur un produit bancaire spécifique ?"
            ])
        
        # Intent-specific clarification
        elif intent == IntentType.CATEGORY_ANALYSIS.value and not any(e.get("type") == "CATEGORY" for e in entities):
            questions.extend([
                "Quelle catégorie voulez-vous analyser ? (restaurant, transport, shopping...)",
                "Sur quelle période souhaitez-vous cette analyse ?"
            ])
        
        elif intent == IntentType.MERCHANT_ANALYSIS.value and not any(e.get("type") == "MERCHANT" for e in entities):
            questions.extend([
                "Chez quel marchand souhaitez-vous voir vos dépenses ?",
                "Voulez-vous une analyse sur une période spécifique ?"
            ])
        
        return questions[:3]  # Limit to 3 questions max