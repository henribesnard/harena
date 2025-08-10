"""
Models package for Conversation Service MVP.

This module provides all the essential data models for the AutoGen-based
conversation service, including agent configurations, conversation context,
financial entities, and service contracts.

Exports:
    Agent Models:
        - AgentConfig: Configuration for AutoGen agents
        - AgentResponse: Standardized agent response format
        - TeamWorkflow: Team workflow configuration
    
    Conversation Models:
        - ConversationRequest: API request model
        - ConversationResponse: API response model
        - ConversationTurn: Individual conversation turn
        - ConversationContext: Complete conversation context
    
    Financial Models:
        - FinancialEntity: Extracted financial entities
        - IntentResult: Intent classification results
    
    Service Contracts:
        - SearchServiceQuery: Search service request contract
        - SearchServiceResponse: Search service response contract
        - QueryMetadata: Query metadata for search
        - SearchParameters: Search parameters configuration
        - SearchFilters: Search filters model
        - ResponseMetadata: Response metadata model
        - TransactionResult: Transaction result model
        - AggregationRequest: Aggregation request model (optional)
        - AggregationResult: Aggregation result model (optional)

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Pydantic V2
"""

# Import agent models
from .agent_models import (
    AgentConfig,
    AgentResponse,
    TeamWorkflow
)

# Import conversation models
from .conversation_models import (
    ConversationTurn,
    ConversationContext,
    ConversationRequest,
    ConversationResponse
)

# Import financial models
from .financial_models import (
    FinancialEntity,
    IntentResult,
    EntityType,
    IntentCategory,
    DetectionMethod
)

# Import service contracts
from .service_contracts import (
    SearchServiceQuery,
    SearchServiceResponse,
    QueryMetadata,
    SearchParameters,
    SearchFilters,
    ResponseMetadata,
    TransactionResult,
    AggregationRequest,
    AggregationResult,
    # Utility functions
    validate_search_query_contract,
    validate_search_response_contract,
    create_minimal_query,
    create_error_response
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Conversation Service Team"

# Export all models for easy importing
__all__ = [
    # Agent Models
    "AgentConfig",
    "AgentResponse", 
    "TeamWorkflow",
    
    # Conversation Models
    "ConversationRequest",
    "ConversationResponse",
    "ConversationTurn",
    "ConversationContext",
    
    # Financial Models
    "FinancialEntity",
    "IntentResult",
    "EntityType",
    "IntentCategory", 
    "DetectionMethod",
    
    # Service Contracts
    "SearchServiceQuery",
    "SearchServiceResponse",
    "QueryMetadata",
    "SearchParameters", 
    "SearchFilters",
    "ResponseMetadata",
    "TransactionResult",
    "AggregationRequest",
    "AggregationResult",
    
    # Utility Functions
    "validate_search_query_contract",
    "validate_search_response_contract",
    "create_minimal_query",
    "create_error_response"
]