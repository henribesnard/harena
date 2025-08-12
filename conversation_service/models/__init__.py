"""Convenience imports for conversation service models.

This package exposes a collection of Pydantic models used throughout the
conversation service.  To keep import time fast and avoid potential side
effects, the individual model modules are imported lazily on first access
instead of eagerly at package import time.
"""

from importlib import import_module
from typing import Any, Dict

_lazy_modules: Dict[str, str] = {
    # Agent models
    "AgentConfig": "agent_models",
    "AgentResponse": "agent_models",
    "TeamWorkflow": "agent_models",

    # Conversation models
    "ConversationRequest": "conversation_models",
    "ConversationResponse": "conversation_models",
    "ConversationTurn": "conversation_models",
    "ConversationContext": "conversation_models",
    "ConversationOut": "conversation_models",
    "ConversationTurnsResponse": "conversation_models",

    # Financial models
    "FinancialEntity": "financial_models",
    "IntentResult": "financial_models",
    "EntityType": "financial_models",
    "IntentCategory": "financial_models",
    "DetectionMethod": "financial_models",

    # Service contracts
    "SearchServiceQuery": "service_contracts",
    "SearchServiceResponse": "service_contracts",
    "QueryMetadata": "service_contracts",
    "SearchParameters": "service_contracts",
    "SearchFilters": "service_contracts",
    "ResponseMetadata": "service_contracts",
    "TransactionResult": "service_contracts",
    "AggregationRequest": "service_contracts",
    "AggregationResult": "service_contracts",
    "validate_search_query_contract": "service_contracts",
    "validate_search_response_contract": "service_contracts",
    "create_minimal_query": "service_contracts",
    "create_error_response": "service_contracts",
}

__all__ = list(_lazy_modules.keys())


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    """Lazily import models when they are accessed.

    This allows ``from conversation_service.models import AgentConfig`` while
    deferring the actual import of ``agent_models`` until ``AgentConfig`` is
    requested, avoiding the overhead of importing every model up front.
    """

    module_name = _lazy_modules.get(name)
    if module_name is None:  # pragma: no cover - standard error path
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> Any:  # pragma: no cover - simple delegation
    return sorted(__all__)

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
    "ConversationOut",
    "ConversationTurnsResponse",
    
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

