"""Backward-compatible aggregator for Harena models."""
from .enums import IntentType, EntityType
from .financial_models import FinancialEntity
from .conversation_models import (
    IntentResult,
    QueryResult,
    ResponseResult,
    ConversationState,
)
from ..core.validators import HarenaValidators
from .contracts import SearchServiceFilter, SearchServiceQuery, SearchServiceResponse
# ``AgentResponse`` is used in production but optional for the
# lightweight testing environment where the full model hierarchy might not be
# available.  Importing it lazily prevents ImportError during tests that stub
# ``conversation_service.models.agent_models`` with limited attributes.
try:  # pragma: no cover - simple import guard
    from .agent_models import AgentResponse
except Exception:  # pragma: no cover - fallback for test stubs
    AgentResponse = object  # type: ignore

__all__ = [
    "IntentType",
    "EntityType",
    "FinancialEntity",
    "IntentResult",
    "QueryResult",
    "ResponseResult",
    "SearchServiceFilter",
    "SearchServiceQuery",
    "SearchServiceResponse",
    "ConversationState",
    "AgentResponse",
    "HarenaValidators",
]
