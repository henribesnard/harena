"""Expose les modèles publics du Search Service."""

from .request import SearchRequest
from .response import SearchResponse, SearchResult
from .llm_models import (
    FlexibleFinancialTransaction,
    DynamicSpendingAnalysis,
    FlexibleSearchCriteria,
    LLMExtractedInsights,
)

__all__ = ["SearchRequest", "SearchResponse", "SearchResult"]
__all__ = [
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "FlexibleFinancialTransaction",
    "DynamicSpendingAnalysis",
    "FlexibleSearchCriteria",
    "LLMExtractedInsights",
]
