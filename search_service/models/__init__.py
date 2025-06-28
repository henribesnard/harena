"""
Modèles de données pour le service de recherche.

Ce module exporte tous les modèles Pydantic utilisés
par l'API de recherche.
"""

# Modèles de requêtes
from .requests import (
    SearchRequest,
    AdvancedSearchRequest,
    SuggestionsRequest,
    StatsRequest,
    BulkSearchRequest,
    ExplainRequest,
    HealthCheckRequest
)

# Modèles de réponses
from .responses import (
    SearchResponse,
    SearchResultItem,
    SuggestionsResponse,
    SuggestionItem,
    StatsResponse,
    SearchStats,
    HealthResponse,
    ServiceHealth,
    BulkSearchResponse,
    ExplainResponse,
    ScoringExplanation,
    ExplanationDetail,
    ErrorResponse,
    MetricsResponse
)

# Types et énumérations
from .search_types import (
    SearchType,
    SortOrder,
    TransactionType,
    FilterOperator,
    CategoryFilterType,
    SearchQuality,
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    FINANCIAL_SYNONYMS
)

# Filtres
from .filters import (
    AmountFilter,
    DateFilter,
    CategoryFilter,
    MerchantFilter,
    TextFilter,
    GeographicFilter,
    AdvancedFilters
)

__all__ = [
    # Requests
    "SearchRequest",
    "AdvancedSearchRequest", 
    "SuggestionsRequest",
    "StatsRequest",
    "BulkSearchRequest",
    "ExplainRequest",
    "HealthCheckRequest",
    
    # Responses
    "SearchResponse",
    "SearchResultItem",
    "SuggestionsResponse",
    "SuggestionItem",
    "StatsResponse",
    "SearchStats",
    "HealthResponse",
    "ServiceHealth",
    "BulkSearchResponse",
    "ExplainResponse",
    "ScoringExplanation",
    "ExplanationDetail",
    "ErrorResponse",
    "MetricsResponse",
    
    # Types
    "SearchType",
    "SortOrder",
    "TransactionType",
    "FilterOperator",
    "CategoryFilterType",
    "SearchQuality",
    "DEFAULT_SEARCH_LIMIT",
    "MAX_SEARCH_LIMIT",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_LEXICAL_WEIGHT",
    "DEFAULT_SEMANTIC_WEIGHT",
    "FINANCIAL_SYNONYMS",
    
    # Filters
    "AmountFilter",
    "DateFilter",
    "CategoryFilter",
    "MerchantFilter",
    "TextFilter",
    "GeographicFilter",
    "AdvancedFilters"
]