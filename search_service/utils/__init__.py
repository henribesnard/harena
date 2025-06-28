"""
Utilitaires pour le service de recherche.

Ce module expose les utilitaires communs utilisés
par le service de recherche.
"""

from .cache import (
    SearchCache,
    MultiLevelCache,
    global_cache,
    get_search_cache,
    get_embedding_cache,
    get_query_analysis_cache,
    get_suggestions_cache,
    generate_cache_key,
    cache_with_ttl,
    get_cache_metrics
)

__all__ = [
    "SearchCache",
    "MultiLevelCache",
    "global_cache",
    "get_search_cache",
    "get_embedding_cache",
    "get_query_analysis_cache", 
    "get_suggestions_cache",
    "generate_cache_key",
    "cache_with_ttl",
    "get_cache_metrics"
]