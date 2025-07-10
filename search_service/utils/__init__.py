"""
Utilitaires pour le Search Service.

Ce module fournit les utilitaires spécialisés pour le service de recherche :
cache LRU optimisé, métriques spécialisées, validation de requêtes,
et helpers Elasticsearch pour optimiser les performances.

ARCHITECTURE UTILITAIRES:
- Cache LRU avec TTL pour résultats de recherche
- Métriques spécialisées recherche lexicale
- Validateurs de requêtes Elasticsearch  
- Helpers Elasticsearch optimisés domaine financier
- Utilitaires performance et monitoring

RESPONSABILITÉS:
✅ Cache intelligent résultats recherche
✅ Métriques performance spécialisées
✅ Validation stricte requêtes Elasticsearch
✅ Optimisations queries financières
✅ Monitoring et observabilité
✅ Helpers formatage et transformation

USAGE:
    from search_service.utils import (
        LRUCache, SearchMetrics, QueryValidator,
        ElasticsearchHelpers, format_search_results
    )
    
    # Cache avec TTL
    cache = LRUCache(max_size=1000, ttl_seconds=300)
    await cache.set("key", result)
    cached_result = await cache.get("key")
    
    # Validation requête
    validator = QueryValidator()
    is_valid = validator.validate_search_query(query_body)
    
    # Helpers Elasticsearch
    formatted_query = ElasticsearchHelpers.build_financial_query(
        query="virement", user_id=123
    )
"""

import logging
from typing import Dict, Any, List, Optional, Union

# Configuration centralisée
from config_service.config import settings

# Cache
from .cache import (
    LRUCache,
    CacheKey,
    CacheStats,
    CacheError,
    CacheKeyError,
    CacheSizeError,
    create_search_cache
)

# Métriques
from .metrics import (
    SearchMetrics,
    QueryMetrics,
    PerformanceMetrics,
    MetricsCollector,
    MetricsExporter,
    create_metrics_collector
)

# Validation
from .validators import (
    QueryValidator,
    FilterValidator,
    ResultValidator,
    ValidationError,
    QueryValidationError,
    FilterValidationError,
    create_query_validator
)

# Helpers Elasticsearch
from .elasticsearch_helpers import (
    ElasticsearchHelpers,
    QueryBuilder,
    ResultFormatter,
    ScoreCalculator,
    HighlightProcessor,
    create_query_builder,
    format_search_results,
    extract_highlights,
    calculate_relevance_score
)

# Logger pour ce module
logger = logging.getLogger(__name__)

# ==================== CONSTANTES ====================

# Tailles de cache par défaut
DEFAULT_CACHE_SIZE = getattr(settings, 'SEARCH_CACHE_SIZE', 1000)
DEFAULT_CACHE_TTL = getattr(settings, 'SEARCH_CACHE_TTL', 300)  # 5 minutes

# Limites de validation
MAX_QUERY_LENGTH = getattr(settings, 'SEARCH_MAX_QUERY_LENGTH', 500)
MAX_RESULTS_LIMIT = getattr(settings, 'SEARCH_MAX_LIMIT', 100)
MAX_FILTER_VALUES = getattr(settings, 'SEARCH_MAX_FILTER_VALUES', 50)

# Champs Elasticsearch optimisés pour les finances
FINANCIAL_SEARCH_FIELDS = [
    "searchable_text^4.0",
    "primary_description^3.0", 
    "clean_description^2.5",
    "provider_description^2.0",
    "merchant_name^3.5"
]

FINANCIAL_HIGHLIGHT_FIELDS = [
    "searchable_text",
    "primary_description", 
    "merchant_name"
]

# ==================== FACTORY FUNCTIONS ====================

def create_search_service_cache(
    max_size: Optional[int] = None,
    ttl_seconds: Optional[int] = None
) -> LRUCache:
    """
    Crée un cache optimisé pour le service de recherche.
    
    Args:
        max_size: Taille maximum du cache
        ttl_seconds: TTL des entrées en secondes
        
    Returns:
        Cache LRU configuré pour les recherches
    """
    if max_size is None:
        max_size = DEFAULT_CACHE_SIZE
    if ttl_seconds is None:
        ttl_seconds = DEFAULT_CACHE_TTL
    
    logger.info(f"Creating search cache: size={max_size}, ttl={ttl_seconds}s")
    
    return LRUCache(
        max_size=max_size,
        ttl_seconds=ttl_seconds,
        name="search_service_cache"
    )

def create_search_metrics_collector() -> MetricsCollector:
    """
    Crée un collecteur de métriques pour le service de recherche.
    
    Returns:
        Collecteur de métriques configuré
    """
    logger.info("Creating search metrics collector")
    
    return MetricsCollector(
        service_name="search_service",
        include_query_metrics=True,
        include_performance_metrics=True,
        include_cache_metrics=True
    )

def create_elasticsearch_query_validator(
    strict_mode: bool = True
) -> QueryValidator:
    """
    Crée un validateur de requêtes Elasticsearch pour les finances.
    
    Args:
        strict_mode: Mode strict de validation
        
    Returns:
        Validateur configuré pour les requêtes financières
    """
    logger.info(f"Creating Elasticsearch query validator: strict_mode={strict_mode}")
    
    return QueryValidator(
        max_query_length=MAX_QUERY_LENGTH,
        max_results_limit=MAX_RESULTS_LIMIT,
        max_filter_values=MAX_FILTER_VALUES,
        allowed_fields=FINANCIAL_SEARCH_FIELDS,
        strict_mode=strict_mode
    )

def create_financial_query_builder() -> QueryBuilder:
    """
    Crée un builder de requêtes optimisé pour les transactions financières.
    
    Returns:
        Builder configuré pour le domaine financier
    """
    logger.info("Creating financial query builder")
    
    return QueryBuilder(
        default_fields=FINANCIAL_SEARCH_FIELDS,
        highlight_fields=FINANCIAL_HIGHLIGHT_FIELDS,
        boost_merchant_name=3.5,
        boost_exact_phrase=10.0,
        enable_fuzzy_matching=True,
        enable_synonym_expansion=True
    )

# ==================== HELPER FUNCTIONS ====================

def generate_cache_key(
    query: str,
    user_id: int,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 20,
    offset: int = 0
) -> str:
    """
    Génère une clé de cache pour une recherche.
    
    Args:
        query: Terme de recherche
        user_id: ID utilisateur
        filters: Filtres appliqués
        limit: Limite de résultats
        offset: Offset de pagination
        
    Returns:
        Clé de cache unique
    """
    import hashlib
    import json
    
    # Créer un objet hashable
    cache_data = {
        "query": query.lower().strip(),
        "user_id": user_id,
        "filters": filters or {},
        "limit": limit,
        "offset": offset
    }
    
    # Sérialiser de manière déterministe
    cache_str = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))
    
    # Hasher pour créer une clé courte
    return f"search:{hashlib.md5(cache_str.encode()).hexdigest()}"

def extract_query_terms(query: str) -> List[str]:
    """
    Extrait les termes d'une requête de recherche.
    
    Args:
        query: Requête de recherche
        
    Returns:
        Liste des termes nettoyés
    """
    import re
    
    # Nettoyer et normaliser
    query = query.lower().strip()
    
    # Supprimer les caractères spéciaux mais garder les espaces
    query = re.sub(r'[^\w\s\-\.]', ' ', query)
    
    # Diviser en mots et filtrer
    terms = [term.strip() for term in query.split() if len(term.strip()) > 1]
    
    # Dédupliquer en préservant l'ordre
    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    return unique_terms

def validate_user_id(user_id: Union[int, str]) -> int:
    """
    Valide et normalise un ID utilisateur.
    
    Args:
        user_id: ID utilisateur à valider
        
    Returns:
        ID utilisateur validé
        
    Raises:
        ValueError: Si l'ID n'est pas valide
    """
    try:
        user_id_int = int(user_id)
        if user_id_int <= 0:
            raise ValueError("User ID must be positive")
        return user_id_int
    except (ValueError, TypeError):
        raise ValueError(f"Invalid user ID: {user_id}")

def sanitize_query(query: str) -> str:
    """
    Sanitise une requête de recherche.
    
    Args:
        query: Requête à sanitiser
        
    Returns:
        Requête sanitisée
    """
    if not query or not isinstance(query, str):
        return ""
    
    # Nettoyer et limiter la longueur
    query = query.strip()[:MAX_QUERY_LENGTH]
    
    # Supprimer les caractères potentiellement dangereux
    import re
    query = re.sub(r'[<>"\'\;\{\}]', '', query)
    
    return query

def calculate_search_score(
    elasticsearch_score: float,
    recency_factor: float = 1.0,
    user_preference_factor: float = 1.0
) -> float:
    """
    Calcule un score de recherche composite.
    
    Args:
        elasticsearch_score: Score Elasticsearch de base
        recency_factor: Facteur de récence (0.0-2.0)
        user_preference_factor: Facteur de préférence utilisateur (0.0-2.0)
        
    Returns:
        Score composite normalisé
    """
    if elasticsearch_score <= 0:
        return 0.0
    
    # Pondération : 70% ES score, 20% récence, 10% préférence
    composite_score = (
        elasticsearch_score * 0.7 +
        elasticsearch_score * recency_factor * 0.2 +
        elasticsearch_score * user_preference_factor * 0.1
    )
    
    return round(composite_score, 2)

# ==================== EXPORTS ====================

__all__ = [
    # Cache
    "LRUCache",
    "CacheKey", 
    "CacheStats",
    "CacheError",
    "CacheKeyError",
    "CacheSizeError",
    "create_search_cache",
    
    # Métriques
    "SearchMetrics",
    "QueryMetrics", 
    "PerformanceMetrics",
    "MetricsCollector",
    "MetricsExporter",
    "create_metrics_collector",
    
    # Validation
    "QueryValidator",
    "FilterValidator",
    "ResultValidator", 
    "ValidationError",
    "QueryValidationError",
    "FilterValidationError",
    "create_query_validator",
    
    # Helpers Elasticsearch
    "ElasticsearchHelpers",
    "QueryBuilder",
    "ResultFormatter",
    "ScoreCalculator", 
    "HighlightProcessor",
    "create_query_builder",
    "format_search_results",
    "extract_highlights",
    "calculate_relevance_score",
    
    # Factory functions
    "create_search_service_cache",
    "create_search_metrics_collector",
    "create_elasticsearch_query_validator",
    "create_financial_query_builder",
    
    # Helper functions
    "generate_cache_key",
    "extract_query_terms",
    "validate_user_id", 
    "sanitize_query",
    "calculate_search_score",
    
    # Constantes
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_CACHE_TTL",
    "MAX_QUERY_LENGTH",
    "MAX_RESULTS_LIMIT", 
    "MAX_FILTER_VALUES",
    "FINANCIAL_SEARCH_FIELDS",
    "FINANCIAL_HIGHLIGHT_FIELDS"
]