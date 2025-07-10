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

logger = logging.getLogger(__name__)

# ==================== CONSTANTES ET CONFIGURATION ====================

# Constantes par défaut
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 300  # 5 minutes
MAX_QUERY_LENGTH = 1000
MAX_RESULTS_LIMIT = 1000
MAX_FILTER_VALUES = 100

# Champs de recherche financière
FINANCIAL_SEARCH_FIELDS = [
    "searchable_text^3",
    "merchant_name^2.5", 
    "clean_description^2",
    "primary_description^1.5",
    "provider_description^1"
]

# Champs pour highlighting
FINANCIAL_HIGHLIGHT_FIELDS = {
    "searchable_text": {"fragment_size": 150, "number_of_fragments": 3},
    "merchant_name": {"fragment_size": 100, "number_of_fragments": 1},
    "clean_description": {"fragment_size": 200, "number_of_fragments": 2}
}

# ==================== IMPORTS AVEC FALLBACKS ====================

# Cache - avec fallback si module manquant
try:
    from .cache import (
        LRUCache, CacheKey, CacheStats, CacheError, 
        CacheKeyError, CacheSizeError, create_search_cache
    )
except ImportError:
    logger.warning("Cache module not available - using basic implementations")
    
    class LRUCache:
        def __init__(self, max_size=1000, ttl_seconds=300):
            self._cache = {}
            self.max_size = max_size
            
        async def get(self, key):
            return self._cache.get(key)
            
        async def set(self, key, value):
            self._cache[key] = value
            if len(self._cache) > self.max_size:
                # Simple eviction
                first_key = next(iter(self._cache))
                del self._cache[first_key]
    
    CacheKey = str
    CacheStats = dict
    CacheError = Exception
    CacheKeyError = KeyError
    CacheSizeError = ValueError
    
    def create_search_cache():
        return LRUCache()

# Métriques - avec fallback si module manquant
try:
    from .metrics import (
        SearchMetrics, QueryMetrics, PerformanceMetrics,
        MetricsCollector, MetricsExporter, create_metrics_collector
    )
except ImportError:
    logger.warning("Metrics module not available - using basic implementations")
    
    class SearchMetrics:
        def __init__(self):
            self.search_count = 0
            
        def record_search(self, query, duration):
            self.search_count += 1
    
    class QueryMetrics:
        def __init__(self):
            self.query_count = 0
    
    class PerformanceMetrics:
        def __init__(self):
            self.avg_response_time = 0.0
    
    class MetricsCollector:
        def __init__(self):
            pass
            
        def collect(self):
            return {}
    
    class MetricsExporter:
        def export(self, metrics):
            pass
    
    def create_metrics_collector():
        return MetricsCollector()

# Validation - toujours disponible
from .validators import (
    QueryValidator, FilterValidator, ResultValidator,
    ValidationError, QueryValidationError, FilterValidationError,
    create_query_validator, validate_search_request,
    validate_user_id, validate_amount, validate_date,
    sanitize_query, is_safe_query, escape_elasticsearch_query
)

# Helpers Elasticsearch - toujours disponible
from .elasticsearch_helpers import (
    ElasticsearchHelpers, QueryBuilder, ResultFormatter,
    ScoreCalculator, HighlightProcessor, QueryStrategy, SortStrategy,
    create_query_builder, format_search_results, extract_highlights,
    calculate_relevance_score, optimize_query_for_performance,
    build_suggestion_query, validate_query_structure
)

# ==================== FACTORY FUNCTIONS ====================

def create_search_service_cache(max_size: int = DEFAULT_CACHE_SIZE, 
                              ttl_seconds: int = DEFAULT_CACHE_TTL) -> LRUCache:
    """
    Crée un cache optimisé pour le service de recherche.
    
    Args:
        max_size: Taille maximale du cache
        ttl_seconds: Durée de vie en secondes
        
    Returns:
        Instance de LRUCache configurée
    """
    return LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)

def create_search_metrics_collector() -> MetricsCollector:
    """
    Crée un collecteur de métriques pour la recherche.
    
    Returns:
        Instance de MetricsCollector configurée
    """
    return create_metrics_collector()

def create_elasticsearch_query_validator(validation_level: str = "standard") -> QueryValidator:
    """
    Crée un validateur de requêtes Elasticsearch.
    
    Args:
        validation_level: Niveau de validation (basic, standard, strict, paranoid)
        
    Returns:
        Instance de QueryValidator
    """
    from .validators import ValidationLevel
    
    level_map = {
        "basic": ValidationLevel.BASIC,
        "standard": ValidationLevel.STANDARD,
        "strict": ValidationLevel.STRICT,
        "paranoid": ValidationLevel.PARANOID
    }
    
    level = level_map.get(validation_level, ValidationLevel.STANDARD)
    return create_query_validator(validation_level=level)

def create_financial_query_builder() -> QueryBuilder:
    """
    Crée un builder de requêtes optimisé pour les données financières.
    
    Returns:
        Instance de QueryBuilder configurée
    """
    return create_query_builder()

# ==================== HELPER FUNCTIONS ====================

def generate_cache_key(query: str, user_id: int, filters: Dict[str, Any] = None) -> str:
    """
    Génère une clé de cache pour une recherche.
    
    Args:
        query: Texte de recherche
        user_id: ID utilisateur
        filters: Filtres optionnels
        
    Returns:
        Clé de cache unique
    """
    import hashlib
    import json
    
    # Création d'un dictionnaire ordonné pour la cohérence
    cache_data = {
        "query": sanitize_query(query),
        "user_id": user_id,
        "filters": filters or {}
    }
    
    # Sérialisation JSON ordonnée
    cache_str = json.dumps(cache_data, sort_keys=True)
    
    # Hash MD5 pour une clé courte
    return hashlib.md5(cache_str.encode()).hexdigest()

def extract_query_terms(query: str) -> List[str]:
    """
    Extrait les termes de recherche d'une requête.
    
    Args:
        query: Texte de recherche
        
    Returns:
        Liste des termes extraits
    """
    import re
    
    if not isinstance(query, str):
        return []
    
    # Nettoyage de base
    cleaned = sanitize_query(query)
    
    # Extraction des mots (minimum 2 caractères)
    terms = re.findall(r'\b\w{2,}\b', cleaned.lower())
    
    # Suppression des doublons en préservant l'ordre
    unique_terms = []
    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)
    
    return unique_terms

def calculate_search_score(elasticsearch_score: float, 
                         recency_factor: float = 1.0,
                         user_preference_factor: float = 1.0) -> float:
    """
    Calcule un score de recherche composite.
    
    Args:
        elasticsearch_score: Score Elasticsearch de base
        recency_factor: Facteur de récence (0.0-2.0)
        user_preference_factor: Facteur de préférence utilisateur (0.0-2.0)
        
    Returns:
        Score composite normalisé
    """
    return calculate_relevance_score(elasticsearch_score, recency_factor, user_preference_factor)

def validate_search_parameters(query: str = None, user_id: int = None, 
                             size: int = None, from_: int = None) -> Dict[str, Any]:
    """
    Valide les paramètres de recherche.
    
    Args:
        query: Texte de recherche
        user_id: ID utilisateur
        size: Nombre de résultats
        from_: Offset
        
    Returns:
        Dictionnaire avec validation et paramètres nettoyés
    """
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "cleaned_params": {}
    }
    
    # Validation query
    if query is not None:
        if not is_safe_query(query):
            validation["errors"].append("Query contains unsafe content")
            validation["is_valid"] = False
        else:
            validation["cleaned_params"]["query"] = sanitize_query(query)
    
    # Validation user_id
    if user_id is not None:
        if not validate_user_id(user_id):
            validation["errors"].append("Invalid user_id")
            validation["is_valid"] = False
        else:
            validation["cleaned_params"]["user_id"] = user_id
    
    # Validation size
    if size is not None:
        if not isinstance(size, int) or size < 0:
            validation["errors"].append("Size must be a non-negative integer")
            validation["is_valid"] = False
        elif size > MAX_RESULTS_LIMIT:
            validation["warnings"].append(f"Size capped at {MAX_RESULTS_LIMIT}")
            validation["cleaned_params"]["size"] = MAX_RESULTS_LIMIT
        else:
            validation["cleaned_params"]["size"] = size
    
    # Validation from_
    if from_ is not None:
        if not isinstance(from_, int) or from_ < 0:
            validation["errors"].append("From must be a non-negative integer")
            validation["is_valid"] = False
        elif from_ > 10000:
            validation["errors"].append("From cannot exceed 10000")
            validation["is_valid"] = False
        else:
            validation["cleaned_params"]["from"] = from_
    
    return validation

def optimize_search_query(query: str, user_preferences: Dict[str, Any] = None) -> str:
    """
    Optimise une requête de recherche basée sur les préférences utilisateur.
    
    Args:
        query: Requête originale
        user_preferences: Préférences utilisateur optionnelles
        
    Returns:
        Requête optimisée
    """
    if not isinstance(query, str):
        return ""
    
    optimized = sanitize_query(query)
    
    # Expansion basée sur les préférences
    if user_preferences:
        # Ajout de synonymes fréquents si activé
        if user_preferences.get("expand_synonyms", False):
            terms = extract_query_terms(optimized)
            expanded_terms = []
            
            for term in terms:
                expanded_terms.append(term)
                # Ajouter des synonymes financiers courants
                if term in ["cafe", "café"]:
                    expanded_terms.append("restaurant")
                elif term in ["essence", "carburant"]:
                    expanded_terms.append("station")
            
            optimized = " ".join(expanded_terms)
    
    return optimized

def build_search_context(query: str, user_id: int, **kwargs) -> Dict[str, Any]:
    """
    Construit un contexte de recherche complet.
    
    Args:
        query: Texte de recherche
        user_id: ID utilisateur
        **kwargs: Paramètres supplémentaires
        
    Returns:
        Contexte de recherche structuré
    """
    context = {
        "query": sanitize_query(query),
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": {},
        "performance": {},
        "security": {}
    }
    
    # Paramètres de recherche
    context["parameters"].update({
        "size": kwargs.get("size", 20),
        "from": kwargs.get("from", 0),
        "highlight": kwargs.get("highlight", True),
        "sort": kwargs.get("sort", "relevance")
    })
    
    # Métriques de performance
    context["performance"].update({
        "query_length": len(query),
        "term_count": len(extract_query_terms(query)),
        "estimated_complexity": "low" if len(query) < 50 else "medium"
    })
    
    # Informations de sécurité
    context["security"].update({
        "is_safe": is_safe_query(query),
        "sanitized": sanitize_query(query) != query
    })
    
    return context

# ==================== CLASSES UTILITAIRES ====================

class SearchHelper:
    """
    Classe helper pour simplifier les opérations de recherche courantes.
    
    Encapsule les opérations fréquentes en une interface simple.
    """
    
    def __init__(self, enable_cache: bool = True, enable_metrics: bool = True):
        self.enable_cache = enable_cache
        self.enable_metrics = enable_metrics
        
        if enable_cache:
            self.cache = create_search_service_cache()
        
        if enable_metrics:
            self.metrics = create_search_metrics_collector()
        
        self.query_builder = create_financial_query_builder()
        self.validator = create_elasticsearch_query_validator()
    
    def build_query(self, query: str, user_id: int, **kwargs) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch optimisée.
        
        Args:
            query: Texte de recherche
            user_id: ID utilisateur
            **kwargs: Options supplémentaires
            
        Returns:
            Requête Elasticsearch
        """
        # Validation des paramètres
        validation = validate_search_parameters(query, user_id, 
                                              kwargs.get("size"), 
                                              kwargs.get("from"))
        
        if not validation["is_valid"]:
            raise ValueError(f"Invalid parameters: {validation['errors']}")
        
        # Construction de la requête
        return ElasticsearchHelpers.build_financial_query(
            query=validation["cleaned_params"]["query"],
            user_id=validation["cleaned_params"]["user_id"],
            **kwargs
        )
    
    def validate_query(self, es_query: Dict[str, Any]) -> bool:
        """
        Valide une requête Elasticsearch.
        
        Args:
            es_query: Requête à valider
            
        Returns:
            True si valide, False sinon
        """
        result = self.validator.validate(es_query)
        return result.is_valid
    
    def format_results(self, es_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formate les résultats Elasticsearch.
        
        Args:
            es_response: Réponse brute d'Elasticsearch
            
        Returns:
            Résultats formatés
        """
        return format_search_results(es_response)
    
    async def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Récupère un résultat du cache.
        
        Args:
            cache_key: Clé de cache
            
        Returns:
            Résultat mis en cache ou None
        """
        if not self.enable_cache:
            return None
        
        try:
            return await self.cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def cache_result(self, cache_key: str, result: Any) -> bool:
        """
        Met en cache un résultat.
        
        Args:
            cache_key: Clé de cache
            result: Résultat à mettre en cache
            
        Returns:
            True si mis en cache avec succès
        """
        if not self.enable_cache:
            return False
        
        try:
            await self.cache.set(cache_key, result)
            return True
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            return False

# ==================== IMPORTS DATETIME ====================

try:
    from datetime import datetime
except ImportError:
    import datetime as dt
    datetime = dt.datetime

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
    "validate_search_request",
    "validate_user_id",
    "validate_amount",
    "validate_date",
    "sanitize_query",
    "is_safe_query",
    "escape_elasticsearch_query",
    
    # Helpers Elasticsearch
    "ElasticsearchHelpers",
    "QueryBuilder",
    "ResultFormatter",
    "ScoreCalculator", 
    "HighlightProcessor",
    "QueryStrategy",
    "SortStrategy",
    "create_query_builder",
    "format_search_results",
    "extract_highlights",
    "calculate_relevance_score",
    "optimize_query_for_performance",
    "build_suggestion_query",
    "validate_query_structure",
    
    # Factory functions
    "create_search_service_cache",
    "create_search_metrics_collector",
    "create_elasticsearch_query_validator",
    "create_financial_query_builder",
    
    # Helper functions
    "generate_cache_key",
    "extract_query_terms",
    "validate_search_parameters",
    "optimize_search_query",
    "build_search_context",
    "calculate_search_score",
    
    # Classes utilitaires
    "SearchHelper",
    
    # Constantes
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_CACHE_TTL",
    "MAX_QUERY_LENGTH",
    "MAX_RESULTS_LIMIT", 
    "MAX_FILTER_VALUES",
    "FINANCIAL_SEARCH_FIELDS",
    "FINANCIAL_HIGHLIGHT_FIELDS"
]