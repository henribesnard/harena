"""
Module utils du Search Service
=============================

Module utilitaires simplifié contenant les outils essentiels :
- Cache LRU pour optimisation performance
- Métriques de monitoring et alertes
- Validateurs de requêtes et sécurité
- Helpers Elasticsearch
- Utilitaires système et performance

Usage:
    from utils import cache_manager, metrics_collector
    from utils import get_system_metrics, cleanup_old_metrics
    from utils.validators import ContractValidator
"""

# === IMPORTS INTERNES ===

# Cache système
from .cache import (
    LRUCache,
    SmartCache,
    CacheManager,
    CacheKeyGenerator,
    cache_manager,
    global_cache_manager
)

# Métriques et monitoring
from .metrics import (
    MetricsCollector,
    AlertManager,
    SearchMetrics,
    ElasticsearchMetrics,
    ApiMetrics,
    metrics_collector,
    alert_manager,
    search_metrics,
    elasticsearch_metrics,
    api_metrics,
    query_metrics,
    result_metrics
)

# Validateurs
from .validators import (
    ValidationError,
    SecurityValidator,
    ContractValidator,
    FilterValidator,
    ElasticsearchQueryValidator,
    PerformanceValidator,
    ValidatorFactory
)

# Helpers Elasticsearch
from .elasticsearch_helpers import (
    ElasticsearchQueryBuilder,
    ResponseFormatter,
    ErrorHandler,
    IndexManager,
    QueryOptimizer,
    build_simple_user_query,
    build_text_search_query
)

# Utilitaires système (importés depuis system_utils.py)
from .system_utils import (
    HealthStatus,
    ComponentType,
    get_system_metrics,
    get_performance_summary,
    get_utils_performance,
    cleanup_old_metrics,
    get_utils_health
)


# === FONCTIONS UTILITAIRES PUBLIQUES ===

def get_cache_manager() -> CacheManager:
    """
    Retourne l'instance globale du gestionnaire de cache
    
    Returns:
        CacheManager: Instance globale du cache manager
    """
    return cache_manager


def get_cache_stats() -> dict:
    """
    Retourne les statistiques globales du cache
    
    Returns:
        dict: Statistiques du cache
    """
    return cache_manager.get_global_stats()


def reset_all_metrics():
    """
    Remet à zéro tous les compteurs de métriques
    """
    metrics_collector._stats.reset()
    for cache in cache_manager.caches.values():
        cache.reset_stats()


def validate_search_contract(contract) -> tuple:
    """
    Valide un contrat de recherche complet
    
    Args:
        contract: Contrat SearchServiceQuery
        
    Returns:
        tuple: (is_valid, errors_list)
    """
    return ContractValidator.validate_search_query(contract)


def build_elasticsearch_query(request_data: dict) -> dict:
    """
    Construit une requête Elasticsearch optimisée
    
    Args:
        request_data: Données de requête
        
    Returns:
        dict: Requête Elasticsearch formatée
    """
    builder = ElasticsearchQueryBuilder()
    return builder.build_query(request_data)


def get_service_health() -> dict:
    """
    Retourne l'état de santé global du service
    
    Returns:
        dict: État de santé des composants
    """
    return get_utils_health()


def initialize_utils():
    """
    Initialise tous les composants utils
    """
    # Le cache manager s'auto-initialise
    # Le metrics collector s'auto-initialise
    pass


def shutdown_utils():
    """
    Arrêt propre de tous les composants utils
    """
    # Nettoyer les métriques anciennes
    cleanup_old_metrics(hours=1)
    
    # Sauvegarder les stats de cache si nécessaire
    cache_stats = get_cache_stats()
    if cache_stats.get("hits", 0) > 0:
        print(f"Cache final stats: {cache_stats['hits']} hits, "
              f"{cache_stats.get('hit_rate', 0):.1%} hit rate")


# === EXPORTS PUBLICS ===

__all__ = [
    # === CACHE ===
    "LRUCache",
    "SmartCache", 
    "CacheManager",
    "CacheKeyGenerator",
    "cache_manager",
    "global_cache_manager",
    
    # === MÉTRIQUES ===
    "MetricsCollector",
    "AlertManager",
    "SearchMetrics",
    "ElasticsearchMetrics", 
    "ApiMetrics",
    "metrics_collector",
    "alert_manager",
    "search_metrics",
    "elasticsearch_metrics",
    "api_metrics",
    "query_metrics",
    "result_metrics",
    
    # === VALIDATEURS ===
    "ValidationError",
    "SecurityValidator",
    "ContractValidator",
    "FilterValidator",
    "ElasticsearchQueryValidator",
    "PerformanceValidator",
    "ValidatorFactory",
    
    # === ELASTICSEARCH HELPERS ===
    "ElasticsearchQueryBuilder",
    "ResponseFormatter",
    "ErrorHandler",
    "IndexManager",
    "QueryOptimizer",
    "build_simple_user_query",
    "build_text_search_query",
    
    # === SYSTÈME ===
    "HealthStatus",
    "ComponentType",
    "get_system_metrics",
    "get_performance_summary",
    "get_utils_performance", 
    "cleanup_old_metrics",
    "get_utils_health",
    
    # === FONCTIONS UTILITAIRES ===
    "get_cache_manager",
    "get_cache_stats",
    "reset_all_metrics",
    "validate_search_contract",
    "build_elasticsearch_query",
    "get_service_health",
    "initialize_utils",
    "shutdown_utils"
]


# === INFORMATIONS MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Module utilitaires Search Service - Cache, Métriques, Validation"

# Auto-initialisation légère
try:
    initialize_utils()
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Erreur initialisation utils: {e}")