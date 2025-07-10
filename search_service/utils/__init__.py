"""
Package Utils pour le Search Service.

Ce module contient tous les utilitaires :
- Cache LRU pour les résultats de recherche
- Métriques et monitoring
- Validators pour les requêtes
- Helpers Elasticsearch
- Configuration et constantes
"""

import logging
from typing import Dict, Any, List, Optional, Union

# Logger pour ce module
logger = logging.getLogger(__name__)

# ==================== CONSTANTES PAR DÉFAUT ====================

# Configuration cache
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 300  # 5 minutes
DEFAULT_BATCH_SIZE = 100

# Configuration recherche
MAX_SEARCH_RESULTS = 1000
DEFAULT_TIMEOUT = 30
DEFAULT_PAGE_SIZE = 20

# Configuration Elasticsearch
HIGHLIGHT_FIELDS = {
    "searchable_text": {"fragment_size": 150, "number_of_fragments": 3},
    "merchant_name": {"fragment_size": 100, "number_of_fragments": 1},
    "clean_description": {"fragment_size": 200, "number_of_fragments": 2}
}

# ==================== GESTION ELASTICSEARCH VERSION ====================

# Gestion des différentes versions d'Elasticsearch
try:
    from elasticsearch.exceptions import ElasticsearchException, ConnectionError as ESConnectionError
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    try:
        # Essayer l'ancienne structure
        from elasticsearch import ElasticsearchException
        from elasticsearch import ConnectionError as ESConnectionError
        ELASTICSEARCH_AVAILABLE = True
    except ImportError:
        # Fallback complet
        class ElasticsearchException(Exception):
            pass
        class ESConnectionError(Exception):
            pass
        ELASTICSEARCH_AVAILABLE = False

logger.info(f"Elasticsearch disponible: {ELASTICSEARCH_AVAILABLE}")

# ==================== IMPORTS AVEC FALLBACKS ====================

# Cache - avec fallback si module manquant
try:
    from .cache import (
        LRUCache, CacheKey, CacheStats, CacheError, 
        CacheKeyError, CacheSizeError, create_search_cache
    )
    logger.info("✅ Module cache chargé")
except ImportError as e:
    logger.warning(f"Cache module non disponible - utilisation fallback: {e}")
    
    class LRUCache:
        """Implémentation fallback simple du cache LRU."""
        def __init__(self, max_size=1000, ttl_seconds=300):
            self._cache = {}
            self.max_size = max_size
            self.ttl_seconds = ttl_seconds
            
        async def get(self, key):
            return self._cache.get(key)
            
        async def set(self, key, value):
            self._cache[key] = value
            if len(self._cache) > self.max_size:
                # Simple eviction - supprimer le premier élément
                first_key = next(iter(self._cache))
                del self._cache[first_key]
                
        async def clear(self):
            self._cache.clear()
            
        def stats(self):
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": 0,
                "misses": 0
            }
    
    # Types fallback
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
    logger.info("✅ Module metrics chargé")
except ImportError as e:
    logger.warning(f"Metrics module non disponible - utilisation fallback: {e}")
    
    class SearchMetrics:
        def __init__(self):
            self.queries_count = 0
            self.cache_hits = 0
            self.cache_misses = 0
            
        def record_query(self, query, duration=0):
            self.queries_count += 1
            
        def record_cache_hit(self, key):
            self.cache_hits += 1
            
        def record_cache_miss(self, key):
            self.cache_misses += 1
            
        def get_stats(self):
            return {
                "queries_total": self.queries_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses
            }
    
    # Alias pour compatibilité
    QueryMetrics = SearchMetrics
    PerformanceMetrics = SearchMetrics
    MetricsCollector = SearchMetrics
    MetricsExporter = SearchMetrics
    
    def create_metrics_collector():
        return SearchMetrics()

# Validators - avec fallback si module manquant
try:
    from .validators import (
        QueryValidator, FilterValidator, ResultValidator,
        ParameterValidator, SecurityValidator
    )
    logger.info("✅ Module validators chargé")
except ImportError as e:
    logger.warning(f"Validators module non disponible - utilisation fallback: {e}")
    
    class BaseValidator:
        """Validateur de base fallback."""
        def validate(self, data):
            return True, []
            
        def validate_query(self, query):
            return isinstance(query, str) and len(query.strip()) > 0
    
    # Alias pour tous les validators
    QueryValidator = BaseValidator
    FilterValidator = BaseValidator
    ResultValidator = BaseValidator
    ParameterValidator = BaseValidator
    SecurityValidator = BaseValidator

# Elasticsearch helpers - avec fallback si module manquant
try:
    from .elasticsearch_helpers import (
        build_query, parse_response, format_filters,
        validate_es_config, create_es_connection
    )
    logger.info("✅ Module elasticsearch_helpers chargé")
except ImportError as e:
    logger.warning(f"Elasticsearch helpers non disponibles - utilisation fallback: {e}")
    
    def build_query(text, filters=None, options=None):
        """Fallback pour construction de requête."""
        if filters is None:
            filters = {}
        if options is None:
            options = {}
            
        return {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"searchable_text": text}}
                    ]
                }
            },
            "size": options.get("size", DEFAULT_PAGE_SIZE)
        }
    
    def parse_response(response):
        """Fallback pour parsing de réponse."""
        if not isinstance(response, dict):
            return {"results": [], "total": 0}
            
        hits = response.get("hits", {}).get("hits", [])
        total = response.get("hits", {}).get("total", {})
        
        if isinstance(total, dict):
            total_value = total.get("value", 0)
        else:
            total_value = total
            
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            source["_score"] = hit.get("_score", 0)
            results.append(source)
            
        return {
            "results": results,
            "total": total_value,
            "took": response.get("took", 0)
        }
    
    def format_filters(filters):
        """Fallback pour formatage de filtres."""
        return filters if isinstance(filters, dict) else {}
    
    def validate_es_config(config):
        """Fallback pour validation config ES."""
        if not isinstance(config, dict):
            return False
        required_keys = ["host", "port"]
        return all(key in config for key in required_keys)
    
    def create_es_connection(config):
        """Fallback pour création connexion ES."""
        return None

# ==================== EXPORTS PRINCIPAUX ====================

__all__ = [
    # Constantes
    'DEFAULT_CACHE_SIZE',
    'DEFAULT_CACHE_TTL', 
    'DEFAULT_BATCH_SIZE',
    'MAX_SEARCH_RESULTS',
    'DEFAULT_TIMEOUT',
    'DEFAULT_PAGE_SIZE',
    'HIGHLIGHT_FIELDS',
    
    # Cache
    'LRUCache',
    'CacheKey',
    'CacheStats', 
    'CacheError',
    'CacheKeyError',
    'CacheSizeError',
    'create_search_cache',
    
    # Métriques
    'SearchMetrics',
    'QueryMetrics',
    'PerformanceMetrics',
    'MetricsCollector',
    'MetricsExporter',
    'create_metrics_collector',
    
    # Validators
    'QueryValidator',
    'FilterValidator',
    'ResultValidator',
    'ParameterValidator',
    'SecurityValidator',
    
    # Elasticsearch helpers
    'build_query',
    'parse_response',
    'format_filters',
    'validate_es_config',
    'create_es_connection'
]

# Log de l'état du module
logger.info(f"Utils module chargé avec {len(__all__)} exports")