"""
💾 Cache LRU Search Service - Optimisation Performance
======================================================

Système de cache multi-niveaux pour optimiser les performances
du Search Service avec stratégies LRU, TTL et limitation mémoire.

Responsabilités:
- Cache résultats recherche avec LRU
- Cache requêtes Elasticsearch fréquentes
- Cache agrégations financières
- Gestion mémoire intelligente
- Métriques cache et monitoring
- Invalidation sélective et cleanup
"""

import logging
import pickle
import hashlib
import json
import threading
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps
import weakref

from search_service.models.service_contracts import SearchServiceQuery, SearchServiceResponse

logger = logging.getLogger(__name__)

# =============================================================================
# 📊 CONFIGURATION CACHE
# =============================================================================

# Tailles par défaut
DEFAULT_CACHE_SIZE = 1000
DEFAULT_TTL_SECONDS = 300  # 5 minutes
MAX_CACHE_MEMORY_MB = 100
MAX_KEY_LENGTH = 250

# Types de cache
CACHE_TYPES = {
    "search_results": {"size": 500, "ttl": 300},
    "query_templates": {"size": 100, "ttl": 3600},
    "aggregations": {"size": 200, "ttl": 600},
    "user_queries": {"size": 1000, "ttl": 1800},
    "performance": {"size": 50, "ttl": 60}
}

# =============================================================================
# 📋 MODÈLES ET CONFIGURATION
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration d'un cache."""
    max_size: int = DEFAULT_CACHE_SIZE
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    max_memory_mb: int = MAX_CACHE_MEMORY_MB
    enable_metrics: bool = True
    auto_cleanup: bool = True
    cleanup_interval: int = 60  # secondes

@dataclass
class CacheKey:
    """Clé de cache structurée."""
    cache_type: str
    user_id: Optional[int] = None
    query_hash: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """Génère la clé string."""
        parts = [self.cache_type]
        
        if self.user_id:
            parts.append(f"user_{self.user_id}")
        
        if self.query_hash:
            parts.append(f"query_{self.query_hash}")
        
        if self.parameters:
            # Hash des paramètres pour clé stable
            params_str = json.dumps(self.parameters, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            parts.append(f"params_{params_hash}")
        
        key = ":".join(parts)
        
        # Limitation longueur
        if len(key) > MAX_KEY_LENGTH:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            key = f"{self.cache_type}:hash_{key_hash}"
        
        return key

@dataclass
class CacheStats:
    """Statistiques de cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    total_items: int = 0
    hit_rate: float = 0.0
    last_cleanup: Optional[datetime] = None
    
    def update_hit_rate(self):
        """Met à jour le taux de hit."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée est expirée."""
        if self.ttl_seconds is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time
    
    def touch(self):
        """Met à jour l'accès."""
        self.last_accessed = datetime.now()
        self.access_count += 1

# =============================================================================
# 🎯 STRATÉGIES DE CACHE
# =============================================================================

class CacheStrategy:
    """Stratégie de cache abstraite."""
    
    def should_cache(self, key: str, value: Any) -> bool:
        """Détermine si une valeur doit être mise en cache."""
        return True
    
    def should_evict(self, entry: CacheEntry) -> bool:
        """Détermine si une entrée doit être évincée."""
        return entry.is_expired()
    
    def get_eviction_candidates(self, cache_data: Dict[str, CacheEntry], count: int) -> List[str]:
        """Retourne les candidats à l'éviction."""
        return []

class LRUCacheStrategy(CacheStrategy):
    """Stratégie LRU (Least Recently Used)."""
    
    def get_eviction_candidates(self, cache_data: Dict[str, CacheEntry], count: int) -> List[str]:
        """Retourne les entrées les moins récemment utilisées."""
        sorted_entries = sorted(
            cache_data.items(),
            key=lambda x: x[1].last_accessed
        )
        return [key for key, _ in sorted_entries[:count]]

class TTLCacheStrategy(CacheStrategy):
    """Stratégie basée sur TTL (Time To Live)."""
    
    def should_evict(self, entry: CacheEntry) -> bool:
        """Évince les entrées expirées."""
        return entry.is_expired()
    
    def get_eviction_candidates(self, cache_data: Dict[str, CacheEntry], count: int) -> List[str]:
        """Retourne les entrées expirées."""
        expired_keys = []
        for key, entry in cache_data.items():
            if entry.is_expired():
                expired_keys.append(key)
        return expired_keys

class SizeLimitedCacheStrategy(CacheStrategy):
    """Stratégie basée sur la taille mémoire."""
    
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
    
    def should_cache(self, key: str, value: Any) -> bool:
        """Vérifie si la valeur n'est pas trop grande."""
        try:
            size = len(pickle.dumps(value))
            return size < self.max_size_bytes * 0.1  # Max 10% de la taille totale
        except:
            return False
    
    def get_eviction_candidates(self, cache_data: Dict[str, CacheEntry], count: int) -> List[str]:
        """Retourne les plus grosses entrées."""
        sorted_entries = sorted(
            cache_data.items(),
            key=lambda x: x[1].size_bytes,
            reverse=True
        )
        return [key for key, _ in sorted_entries[:count]]

# =============================================================================
# 🏗️ CACHE CORE
# =============================================================================

class SearchResultsCache:
    """Cache principal pour les résultats de recherche."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Stratégies combinées
        self._strategies = [
            TTLCacheStrategy(),
            LRUCacheStrategy(),
            SizeLimitedCacheStrategy(self.config.max_memory_mb * 1024 * 1024)
        ]
        
        # Cleanup automatique
        if self.config.auto_cleanup:
            self._start_cleanup_thread()
        
        logger.info(f"SearchResultsCache initialized with size={self.config.max_size}")
    
    def get(self, key: Union[str, CacheKey]) -> Optional[Any]:
        """Récupère une valeur du cache."""
        str_key = str(key)
        
        with self._lock:
            if str_key not in self._data:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return None
            
            entry = self._data[str_key]
            
            # Vérification expiration
            if entry.is_expired():
                del self._data[str_key]
                self._stats.misses += 1
                self._stats.total_items -= 1
                self._stats.update_hit_rate()
                return None
            
            # Touch et move to end (LRU)
            entry.touch()
            self._data.move_to_end(str_key)
            
            self._stats.hits += 1
            self._stats.update_hit_rate()
            
            return entry.value
    
    def set(self, key: Union[str, CacheKey], value: Any, ttl: Optional[int] = None) -> bool:
        """Met une valeur en cache."""
        str_key = str(key)
        
        # Vérification stratégies
        for strategy in self._strategies:
            if not strategy.should_cache(str_key, value):
                return False
        
        with self._lock:
            # Calcul taille
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0
            
            # Création entrée
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.config.ttl_seconds
            )
            
            # Éviction si nécessaire
            if len(self._data) >= self.config.max_size:
                self._evict_entries(1)
            
            # Stockage
            self._data[str_key] = entry
            self._data.move_to_end(str_key)
            
            # Mise à jour stats
            self._stats.total_items = len(self._data)
            self._stats.memory_usage_bytes += size_bytes
            
            return True
    
    def delete(self, key: Union[str, CacheKey]) -> bool:
        """Supprime une entrée du cache."""
        str_key = str(key)
        
        with self._lock:
            if str_key in self._data:
                entry = self._data.pop(str_key)
                self._stats.total_items -= 1
                self._stats.memory_usage_bytes -= entry.size_bytes
                return True
            return False
    
    def clear(self) -> None:
        """Vide complètement le cache."""
        with self._lock:
            self._data.clear()
            self._stats = CacheStats()
            logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStats:
        """Retourne les statistiques du cache."""
        with self._lock:
            self._stats.total_items = len(self._data)
            return self._stats
    
    def _evict_entries(self, count: int) -> None:
        """Évince des entrées selon les stratégies."""
        candidates = set()
        
        # Collecte candidats de toutes les stratégies
        for strategy in self._strategies:
            strategy_candidates = strategy.get_eviction_candidates(self._data, count * 2)
            candidates.update(strategy_candidates)
        
        # Éviction LRU si pas assez de candidats
        if len(candidates) < count:
            lru_candidates = list(self._data.keys())[:count]
            candidates.update(lru_candidates)
        
        # Éviction effective
        evicted = 0
        for key in list(candidates):
            if evicted >= count:
                break
            if key in self._data:
                entry = self._data.pop(key)
                self._stats.memory_usage_bytes -= entry.size_bytes
                self._stats.evictions += 1
                evicted += 1
        
        self._stats.total_items = len(self._data)
    
    def cleanup_expired(self) -> int:
        """Nettoie les entrées expirées."""
        expired_keys = []
        
        with self._lock:
            for key, entry in self._data.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
    def cleanup_expired(self) -> int:
        """Nettoie les entrées expirées."""
        expired_keys = []
        
        with self._lock:
            for key, entry in self._data.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self._data:
                    entry = self._data.pop(key)
                    self._stats.memory_usage_bytes -= entry.size_bytes
            
            self._stats.total_items = len(self._data)
            self._stats.last_cleanup = datetime.now()
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
        return len(expired_keys)
    
    def _start_cleanup_thread(self):
        """Démarre le thread de nettoyage automatique."""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval)
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

# =============================================================================
# 🎯 CACHES SPÉCIALISÉS
# =============================================================================

class QueryCache(SearchResultsCache):
    """Cache spécialisé pour les requêtes."""
    
    def __init__(self):
        config = CacheConfig(
            max_size=CACHE_TYPES["query_templates"]["size"],
            ttl_seconds=CACHE_TYPES["query_templates"]["ttl"]
        )
        super().__init__(config)

class AggregationCache(SearchResultsCache):
    """Cache spécialisé pour les agrégations."""
    
    def __init__(self):
        config = CacheConfig(
            max_size=CACHE_TYPES["aggregations"]["size"],
            ttl_seconds=CACHE_TYPES["aggregations"]["ttl"]
        )
        super().__init__(config)

class UserQueryCache(SearchResultsCache):
    """Cache spécialisé par utilisateur."""
    
    def __init__(self):
        config = CacheConfig(
            max_size=CACHE_TYPES["user_queries"]["size"],
            ttl_seconds=CACHE_TYPES["user_queries"]["ttl"]
        )
        super().__init__(config)
    
    def get_user_key(self, user_id: int, base_key: str) -> CacheKey:
        """Génère une clé spécifique utilisateur."""
        return CacheKey(
            cache_type="user_query",
            user_id=user_id,
            query_hash=hashlib.md5(base_key.encode()).hexdigest()[:16]
        )

class FinancialDataCache(SearchResultsCache):
    """Cache spécialisé pour données financières."""
    
    def __init__(self):
        config = CacheConfig(
            max_size=500,
            ttl_seconds=600,  # 10 minutes pour données financières
            max_memory_mb=50
        )
        super().__init__(config)
    
    def cache_transaction_results(self, query: SearchServiceQuery, response: SearchServiceResponse) -> bool:
        """Cache les résultats de transaction."""
        key = CacheKey(
            cache_type="financial_transactions",
            user_id=query.query_metadata.user_id,
            query_hash=self._generate_query_hash(query)
        )
        
        return self.set(key, response, ttl=600)
    
    def get_transaction_results(self, query: SearchServiceQuery) -> Optional[SearchServiceResponse]:
        """Récupère les résultats de transaction."""
        key = CacheKey(
            cache_type="financial_transactions",
            user_id=query.query_metadata.user_id,
            query_hash=self._generate_query_hash(query)
        )
        
        return self.get(key)
    
    def _generate_query_hash(self, query: SearchServiceQuery) -> str:
        """Génère un hash de requête financière."""
        # Sérialisation déterministe pour hash stable
        query_dict = {
            "intent": query.query_metadata.intent_type.value,
            "fields": sorted(query.search_parameters.fields),
            "filters": self._serialize_filters(query.filters),
            "aggregations": query.aggregations.enabled,
            "limit": query.search_parameters.limit
        }
        
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()[:16]
    
    def _serialize_filters(self, filters) -> Dict:
        """Sérialise les filtres pour hash."""
        return {
            "required": len(filters.required),
            "optional": len(filters.optional),
            "ranges": len(filters.ranges),
            "text_search": bool(filters.text_search and filters.text_search.query)
        }

class PerformanceCache(SearchResultsCache):
    """Cache pour métriques de performance."""
    
    def __init__(self):
        config = CacheConfig(
            max_size=CACHE_TYPES["performance"]["size"],
            ttl_seconds=CACHE_TYPES["performance"]["ttl"]
        )
        super().__init__(config)

# =============================================================================
# 🏗️ GESTIONNAIRE CACHE GLOBAL
# =============================================================================

class CacheManager:
    """Gestionnaire centralisé de tous les caches."""
    
    def __init__(self):
        self._caches: Dict[str, SearchResultsCache] = {
            "search_results": SearchResultsCache(),
            "queries": QueryCache(),
            "aggregations": AggregationCache(),
            "user_queries": UserQueryCache(),
            "financial_data": FinancialDataCache(),
            "performance": PerformanceCache()
        }
        
        self._global_stats = CacheStats()
        logger.info("CacheManager initialized with all cache types")
    
    def get_cache(self, cache_type: str) -> Optional[SearchResultsCache]:
        """Récupère un cache par type."""
        return self._caches.get(cache_type)
    
    def get_global_stats(self) -> Dict[str, CacheStats]:
        """Récupère les stats de tous les caches."""
        stats = {}
        total_hits = 0
        total_misses = 0
        total_memory = 0
        total_items = 0
        
        for cache_type, cache in self._caches.items():
            cache_stats = cache.get_stats()
            stats[cache_type] = cache_stats
            
            total_hits += cache_stats.hits
            total_misses += cache_stats.misses
            total_memory += cache_stats.memory_usage_bytes
            total_items += cache_stats.total_items
        
        # Stats globales
        self._global_stats.hits = total_hits
        self._global_stats.misses = total_misses
        self._global_stats.memory_usage_bytes = total_memory
        self._global_stats.total_items = total_items
        self._global_stats.update_hit_rate()
        
        stats["global"] = self._global_stats
        return stats
    
    def clear_all_caches(self) -> None:
        """Vide tous les caches."""
        for cache in self._caches.values():
            cache.clear()
        logger.info("All caches cleared")
    
    def cleanup_all_expired(self) -> int:
        """Nettoie les entrées expirées de tous les caches."""
        total_cleaned = 0
        for cache_type, cache in self._caches.items():
            cleaned = cache.cleanup_expired()
            total_cleaned += cleaned
        
        logger.info(f"Cleaned {total_cleaned} expired entries across all caches")
        return total_cleaned
    
    def invalidate_user_cache(self, user_id: int) -> int:
        """Invalide le cache d'un utilisateur spécifique."""
        invalidated = 0
        
        for cache_type, cache in self._caches.items():
            with cache._lock:
                keys_to_remove = []
                for key in cache._data.keys():
                    if f"user_{user_id}" in key:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    cache.delete(key)
                    invalidated += 1
        
        logger.info(f"Invalidated {invalidated} cache entries for user {user_id}")
        return invalidated

# =============================================================================
# 🔧 FONCTIONS UTILITAIRES
# =============================================================================

# Instance globale
_cache_manager = CacheManager()

def cache_search_results(
    query: SearchServiceQuery, 
    response: SearchServiceResponse,
    cache_type: str = "search_results"
) -> bool:
    """Cache les résultats de recherche."""
    cache = _cache_manager.get_cache(cache_type)
    if not cache:
        return False
    
    # Génération clé
    key = CacheKey(
        cache_type="search_results",
        user_id=query.query_metadata.user_id,
        query_hash=_generate_search_query_hash(query)
    )
    
    return cache.set(key, response)

def get_cached_search_results(
    query: SearchServiceQuery,
    cache_type: str = "search_results"
) -> Optional[SearchServiceResponse]:
    """Récupère les résultats de recherche mis en cache."""
    cache = _cache_manager.get_cache(cache_type)
    if not cache:
        return None
    
    key = CacheKey(
        cache_type="search_results",
        user_id=query.query_metadata.user_id,
        query_hash=_generate_search_query_hash(query)
    )
    
    return cache.get(key)

def cache_aggregations(
    query: SearchServiceQuery,
    aggregation_results: Dict[str, Any]
) -> bool:
    """Cache les résultats d'agrégation."""
    cache = _cache_manager.get_cache("aggregations")
    if not cache:
        return False
    
    key = CacheKey(
        cache_type="aggregations",
        user_id=query.query_metadata.user_id,
        query_hash=_generate_aggregation_hash(query)
    )
    
    return cache.set(key, aggregation_results, ttl=600)

def cache_query_templates(template_name: str, template_data: Dict[str, Any]) -> bool:
    """Cache les templates de requête."""
    cache = _cache_manager.get_cache("queries")
    if not cache:
        return False
    
    key = CacheKey(
        cache_type="query_template",
        parameters={"template": template_name}
    )
    
    return cache.set(key, template_data, ttl=3600)

def invalidate_user_cache(user_id: int) -> int:
    """Invalide le cache d'un utilisateur."""
    return _cache_manager.invalidate_user_cache(user_id)

def invalidate_query_cache(query_pattern: str) -> int:
    """Invalide les requêtes correspondant à un pattern."""
    invalidated = 0
    
    for cache_type in ["search_results", "aggregations", "user_queries"]:
        cache = _cache_manager.get_cache(cache_type)
        if cache:
            with cache._lock:
                keys_to_remove = []
                for key in cache._data.keys():
                    if query_pattern in key:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    cache.delete(key)
                    invalidated += 1
    
    return invalidated

def clear_all_caches() -> None:
    """Vide tous les caches."""
    _cache_manager.clear_all_caches()

# =============================================================================
# 📊 MÉTRIQUES CACHE
# =============================================================================

def get_cache_hit_rate(cache_type: str = None) -> float:
    """Récupère le taux de hit d'un cache."""
    if cache_type:
        cache = _cache_manager.get_cache(cache_type)
        if cache:
            stats = cache.get_stats()
            return stats.hit_rate
        return 0.0
    else:
        # Taux global
        global_stats = _cache_manager.get_global_stats()
        return global_stats["global"].hit_rate

def get_cache_memory_usage(cache_type: str = None) -> int:
    """Récupère l'usage mémoire d'un cache en bytes."""
    if cache_type:
        cache = _cache_manager.get_cache(cache_type)
        if cache:
            stats = cache.get_stats()
            return stats.memory_usage_bytes
        return 0
    else:
        # Usage global
        global_stats = _cache_manager.get_global_stats()
        return global_stats["global"].memory_usage_bytes

def get_cache_statistics() -> Dict[str, Any]:
    """Récupère toutes les statistiques de cache."""
    stats = _cache_manager.get_global_stats()
    
    # Format pour export
    formatted_stats = {}
    for cache_type, cache_stats in stats.items():
        formatted_stats[cache_type] = {
            "hits": cache_stats.hits,
            "misses": cache_stats.misses,
            "hit_rate": round(cache_stats.hit_rate, 3),
            "evictions": cache_stats.evictions,
            "total_items": cache_stats.total_items,
            "memory_usage_mb": round(cache_stats.memory_usage_bytes / (1024 * 1024), 2),
            "last_cleanup": cache_stats.last_cleanup.isoformat() if cache_stats.last_cleanup else None
        }
    
    return formatted_stats

# =============================================================================
# 🔧 HELPERS INTERNES
# =============================================================================

def _generate_search_query_hash(query: SearchServiceQuery) -> str:
    """Génère un hash pour une requête de recherche."""
    # Éléments clés pour le hash
    hash_elements = {
        "intent": query.query_metadata.intent_type.value,
        "query_type": query.search_parameters.query_type.value,
        "fields": sorted(query.search_parameters.fields),
        "limit": query.search_parameters.limit,
        "filters_count": {
            "required": len(query.filters.required),
            "optional": len(query.filters.optional),
            "ranges": len(query.filters.ranges)
        }
    }
    
    # Ajout text search si présent
    if query.filters.text_search and query.filters.text_search.query:
        hash_elements["text_search"] = query.filters.text_search.query
    
    hash_str = json.dumps(hash_elements, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:16]

def _generate_aggregation_hash(query: SearchServiceQuery) -> str:
    """Génère un hash pour les agrégations."""
    if not query.aggregations.enabled:
        return "no_agg"
    
    hash_elements = {
        "user_id": query.query_metadata.user_id,
        "agg_types": [req.agg_type.value for req in query.aggregations.requests],
        "agg_fields": [req.field for req in query.aggregations.requests],
        "group_by": sorted(query.aggregations.group_by)
    }
    
    hash_str = json.dumps(hash_elements, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:16]

def generate_cache_key(
    cache_type: str,
    user_id: Optional[int] = None,
    **kwargs
) -> CacheKey:
    """Génère une clé de cache standardisée."""
    return CacheKey(
        cache_type=cache_type,
        user_id=user_id,
        parameters=kwargs if kwargs else None
    )

def serialize_cache_value(value: Any) -> bytes:
    """Sérialise une valeur pour le cache."""
    try:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Cache serialization error: {e}")
        return b""

def deserialize_cache_value(data: bytes) -> Any:
    """Désérialise une valeur du cache."""
    try:
        return pickle.loads(data)
    except Exception as e:
        logger.error(f"Cache deserialization error: {e}")
        return None

# =============================================================================
# 🎯 DÉCORATEURS CACHE
# =============================================================================

def cached_search(cache_type: str = "search_results", ttl: Optional[int] = None):
    """Décorateur pour mettre en cache les résultats de recherche."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(query: SearchServiceQuery, *args, **kwargs):
            # Tentative récupération cache
            cached_result = get_cached_search_results(query, cache_type)
            if cached_result:
                logger.debug(f"Cache hit for query {query.query_metadata.query_id}")
                return cached_result
            
            # Exécution fonction
            result = await func(query, *args, **kwargs)
            
            # Mise en cache du résultat
            if result:
                cache_search_results(query, result, cache_type)
                logger.debug(f"Cached result for query {query.query_metadata.query_id}")
            
            return result
        return wrapper
    return decorator

# =============================================================================
# 📊 EXPORTS
# =============================================================================

__all__ = [
    # Cache principal
    "SearchResultsCache", "QueryCache", "AggregationCache",
    # Cache managers
    "CacheManager", "CacheKey", "CacheStats", "CacheConfig",
    # Cache spécialisé
    "UserQueryCache", "FinancialDataCache", "PerformanceCache",
    # Cache operations
    "cache_search_results", "get_cached_search_results", "cache_aggregations", "cache_query_templates",
    "invalidate_user_cache", "invalidate_query_cache", "clear_all_caches",
    # Cache metrics
    "get_cache_hit_rate", "get_cache_memory_usage", "get_cache_statistics",
    # Cache strategies
    "CacheStrategy", "LRUCacheStrategy", "TTLCacheStrategy", "SizeLimitedCacheStrategy",
    # Helpers
    "generate_cache_key", "serialize_cache_value", "deserialize_cache_value",
    # Décorateurs
    "cached_search",
    # Constantes
    "DEFAULT_CACHE_SIZE", "DEFAULT_TTL_SECONDS", "MAX_CACHE_MEMORY_MB",
]