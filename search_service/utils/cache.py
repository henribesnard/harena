"""
Système de cache LRU pour le service de recherche - VERSION CENTRALISÉE.

Ce module implémente un cache intelligent avec TTL et LRU
pour optimiser les performances de recherche.

AMÉLIORATION:
- Configuration entièrement centralisée via config_service
- Paramètres de cache contrôlés par .env
- Plus de valeurs hardcodées
"""
import time
import threading
import asyncio
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from collections import OrderedDict

# ✅ CONFIGURATION CENTRALISÉE
from config_service.config import settings


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    ttl: Optional[float] = None


class SearchCache:
    """
    Cache LRU thread-safe avec TTL pour les résultats de recherche.
    
    Configuration entièrement centralisée via config_service.
    """
    
    def __init__(self, max_size: Optional[int] = None, ttl_seconds: Optional[float] = None, cache_type: str = "search"):
        """
        Initialise le cache avec configuration centralisée.
        
        Args:
            max_size: Taille max (utilise config centralisée si None)
            ttl_seconds: TTL par défaut (utilise config centralisée si None)
            cache_type: Type de cache pour récupérer la config appropriée
        """
        # ✅ Utiliser la configuration centralisée selon le type de cache
        if cache_type == "search":
            self.max_size = max_size or settings.SEARCH_CACHE_MAX_SIZE
            self.default_ttl = ttl_seconds or settings.SEARCH_CACHE_TTL
        elif cache_type == "embedding":
            self.max_size = max_size or settings.EMBEDDING_CACHE_MAX_SIZE
            self.default_ttl = ttl_seconds or settings.EMBEDDING_CACHE_TTL
        elif cache_type == "query_analysis":
            self.max_size = max_size or settings.QUERY_ANALYSIS_CACHE_MAX_SIZE
            self.default_ttl = ttl_seconds or settings.QUERY_ANALYSIS_CACHE_TTL
        else:
            # Fallback vers la config de recherche
            self.max_size = max_size or settings.SEARCH_CACHE_MAX_SIZE
            self.default_ttl = ttl_seconds or settings.SEARCH_CACHE_TTL
        
        self.cache_type = cache_type
        
        # Stockage principal (OrderedDict pour LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Métriques
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_items = 0
        self.puts = 0
        
        # ✅ Nettoyage automatique basé sur la config
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # 1 minute (peut être configuré plus tard)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            Valeur si trouvée et valide, None sinon
        """
        with self._lock:
            # Nettoyage périodique
            self._cleanup_if_needed()
            
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Vérifier l'expiration
            if self._is_expired(entry):
                del self._cache[key]
                self.expired_items += 1
                self.misses += 1
                return None
            
            # Mettre à jour les statistiques d'accès
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Déplacer en fin (most recently used)
            self._cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: TTL spécifique (utilise default_ttl si None)
        """
        with self._lock:
            current_time = time.time()
            
            # Créer l'entrée
            entry = CacheEntry(
                value=value,
                timestamp=current_time,
                access_count=1,
                last_access=current_time,
                ttl=ttl or self.default_ttl
            )
            
            # Si la clé existe déjà, la remplacer
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Éviction si nécessaire
                while len(self._cache) >= self.max_size:
                    self._evict_oldest()
                
                self._cache[key] = entry
            
            self.puts += 1
    
    def delete(self, key: str) -> bool:
        """
        Supprime une entrée du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si supprimée, False si n'existait pas
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Vide complètement le cache."""
        with self._lock:
            self._cache.clear()
            # Garder les métriques totales mais reset les compteurs relatifs
            self.last_cleanup = time.time()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Vérifie si une entrée a expiré."""
        if entry.ttl is None:
            return False
        
        return (time.time() - entry.timestamp) > entry.ttl
    
    def _evict_oldest(self) -> None:
        """Évince l'entrée la plus ancienne (LRU)."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.evictions += 1
    
    def _cleanup_if_needed(self) -> None:
        """Nettoie les entrées expirées si nécessaire."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self._cleanup_expired()
        self.last_cleanup = current_time
    
    def _cleanup_expired(self) -> None:
        """Nettoie toutes les entrées expirées."""
        expired_keys = []
        
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self.expired_items += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "cache_type": self.cache_type,
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "expired_items": self.expired_items,
                "puts": self.puts,
                "default_ttl_seconds": self.default_ttl,
                "cleanup_interval_seconds": self.cleanup_interval,
                "config_source": "centralized"
            }
    
    def get_keys(self) -> List[str]:
        """Retourne toutes les clés du cache (pour debug)."""
        with self._lock:
            return list(self._cache.keys())
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Retourne les informations détaillées d'une entrée."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            return {
                "key": key,
                "timestamp": entry.timestamp,
                "age_seconds": current_time - entry.timestamp,
                "ttl_seconds": entry.ttl,
                "remaining_ttl": entry.ttl - (current_time - entry.timestamp) if entry.ttl else None,
                "access_count": entry.access_count,
                "last_access": entry.last_access,
                "is_expired": self._is_expired(entry),
                "value_type": type(entry.value).__name__
            }
    
    def force_cleanup(self) -> int:
        """Force le nettoyage des entrées expirées."""
        with self._lock:
            initial_size = len(self._cache)
            self._cleanup_expired()
            return initial_size - len(self._cache)
    
    def resize(self, new_max_size: int) -> None:
        """Redimensionne le cache."""
        with self._lock:
            self.max_size = new_max_size
            
            # Éviction si nécessaire
            while len(self._cache) > self.max_size:
                self._evict_oldest()
    
    def get_most_accessed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retourne les entrées les plus accédées."""
        with self._lock:
            entries_with_stats = []
            
            for key, entry in self._cache.items():
                entries_with_stats.append({
                    "key": key,
                    "access_count": entry.access_count,
                    "last_access": entry.last_access,
                    "age_seconds": time.time() - entry.timestamp
                })
            
            # Trier par nombre d'accès
            entries_with_stats.sort(key=lambda x: x["access_count"], reverse=True)
            
            return entries_with_stats[:limit]
    
    def get_cache_efficiency(self) -> Dict[str, float]:
        """Calcule l'efficacité du cache."""
        with self._lock:
            total_requests = self.hits + self.misses
            
            if total_requests == 0:
                return {
                    "hit_rate": 0.0,
                    "miss_rate": 0.0,
                    "eviction_rate": 0.0,
                    "expiration_rate": 0.0
                }
            
            return {
                "hit_rate": self.hits / total_requests,
                "miss_rate": self.misses / total_requests,
                "eviction_rate": self.evictions / self.puts if self.puts > 0 else 0.0,
                "expiration_rate": self.expired_items / self.puts if self.puts > 0 else 0.0
            }


class MultiLevelCache:
    """
    Cache multi-niveaux pour différents types de données.
    
    Configuration automatique via config_service.
    """
    
    def __init__(self):
        self.caches: Dict[str, SearchCache] = {}
        
        # ✅ Configuration depuis config_service
        self.default_configs = {
            "search_results": {
                "max_size": settings.SEARCH_CACHE_MAX_SIZE,
                "ttl_seconds": settings.SEARCH_CACHE_TTL,
                "enabled": settings.SEARCH_CACHE_ENABLED
            },
            "embeddings": {
                "max_size": settings.EMBEDDING_CACHE_MAX_SIZE,
                "ttl_seconds": settings.EMBEDDING_CACHE_TTL,
                "enabled": settings.EMBEDDING_CACHE_ENABLED
            },
            "query_analysis": {
                "max_size": settings.QUERY_ANALYSIS_CACHE_MAX_SIZE,
                "ttl_seconds": settings.QUERY_ANALYSIS_CACHE_TTL,
                "enabled": settings.QUERY_ANALYSIS_CACHE_ENABLED
            },
            "suggestions": {
                "max_size": 200,  # Pas encore configuré dans settings
                "ttl_seconds": 600,
                "enabled": True
            }
        }
    
    def get_cache(self, cache_type: str) -> Optional[SearchCache]:
        """
        Récupère ou crée un cache d'un type donné.
        
        Args:
            cache_type: Type de cache demandé
            
        Returns:
            Cache si activé, None sinon
        """
        config = self.default_configs.get(cache_type, {
            "max_size": 100, 
            "ttl_seconds": 300,
            "enabled": True
        })
        
        # Vérifier si le cache est activé
        if not config.get("enabled", True):
            return None
        
        if cache_type not in self.caches:
            self.caches[cache_type] = SearchCache(
                max_size=config["max_size"],
                ttl_seconds=config["ttl_seconds"],
                cache_type=cache_type
            )
        
        return self.caches[cache_type]
    
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Récupère une valeur d'un cache spécifique."""
        cache = self.get_cache(cache_type)
        if cache is None:
            return None
        return cache.get(key)
    
    def put(self, cache_type: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Stocke une valeur dans un cache spécifique."""
        cache = self.get_cache(cache_type)
        if cache is not None:
            cache.put(key, value, ttl)
    
    def delete(self, cache_type: str, key: str) -> bool:
        """Supprime une entrée d'un cache spécifique."""
        if cache_type in self.caches:
            return self.caches[cache_type].delete(key)
        return False
    
    def clear_cache(self, cache_type: str) -> None:
        """Vide un cache spécifique."""
        if cache_type in self.caches:
            self.caches[cache_type].clear()
    
    def clear_all(self) -> None:
        """Vide tous les caches."""
        for cache in self.caches.values():
            cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retourne les statistiques de tous les caches."""
        stats = {}
        for cache_type, config in self.default_configs.items():
            if config.get("enabled", True) and cache_type in self.caches:
                stats[cache_type] = self.caches[cache_type].get_stats()
            else:
                stats[cache_type] = {
                    "enabled": config.get("enabled", True),
                    "status": "disabled" if not config.get("enabled", True) else "not_created"
                }
        return stats
    
    def get_total_size(self) -> int:
        """Retourne la taille totale de tous les caches."""
        return sum(len(cache._cache) for cache in self.caches.values())
    
    def force_cleanup_all(self) -> Dict[str, int]:
        """Force le nettoyage de tous les caches."""
        cleanup_results = {}
        for cache_type, cache in self.caches.items():
            cleanup_results[cache_type] = cache.force_cleanup()
        return cleanup_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la configuration des caches."""
        return {
            "cache_configs": self.default_configs,
            "active_caches": list(self.caches.keys()),
            "total_active_caches": len(self.caches),
            "config_source": "centralized (config_service)"
        }


# ==========================================
# 🎯 INSTANCE GLOBALE AVEC CONFIG CENTRALISÉE
# ==========================================

# Instance globale pour l'utilisation dans l'application
global_cache = MultiLevelCache()


def get_search_cache() -> Optional[SearchCache]:
    """Raccourci pour le cache de résultats de recherche."""
    return global_cache.get_cache("search_results")


def get_embedding_cache() -> Optional[SearchCache]:
    """Raccourci pour le cache d'embeddings."""
    return global_cache.get_cache("embeddings")


def get_query_analysis_cache() -> Optional[SearchCache]:
    """Raccourci pour le cache d'analyses de requêtes."""
    return global_cache.get_cache("query_analysis")


def get_suggestions_cache() -> Optional[SearchCache]:
    """Raccourci pour le cache de suggestions."""
    return global_cache.get_cache("suggestions")


# ==========================================
# 🛠️ FONCTIONS UTILITAIRES
# ==========================================

def generate_cache_key(*args, **kwargs) -> str:
    """
    Génère une clé de cache unique à partir des arguments.
    
    Args:
        *args: Arguments positionnels
        **kwargs: Arguments nommés
        
    Returns:
        Clé de cache unique
    """
    import hashlib
    import json
    
    # Convertir tous les arguments en string
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))
    
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (dict, list)):
            key_parts.append(f"{key}={json.dumps(value, sort_keys=True)}")
        else:
            key_parts.append(f"{key}={value}")
    
    # Créer un hash MD5 de la clé complète
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()


def cache_with_ttl(cache_type: str, ttl: Optional[float] = None):
    """
    Décorateur pour mettre en cache le résultat d'une fonction.
    
    Args:
        cache_type: Type de cache à utiliser
        ttl: TTL spécifique (optionnel)
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Vérifier si le cache est activé
            cache = global_cache.get_cache(cache_type)
            if cache is None:
                # Cache désactivé, exécuter directement
                return await func(*args, **kwargs)
            
            # Générer la clé de cache
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Essayer de récupérer du cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Exécuter la fonction et mettre en cache
            result = await func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            return result
        
        def sync_wrapper(*args, **kwargs):
            cache = global_cache.get_cache(cache_type)
            if cache is None:
                return func(*args, **kwargs)
            
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            return result
        
        # Détecter si la fonction est async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ==========================================
# 📊 MÉTRIQUES DE CACHE POUR MONITORING
# ==========================================

def get_cache_metrics() -> Dict[str, Any]:
    """Retourne les métriques consolidées de tous les caches."""
    all_stats = global_cache.get_all_stats()
    
    # Filtrer seulement les caches actifs
    active_stats = {k: v for k, v in all_stats.items() if isinstance(v, dict) and "size" in v}
    
    if not active_stats:
        return {
            "overall": {
                "total_size": 0,
                "total_hits": 0,
                "total_misses": 0,
                "overall_hit_rate": 0.0,
                "active_cache_types": 0
            },
            "by_type": all_stats,
            "configuration": global_cache.get_configuration_summary(),
            "efficiency": {
                "memory_efficiency": 0,
                "hit_rate_variance": 0
            }
        }
    
    total_size = sum(stats["size"] for stats in active_stats.values())
    total_hits = sum(stats["hits"] for stats in active_stats.values())
    total_misses = sum(stats["misses"] for stats in active_stats.values())
    total_requests = total_hits + total_misses
    
    overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
    
    return {
        "overall": {
            "total_size": total_size,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": overall_hit_rate,
            "active_cache_types": len(active_stats)
        },
        "by_type": all_stats,
        "configuration": global_cache.get_configuration_summary(),
        "efficiency": {
            "memory_efficiency": total_size / sum(stats["max_size"] for stats in active_stats.values()) if active_stats else 0,
            "hit_rate_variance": max(stats["hit_rate"] for stats in active_stats.values()) - min(stats["hit_rate"] for stats in active_stats.values()) if active_stats else 0
        }
    }


def is_cache_enabled(cache_type: str) -> bool:
    """Vérifie si un type de cache est activé."""
    config = global_cache.default_configs.get(cache_type, {})
    return config.get("enabled", True)


def get_cache_config_summary() -> Dict[str, Any]:
    """Retourne un résumé de la configuration centralisée des caches."""
    return {
        "search_cache": {
            "enabled": settings.SEARCH_CACHE_ENABLED,
            "max_size": settings.SEARCH_CACHE_MAX_SIZE,
            "ttl_seconds": settings.SEARCH_CACHE_TTL
        },
        "embedding_cache": {
            "enabled": settings.EMBEDDING_CACHE_ENABLED,
            "max_size": settings.EMBEDDING_CACHE_MAX_SIZE,
            "ttl_seconds": settings.EMBEDDING_CACHE_TTL
        },
        "query_analysis_cache": {
            "enabled": settings.QUERY_ANALYSIS_CACHE_ENABLED,
            "max_size": settings.QUERY_ANALYSIS_CACHE_MAX_SIZE,
            "ttl_seconds": settings.QUERY_ANALYSIS_CACHE_TTL
        },
        "config_source": "config_service (centralized)"
    }