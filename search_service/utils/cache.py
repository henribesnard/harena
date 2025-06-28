"""
Système de cache LRU pour le service de recherche.

Ce module implémente un cache intelligent avec TTL et LRU
pour optimiser les performances de recherche.
"""
import time
import threading
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from collections import OrderedDict


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
    
    Fonctionnalités:
    - LRU (Least Recently Used) eviction
    - TTL (Time To Live) par entrée
    - Thread-safe
    - Métriques détaillées
    - Nettoyage automatique
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300):
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        
        # Stockage principal (OrderedDict pour LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Métriques
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_items = 0
        self.puts = 0
        
        # Nettoyage automatique
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # 1 minute
    
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
                "cleanup_interval_seconds": self.cleanup_interval
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
    
    Permet d'avoir des caches séparés avec des TTL différents
    pour différents types de données (requêtes, embeddings, etc.)
    """
    
    def __init__(self):
        self.caches: Dict[str, SearchCache] = {}
        self.default_configs = {
            "search_results": {"max_size": 1000, "ttl_seconds": 300},
            "embeddings": {"max_size": 5000, "ttl_seconds": 3600},
            "query_analysis": {"max_size": 500, "ttl_seconds": 1800},
            "suggestions": {"max_size": 200, "ttl_seconds": 600}
        }
    
    def get_cache(self, cache_type: str) -> SearchCache:
        """Récupère ou crée un cache d'un type donné."""
        if cache_type not in self.caches:
            config = self.default_configs.get(cache_type, {"max_size": 100, "ttl_seconds": 300})
            self.caches[cache_type] = SearchCache(**config)
        
        return self.caches[cache_type]
    
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Récupère une valeur d'un cache spécifique."""
        cache = self.get_cache(cache_type)
        return cache.get(key)
    
    def put(self, cache_type: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Stocke une valeur dans un cache spécifique."""
        cache = self.get_cache(cache_type)
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
        return {
            cache_type: cache.get_stats()
            for cache_type, cache in self.caches.items()
        }
    
    def get_total_size(self) -> int:
        """Retourne la taille totale de tous les caches."""
        return sum(len(cache._cache) for cache in self.caches.values())
    
    def force_cleanup_all(self) -> Dict[str, int]:
        """Force le nettoyage de tous les caches."""
        cleanup_results = {}
        for cache_type, cache in self.caches.items():
            cleanup_results[cache_type] = cache.force_cleanup()
        return cleanup_results


# Instance globale pour l'utilisation dans l'application
global_cache = MultiLevelCache()


def get_search_cache() -> SearchCache:
    """Raccourci pour le cache de résultats de recherche."""
    return global_cache.get_cache("search_results")


def get_embedding_cache() -> SearchCache:
    """Raccourci pour le cache d'embeddings."""
    return global_cache.get_cache("embeddings")


def get_query_analysis_cache() -> SearchCache:
    """Raccourci pour le cache d'analyses de requêtes."""
    return global_cache.get_cache("query_analysis")


def get_suggestions_cache() -> SearchCache:
    """Raccourci pour le cache de suggestions."""
    return global_cache.get_cache("suggestions")


# Fonctions utilitaires

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
            # Générer la clé de cache
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Essayer de récupérer du cache
            cached_result = global_cache.get(cache_type, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Exécuter la fonction et mettre en cache
            result = await func(*args, **kwargs)
            global_cache.put(cache_type, cache_key, result, ttl)
            return result
        
        def sync_wrapper(*args, **kwargs):
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            cached_result = global_cache.get(cache_type, cache_key)
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            global_cache.put(cache_type, cache_key, result, ttl)
            return result
        
        # Détecter si la fonction est async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Métriques de cache pour monitoring

def get_cache_metrics() -> Dict[str, Any]:
    """Retourne les métriques consolidées de tous les caches."""
    all_stats = global_cache.get_all_stats()
    
    total_size = sum(stats["size"] for stats in all_stats.values())
    total_hits = sum(stats["hits"] for stats in all_stats.values())
    total_misses = sum(stats["misses"] for stats in all_stats.values())
    total_requests = total_hits + total_misses
    
    overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
    
    return {
        "overall": {
            "total_size": total_size,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": overall_hit_rate,
            "cache_types": len(all_stats)
        },
        "by_type": all_stats,
        "efficiency": {
            "memory_efficiency": total_size / sum(stats["max_size"] for stats in all_stats.values()) if all_stats else 0,
            "hit_rate_variance": max(stats["hit_rate"] for stats in all_stats.values()) - min(stats["hit_rate"] for stats in all_stats.values()) if all_stats else 0
        }
    }


# Import pour le décorateur
import asyncio