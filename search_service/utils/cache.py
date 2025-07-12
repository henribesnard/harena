"""
Système de cache LRU pour le Search Service
Cache intelligent pour requêtes fréquentes et optimisation performance
"""

import json
import logging
import hashlib
import time
import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
from threading import Lock


logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Stratégies de cache disponibles"""
    DISABLED = "disabled"
    MEMORY_ONLY = "memory_only"
    REDIS_ONLY = "redis_only"
    HYBRID = "hybrid"  # Memory L1 + Redis L2


class CacheLevel(str, Enum):
    """Niveaux de cache"""
    L1_MEMORY = "l1_memory"    # Cache mémoire rapide
    L2_REDIS = "l2_redis"      # Cache Redis persistant
    DISK = "disk"              # Cache disque (futur)


class EvictionPolicy(str, Enum):
    """Politiques d'éviction"""
    LRU = "lru"               # Least Recently Used
    LFU = "lfu"               # Least Frequently Used
    TTL = "ttl"               # Time To Live
    SIZE = "size"             # Taille maximale


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calcule la taille approximative de l'entrée"""
        try:
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            elif isinstance(self.value, (dict, list)):
                return len(json.dumps(self.value, separators=(',', ':')).encode('utf-8'))
            else:
                return len(str(self.value).encode('utf-8'))
        except:
            return 1024  # Taille par défaut
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré"""
        if not self.ttl_seconds:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time
    
    def touch(self):
        """Met à jour le timestamp d'accès"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistiques du cache"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired_entries: int = 0
    total_size_bytes: int = 0
    current_entries: int = 0
    max_entries: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    average_access_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    def update_rates(self):
        """Met à jour les taux de hit/miss"""
        total_requests = self.hits + self.misses
        if total_requests > 0:
            self.hit_rate = self.hits / total_requests
            self.miss_rate = self.misses / total_requests
        else:
            self.hit_rate = 0.0
            self.miss_rate = 0.0
    
    def reset(self):
        """Remet à zéro les statistiques"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_entries = 0
        self.update_rates()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les stats en dictionnaire"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
            "miss_rate": round(self.miss_rate, 3),
            "evictions": self.evictions,
            "expired_entries": self.expired_entries,
            "current_entries": self.current_entries,
            "max_entries": self.max_entries,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "average_access_time_ms": round(self.average_access_time_ms, 2)
        }


class LRUCache:
    """
    Cache LRU (Least Recently Used) thread-safe
    Optimisé pour les résultats de recherche Elasticsearch
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: int = 300,  # 5 minutes
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        name: str = "default"
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.name = name
        
        # Stockage principal
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        
        # Métriques et statistiques
        self._stats = CacheStats(max_entries=max_size)
        self._current_memory = 0
        self._access_times: List[float] = []
        
        # Tags pour invalidation groupée
        self._tag_map: Dict[str, set] = {}
        
        logger.info(f"LRU Cache '{name}' initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache
        
        Args:
            key: Clé de cache
            
        Returns:
            Valeur si trouvée, None sinon
        """
        start_time = time.time()
        
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                self._stats.update_rates()
                return None
            
            entry = self._cache[key]
            
            # Vérifier expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                self._stats.expired_entries += 1
                self._stats.update_rates()
                return None
            
            # Mettre à jour l'accès
            entry.touch()
            
            # Réorganiser pour LRU (move to end)
            if self.eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            self._stats.hits += 1
            self._stats.update_rates()
            
            # Enregistrer le temps d'accès
            access_time = (time.time() - start_time) * 1000  # ms
            self._access_times.append(access_time)
            if len(self._access_times) > 100:  # Garder seulement les 100 derniers
                self._access_times = self._access_times[-100:]
            
            return entry.value
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Ajoute une valeur au cache
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Time to live en secondes (optionnel)
            tags: Tags pour invalidation groupée
            
        Returns:
            True si ajouté avec succès
        """
        if not key or value is None:
            return False
        
        ttl = ttl or self.default_ttl
        tags = tags or []
        
        with self._lock:
            # Créer l'entrée
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                ttl_seconds=ttl,
                tags=tags
            )
            
            # Vérifier si l'entrée existe déjà
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory -= old_entry.size_bytes
                self._remove_from_tags(key, old_entry.tags)
            
            # Vérifier les limites avant ajout
            if not self._can_add_entry(entry):
                self._evict_entries()
                
                # Recheck après éviction
                if not self._can_add_entry(entry):
                    logger.warning(f"Cannot add entry to cache: {key}")
                    return False
            
            # Ajouter au cache
            self._cache[key] = entry
            self._current_memory += entry.size_bytes
            self._stats.total_size_bytes = self._current_memory
            self._stats.current_entries = len(self._cache)
            
            # Ajouter aux tags
            self._add_to_tags(key, tags)
            
            # Move to end pour LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Vide complètement le cache"""
        with self._lock:
            self._cache.clear()
            self._tag_map.clear()
            self._current_memory = 0
            self._stats.current_entries = 0
            self._stats.total_size_bytes = 0
            logger.info(f"Cache '{self.name}' cleared")
    
    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalide toutes les entrées avec un tag donné
        
        Args:
            tag: Tag à invalider
            
        Returns:
            Nombre d'entrées supprimées
        """
        with self._lock:
            if tag not in self._tag_map:
                return 0
            
            keys_to_remove = list(self._tag_map[tag])
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def cleanup_expired(self) -> int:
        """
        Nettoie les entrées expirées
        
        Returns:
            Nombre d'entrées supprimées
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                self._stats.expired_entries += 1
            
            return len(expired_keys)
    
    def _can_add_entry(self, entry: CacheEntry) -> bool:
        """Vérifie si une entrée peut être ajoutée"""
        # Vérification taille
        if len(self._cache) >= self.max_size:
            return False
        
        # Vérification mémoire
        if self._current_memory + entry.size_bytes > self.max_memory_bytes:
            return False
        
        return True
    
    def _evict_entries(self):
        """Évince des entrées selon la politique configurée"""
        if self.eviction_policy == EvictionPolicy.LRU:
            self._evict_lru()
        elif self.eviction_policy == EvictionPolicy.LFU:
            self._evict_lfu()
        elif self.eviction_policy == EvictionPolicy.TTL:
            self._evict_expired()
        else:
            self._evict_lru()  # Fallback
    
    def _evict_lru(self):
        """Évince les entrées les moins récemment utilisées"""
        while (len(self._cache) >= self.max_size or 
               self._current_memory > self.max_memory_bytes * 0.8):
            if not self._cache:
                break
            
            # Supprimer le premier (plus ancien)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._stats.evictions += 1
    
    def _evict_lfu(self):
        """Évince les entrées les moins fréquemment utilisées"""
        if not self._cache:
            return
        
        # Trier par access_count croissant
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count
        )
        
        # Supprimer jusqu'à 25% du cache
        entries_to_remove = len(sorted_entries) // 4
        for i in range(min(entries_to_remove, len(sorted_entries))):
            key = sorted_entries[i][0]
            self._remove_entry(key)
            self._stats.evictions += 1
    
    def _evict_expired(self):
        """Évince toutes les entrées expirées"""
        expired_count = self.cleanup_expired()
        self._stats.evictions += expired_count
    
    def _remove_entry(self, key: str):
        """Supprime une entrée et met à jour les métadonnées"""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory -= entry.size_bytes
            self._stats.total_size_bytes = self._current_memory
            self._remove_from_tags(key, entry.tags)
            del self._cache[key]
            self._stats.current_entries = len(self._cache)
    
    def _add_to_tags(self, key: str, tags: List[str]):
        """Ajoute une clé aux tags"""
        for tag in tags:
            if tag not in self._tag_map:
                self._tag_map[tag] = set()
            self._tag_map[tag].add(key)
    
    def _remove_from_tags(self, key: str, tags: List[str]):
        """Supprime une clé des tags"""
        for tag in tags:
            if tag in self._tag_map:
                self._tag_map[tag].discard(key)
                if not self._tag_map[tag]:
                    del self._tag_map[tag]
    
    def get_stats(self) -> CacheStats:
        """Retourne les statistiques du cache"""
        with self._lock:
            # Mettre à jour les statistiques calculées
            if self._access_times:
                self._stats.average_access_time_ms = sum(self._access_times) / len(self._access_times)
            
            self._stats.memory_usage_mb = self._current_memory / (1024 * 1024)
            self._stats.current_entries = len(self._cache)
            self._stats.total_size_bytes = self._current_memory
            
            return self._stats
    
    def get_info(self) -> Dict[str, Any]:
        """Informations détaillées du cache"""
        stats = self.get_stats()
        return {
            "name": self.name,
            "eviction_policy": self.eviction_policy.value,
            "max_size": self.max_size,
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "default_ttl": self.default_ttl,
            "stats": stats.to_dict()
        }
    
    def reset_stats(self):
        """Remet à zéro les statistiques"""
        with self._lock:
            self._stats.reset()
            self._access_times.clear()


class SmartCache:
    """
    Cache intelligent pour requêtes de recherche
    Optimisé spécifiquement pour les patterns du Search Service
    """
    
    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.MEMORY_ONLY,
        memory_cache_size: int = 1000,
        memory_cache_mb: int = 100,
        default_ttl: int = 300,
        name: str = "smart_cache"
    ):
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.name = name
        
        # Cache L1 (mémoire)
        if strategy in [CacheStrategy.MEMORY_ONLY, CacheStrategy.HYBRID]:
            self.memory_cache = LRUCache(
                max_size=memory_cache_size,
                max_memory_mb=memory_cache_mb,
                default_ttl=default_ttl,
                name=f"{name}_memory"
            )
        else:
            self.memory_cache = None
        
        # Cache L2 (Redis) - TODO: Implémenter
        if strategy in [CacheStrategy.REDIS_ONLY, CacheStrategy.HYBRID]:
            self.redis_cache = None  # RedisCache() - à implémenter
        else:
            self.redis_cache = None
        
        logger.info(f"SmartCache '{name}' initialized with strategy: {strategy}")
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur depuis le cache (L1 puis L2)"""
        if self.strategy == CacheStrategy.DISABLED:
            return None
        
        # Essayer L1 (mémoire) d'abord
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                return value
        
        # Essayer L2 (Redis) si configuré
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                # Promouvoir vers L1
                if self.memory_cache:
                    self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Stocke une valeur dans le cache"""
        if self.strategy == CacheStrategy.DISABLED:
            return False
        
        ttl = ttl or self.default_ttl
        success = True
        
        # Stocker en L1 (mémoire)
        if self.memory_cache:
            success &= self.memory_cache.set(key, value, ttl, tags)
        
        # Stocker en L2 (Redis)
        if self.redis_cache:
            success &= self.redis_cache.set(key, value, ttl, tags)
        
        return success
    
    def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        if self.strategy == CacheStrategy.DISABLED:
            return False
        
        success = True
        
        if self.memory_cache:
            success &= self.memory_cache.delete(key)
        
        if self.redis_cache:
            success &= self.redis_cache.delete(key)
        
        return success
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalide toutes les entrées avec un tag"""
        if self.strategy == CacheStrategy.DISABLED:
            return 0
        
        total_removed = 0
        
        if self.memory_cache:
            total_removed += self.memory_cache.invalidate_by_tag(tag)
        
        if self.redis_cache:
            total_removed += self.redis_cache.invalidate_by_tag(tag)
        
        return total_removed
    
    def clear(self):
        """Vide tous les niveaux de cache"""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.redis_cache:
            self.redis_cache.clear()
    
    def cleanup(self) -> Dict[str, int]:
        """Nettoie les entrées expirées"""
        cleanup_stats = {}
        
        if self.memory_cache:
            cleanup_stats["memory"] = self.memory_cache.cleanup_expired()
        
        if self.redis_cache:
            cleanup_stats["redis"] = self.redis_cache.cleanup_expired()
        
        return cleanup_stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de tous les niveaux"""
        stats = {"strategy": self.strategy.value, "name": self.name}
        
        if self.memory_cache:
            stats["memory"] = self.memory_cache.get_stats().to_dict()
        
        if self.redis_cache:
            stats["redis"] = self.redis_cache.get_stats().to_dict()
        
        return stats


class CacheKeyGenerator:
    """Générateur de clés de cache intelligentes"""
    
    @staticmethod
    def generate_search_key(
        user_id: int,
        query_hash: str,
        filters_hash: Optional[str] = None,
        sort_hash: Optional[str] = None
    ) -> str:
        """Génère une clé pour les résultats de recherche"""
        components = [f"search", f"user:{user_id}", f"query:{query_hash}"]
        
        if filters_hash:
            components.append(f"filters:{filters_hash}")
        
        if sort_hash:
            components.append(f"sort:{sort_hash}")
        
        return ":".join(components)
    
    @staticmethod
    def generate_aggregation_key(
        user_id: int,
        agg_type: str,
        filters_hash: Optional[str] = None
    ) -> str:
        """Génère une clé pour les agrégations"""
        components = [f"agg", f"user:{user_id}", f"type:{agg_type}"]
        
        if filters_hash:
            components.append(f"filters:{filters_hash}")
        
        return ":".join(components)
    
    @staticmethod
    def generate_count_key(user_id: int, filters_hash: str) -> str:
        """Génère une clé pour les comptages"""
        return f"count:user:{user_id}:filters:{filters_hash}"
    
    @staticmethod
    def hash_dict(data: Dict[str, Any]) -> str:
        """Crée un hash stable d'un dictionnaire"""
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(json_str.encode()).hexdigest()[:12]
    
    @staticmethod
    def get_user_tag(user_id: int) -> str:
        """Tag pour invalider toutes les données d'un utilisateur"""
        return f"user:{user_id}"
    
    @staticmethod
    def get_category_tag(category: str) -> str:
        """Tag pour invalider par catégorie"""
        return f"category:{category}"


class CacheManager:
    """
    Gestionnaire central du cache pour le Search Service
    Coordonne les différents types de cache et stratégies
    """
    
    def __init__(self):
        self.caches: Dict[str, SmartCache] = {}
        self._init_default_caches()
        logger.info("CacheManager initialized")
    
    def _init_default_caches(self):
        """Initialise les caches par défaut"""
        # Cache pour résultats de recherche
        self.caches["search"] = SmartCache(
            strategy=CacheStrategy.MEMORY_ONLY,
            memory_cache_size=500,
            memory_cache_mb=50,
            default_ttl=300,  # 5 minutes
            name="search_results"
        )
        
        # Cache pour agrégations (plus long TTL)
        self.caches["aggregations"] = SmartCache(
            strategy=CacheStrategy.MEMORY_ONLY,
            memory_cache_size=200,
            memory_cache_mb=20,
            default_ttl=900,  # 15 minutes
            name="aggregations"
        )
        
        # Cache pour comptages (très long TTL)
        self.caches["counts"] = SmartCache(
            strategy=CacheStrategy.MEMORY_ONLY,
            memory_cache_size=100,
            memory_cache_mb=10,
            default_ttl=1800,  # 30 minutes
            name="counts"
        )
    
    def get_cache(self, cache_type: str) -> Optional[SmartCache]:
        """Récupère un cache par type"""
        return self.caches.get(cache_type)
    
    def cache_search_results(
        self,
        user_id: int,
        query_data: Dict[str, Any],
        results: Any,
        ttl: Optional[int] = None
    ) -> str:
        """Cache des résultats de recherche"""
        cache = self.get_cache("search")
        if not cache:
            return ""
        
        # Générer la clé
        query_hash = CacheKeyGenerator.hash_dict(query_data.get("query", {}))
        filters_hash = CacheKeyGenerator.hash_dict(query_data.get("filters", {}))
        
        key = CacheKeyGenerator.generate_search_key(
            user_id, query_hash, filters_hash
        )
        
        # Tags pour invalidation
        tags = [
            CacheKeyGenerator.get_user_tag(user_id),
            "search_results"
        ]
        
        # Ajouter tags par catégorie si applicable
        if "filters" in query_data:
            for filter_obj in query_data["filters"].get("required", []):
                if filter_obj.get("field") == "category_name":
                    category_tag = CacheKeyGenerator.get_category_tag(filter_obj.get("value"))
                    tags.append(category_tag)
        
        # Stocker
        cache.set(key, results, ttl, tags)
        return key
    
    def get_search_results(
        self,
        user_id: int,
        query_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Récupère des résultats de recherche en cache"""
        cache = self.get_cache("search")
        if not cache:
            return None
        
        # Générer la même clé
        query_hash = CacheKeyGenerator.hash_dict(query_data.get("query", {}))
        filters_hash = CacheKeyGenerator.hash_dict(query_data.get("filters", {}))
        
        key = CacheKeyGenerator.generate_search_key(
            user_id, query_hash, filters_hash
        )
        
        return cache.get(key)
    
    def invalidate_user_cache(self, user_id: int) -> int:
        """Invalide tout le cache d'un utilisateur"""
        total_removed = 0
        user_tag = CacheKeyGenerator.get_user_tag(user_id)
        
        for cache in self.caches.values():
            total_removed += cache.invalidate_by_tag(user_tag)
        
        logger.info(f"Invalidated {total_removed} cache entries for user {user_id}")
        return total_removed
    
    def cleanup_all_caches(self) -> Dict[str, Dict[str, int]]:
        """Nettoie tous les caches"""
        cleanup_stats = {}
        
        for cache_name, cache in self.caches.items():
            cleanup_stats[cache_name] = cache.cleanup()
        
        return cleanup_stats
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Statistiques globales de tous les caches"""
        stats = {
            "total_caches": len(self.caches),
            "caches": {}
        }
        
        for cache_name, cache in self.caches.items():
            stats["caches"][cache_name] = cache.get_stats()
        
        return stats


# === INSTANCE GLOBALE ===

# Instance globale du gestionnaire de cache
cache_manager = CacheManager()

# Alias pour compatibilité avec l'import attendu
global_cache_manager = cache_manager


# === DÉCORATEURS UTILES ===

def cached(
    cache_type: str = "search",
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """
    Décorateur pour mettre en cache le résultat d'une fonction
    
    Args:
        cache_type: Type de cache à utiliser
        ttl: Time to live custom
        key_func: Fonction pour générer la clé de cache
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_type)
            if not cache:
                return await func(*args, **kwargs)
            
            # Générer la clé
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Clé par défaut basée sur les arguments
                key_data = {"args": args, "kwargs": kwargs}
                cache_key = f"{func.__name__}:{CacheKeyGenerator.hash_dict(key_data)}"
            
            # Essayer de récupérer depuis le cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Exécuter la fonction et cacher le résultat
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# === FONCTIONS UTILITAIRES ===

def create_cache_key(query: str, filters: Dict[str, Any] = None, user_id: str = None) -> str:
    """Fonction utilitaire pour créer des clés de cache"""
    filters = filters or {}
    query_hash = CacheKeyGenerator.hash_dict({"query": query})
    filters_hash = CacheKeyGenerator.hash_dict(filters)
    
    if user_id:
        return CacheKeyGenerator.generate_search_key(int(user_id), query_hash, filters_hash)
    else:
        return f"search:query:{query_hash}:filters:{filters_hash}"


def serialize_cache_value(value: Any) -> str:
    """Sérialise une valeur pour le cache"""
    try:
        if isinstance(value, (dict, list)):
            return json.dumps(value, separators=(',', ':'))
        else:
            return str(value)
    except Exception as e:
        logger.warning(f"Could not serialize cache value: {e}")
        return ""


def deserialize_cache_value(data: str) -> Any:
    """Désérialise une valeur du cache"""
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        # Si ce n'est pas du JSON, retourner tel quel
        return data
    except Exception as e:
        logger.warning(f"Could not deserialize cache value: {e}")
        return None


def get_cache_statistics() -> Dict[str, Any]:
    """Récupère les statistiques globales du cache"""
    return cache_manager.get_global_stats()


def get_cache_stats() -> Dict[str, Any]:
    """Récupère les statistiques globales du cache (alias)"""
    return get_cache_statistics()


def clear_all_caches():
    """Vide tous les caches"""
    for cache in cache_manager.caches.values():
        cache.clear()
    logger.info("All caches cleared")


def cleanup_expired_entries() -> Dict[str, Dict[str, int]]:
    """Nettoie les entrées expirées de tous les caches"""
    return cache_manager.cleanup_all_caches()


async def periodic_cleanup(interval_seconds: int = 300):
    """Tâche de nettoyage périodique du cache"""
    while True:
        try:
            cleanup_stats = cleanup_expired_entries()
            total_cleaned = sum(
                sum(cache_stats.values()) if isinstance(cache_stats, dict) else 0
                for cache_stats in cleanup_stats.values()
            )
            if total_cleaned > 0:
                logger.info(f"Periodic cleanup removed {total_cleaned} expired entries")
        except Exception as e:
            logger.error(f"Error during periodic cache cleanup: {e}")
        
        await asyncio.sleep(interval_seconds)


# === EXPORTS ===

__all__ = [
    # Enums
    "CacheStrategy",
    "CacheLevel", 
    "EvictionPolicy",
    
    # Classes principales
    "LRUCache",
    "SmartCache",
    "CacheManager",
    "CacheKeyGenerator",
    
    # Structures de données
    "CacheEntry",
    "CacheStats",
    
    # Instances globales
    "cache_manager",
    "global_cache_manager",
    
    # Décorateurs
    "cached",
    
    # Fonctions utilitaires
    "create_cache_key",
    "serialize_cache_value",
    "deserialize_cache_value",
    "get_cache_statistics",
    "get_cache_stats",
    "clear_all_caches",
    "cleanup_expired_entries",
    "periodic_cleanup"
]