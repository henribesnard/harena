"""
Cache multi-niveaux intelligent avec Redis pour Conversation Service MVP.

Ce module implémente un système de cache haute performance avec plusieurs niveaux :
- LRU Cache : Cache mémoire local avec expiration TTL
- Redis Cache : Cache distribué pour production avec persistance
- MultiLevel Cache : Combinaison L1 (mémoire) + L2 (Redis)

Performance :
- L1 : <0.1ms access time (mémoire)
- L2 : <5ms access time (Redis)
- Éviction intelligente avec LRU + TTL
- Fallback gracieux si Redis indisponible

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP avec Redis - CORRIGÉ ASYNC
"""

import time
import logging
import hashlib
import json
import pickle
from typing import Dict, Optional, Any, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import asyncio

from config.settings import settings

# Redis import avec fallback gracieux
try:
    import redis.asyncio as redis_module
    from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis_module = None
    RedisConnectionError = Exception
    RedisTimeoutError = Exception
    RedisError = Exception

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistiques de performance du cache."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = field(init=False)
    memory_usage_bytes: int = 0
    redis_hits: int = 0
    redis_misses: int = 0
    redis_errors: int = 0
    
    def __post_init__(self):
        """Calcule le taux de hit après initialisation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    
    value: Any
    created_at: float
    ttl: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré."""
        if self.ttl <= 0:  # TTL infini
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self) -> None:
        """Met à jour les statistiques d'accès."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """
    Cache LRU (Least Recently Used) avec expiration TTL.
    
    Features :
    - Éviction LRU automatique
    - Expiration TTL configurable par entrée
    - Thread-safe
    - Statistiques détaillées
    - Nettoyage automatique des entrées expirées
    """
    
    def __init__(
        self,
        maxsize: int = None,
        default_ttl: int = None
    ):
        """
        Initialise le cache LRU.
        
        Args:
            maxsize: Taille maximale du cache (défaut: MEMORY_CACHE_SIZE env var)
            default_ttl: TTL par défaut en secondes (défaut: MEMORY_CACHE_TTL env var)
        """
        # Configuration depuis les paramètres
        self.maxsize = maxsize or settings.MEMORY_CACHE_SIZE
        self.default_ttl = default_ttl or settings.MEMORY_CACHE_TTL
        
        # Stockage principal
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistiques
        self._stats = CacheStats(max_size=self.maxsize)
        
        logger.info(f"LRUCache initialized: maxsize={self.maxsize}, default_ttl={self.default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de l'entrée
            
        Returns:
            Valeur si trouvée et non expirée, None sinon
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Vérification expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._stats.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de l'entrée
            value: Valeur à stocker
            ttl: TTL en secondes (optionnel, utilise default_ttl si None)
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        with self._lock:
            # Supprime l'ancienne entrée si elle existe
            if key in self._cache:
                del self._cache[key]
            
            # Crée nouvelle entrée
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=effective_ttl
            )
            
            self._cache[key] = entry
            
            # Éviction si nécessaire
            while len(self._cache) > self.maxsize:
                # Supprime la plus ancienne (LRU)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
            
            self._stats.size = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """
        Supprime une entrée du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si l'entrée existait, False sinon
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Vide complètement le cache."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats(max_size=self.maxsize)
            
        logger.info("LRUCache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Nettoie les entrées expirées.
        
        Returns:
            Nombre d'entrées supprimées
        """
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            self._stats.size = len(self._cache)
            self._stats.evictions += len(expired_keys)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dictionnaire avec les métriques de performance
        """
        with self._lock:
            self._stats.size = len(self._cache)
            total_requests = self._stats.hits + self._stats.misses
            
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_rate": self._stats.hit_rate,
                "evictions": self._stats.evictions,
                "size": self._stats.size,
                "max_size": self._stats.max_size,
                "total_requests": total_requests,
                "memory_efficiency": f"{self._stats.size}/{self._stats.max_size} ({self._stats.size/self._stats.max_size:.1%})" if self._stats.max_size > 0 else "0%"
            }
    
    def keys(self) -> list:
        """Retourne toutes les clés actives (non expirées)."""
        with self._lock:
            active_keys = []
            for key, entry in self._cache.items():
                if not entry.is_expired():
                    active_keys.append(key)
            return active_keys


class RedisCache:
    """
    Cache Redis distribué avec sérialisation automatique.
    
    Features :
    - Connexion asynchrone Redis
    - Sérialisation pickle/json intelligente
    - Fallback gracieux si Redis indisponible
    - Préfixe configurable pour isolation
    - Monitoring des erreurs Redis
    """
    
    def __init__(
        self,
        redis_url: str = None,
        key_prefix: str = None,
        default_ttl: int = None,
        max_connections: int = None
    ):
        """
        Initialise le cache Redis.
        
        Args:
            redis_url: URL Redis (défaut: REDIS_URL env var)
            key_prefix: Préfixe des clés (défaut: REDIS_CACHE_PREFIX env var)
            default_ttl: TTL par défaut (défaut: CACHE_TTL env var)
            max_connections: Max connexions pool (défaut: REDIS_MAX_CONNECTIONS env var)
        """
        # Configuration depuis env vars
        self.redis_url = redis_url or settings.REDIS_URL or settings.REDISCLOUD_URL
        self.key_prefix = key_prefix or settings.REDIS_CACHE_PREFIX
        self.default_ttl = default_ttl or settings.CACHE_TTL
        self.max_connections = max_connections or settings.REDIS_MAX_CONNECTIONS
        
        # État Redis
        self.redis_client: Optional[Any] = None
        self.redis_available = False
        self._stats = CacheStats()
        
        # Initialisation différée (sera faite au premier appel)
        self._initialized = False
        
        logger.info(f"RedisCache configured: prefix={self.key_prefix}, ttl={self.default_ttl}s")
    
    async def _ensure_connection(self) -> bool:
        """S'assure que la connexion Redis est établie."""
        if self._initialized:
            return self.redis_available
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis package not available - RedisCache disabled")
            self._initialized = True
            return False
        
        if not self.redis_url:
            logger.warning("No Redis URL configured - RedisCache disabled")
            self._initialized = True
            return False
        
        try:
            # Configuration connexion Redis
            self.redis_client = redis_module.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                decode_responses=False  # Important pour pickle
            )
            
            # Test connexion
            await self.redis_client.ping()
            self.redis_available = True
            
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_available = False
            self.redis_client = None
        
        self._initialized = True
        return self.redis_available
    
    def _make_key(self, key: str) -> str:
        """Génère la clé Redis complète avec préfixe."""
        return f"{self.key_prefix}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Sérialise une valeur pour Redis."""
        try:
            # Essaie JSON d'abord (plus lisible, compatible)
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                json_str = json.dumps(value, ensure_ascii=False)
                return f"json:{json_str}".encode('utf-8')
            else:
                # Fallback pickle pour objets complexes
                pickled = pickle.dumps(value)
                return b"pickle:" + pickled
        except Exception as e:
            logger.warning(f"Serialization error: {e}, falling back to pickle")
            return b"pickle:" + pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Désérialise une valeur depuis Redis."""
        try:
            if data.startswith(b"json:"):
                json_str = data[5:].decode('utf-8')
                return json.loads(json_str)
            elif data.startswith(b"pickle:"):
                return pickle.loads(data[7:])
            else:
                # Format legacy - assume pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur de Redis.
        
        Args:
            key: Clé à récupérer
            
        Returns:
            Valeur si trouvée, None sinon
        """
        if not await self._ensure_connection():
            self._stats.misses += 1
            return None
        
        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)
            
            if data is None:
                self._stats.redis_misses += 1
                self._stats.misses += 1
                return None
            
            value = self._deserialize_value(data)
            if value is not None:
                self._stats.redis_hits += 1
                self._stats.hits += 1
                return value
            else:
                self._stats.redis_errors += 1
                self._stats.misses += 1
                return None
                
        except (RedisConnectionError, RedisTimeoutError, RedisError) as e:
            logger.warning(f"Redis get error: {e}")
            self._stats.redis_errors += 1
            self._stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Stocke une valeur dans Redis.
        
        Args:
            key: Clé à stocker
            value: Valeur à stocker
            ttl: TTL en secondes (optionnel)
            
        Returns:
            True si stocké avec succès
        """
        if not await self._ensure_connection():
            return False
        
        try:
            redis_key = self._make_key(key)
            serialized_value = self._serialize_value(value)
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            result = await self.redis_client.setex(
                redis_key,
                effective_ttl,
                serialized_value
            )
            
            return bool(result)
            
        except (RedisConnectionError, RedisTimeoutError, RedisError) as e:
            logger.warning(f"Redis set error: {e}")
            self._stats.redis_errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Supprime une clé de Redis.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si supprimée
        """
        if not await self._ensure_connection():
            return False
        
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            return result > 0
            
        except (RedisConnectionError, RedisTimeoutError, RedisError) as e:
            logger.warning(f"Redis delete error: {e}")
            self._stats.redis_errors += 1
            return False
    
    async def clear(self) -> None:
        """Vide toutes les clés avec le préfixe."""
        if not await self._ensure_connection():
            return
        
        try:
            pattern = f"{self.key_prefix}:*"
            keys = []
            
            # Scan pour éviter KEYS * sur gros datasets
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Redis cleared {len(keys)} keys with prefix {self.key_prefix}")
            
        except (RedisConnectionError, RedisTimeoutError, RedisError) as e:
            logger.warning(f"Redis clear error: {e}")
            self._stats.redis_errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques Redis."""
        total_requests = self._stats.hits + self._stats.misses
        redis_total = self._stats.redis_hits + self._stats.redis_misses
        
        return {
            "redis_available": self.redis_available,
            "redis_hits": self._stats.redis_hits,
            "redis_misses": self._stats.redis_misses,
            "redis_errors": self._stats.redis_errors,
            "redis_hit_rate": self._stats.redis_hits / redis_total if redis_total > 0 else 0.0,
            "total_requests": total_requests,
            "overall_hit_rate": self._stats.hits / total_requests if total_requests > 0 else 0.0
        }
    
    async def close(self) -> None:
        """Ferme la connexion Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")


class MultiLevelCache:
    """
    Cache multi-niveaux avec L1 (mémoire) et L2 (Redis).
    
    Architecture :
    - L1 : Cache mémoire ultra-rapide (LRUCache)
    - L2 : Cache Redis distribué avec persistence
    
    Stratégie :
    - Get : L1 → L2 → Miss (avec promotion L2→L1)
    - Set : L1 + L2 simultané
    - Éviction : LRU sur L1, TTL sur L2
    """
    
    def __init__(
        self,
        l1_size: int = None,
        l2_redis_url: str = None,
        l1_ttl: int = None,
        l2_ttl: int = None,
        key_prefix: str = None
    ):
        """
        Initialise le cache multi-niveaux.
        
        Args:
            l1_size: Taille L1 (défaut: env var)
            l2_redis_url: URL Redis L2 (défaut: env var)
            l1_ttl: TTL L1 (défaut: CACHE_TTL_RESPONSE env var)
            l2_ttl: TTL L2 (défaut: CACHE_TTL env var)
            key_prefix: Préfixe Redis (défaut: env var)
        """
        # Configuration depuis les paramètres avec fallbacks
        self.l1_size = l1_size or settings.MEMORY_CACHE_SIZE
        self.l1_ttl = l1_ttl or settings.CACHE_TTL_RESPONSE
        self.l2_ttl = l2_ttl or settings.CACHE_TTL
        
        # Création des caches
        self.l1_cache = LRUCache(maxsize=self.l1_size, default_ttl=self.l1_ttl)
        
        # L2 Redis seulement si activé
        redis_enabled = settings.REDIS_CACHE_ENABLED
        if redis_enabled:
            self.l2_cache = RedisCache(
                redis_url=l2_redis_url,
                key_prefix=key_prefix,
                default_ttl=self.l2_ttl
            )
        else:
            self.l2_cache = None
        
        logger.info(f"MultiLevelCache initialized: L1={self.l1_size}({self.l1_ttl}s), L2={'Redis' if self.l2_cache else 'Disabled'}({self.l2_ttl}s)")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur avec promotion L2→L1.
        
        Args:
            key: Clé à récupérer
            
        Returns:
            Valeur si trouvée, None sinon
        """
        # Essai L1 d'abord (synchrone, ultra-rapide)
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Essai L2 (Redis) si disponible
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Promotion vers L1 avec TTL court
                self.l1_cache.set(key, value, ttl=self.l1_ttl)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Stocke dans L1 et L2 simultanément.
        
        Args:
            key: Clé à stocker
            value: Valeur à stocker
            ttl: TTL optionnel (utilise TTL par défaut de chaque niveau)
        """
        # Stockage L1 avec TTL court ou spécifié
        l1_ttl = ttl if ttl is not None and ttl <= self.l1_ttl else self.l1_ttl
        self.l1_cache.set(key, value, ttl=l1_ttl)
        
        # Stockage L2 avec TTL long ou spécifié
        if self.l2_cache:
            l2_ttl = ttl if ttl is not None else self.l2_ttl
            await self.l2_cache.set(key, value, ttl=l2_ttl)
    
    async def delete(self, key: str) -> bool:
        """Supprime de tous les niveaux."""
        l1_deleted = self.l1_cache.delete(key)
        l2_deleted = False
        
        if self.l2_cache:
            l2_deleted = await self.l2_cache.delete(key)
        
        return l1_deleted or l2_deleted
    
    async def clear(self) -> None:
        """Vide tous les niveaux."""
        self.l1_cache.clear()
        
        if self.l2_cache:
            await self.l2_cache.clear()
        
        logger.info("MultiLevelCache cleared")
    
    def cleanup_expired(self) -> int:
        """Nettoie les entrées expirées L1 (L2 gère TTL automatiquement)."""
        return self.l1_cache.cleanup_expired()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statistiques combinées des deux niveaux."""
        l1_stats = self.l1_cache.get_stats()
        
        if self.l2_cache:
            l2_stats = self.l2_cache.get_stats()
            
            total_hits = l1_stats["hits"] + l2_stats["redis_hits"]
            total_misses = l1_stats["misses"] + l2_stats["redis_misses"]
            total_requests = total_hits + total_misses
            overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "overall": {
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "hit_rate": overall_hit_rate,
                    "total_requests": total_requests
                },
                "l1_cache": l1_stats,
                "l2_cache": l2_stats,
                "promotion_efficiency": f"L1 serves {l1_stats['hit_rate']:.1%} of hits"
            }
        else:
            return {
                "overall": l1_stats,
                "l1_cache": l1_stats,
                "l2_cache": {"status": "disabled"},
                "promotion_efficiency": "L2 disabled"  
            }
    
    def get_stats_sync(self) -> Dict[str, Any]:
        """Version synchrone des statistiques (L1 seulement)."""
        l1_stats = self.l1_cache.get_stats()
        
        return {
            "overall": l1_stats,
            "l1_cache": l1_stats,
            "l2_cache": {"status": "async_required"},
            "promotion_efficiency": "L1 only (sync mode)"
        }
    
    async def close(self) -> None:
        """Ferme proprement les connexions."""
        if self.l2_cache:
            await self.l2_cache.close()


# Factory functions
def create_cache(cache_type: str = "multi_level", **kwargs) -> Union[LRUCache, RedisCache, MultiLevelCache]:
    """
    Factory function pour créer un cache.
    
    Args:
        cache_type: Type de cache ("lru", "redis", "multi_level")
        **kwargs: Arguments spécifiques au type de cache
        
    Returns:
        Instance de cache configurée
    """
    if cache_type == "lru":
        return LRUCache(**kwargs)
    elif cache_type == "redis":
        return RedisCache(**kwargs)
    elif cache_type == "multi_level":
        return MultiLevelCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


def generate_cache_key(*args, **kwargs) -> str:
    """
    Génère une clé de cache standardisée.
    
    Args:
        *args: Arguments positionnels
        **kwargs: Arguments nommés
        
    Returns:
        Clé de cache SHA256 tronquée
    """
    # Combine tous les arguments en string
    key_parts = []
    
    for arg in args:
        key_parts.append(str(arg))
    
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"{key}={value}")
    
    combined = "|".join(key_parts)
    
    # SHA256 hash tronqué pour économiser la mémoire
    hash_object = hashlib.sha256(combined.encode())
    return hash_object.hexdigest()[:16]


# Instance globale pour usage package-level
_default_cache: Optional[MultiLevelCache] = None

async def get_default_cache() -> MultiLevelCache:
    """Retourne l'instance de cache par défaut du package."""
    global _default_cache
    
    if _default_cache is None:
        _default_cache = MultiLevelCache()
    
    return _default_cache


def get_default_cache_sync() -> MultiLevelCache:
    """Retourne l'instance de cache par défaut (version synchrone)."""
    global _default_cache
    
    if _default_cache is None:
        _default_cache = MultiLevelCache()
    
    return _default_cache


async def close_default_cache() -> None:
    """Ferme l'instance de cache par défaut."""
    global _default_cache
    
    if _default_cache is not None:
        await _default_cache.close()
        _default_cache = None