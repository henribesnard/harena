"""
Cache LRU optimisé pour le Search Service.

Ce module fournit un cache LRU (Least Recently Used) avec TTL (Time To Live)
spécialement optimisé pour les résultats de recherche Elasticsearch.

FONCTIONNALITÉS:
- Cache LRU avec éviction automatique
- TTL configurable par entrée
- Statistiques détaillées (hit rate, miss rate, etc.)
- Thread-safe avec async/await
- Sérialisation automatique JSON
- Monitoring et métriques intégrés
- Nettoyage automatique des entrées expirées

OPTIMISATIONS RECHERCHE:
- Clés optimisées pour les requêtes de recherche
- Compression automatique des gros résultats
- Invalidation sélective par user_id
- Préchauffage du cache pour requêtes fréquentes

USAGE:
    cache = LRUCache(max_size=1000, ttl_seconds=300)
    
    # Stocker un résultat
    await cache.set("search:query:123", search_results)
    
    # Récupérer un résultat
    results = await cache.get("search:query:123")
    
    # Vérifier l'existence
    exists = await cache.contains("search:query:123")
    
    # Statistiques
    stats = cache.get_stats()
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from weakref import WeakSet
import gzip
import base64

# Configuration centralisée
from config_service.config import settings

T = TypeVar('T')

logger = logging.getLogger(__name__)

# ==================== EXCEPTIONS ====================

class CacheError(Exception):
    """Exception de base pour les erreurs de cache."""
    pass

class CacheKeyError(CacheError):
    """Erreur de clé de cache."""
    pass

class CacheSizeError(CacheError):
    """Erreur de taille de cache."""
    pass

# ==================== TYPES ET STRUCTURES ====================

@dataclass
class CacheEntry:
    """Entrée dans le cache."""
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self):
        """Met à jour le timestamp d'accès."""
        self.accessed_at = time.time()
        self.access_count += 1

@dataclass
class CacheStats:
    """Statistiques du cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expires: int = 0
    size: int = 0
    max_size: int = 0
    memory_usage_bytes: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    
    def update_rates(self):
        """Met à jour les taux de succès/échec."""
        total = self.hits + self.misses
        if total > 0:
            self.hit_rate = self.hits / total
            self.miss_rate = self.misses / total
        else:
            self.hit_rate = 0.0
            self.miss_rate = 0.0

@dataclass 
class CacheKey:
    """Clé de cache structurée."""
    prefix: str
    identifier: str
    user_id: Optional[int] = None
    version: Optional[str] = None
    
    def __str__(self) -> str:
        """Conversion en string pour utilisation comme clé."""
        parts = [self.prefix, self.identifier]
        if self.user_id is not None:
            parts.append(f"user:{self.user_id}")
        if self.version is not None:
            parts.append(f"v:{self.version}")
        return ":".join(parts)
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, other) -> bool:
        return str(self) == str(other)

# ==================== CACHE LRU ====================

class LRUCache(Generic[T]):
    """
    Cache LRU avec TTL optimisé pour les résultats de recherche.
    
    Fonctionnalités:
    - Éviction LRU automatique
    - TTL par entrée
    - Compression automatique
    - Statistiques détaillées
    - Thread-safe
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
        name: str = "cache",
        enable_compression: bool = True,
        compression_threshold: int = 1024,  # Compresser si > 1KB
        cleanup_interval: float = 60.0  # Nettoyage toutes les minutes
    ):
        if max_size <= 0:
            raise CacheSizeError("max_size must be positive")
        
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        self.name = name
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # Stockage ordonné pour LRU
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Statistiques
        self._stats = CacheStats(max_size=max_size)
        
        # Tâche de nettoyage
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = cleanup_interval
        self._start_cleanup_task()
        
        logger.info(
            f"LRU Cache '{name}' initialized: "
            f"max_size={max_size}, ttl={ttl_seconds}s, compression={enable_compression}"
        )
    
    def _start_cleanup_task(self):
        """Démarre la tâche de nettoyage périodique."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage des entrées expirées."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _cleanup_expired(self):
        """Nettoie les entrées expirées."""
        async with self._lock:
            expired_keys = []
            
            for key, entry in self._data.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._data[key]
                self._stats.expires += 1
                self._stats.size -= 1
            
            if expired_keys:
                logger.debug(f"Cache '{self.name}' cleaned {len(expired_keys)} expired entries")
            
            self._update_memory_usage()
    
    def _serialize_value(self, value: Any) -> tuple[bytes, bool]:
        """Sérialise et optionnellement compresse une valeur."""
        # Sérialisation JSON
        json_str = json.dumps(value, default=str, ensure_ascii=False)
        json_bytes = json_str.encode('utf-8')
        
        # Compression si nécessaire
        if self.enable_compression and len(json_bytes) > self.compression_threshold:
            compressed = gzip.compress(json_bytes)
            # Ne compresser que si ça réduit vraiment la taille
            if len(compressed) < len(json_bytes) * 0.9:
                return compressed, True
        
        return json_bytes, False
    
    def _deserialize_value(self, data: bytes, compressed: bool) -> Any:
        """Désérialise et décompresse une valeur."""
        if compressed:
            data = gzip.decompress(data)
        
        json_str = data.decode('utf-8')
        return json.loads(json_str)
    
    def _evict_lru(self):
        """Évince l'entrée la moins récemment utilisée."""
        if self._data:
            key, _ = self._data.popitem(last=False)  # FIFO = LRU
            self._stats.evictions += 1
            self._stats.size -= 1
            logger.debug(f"Cache '{self.name}' evicted LRU entry: {key}")
    
    def _update_memory_usage(self):
        """Met à jour l'estimation de l'usage mémoire."""
        total_bytes = sum(entry.size_bytes for entry in self._data.values())
        self._stats.memory_usage_bytes = total_bytes
    
    async def get(self, key: Union[str, CacheKey]) -> Optional[T]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            Valeur si trouvée, None sinon
        """
        key_str = str(key)
        
        async with self._lock:
            entry = self._data.get(key_str)
            
            if entry is None:
                self._stats.misses += 1
                self._stats.update_rates()
                return None
            
            if entry.is_expired():
                del self._data[key_str]
                self._stats.expires += 1
                self._stats.misses += 1
                self._stats.size -= 1
                self._stats.update_rates()
                return None
            
            # Déplacer en fin (plus récent)
            self._data.move_to_end(key_str)
            entry.touch()
            
            self._stats.hits += 1
            self._stats.update_rates()
            
            try:
                return self._deserialize_value(entry.value, entry.compressed)
            except Exception as e:
                logger.error(f"Cache deserialization error for key {key_str}: {e}")
                # Supprimer l'entrée corrompue
                del self._data[key_str]
                self._stats.size -= 1
                return None
    
    async def set(
        self, 
        key: Union[str, CacheKey], 
        value: T, 
        ttl_seconds: Optional[float] = None
    ) -> bool:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl_seconds: TTL spécifique pour cette entrée
            
        Returns:
            True si stocké avec succès
        """
        key_str = str(key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        
        try:
            # Sérialiser la valeur
            serialized_value, compressed = self._serialize_value(value)
            size_bytes = len(serialized_value)
            
            async with self._lock:
                # Éviction si nécessaire
                while len(self._data) >= self.max_size:
                    self._evict_lru()
                
                # Créer l'entrée
                now = time.time()
                entry = CacheEntry(
                    value=serialized_value,
                    created_at=now,
                    accessed_at=now,
                    ttl_seconds=ttl,
                    size_bytes=size_bytes,
                    compressed=compressed
                )
                
                # Stocker (remplace si existe)
                if key_str in self._data:
                    # Mise à jour
                    self._data[key_str] = entry
                    self._data.move_to_end(key_str)
                else:
                    # Nouvelle entrée
                    self._data[key_str] = entry
                    self._stats.size += 1
                
                self._update_memory_usage()
                
                logger.debug(
                    f"Cache '{self.name}' stored key {key_str}: "
                    f"{size_bytes} bytes, compressed={compressed}, ttl={ttl}"
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key {key_str}: {e}")
            return False
    
    async def delete(self, key: Union[str, CacheKey]) -> bool:
        """
        Supprime une entrée du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si supprimé, False si non trouvé
        """
        key_str = str(key)
        
        async with self._lock:
            if key_str in self._data:
                del self._data[key_str]
                self._stats.size -= 1
                self._update_memory_usage()
                logger.debug(f"Cache '{self.name}' deleted key: {key_str}")
                return True
            return False
    
    async def contains(self, key: Union[str, CacheKey]) -> bool:
        """
        Vérifie si une clé existe dans le cache.
        
        Args:
            key: Clé à vérifier
            
        Returns:
            True si la clé existe et n'a pas expiré
        """
        key_str = str(key)
        
        async with self._lock:
            entry = self._data.get(key_str)
            if entry is None:
                return False
            
            if entry.is_expired():
                del self._data[key_str]
                self._stats.expires += 1
                self._stats.size -= 1
                return False
            
            return True
    
    async def clear(self):
        """Vide complètement le cache."""
        async with self._lock:
            self._data.clear()
            self._stats.size = 0
            self._stats.memory_usage_bytes = 0
            logger.info(f"Cache '{self.name}' cleared")
    
    async def invalidate_user(self, user_id: int):
        """
        Invalide toutes les entrées d'un utilisateur.
        
        Args:
            user_id: ID utilisateur
        """
        user_pattern = f":user:{user_id}"
        
        async with self._lock:
            keys_to_delete = [
                key for key in self._data.keys()
                if user_pattern in key
            ]
            
            for key in keys_to_delete:
                del self._data[key]
                self._stats.size -= 1
            
            if keys_to_delete:
                self._update_memory_usage()
                logger.info(f"Cache '{self.name}' invalidated {len(keys_to_delete)} entries for user {user_id}")
    
    def get_stats(self) -> CacheStats:
        """Retourne les statistiques du cache."""
        self._stats.update_rates()
        return self._stats
    
    async def get_info(self) -> Dict[str, Any]:
        """Retourne des informations détaillées sur le cache."""
        stats = self.get_stats()
        
        async with self._lock:
            # Calculer la distribution des TTL
            ttl_distribution = {}
            for entry in self._data.values():
                if entry.ttl_seconds is None:
                    key = "no_ttl"
                else:
                    key = f"{int(entry.ttl_seconds)}s"
                ttl_distribution[key] = ttl_distribution.get(key, 0) + 1
            
            # Calculer l'âge moyen des entrées
            now = time.time()
            ages = [now - entry.created_at for entry in self._data.values()]
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return {
                "name": self.name,
                "stats": {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "hit_rate": stats.hit_rate,
                    "evictions": stats.evictions,
                    "expires": stats.expires,
                    "size": stats.size,
                    "max_size": stats.max_size,
                    "memory_usage_mb": stats.memory_usage_bytes / 1024 / 1024
                },
                "config": {
                    "default_ttl": self.default_ttl,
                    "enable_compression": self.enable_compression,
                    "compression_threshold": self.compression_threshold
                },
                "runtime": {
                    "avg_entry_age_seconds": avg_age,
                    "ttl_distribution": ttl_distribution,
                    "cleanup_interval": self._cleanup_interval
                }
            }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

# ==================== FACTORY ====================

def create_search_cache(
    max_size: Optional[int] = None,
    ttl_seconds: Optional[float] = None,
    name: str = "search_cache"
) -> LRUCache:
    """
    Crée un cache optimisé pour les résultats de recherche.
    
    Args:
        max_size: Taille maximum (défaut depuis config)
        ttl_seconds: TTL par défaut (défaut depuis config)
        name: Nom du cache
        
    Returns:
        Cache LRU configuré
    """
    if max_size is None:
        max_size = getattr(settings, 'SEARCH_CACHE_SIZE', 1000)
    if ttl_seconds is None:
        ttl_seconds = getattr(settings, 'SEARCH_CACHE_TTL', 300)
    
    return LRUCache(
        max_size=max_size,
        ttl_seconds=ttl_seconds,
        name=name,
        enable_compression=True,
        compression_threshold=1024
    )