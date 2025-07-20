"""
Cache local en mémoire pour patterns fréquents
Complément au cache Redis pour ultra-performance
"""

import asyncio
from typing import Dict, Any, Optional
from collections import OrderedDict

from .cache_strategies import CacheStrategy, HybridStrategy, CacheEntry
from conversation_service.utils.logging import get_logger

logger = get_logger(__name__)


class LocalCache:
    """
    Cache local LRU avec TTL pour patterns ultra-fréquents
    Utilisé en complément Redis pour réduire latence réseau
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.strategy: CacheStrategy = HybridStrategy()
        self._lock = asyncio.Lock()
        
        # Métriques locales
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Récupération avec gestion TTL et LRU"""
        async with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Vérification expiration
            if entry.is_expired:
                del self.cache[key]
                self.misses += 1
                return None
            
            # Marquer accès et réorganiser LRU
            entry.mark_accessed()
            self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Stockage avec éviction automatique"""
        async with self._lock:
            ttl = ttl or self.default_ttl
            
            # Création nouvelle entrée
            entry = CacheEntry(value=value, ttl_seconds=ttl)
            
            # Si clé existe, mise à jour
            if key in self.cache:
                self.cache[key] = entry
                self.cache.move_to_end(key)
                return True
            
            # Éviction si taille max atteinte
            if len(self.cache) >= self.max_size:
                await self._evict_entries()
            
            # Ajout nouvelle entrée
            self.cache[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """Suppression entrée"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> int:
        """Nettoyage complet"""
        async with self._lock:
            count = len(self.cache)
            self.cache.clear()
            return count
    
    async def _evict_entries(self):
        """Éviction intelligente basée sur stratégie"""
        
        # Éviction entrées expirées d'abord
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.evictions += 1
        
        # Si encore trop d'entrées, éviction basée stratégie
        if len(self.cache) >= self.max_size:
            # Calcul scores éviction pour toutes les entrées
            eviction_candidates = []
            for key, entry in self.cache.items():
                priority = await self.strategy.get_eviction_priority(entry)
                eviction_candidates.append((priority, key))
            
            # Tri par priorité décroissante et éviction
            eviction_candidates.sort(reverse=True)
            
            # Éviction 10% des entrées
            evict_count = max(1, len(self.cache) // 10)
            for i in range(min(evict_count, len(eviction_candidates))):
                _, key_to_evict = eviction_candidates[i]
                del self.cache[key_to_evict]
                self.evictions += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statistiques cache local"""
        async with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "memory_usage_estimate": len(self.cache) * 1024  # Estimation grossière
            }
    
    async def cleanup_expired(self):
        """Nettoyage périodique entrées expirées"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
