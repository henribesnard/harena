"""
Module de cache en mémoire pour le service de recherche.

Ce module fournit un remplacement pour Redis en utilisant un cache en mémoire simple
pour éviter les coûts liés à Redis.
"""
import logging
import time
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MemoryCache:
    """Cache en mémoire simple pour remplacer Redis."""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.lock = threading.RLock()  # Utilisation d'un RLock pour permettre un verrouillage récursif
        
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache."""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                if item["expires_at"] > time.time():
                    return item["value"]
                else:
                    # Supprimer l'entrée expirée
                    del self.cache[key]
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Met une valeur en cache."""
        ttl = ttl or self.default_ttl
        with self.lock:
            self.cache[key] = {
                "value": value,
                "expires_at": time.time() + ttl
            }
        return True
        
    async def delete(self, key: str) -> bool:
        """Supprime une entrée du cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
        return False
    
    async def clear(self) -> None:
        """Vide le cache."""
        with self.lock:
            self.cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtient des statistiques sur l'utilisation du cache."""
        with self.lock:
            total_keys = len(self.cache)
            current_time = time.time()
            expired_keys = sum(1 for v in self.cache.values() if v["expires_at"] <= current_time)
            valid_keys = total_keys - expired_keys
            
            # Nettoyer les clés expirées pendant qu'on y est
            if expired_keys > 0:
                self.cache = {k: v for k, v in self.cache.items() if v["expires_at"] > current_time}
            
            return {
                "total_keys": total_keys,
                "valid_keys": valid_keys,
                "expired_keys": expired_keys,
                "memory_usage_estimate_kb": self._estimate_memory_usage_kb()
            }
    
    def _estimate_memory_usage_kb(self) -> int:
        """Estimation grossière de l'utilisation mémoire (en KB)."""
        try:
            import sys
            # Estimation très approximative basée sur un échantillonnage
            if not self.cache:
                return 0
            
            # Prendre au maximum 10 clés pour estimer
            sample_keys = list(self.cache.keys())[:min(10, len(self.cache))]
            sample_size_bytes = sum(sys.getsizeof(k) + sys.getsizeof(self.cache[k]["value"]) for k in sample_keys)
            avg_entry_size = sample_size_bytes / len(sample_keys)
            
            total_estimate_bytes = avg_entry_size * len(self.cache)
            return int(total_estimate_bytes / 1024)  # Convertir en KB
        except Exception as e:
            logger.warning(f"Erreur lors de l'estimation de la mémoire du cache: {e}")
            return -1

# Instance singleton du cache
_memory_cache = MemoryCache()

# Fonctions compatibles avec l'interface existante
async def get_cache(key: str) -> Optional[Any]:
    return await _memory_cache.get(key)

async def set_cache(key: str, value: Any, ttl: int = 3600) -> bool:
    return await _memory_cache.set(key, value, ttl)

async def invalidate_cache(key: str) -> bool:
    return await _memory_cache.delete(key)

async def get_cache_stats() -> Dict[str, Any]:
    return await _memory_cache.get_stats()