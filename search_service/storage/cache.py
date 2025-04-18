"""
Module de gestion du cache pour le service de recherche.

Ce module fournit des fonctionnalités pour mettre en cache les résultats
de recherche et les embeddings fréquemment utilisés afin d'améliorer
les performances.
"""
import logging
import json
import pickle
from typing import Any, Optional, Dict, List, Union
import asyncio

# Import conditionnel de Redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from search_service.core.config import settings

logger = logging.getLogger(__name__)

# Client Redis global
_redis_client = None

async def get_redis_client():
    """
    Obtient une instance du client Redis (singleton).
    
    Returns:
        Client Redis ou None si Redis n'est pas disponible
    """
    global _redis_client
    
    if not REDIS_AVAILABLE:
        logger.warning("Module Redis non disponible. Cache désactivé.")
        return None
    
    if _redis_client is None:
        try:
            _redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=False  # Désactivé car on utilise pickle
            )
            logger.info(f"Client Redis connecté à {settings.REDIS_URL}")
        except Exception as e:
            logger.error(f"Impossible de se connecter à Redis: {str(e)}")
            return None
    
    return _redis_client

async def get_cache(key: str) -> Optional[Any]:
    """
    Récupère une valeur du cache.
    
    Args:
        key: Clé de cache
        
    Returns:
        Valeur mise en cache ou None si pas trouvée
    """
    redis = await get_redis_client()
    if not redis:
        return None
    
    try:
        cached_value = await redis.get(key)
        if cached_value:
            # Désérialiser avec pickle pour les types complexes
            return pickle.loads(cached_value)
        return None
    except Exception as e:
        logger.warning(f"Erreur lors de la récupération du cache pour '{key}': {str(e)}")
        return None

async def set_cache(key: str, value: Any, ttl: int = 3600) -> bool:
    """
    Met une valeur en cache.
    
    Args:
        key: Clé de cache
        value: Valeur à mettre en cache
        ttl: Durée de vie en secondes (par défaut: 1 heure)
        
    Returns:
        True si la mise en cache a réussi, False sinon
    """
    redis = await get_redis_client()
    if not redis:
        return False
    
    try:
        # Sérialiser avec pickle pour les types complexes
        serialized_value = pickle.dumps(value)
        await redis.set(key, serialized_value, ex=ttl)
        return True
    except Exception as e:
        logger.warning(f"Erreur lors de la mise en cache pour '{key}': {str(e)}")
        return False

async def invalidate_cache(key: str) -> bool:
    """
    Invalide une entrée du cache.
    
    Args:
        key: Clé de cache à invalider
        
    Returns:
        True si l'invalidation a réussi, False sinon
    """
    redis = await get_redis_client()
    if not redis:
        return False
    
    try:
        await redis.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Erreur lors de l'invalidation du cache pour '{key}': {str(e)}")
        return False

async def get_cache_stats() -> Dict[str, Any]:
    """
    Obtient des statistiques sur le cache.
    
    Returns:
        Dictionnaire de statistiques
    """
    redis = await get_redis_client()
    if not redis:
        return {"status": "unavailable"}
    
    try:
        info = await redis.info()
        keys_count = await redis.dbsize()
        
        return {
            "status": "available",
            "keys_count": keys_count,
            "used_memory": info.get("used_memory_human", "unknown"),
            "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1) + 0.001),
            "uptime_seconds": info.get("uptime_in_seconds", 0)
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques de cache: {str(e)}")
        return {"status": "error", "message": str(e)}