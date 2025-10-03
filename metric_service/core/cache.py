"""
Redis Cache Manager pour les métriques
"""
import redis.asyncio as redis
import json
import logging
from typing import Optional, Any
from datetime import timedelta
import os

logger = logging.getLogger(__name__)

class CacheManager:
    """Gestionnaire de cache Redis pour les métriques"""

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.default_ttl = 300  # 5 minutes par défaut

    async def connect(self):
        """Connexion à Redis"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis = await redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info(f"✅ Connecté à Redis: {redis_url}")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Redis: {e}")
            self.redis = None

    async def disconnect(self):
        """Déconnexion de Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("✅ Déconnecté de Redis")

    async def ping(self) -> bool:
        """Vérifier la connexion Redis"""
        try:
            if self.redis:
                await self.redis.ping()
                return True
        except:
            pass
        return False

    async def get(self, key: str) -> Optional[Any]:
        """Récupérer une valeur depuis le cache"""
        if not self.redis:
            return None

        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"❌ Erreur lecture cache {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Stocker une valeur dans le cache"""
        if not self.redis:
            return

        try:
            ttl = ttl or self.default_ttl
            await self.redis.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
        except Exception as e:
            logger.error(f"❌ Erreur écriture cache {key}: {e}")

    async def delete(self, key: str):
        """Supprimer une clé du cache"""
        if not self.redis:
            return

        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"❌ Erreur suppression cache {key}: {e}")

    async def clear_user_cache(self, user_id: int):
        """Supprimer tout le cache d'un utilisateur"""
        if not self.redis:
            return

        try:
            pattern = f"metrics:user:{user_id}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"✅ Cache utilisateur {user_id} supprimé ({len(keys)} clés)")
        except Exception as e:
            logger.error(f"❌ Erreur suppression cache utilisateur {user_id}: {e}")

    def make_key(self, user_id: int, metric_type: str, **params) -> str:
        """Générer une clé de cache"""
        param_str = ":".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        if param_str:
            return f"metrics:user:{user_id}:{metric_type}:{param_str}"
        return f"metrics:user:{user_id}:{metric_type}"

# Instance globale
cache_manager = CacheManager()
