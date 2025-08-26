"""
Gestionnaire de cache sémantique Redis pour conversation service
"""
import logging
import json
import hashlib
from typing import Any, Dict, Optional
import redis.asyncio as redis
from datetime import datetime, timezone
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.cache")

class CacheManager:
    """Gestionnaire de cache sémantique avec Redis"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = settings.REDIS_CONVERSATION_TTL
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialisation connexion Redis"""
        if self._initialized:
            return
        
        try:
            # Utilisation de la configuration Redis existante
            if hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
                redis_url = settings.REDIS_URL
            else:
                # Configuration par défaut locale
                redis_url = "redis://localhost:6379/0"
            
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
                retry_on_timeout=True,
                socket_timeout=5
            )
            
            # Test connexion
            await self.redis_client.ping()
            self._initialized = True
            logger.info("Cache Manager Redis initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation Redis: {str(e)}")
            # Cache désactivé si Redis indisponible
            self.redis_client = None
            self._initialized = False
    
    async def close(self) -> None:
        """Fermeture propre Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        self._initialized = False
    
    async def health_check(self) -> bool:
        """Vérification santé Redis"""
        try:
            if not self._initialized or not self.redis_client:
                return False
            
            await self.redis_client.ping()
            return True
            
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            return False
    
    def _generate_cache_key(self, key: str, prefix: str = "conv") -> str:
        """Génération clé cache avec hash"""
        # Hash pour éviter problèmes de caractères spéciaux
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get_semantic_cache(
        self, 
        key: str, 
        similarity_threshold: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """Récupération cache avec similarité sémantique basique"""
        
        if not self._initialized or not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(key)
            result = await self.redis_client.get(cache_key)
            
            if result:
                cached_data = json.loads(result)
                
                # Vérification TTL
                cached_time = datetime.fromisoformat(cached_data.get("cached_at", ""))
                now = datetime.now(timezone.utc)
                
                if (now - cached_time).total_seconds() > self.default_ttl:
                    await self.redis_client.delete(cache_key)
                    return None
                
                # Pour Phase 1, pas de calcul de similarité complexe
                # Retourner directement les données
                return cached_data.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération cache: {str(e)}")
            return None
    
    async def set_semantic_cache(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Sauvegarde en cache avec TTL"""
        
        if not self._initialized or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(key)
            cache_ttl = ttl or self.default_ttl
            
            cache_data = {
                "data": data,
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "ttl": cache_ttl
            }
            
            await self.redis_client.setex(
                cache_key,
                cache_ttl,
                json.dumps(cache_data, ensure_ascii=False)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde cache: {str(e)}")
            return False
    
    async def delete_cache(self, key: str) -> bool:
        """Suppression cache"""
        
        if not self._initialized or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(key)
            result = await self.redis_client.delete(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Erreur suppression cache: {str(e)}")
            return False
    
    async def clear_all_cache(self) -> bool:
        """Nettoyage complet cache conversation"""
        
        if not self._initialized or not self.redis_client:
            return False
        
        try:
            # Suppression clés avec préfixe conversation
            keys = await self.redis_client.keys("conv:*")
            if keys:
                await self.redis_client.delete(*keys)
            
            logger.info(f"Cache cleared: {len(keys)} keys deleted")
            return True
            
        except Exception as e:
            logger.error(f"Erreur nettoyage cache: {str(e)}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques cache pour monitoring"""
        
        if not self._initialized or not self.redis_client:
            return {"status": "disabled"}
        
        try:
            info = await self.redis_client.info()
            keys_count = len(await self.redis_client.keys("conv:*"))
            
            return {
                "status": "active",
                "keys_count": keys_count,
                "memory_usage": info.get("used_memory_human", "N/A"),
                "hit_rate": "N/A",  # Sera implémenté avec métriques
                "connection_count": info.get("connected_clients", 0)
            }
            
        except Exception as e:
            logger.error(f"Erreur stats cache: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()