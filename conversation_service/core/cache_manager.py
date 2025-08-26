"""
Gestionnaire de cache sémantique Redis optimisé pour conversation service
"""
import logging
import json
import hashlib
from typing import Any, Dict, Optional, List
import redis.asyncio as redis
from datetime import datetime, timezone
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.cache")

class CacheManager:
    """Gestionnaire de cache sémantique avec Redis optimisé et robuste"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = getattr(settings, 'REDIS_CONVERSATION_TTL', 3600)
        self.cache_ttl_intent = getattr(settings, 'CACHE_TTL_INTENT', 300)
        self.cache_ttl_entity = getattr(settings, 'CACHE_TTL_ENTITY', 180)
        self.cache_ttl_response = getattr(settings, 'CACHE_TTL_RESPONSE', 60)
        self._initialized = False
        self._health_status = "unknown"
        self._connection_pool = None
        
        # Métriques cache
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "total_operations": 0
        }
        
        logger.info("CacheManager initialisé avec TTLs configurables")
    
    async def initialize(self) -> None:
        """Initialisation connexion Redis avec pool optimisé"""
        if self._initialized:
            return
        
        try:
            # Configuration Redis URL
            redis_url = self._get_redis_url()
            
            # Pool de connexions optimisé
            self._connection_pool = redis.ConnectionPool.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30
            )
            
            self.redis_client = redis.Redis(
                connection_pool=self._connection_pool,
                retry_on_error=[redis.BusyLoadingError, redis.ConnectionError],
                retry=redis.Retry(backoff=redis.ExponentialBackoff(), retries=3)
            )
            
            # Test connexion initial
            await self._test_connection()
            
            self._initialized = True
            self._health_status = "healthy"
            logger.info("Cache Manager Redis initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation Redis: {str(e)}")
            self.redis_client = None
            self._initialized = False
            self._health_status = "unhealthy"
    
    def _get_redis_url(self) -> str:
        """Récupération URL Redis avec fallbacks"""
        if hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
            return settings.REDIS_URL
        
        # Construction URL depuis composants
        host = getattr(settings, 'REDIS_HOST', 'localhost')
        port = getattr(settings, 'REDIS_PORT', 6379)
        db = getattr(settings, 'REDIS_DB', 0)
        password = getattr(settings, 'REDIS_PASSWORD', None)
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"
    
    async def _test_connection(self) -> None:
        """Test connexion Redis avec validation fonctionnelle"""
        if not self.redis_client:
            raise Exception("Redis client non initialisé")
        
        # Test ping
        await self.redis_client.ping()
        
        # Test opérations basiques
        test_key = "harena:cache:test"
        test_value = {"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
        
        await self.redis_client.setex(test_key, 10, json.dumps(test_value))
        retrieved = await self.redis_client.get(test_key)
        
        if not retrieved:
            raise Exception("Test set/get échoué")
        
        parsed = json.loads(retrieved)
        if not parsed.get("test"):
            raise Exception("Test données incorrectes")
        
        # Nettoyage
        await self.redis_client.delete(test_key)
        
        logger.debug("Test connexion Redis réussi")
    
    async def close(self) -> None:
        """Fermeture propre Redis avec métriques finales"""
        if self.redis_client:
            try:
                # Log métriques finales
                logger.info(
                    f"Fermeture cache - Hits: {self._metrics['hits']}, "
                    f"Misses: {self._metrics['misses']}, "
                    f"Errors: {self._metrics['errors']}, "
                    f"Hit Rate: {self._calculate_hit_rate():.2f}%"
                )
                
                await self.redis_client.close()
                
            except Exception as e:
                logger.warning(f"Erreur fermeture Redis: {str(e)}")
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
        
        self.redis_client = None
        self._connection_pool = None
        self._initialized = False
        self._health_status = "closed"
    
    async def health_check(self) -> bool:
        """Vérification santé Redis avec diagnostic détaillé"""
        try:
            if not self._initialized or not self.redis_client:
                self._health_status = "not_initialized"
                return False
            
            # Test ping avec timeout court
            pong = await self.redis_client.ping()
            if not pong:
                self._health_status = "ping_failed"
                return False
            
            # Test mémoire disponible
            info = await self.redis_client.info("memory")
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)
            
            if max_memory > 0:
                memory_usage = (used_memory / max_memory) * 100
                if memory_usage > 95:
                    logger.warning(f"Redis mémoire critique: {memory_usage:.1f}%")
                    self._health_status = "memory_critical"
                    return False
            
            self._health_status = "healthy"
            return True
            
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            self._health_status = f"error: {str(e)}"
            return False
    
    def _generate_cache_key(self, key: str, prefix: str = "harena:conv") -> str:
        """Génération clé cache avec hash et préfixe intelligent"""
        # Hash SHA-256 pour sécurité et longueur optimale
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]  # 16 chars suffisent
        return f"{prefix}:{key_hash}"
    
    def _get_cache_ttl(self, cache_type: str) -> int:
        """Récupération TTL selon type de cache"""
        ttl_mapping = {
            "intent": self.cache_ttl_intent,
            "entity": self.cache_ttl_entity,
            "response": self.cache_ttl_response,
            "default": self.default_ttl
        }
        return ttl_mapping.get(cache_type, self.default_ttl)
    
    async def get_semantic_cache(
        self, 
        key: str, 
        similarity_threshold: float = 0.8,
        cache_type: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Récupération cache avec sémantique basique et validation TTL
        
        Args:
            key: Clé de cache
            similarity_threshold: Seuil similarité (non utilisé en Phase 1)
            cache_type: Type de cache pour TTL spécialisé
            
        Returns:
            Données cachées ou None
        """
        
        if not self._initialized or not self.redis_client:
            self._metrics["misses"] += 1
            return None
        
        try:
            self._metrics["total_operations"] += 1
            cache_key = self._generate_cache_key(key, f"harena:{cache_type}")
            
            # Récupération avec timeout
            result = await self.redis_client.get(cache_key)
            
            if not result:
                self._metrics["misses"] += 1
                logger.debug(f"Cache miss: {key[:50]}...")
                return None
            
            # Parsing et validation données
            cached_data = json.loads(result)
            
            # Validation structure
            if not isinstance(cached_data, dict):
                logger.warning("Données cache format invalide")
                await self.redis_client.delete(cache_key)
                self._metrics["errors"] += 1
                return None
            
            # Validation TTL manuelle si nécessaire
            if "cached_at" in cached_data:
                cached_time = datetime.fromisoformat(cached_data["cached_at"])
                now = datetime.now(timezone.utc)
                ttl = self._get_cache_ttl(cache_type)
                
                if (now - cached_time).total_seconds() > ttl:
                    logger.debug("Données cache expirées")
                    await self.redis_client.delete(cache_key)
                    self._metrics["misses"] += 1
                    return None
            
            # Succès
            self._metrics["hits"] += 1
            logger.debug(f"Cache hit: {key[:50]}...")
            return cached_data.get("data", cached_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur JSON cache: {str(e)}")
            self._metrics["errors"] += 1
            return None
        except Exception as e:
            logger.error(f"Erreur récupération cache: {str(e)}")
            self._metrics["errors"] += 1
            return None
    
    async def set_semantic_cache(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        cache_type: str = "default"
    ) -> bool:
        """
        Sauvegarde en cache avec TTL et métadonnées
        
        Args:
            key: Clé de cache
            data: Données à cacher
            ttl: TTL spécifique ou None pour défaut
            cache_type: Type de cache pour TTL spécialisé
            
        Returns:
            bool: Succès de l'opération
        """
        
        if not self._initialized or not self.redis_client:
            return False
        
        try:
            self._metrics["total_operations"] += 1
            cache_key = self._generate_cache_key(key, f"harena:{cache_type}")
            cache_ttl = ttl or self._get_cache_ttl(cache_type)
            
            # Enrichissement données avec métadonnées
            cache_data = {
                "data": data,
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "ttl": cache_ttl,
                "cache_type": cache_type,
                "version": "1.0"
            }
            
            # Sérialisation optimisée
            serialized = json.dumps(cache_data, ensure_ascii=False, separators=(',', ':'))
            
            # Validation taille
            if len(serialized) > 1024 * 1024:  # 1MB limite
                logger.warning(f"Données cache trop volumineuses: {len(serialized)} bytes")
                return False
            
            # Sauvegarde avec TTL
            await self.redis_client.setex(cache_key, cache_ttl, serialized)
            
            logger.debug(f"Cache set: {key[:50]}... (TTL: {cache_ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde cache: {str(e)}")
            self._metrics["errors"] += 1
            return False
    
    async def delete_cache(self, key: str, cache_type: str = "default") -> bool:
        """Suppression cache avec type spécialisé"""
        
        if not self._initialized or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(key, f"harena:{cache_type}")
            result = await self.redis_client.delete(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Erreur suppression cache: {str(e)}")
            return False
    
    async def clear_all_cache(self, cache_type: Optional[str] = None) -> bool:
        """Nettoyage cache avec type optionnel"""
        
        if not self._initialized or not self.redis_client:
            return False
        
        try:
            if cache_type:
                pattern = f"harena:{cache_type}:*"
            else:
                pattern = "harena:*"
            
            # Récupération clés avec pattern
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
            
            logger.info(f"Cache cleared: {len(keys)} keys deleted (pattern: {pattern})")
            return True
            
        except Exception as e:
            logger.error(f"Erreur nettoyage cache: {str(e)}")
            return False
    
    async def get_cache_info(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """Informations détaillées cache"""
        
        if not self._initialized or not self.redis_client:
            return {"status": "disabled"}
        
        try:
            # Info Redis globale
            info = await self.redis_client.info()
            
            # Comptage clés par type
            if cache_type:
                pattern = f"harena:{cache_type}:*"
            else:
                pattern = "harena:*"
            
            keys = await self.redis_client.keys(pattern)
            
            # Analyse utilisation mémoire
            memory_info = await self.redis_client.info("memory")
            
            return {
                "status": self._health_status,
                "keys_count": len(keys),
                "pattern": pattern,
                "memory_usage_human": memory_info.get("used_memory_human", "N/A"),
                "memory_usage_bytes": memory_info.get("used_memory", 0),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "operations": self._metrics["total_operations"],
                "hit_rate_percent": self._calculate_hit_rate(),
                "cache_metrics": self._metrics.copy(),
                "ttl_config": {
                    "intent": self.cache_ttl_intent,
                    "entity": self.cache_ttl_entity,
                    "response": self.cache_ttl_response,
                    "default": self.default_ttl
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur info cache: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques cache pour monitoring"""
        return await self.get_cache_info()
    
    def _calculate_hit_rate(self) -> float:
        """Calcul taux de hit cache"""
        total_gets = self._metrics["hits"] + self._metrics["misses"]
        if total_gets == 0:
            return 0.0
        return (self._metrics["hits"] / total_gets) * 100
    
    async def warm_up_cache(self, sample_data: List[Dict[str, Any]]) -> int:
        """Pré-chargement cache avec données d'exemple"""
        
        if not self._initialized or not self.redis_client:
            return 0
        
        warmed_count = 0
        
        try:
            for item in sample_data:
                key = item.get("key", "")
                data = item.get("data", {})
                cache_type = item.get("type", "default")
                
                if key and data:
                    success = await self.set_semantic_cache(key, data, cache_type=cache_type)
                    if success:
                        warmed_count += 1
            
            logger.info(f"Cache warmed up with {warmed_count} items")
            return warmed_count
            
        except Exception as e:
            logger.error(f"Erreur warm-up cache: {str(e)}")
            return warmed_count
    
    async def cleanup_expired_keys(self) -> int:
        """Nettoyage manuel clés expirées"""
        
        if not self._initialized or not self.redis_client:
            return 0
        
        try:
            # Redis gère automatiquement l'expiration, mais on peut forcer
            cleaned_count = 0
            
            # Scan des clés Harena
            async for key in self.redis_client.scan_iter(match="harena:*", count=100):
                ttl = await self.redis_client.ttl(key)
                if ttl == -2:  # Clé expirée mais pas encore supprimée
                    await self.redis_client.delete(key)
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Nettoyage manuel: {cleaned_count} clés expirées supprimées")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Erreur nettoyage expire: {str(e)}")
            return 0
    
    def reset_metrics(self) -> None:
        """Reset métriques cache"""
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "total_operations": 0
        }
        logger.info("Métriques cache réinitialisées")
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()