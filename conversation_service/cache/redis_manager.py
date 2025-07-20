"""
Gestionnaire Redis Cloud optimisé pour Heroku
Configuration haute performance avec compression et pooling
"""

import asyncio
import json
import gzip
import time
from typing import Optional, Dict, Any, List
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from ..config import settings
from ..utils.logging import get_logger
from ..utils.metrics import CacheMetrics

logger = get_logger(__name__)


class RedisManager:
    """
    Gestionnaire Redis Cloud optimisé pour les contraintes Heroku
    
    Features:
    - Connection pooling optimisé dyno limitations
    - Compression gzip automatique
    - Retry logic avec circuit breaker
    - Métriques de performance temps réel
    """
    
    def __init__(self):
        self.connection_pool = None
        self.redis_client = None
        self.metrics = CacheMetrics()
        self._circuit_breaker_count = 0
        self._circuit_breaker_threshold = 5
        self._is_circuit_open = False
        
    async def initialize(self):
        """Initialisation connexion Redis Cloud"""
        try:
            # Configuration pool optimisée Heroku dyno
            self.connection_pool = ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                max_connections=settings.REDIS_MAX_CONNECTIONS,  # 10 pour dyno standard
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=False  # Pour compression binaire
            )
            
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False
            )
            
            # Test connexion
            await self.redis_client.ping()
            logger.info("Redis Cloud connexion établie avec succès")
            
            # Configuration optimisations mémoire
            await self._configure_redis_optimizations()
            
        except Exception as e:
            logger.error(f"Erreur initialisation Redis: {e}")
            raise
    
    async def _configure_redis_optimizations(self):
        """Configuration optimisations Redis pour Heroku"""
        try:
            # Configuration éviction LRU pour gestion mémoire limitée
            await self.redis_client.config_set("maxmemory-policy", "allkeys-lru")
            
            # Compression automatique pour économiser bande passante
            await self.redis_client.config_set("save", "")  # Disable snapshots sur Heroku
            
            logger.info("Optimisations Redis Cloud configurées")
            
        except Exception as e:
            logger.warning(f"Impossible de configurer optimisations Redis: {e}")
    
    async def get(self, key: str, decompress: bool = True) -> Optional[Any]:
        """
        Récupération avec décompression automatique
        
        Args:
            key: Clé Redis
            decompress: Décompression gzip automatique
            
        Returns:
            Valeur décompressée ou None si non trouvée
        """
        if self._is_circuit_open:
            self.metrics.record_circuit_breaker_hit()
            return None
        
        start_time = time.time()
        
        try:
            value = await self.redis_client.get(key)
            
            if value is None:
                self.metrics.record_cache_miss(key)
                return None
            
            # Décompression automatique si activée
            if decompress and value.startswith(b'\x1f\x8b'):  # Magic number gzip
                value = gzip.decompress(value)
            
            # Désérialisation JSON
            result = json.loads(value.decode('utf-8'))
            
            latency_ms = int((time.time() - start_time) * 1000)
            self.metrics.record_cache_hit(key, latency_ms)
            
            return result
            
        except Exception as e:
            self._handle_redis_error(e)
            self.metrics.record_cache_error(key, str(e))
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600,
        compress: bool = True
    ) -> bool:
        """
        Stockage avec compression automatique
        
        Args:
            key: Clé Redis
            value: Valeur à stocker
            ttl: TTL en secondes
            compress: Compression gzip automatique
            
        Returns:
            True si succès, False sinon
        """
        if self._is_circuit_open:
            return False
        
        try:
            # Sérialisation JSON
            serialized = json.dumps(value, ensure_ascii=False).encode('utf-8')
            
            # Compression automatique si activée et profitable
            if compress and len(serialized) > 1024:  # Seuil 1KB
                compressed = gzip.compress(serialized, compresslevel=6)
                if len(compressed) < len(serialized) * 0.8:  # 20% économie minimum
                    serialized = compressed
                    self.metrics.record_compression_saving(
                        original_size=len(json.dumps(value).encode()),
                        compressed_size=len(compressed)
                    )
            
            # Stockage avec TTL
            success = await self.redis_client.setex(key, ttl, serialized)
            
            if success:
                self.metrics.record_cache_set(key, len(serialized))
                
            return bool(success)
            
        except Exception as e:
            self._handle_redis_error(e)
            self.metrics.record_cache_error(key, str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Suppression clé"""
        if self._is_circuit_open:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            self.metrics.record_cache_delete(key)
            return bool(result)
            
        except Exception as e:
            self._handle_redis_error(e)
            return False
    
    async def exists(self, key: str) -> bool:
        """Vérification existence clé"""
        if self._is_circuit_open:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return bool(result)
            
        except Exception as e:
            self._handle_redis_error(e)
            return False
    
    async def mget(self, keys: List[str], decompress: bool = True) -> Dict[str, Any]:
        """Récupération multiple optimisée"""
        if self._is_circuit_open or not keys:
            return {}
        
        try:
            values = await self.redis_client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        if decompress and value.startswith(b'\x1f\x8b'):
                            value = gzip.decompress(value)
                        result[key] = json.loads(value.decode('utf-8'))
                        self.metrics.record_cache_hit(key, 0)
                    except Exception:
                        self.metrics.record_cache_miss(key)
                else:
                    self.metrics.record_cache_miss(key)
            
            return result
            
        except Exception as e:
            self._handle_redis_error(e)
            return {}
    
    async def pipeline_operations(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """
        Exécution pipeline pour optimiser latence réseau
        
        Args:
            operations: Liste d'opérations [{"op": "set", "key": "...", "value": "...", "ttl": 3600}]
            
        Returns:
            Liste des résultats
        """
        if self._is_circuit_open:
            return []
        
        try:
            pipeline = self.redis_client.pipeline()
            
            for op in operations:
                if op["op"] == "set":
                    serialized = json.dumps(op["value"]).encode('utf-8')
                    pipeline.setex(op["key"], op.get("ttl", 3600), serialized)
                elif op["op"] == "get":
                    pipeline.get(op["key"])
                elif op["op"] == "delete":
                    pipeline.delete(op["key"])
            
            results = await pipeline.execute()
            self.metrics.record_pipeline_operation(len(operations))
            
            return results
            
        except Exception as e:
            self._handle_redis_error(e)
            return []
    
    def _handle_redis_error(self, error: Exception):
        """Gestion erreurs avec circuit breaker"""
        self._circuit_breaker_count += 1
        logger.warning(f"Erreur Redis: {error}")
        
        if self._circuit_breaker_count >= self._circuit_breaker_threshold:
            self._is_circuit_open = True
            logger.error("Circuit breaker Redis ouvert - fallback mode activé")
            
            # Auto-reset après 60 secondes
            asyncio.create_task(self._reset_circuit_breaker())
    
    async def _reset_circuit_breaker(self):
        """Reset automatique circuit breaker"""
        await asyncio.sleep(60)
        self._circuit_breaker_count = 0
        self._is_circuit_open = False
        logger.info("Circuit breaker Redis resetté")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statistiques Redis et cache"""
        try:
            info = await self.redis_client.info()
            return {
                "redis_info": {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                },
                "cache_metrics": await self.metrics.get_current_metrics(),
                "circuit_breaker": {
                    "is_open": self._is_circuit_open,
                    "error_count": self._circuit_breaker_count
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        """Fermeture propre connexions"""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()