"""
Gestionnaire de cache sémantique Redis optimisé pour conversation service
Version corrigée - Compatible redis-py moderne
"""
import logging
import json
import hashlib
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass
import asyncio

# Import correct pour redis avec gestion des versions
import redis.asyncio as redis
from redis import BusyLoadingError, ConnectionError as RedisConnectionError
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.cache")

@dataclass
class CacheMetrics:
    """Métriques de cache typées pour un meilleur suivi"""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_operations: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calcule le taux de réussite du cache"""
        total_gets = self.hits + self.misses
        if total_gets == 0:
            return 0.0
        return (self.hits / total_gets) * 100

@dataclass
class CacheConfig:
    """Configuration cache centralisée et validée"""
    default_ttl: int = 3600
    intent_ttl: int = 300
    entity_ttl: int = 180
    response_ttl: int = 60
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    health_check_interval: int = 30
    max_key_size: int = 1024 * 1024  # 1MB limite par clé
    retry_attempts: int = 3
    retry_backoff_base: float = 0.1

class CacheManager:
    """
    Gestionnaire de cache sémantique avec Redis optimisé et robuste
    
    Features:
    - Pool de connexions optimisé
    - Retry automatique avec backoff exponentiel
    - Validation et monitoring complet
    - Gestion gracieuse des erreurs
    - TTL configurables par type de cache
    - Métriques détaillées
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialise le gestionnaire de cache
        
        Args:
            config: Configuration optionnelle, sinon utilise les settings par défaut
        """
        self.config = config or self._load_config_from_settings()
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False
        self._health_status = "unknown"
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()
        
        logger.info(
            f"CacheManager initialisé - TTLs: intent({self.config.intent_ttl}s), "
            f"entity({self.config.entity_ttl}s), response({self.config.response_ttl}s)"
        )
    
    def _load_config_from_settings(self) -> CacheConfig:
        """Charge la configuration depuis les settings avec fallbacks sécurisés"""
        return CacheConfig(
            default_ttl=getattr(settings, 'REDIS_CONVERSATION_TTL', 3600),
            intent_ttl=getattr(settings, 'CACHE_TTL_INTENT', 300),
            entity_ttl=getattr(settings, 'CACHE_TTL_ENTITY', 180),
            response_ttl=getattr(settings, 'CACHE_TTL_RESPONSE', 60),
            max_connections=getattr(settings, 'REDIS_MAX_CONNECTIONS', 20),
            socket_timeout=getattr(settings, 'REDIS_SOCKET_TIMEOUT', 5),
            socket_connect_timeout=getattr(settings, 'REDIS_SOCKET_CONNECT_TIMEOUT', 5),
            health_check_interval=getattr(settings, 'REDIS_HEALTH_CHECK_INTERVAL', 30),
            retry_attempts=getattr(settings, 'REDIS_RETRY_ATTEMPTS', 3),
            retry_backoff_base=getattr(settings, 'REDIS_RETRY_BACKOFF_BASE', 0.1)
        )
    
    async def initialize(self) -> None:
        """Initialisation sécurisée de la connexion Redis avec pool optimisé"""
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Configuration Redis URL avec validation
                redis_url = self._get_validated_redis_url()
                logger.debug(f"Connexion Redis: {self._mask_credentials(redis_url)}")
                
                # Pool de connexions optimisé avec configuration robuste
                self._connection_pool = redis.ConnectionPool.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=self.config.max_connections,
                    retry_on_timeout=True,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    health_check_interval=self.config.health_check_interval
                )
                
                # Client Redis avec retry policy configuré
                self.redis_client = redis.Redis(
                    connection_pool=self._connection_pool,
                    retry_on_error=[BusyLoadingError, RedisConnectionError],
                    retry=Retry(
                        backoff=ExponentialBackoff(base=self.config.retry_backoff_base),
                        retries=self.config.retry_attempts
                    )
                )
                
                # Test connexion complet avec validation
                await self._comprehensive_connection_test()
                
                self._initialized = True
                self._health_status = "healthy"
                logger.info(" Cache Manager Redis initialisé avec succès")
                
            except Exception as e:
                logger.error(f"ERROR Erreur initialisation Redis: {str(e)}")
                await self._cleanup_failed_initialization()
                raise
    
    def _get_validated_redis_url(self) -> str:
        """Récupération et validation de l'URL Redis avec fallbacks"""
        # Priorité REDISCLOUD_URL (Heroku) puis REDIS_URL
        redis_url = (
            getattr(settings, 'REDISCLOUD_URL', '') or 
            getattr(settings, 'REDIS_URL', '')
        )
        
        if redis_url:
            # Nettoyage URL (suppression quotes si présentes)
            redis_url = redis_url.strip("'\"")
            if redis_url.startswith('redis://'):
                return redis_url
        
        # Construction URL depuis composants individuels
        host = getattr(settings, 'REDIS_HOST', 'localhost')
        port = getattr(settings, 'REDIS_PORT', 6379)
        db = getattr(settings, 'REDIS_DB', 0)
        password = getattr(settings, 'REDIS_PASSWORD', None)
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"
    
    def _mask_credentials(self, url: str) -> str:
        """Masque les credentials dans l'URL pour les logs"""
        if '@' in url and '://' in url:
            scheme, rest = url.split('://', 1)
            if '@' in rest:
                credentials, host_part = rest.split('@', 1)
                return f"{scheme}://***:***@{host_part}"
        return url
    
    async def _comprehensive_connection_test(self) -> None:
        """Test de connexion complet avec validation fonctionnelle"""
        if not self.redis_client:
            raise Exception("Redis client non initialisé")
        
        try:
            # Test 1: Ping basique
            pong = await self.redis_client.ping()
            if not pong:
                raise Exception("Ping Redis échoué")
            
            # Test 2: Opérations CRUD complètes
            test_key = "harena:cache:startup_test"
            test_data = {
                "test": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "conversation_service",
                "version": "2.0"
            }
            
            # SET
            serialized_data = json.dumps(test_data, ensure_ascii=False)
            await self.redis_client.setex(test_key, 30, serialized_data)
            
            # GET
            retrieved = await self.redis_client.get(test_key)
            if not retrieved:
                raise Exception("Test GET échoué - données non récupérées")
            
            # VALIDATION
            parsed_data = json.loads(retrieved)
            if not parsed_data.get("test") or parsed_data.get("service") != "conversation_service":
                raise Exception("Test données incorrectes")
            
            # TTL CHECK
            ttl = await self.redis_client.ttl(test_key)
            if ttl <= 0:
                raise Exception("TTL non défini correctement")
            
            # DELETE
            deleted = await self.redis_client.delete(test_key)
            if not deleted:
                raise Exception("Test DELETE échoué")
            
            # Vérification suppression
            check = await self.redis_client.get(test_key)
            if check is not None:
                raise Exception("Clé non supprimée après DELETE")
            
            logger.debug(" Test connexion Redis complet réussi")
            
        except Exception as e:
            logger.error(f"ERROR Test connexion Redis échoué: {str(e)}")
            raise
    
    async def _cleanup_failed_initialization(self) -> None:
        """Nettoyage en cas d'échec d'initialisation"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self._connection_pool:
                await self._connection_pool.disconnect()
        except Exception as e:
            logger.warning(f"Erreur nettoyage initialisation échouée: {str(e)}")
        finally:
            self.redis_client = None
            self._connection_pool = None
            self._initialized = False
            self._health_status = "initialization_failed"
    
    async def close(self) -> None:
        """Fermeture propre avec métriques et nettoyage complet"""
        async with self._lock:
            if not self._initialized:
                return
            
            try:
                # Log métriques finales
                logger.info(
                    f" Fermeture cache - Hits: {self._metrics.hits}, "
                    f"Misses: {self._metrics.misses}, Errors: {self._metrics.errors}, "
                    f"Hit Rate: {self._metrics.hit_rate:.2f}%"
                )
                
                # Fermeture client Redis
                if self.redis_client:
                    await self.redis_client.close()
                
                # Fermeture pool de connexions
                if self._connection_pool:
                    await self._connection_pool.disconnect()
                    
            except Exception as e:
                logger.warning(f" Erreur fermeture Redis: {str(e)}")
            finally:
                self.redis_client = None
                self._connection_pool = None
                self._initialized = False
                self._health_status = "closed"
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification santé complète avec diagnostic détaillé"""
        health_data = {
            "status": self._health_status,
            "initialized": self._initialized,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if not self._initialized or not self.redis_client:
            health_data.update({
                "healthy": False,
                "error": "Service non initialisé"
            })
            return health_data
        
        try:
            start_time = datetime.now()
            
            # Test ping avec mesure de latence
            pong = await self.redis_client.ping()
            ping_latency = (datetime.now() - start_time).total_seconds() * 1000
            
            if not pong:
                health_data.update({
                    "healthy": False,
                    "error": "Ping Redis échoué"
                })
                return health_data
            
            # Informations serveur
            info = await self.redis_client.info()
            memory_info = await self.redis_client.info("memory")
            
            # Calculs métriques
            used_memory = memory_info.get("used_memory", 0)
            max_memory = memory_info.get("maxmemory", 0)
            memory_usage_percent = (used_memory / max_memory * 100) if max_memory > 0 else 0
            
            # Comptage clés Harena
            harena_keys = len(await self.redis_client.keys("harena:*"))
            
            health_data.update({
                "healthy": True,
                "ping_latency_ms": round(ping_latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "memory_usage_human": memory_info.get("used_memory_human", "N/A"),
                "memory_usage_percent": round(memory_usage_percent, 1),
                "harena_keys_count": harena_keys,
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "redis_version": info.get("redis_version", "unknown")
            })
            
            # Alertes critiques
            if memory_usage_percent > 90:
                health_data["warning"] = f"Mémoire critique: {memory_usage_percent:.1f}%"
            
            if ping_latency > 1000:  # > 1s
                health_data["warning"] = f"Latence élevée: {ping_latency:.1f}ms"
            
            self._health_status = "healthy"
            
        except Exception as e:
            logger.error(f"ERROR Health check échoué: {str(e)}")
            health_data.update({
                "healthy": False,
                "error": str(e)
            })
            self._health_status = f"unhealthy: {str(e)}"
        
        return health_data
    
    def _generate_cache_key(self, key: str, prefix: str = "harena:conv") -> str:
        """Génération clé cache sécurisée avec hash et préfixe intelligent"""
        # Préfixage par user_id si disponible (sécurité)
        user_prefix = getattr(settings, 'CURRENT_USER_ID', '')
        if user_prefix:
            prefix = f"{prefix}:user:{user_prefix}"
        
        # Hash SHA-256 pour sécurité et longueur optimale
        key_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]
        return f"{prefix}:{key_hash}"
    
    def _get_cache_ttl(self, cache_type: str) -> int:
        """Récupération TTL selon type de cache avec validation"""
        ttl_mapping = {
            "intent": self.config.intent_ttl,
            "entity": self.config.entity_ttl,
            "response": self.config.response_ttl,
            "default": self.config.default_ttl
        }
        
        ttl = ttl_mapping.get(cache_type, self.config.default_ttl)
        
        # Validation TTL (entre 1 sec et 24h)
        return max(1, min(ttl, 86400))
    
    def _validate_cache_data(self, data: Any) -> bool:
        """Validation des données avant mise en cache"""
        if data is None:
            return False
        
        try:
            # Test sérialisation
            serialized = json.dumps(data, ensure_ascii=False)
            
            # Validation taille
            if len(serialized) > self.config.max_key_size:
                logger.warning(f"Données trop volumineuses: {len(serialized)} bytes")
                return False
            
            # Test désérialisation
            json.loads(serialized)
            return True
            
        except (TypeError, ValueError) as e:
            logger.warning(f"Données non sérialisables: {str(e)}")
            return False
    
    async def get_semantic_cache(
        self, 
        key: str, 
        similarity_threshold: float = 0.8,
        cache_type: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Récupération cache sémantique avec validation complète
        
        Args:
            key: Clé de cache
            similarity_threshold: Seuil similarité (réservé pour futur)
            cache_type: Type de cache pour TTL spécialisé
            
        Returns:
            Données cachées validées ou None
        """
        if not self._initialized or not self.redis_client:
            self._metrics.misses += 1
            return None
        
        try:
            self._metrics.total_operations += 1
            cache_key = self._generate_cache_key(key, f"harena:{cache_type}")
            
            # Récupération avec gestion timeout
            result = await asyncio.wait_for(
                self.redis_client.get(cache_key), 
                timeout=2.0  # Timeout court pour éviter blocage
            )
            
            if not result:
                self._metrics.misses += 1
                logger.debug(f" Cache miss: {key[:50]}...")
                return None
            
            # Parsing sécurisé
            try:
                cached_data = json.loads(result)
            except json.JSONDecodeError as e:
                logger.warning(f" Données cache JSON invalides: {str(e)}")
                await self.redis_client.delete(cache_key)
                self._metrics.errors += 1
                return None
            
            # Validation structure
            if not isinstance(cached_data, dict):
                logger.warning(" Structure cache invalide")
                await self.redis_client.delete(cache_key)
                self._metrics.errors += 1
                return None
            
            # Validation expiration manuelle (double sécurité)
            if "cached_at" in cached_data:
                cached_time = datetime.fromisoformat(cached_data["cached_at"])
                now = datetime.now(timezone.utc)
                ttl = self._get_cache_ttl(cache_type)
                
                if (now - cached_time).total_seconds() > ttl:
                    logger.debug("⏰ Données cache expirées manuellement")
                    await self.redis_client.delete(cache_key)
                    self._metrics.misses += 1
                    return None
            
            # Succès avec validation version
            cache_version = cached_data.get("version", "1.0")
            if cache_version != "2.0":
                # Migration ou invalidation cache ancienne version
                logger.debug(f" Cache version obsolète: {cache_version}")
                await self.redis_client.delete(cache_key)
                self._metrics.misses += 1
                return None
            
            self._metrics.hits += 1
            logger.debug(f" Cache hit: {key[:50]}...")
            return cached_data.get("data", cached_data)
            
        except asyncio.TimeoutError:
            logger.warning("⏰ Timeout récupération cache")
            self._metrics.errors += 1
            return None
        except Exception as e:
            logger.error(f"ERROR Erreur récupération cache: {str(e)}")
            self._metrics.errors += 1
            return None
    
    async def set_semantic_cache(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        cache_type: str = "default"
    ) -> bool:
        """
        Sauvegarde cache avec validation, métadonnées et monitoring
        
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
        
        # Validation données avant traitement
        if not self._validate_cache_data(data):
            logger.warning(f" Données invalides pour cache: {key[:50]}...")
            return False
        
        try:
            self._metrics.total_operations += 1
            cache_key = self._generate_cache_key(key, f"harena:{cache_type}")
            cache_ttl = ttl or self._get_cache_ttl(cache_type)
            
            # Enrichissement avec métadonnées complètes
            enriched_data = {
                "data": data,
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "ttl": cache_ttl,
                "cache_type": cache_type,
                "version": "2.0",
                "service": "conversation_service",
                "key_hash": hashlib.sha256(key.encode('utf-8')).hexdigest()[:8]
            }
            
            # Sérialisation optimisée
            serialized = json.dumps(
                enriched_data, 
                ensure_ascii=False, 
                separators=(',', ':'),
                sort_keys=True
            )
            
            # Sauvegarde avec timeout et retry
            await asyncio.wait_for(
                self.redis_client.setex(cache_key, cache_ttl, serialized),
                timeout=3.0
            )
            
            logger.debug(f" Cache set: {key[:50]}... (TTL: {cache_ttl}s, Size: {len(serialized)}b)")
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout sauvegarde cache: {key[:50]}...")
            self._metrics.errors += 1
            return False
        except Exception as e:
            logger.error(f"ERROR Erreur sauvegarde cache: {str(e)}")
            self._metrics.errors += 1
            return False
    
    async def delete_cache(self, key: str, cache_type: str = "default") -> bool:
        """Suppression cache sécurisée avec validation"""
        if not self._initialized or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(key, f"harena:{cache_type}")
            result = await asyncio.wait_for(
                self.redis_client.delete(cache_key),
                timeout=2.0
            )
            
            if result:
                logger.debug(f"🗑️ Cache deleted: {key[:50]}...")
                
            return bool(result)
            
        except Exception as e:
            logger.error(f"ERROR Erreur suppression cache: {str(e)}")
            return False
    
    async def clear_all_cache(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """Nettoyage cache avec rapport détaillé"""
        if not self._initialized or not self.redis_client:
            return {"success": False, "error": "Service non initialisé"}
        
        try:
            pattern = f"harena:{cache_type}:*" if cache_type else "harena:*"
            
            # Récupération clés avec pattern et comptage
            keys = await self.redis_client.keys(pattern)
            deleted_count = 0
            
            if keys:
                # Suppression par batch pour éviter surcharge
                batch_size = 100
                for i in range(0, len(keys), batch_size):
                    batch = keys[i:i + batch_size]
                    deleted = await self.redis_client.delete(*batch)
                    deleted_count += deleted
            
            logger.info(f"🧹 Cache cleared: {deleted_count} keys deleted (pattern: {pattern})")
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "pattern": pattern,
                "total_keys_found": len(keys)
            }
            
        except Exception as e:
            logger.error(f"ERROR Erreur nettoyage cache: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_cache_info(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """Informations complètes cache pour monitoring et debug"""
        if not self._initialized or not self.redis_client:
            return {"status": "disabled", "initialized": False}
        
        try:
            # Info Redis globale
            info = await self.redis_client.info()
            memory_info = await self.redis_client.info("memory")
            
            # Comptage clés par pattern
            pattern = f"harena:{cache_type}:*" if cache_type else "harena:*"
            keys = await self.redis_client.keys(pattern)
            
            # Analyse utilisation par type de cache
            cache_breakdown = {}
            if not cache_type:
                for ct in ["intent", "entity", "response", "default"]:
                    ct_keys = await self.redis_client.keys(f"harena:{ct}:*")
                    cache_breakdown[ct] = len(ct_keys)
            
            return {
                "status": self._health_status,
                "healthy": self._health_status == "healthy",
                "initialized": self._initialized,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                
                # Compteurs clés
                "keys_count": len(keys),
                "pattern_used": pattern,
                "cache_breakdown": cache_breakdown,
                
                # Métriques cache
                "cache_metrics": {
                    "hits": self._metrics.hits,
                    "misses": self._metrics.misses,
                    "errors": self._metrics.errors,
                    "total_operations": self._metrics.total_operations,
                    "hit_rate_percent": round(self._metrics.hit_rate, 2)
                },
                
                # Configuration TTL
                "ttl_config": {
                    "intent": self.config.intent_ttl,
                    "entity": self.config.entity_ttl,
                    "response": self.config.response_ttl,
                    "default": self.config.default_ttl
                },
                
                # Infos Redis serveur
                "redis_info": {
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                    "redis_version": info.get("redis_version", "unknown"),
                    "memory_usage_human": memory_info.get("used_memory_human", "N/A"),
                    "memory_usage_bytes": memory_info.get("used_memory", 0),
                    "memory_peak_human": memory_info.get("used_memory_peak_human", "N/A")
                }
            }
            
        except Exception as e:
            logger.error(f"ERROR Erreur info cache: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "initialized": self._initialized
            }
    
    def reset_metrics(self) -> CacheMetrics:
        """Reset et retour des métriques actuelles"""
        old_metrics = CacheMetrics(
            hits=self._metrics.hits,
            misses=self._metrics.misses,
            errors=self._metrics.errors,
            total_operations=self._metrics.total_operations
        )
        
        self._metrics = CacheMetrics()
        logger.info(" Métriques cache réinitialisées")
        return old_metrics
    
    async def warm_up_cache(self, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pré-chargement cache avec rapport détaillé"""
        if not self._initialized or not self.redis_client:
            return {"success": False, "error": "Service non initialisé"}
        
        warmed_count = 0
        error_count = 0
        start_time = datetime.now()
        
        try:
            for item in sample_data:
                key = item.get("key", "")
                data = item.get("data", {})
                cache_type = item.get("type", "default")
                
                if key and data:
                    success = await self.set_semantic_cache(key, data, cache_type=cache_type)
                    if success:
                        warmed_count += 1
                    else:
                        error_count += 1
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f" Cache warmed up: {warmed_count} items, {error_count} errors, {duration:.2f}s")
            
            return {
                "success": True,
                "warmed_count": warmed_count,
                "error_count": error_count,
                "duration_seconds": round(duration, 2),
                "total_items": len(sample_data)
            }
            
        except Exception as e:
            logger.error(f"ERROR Erreur warm-up cache: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Context manager pour utilisation propre
    async def __aenter__(self):
        """Context manager entry avec initialisation automatique"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit avec fermeture propre"""
        await self.close()
        if exc_type:
            logger.error(f"ERROR Exception dans context manager cache: {exc_val}")