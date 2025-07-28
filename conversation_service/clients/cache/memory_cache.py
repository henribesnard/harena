"""
🧠 Cache Intelligent - Redis + Fallback Mémoire

Cache hybride optimisé pour résultats de détection d'intention :
- Redis comme cache principal (production)
- Fallback mémoire LRU (développement/erreur Redis)
- Stratégies adaptatives et métriques détaillées
"""

import time
import threading
import json
import os
from collections import OrderedDict
import logging

# Redis imports avec fallback gracieux
try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisError = Exception
    RedisConnectionError = Exception

from conversation_service.models.enums import CacheStrategy, IntentType
from conversation_service.models.intent import CacheEntry
from conversation_service.models.exceptions import CacheError, CacheFullError
from conversation_service.config import config

logger = logging.getLogger(__name__)


class CacheStats:
    """Statistiques détaillées du cache"""
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.cache_size = 0
        self.hit_rate = 0.0
        self.avg_confidence = 0.0
        self.top_queries = {}
        self.redis_hits = 0
        self.memory_hits = 0
        self.redis_errors = 0


class RedisConfig:
    """Configuration Redis depuis variables d'environnement"""
    
    def __init__(self):
        self.enabled = os.getenv("REDIS_CACHE_ENABLED", "false").lower() == "true"
        self.url = os.getenv("REDIS_URL", os.getenv("REDISCLOUD_URL", ""))
        self.prefix = os.getenv("REDIS_CACHE_PREFIX", "harena_conv")
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
        self.retry_on_timeout = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
        self.socket_timeout = 5.0
        self.socket_connect_timeout = 5.0


class HybridIntelligentCache:
    """
    Cache hybride intelligent Redis + Mémoire
    
    Architecture:
    - Redis: Cache principal distribué (production)
    - Mémoire: Fallback local LRU (développement/erreur)
    - TTL adaptatif selon confiance résultat
    - Thread-safe pour environnement concurrent
    - Métriques détaillées et monitoring
    """
    
    def __init__(self, max_size=None, default_ttl=3600, strategy=CacheStrategy.SMART):
        self.max_size = max_size or config.performance.cache_max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        # Configuration Redis
        self.redis_config = RedisConfig()
        self.redis_client = None
        self.redis_available = REDIS_AVAILABLE and self.redis_config.enabled  # ✅ UTILISÉ
        
        # Cache mémoire fallback
        self._memory_cache = OrderedDict()
        self._lock = threading.RLock()
        
        # Métriques
        self._stats = CacheStats()
        
        # Configuration TTL adaptatif
        self._ttl_by_confidence = {
            "very_high": int(os.getenv("CACHE_TTL", "3600")) * 2,
            "high": int(os.getenv("CACHE_TTL", "3600")),
            "medium": int(os.getenv("CACHE_TTL_INTENT", "300")),
            "low": int(os.getenv("CACHE_TTL_RESPONSE", "60"))
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialisation Redis avec gestion d'erreurs
        try:
            self._initialize_redis()
        except CacheError as e:
            self.logger.error(f"Erreur initialisation cache: {e}")
            raise
        
        self.logger.info(
            f"Cache hybride initialisé: Redis={self.redis_available}, "
            f"taille max mémoire={self.max_size}, TTL={default_ttl}s, stratégie={strategy.value}"
        )
    
    def _initialize_redis(self):
        """
        Initialise connexion Redis avec fallback gracieux
        
        Raises:
            CacheError: Si Redis requis mais indisponible
        """
        if not REDIS_AVAILABLE:
            if self.redis_config.enabled:
                raise CacheError(
                    "Redis activé dans config mais package redis non installé",
                    cache_operation="initialization"
                )
            self.logger.info("Redis non disponible: package redis non installé")
            return
        
        if not self.redis_config.enabled:
            self.logger.info("Redis désactivé par configuration")
            return
        
        if not self.redis_config.url:
            raise CacheError(
                "Redis activé mais URL manquante dans configuration",
                cache_operation="initialization"
            )
        
        try:
            self.redis_client = redis.from_url(
                self.redis_config.url,
                db=self.redis_config.db,
                max_connections=self.redis_config.max_connections,
                retry_on_timeout=self.redis_config.retry_on_timeout,
                socket_timeout=self.redis_config.socket_timeout,
                socket_connect_timeout=self.redis_config.socket_connect_timeout,
                decode_responses=True
            )
            
            # Test de connexion
            self.redis_client.ping()
            self.redis_available = True
            
            self.logger.info(f"✅ Redis connecté: {self.redis_config.url.split('@')[-1] if '@' in self.redis_config.url else 'localhost'}")
            
        except RedisConnectionError as e:  # ✅ UTILISÉ
            self.logger.warning(f"⚠️ Redis connexion échouée: {e}")
            if self.redis_config.enabled:
                raise CacheError(
                    f"Impossible de se connecter à Redis: {str(e)}",
                    cache_operation="connection",
                    details={"redis_url_masked": self.redis_config.url.split('@')[-1] if '@' in self.redis_config.url else 'localhost'}
                )
            self.redis_client = None
            self.redis_available = False
        except Exception as e:
            self.logger.warning(f"⚠️ Redis erreur inattendue: {e}")
            raise CacheError(
                f"Erreur initialisation Redis: {str(e)}",
                cache_operation="initialization"
            )
    
    def _get_redis_key(self, key):
        """Génère clé Redis avec préfixe"""
        return f"{self.redis_config.prefix}:intent:{key}"
    
    def get(self, key):
        """
        Récupère un élément du cache (Redis puis mémoire)
        
        Raises:
            CacheError: Si erreur critique de cache
        """
        if not key or not isinstance(key, str):
            raise CacheError(
                f"Clé de cache invalide: {key}",
                cache_operation="get",
                cache_key=str(key)
            )
        
        with self._lock:
            self._stats.total_requests += 1
            
            # 1. Tentative Redis d'abord
            if self.redis_available:
                try:
                    redis_result = self._get_from_redis(key)
                    if redis_result:
                        self._stats.cache_hits += 1
                        self._stats.redis_hits += 1
                        self._update_hit_rate()
                        self.logger.debug(f"Redis hit: {key[:50]}...")
                        return redis_result
                        
                except RedisConnectionError as e:  # ✅ UTILISÉ
                    self.logger.error(f"Redis connexion perdue: {e}")
                    self.redis_available = False
                    self._stats.redis_errors += 1
                    # Continue avec cache mémoire
                except RedisError as e:
                    self.logger.warning(f"Redis get error: {e}")
                    self._stats.redis_errors += 1
                    # Continue avec cache mémoire
                except Exception as e:
                    self.logger.warning(f"Redis erreur inattendue: {e}")
                    self._stats.redis_errors += 1
            
            # 2. Fallback cache mémoire
            try:
                memory_result = self._get_from_memory(key)
                if memory_result:
                    self._stats.cache_hits += 1
                    self._stats.memory_hits += 1
                    self._update_hit_rate()
                    self.logger.debug(f"Memory hit: {key[:50]}...")
                    return memory_result
            except Exception as e:
                raise CacheError(
                    f"Erreur critique cache mémoire: {str(e)}",
                    cache_operation="memory_get",
                    cache_key=key
                )
            
            # 3. Cache miss
            self._stats.cache_misses += 1
            return None
    
    def _get_from_redis(self, key):
        """
        Récupère depuis Redis avec gestion TTL
        
        Raises:
            RedisError: Pour erreurs Redis spécifiques
        """
        if not self.redis_available:
            return None
        
        redis_key = self._get_redis_key(key)
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.get(redis_key)
            pipe.ttl(redis_key)
            redis_data, ttl_remaining = pipe.execute()
            
            if not redis_data or ttl_remaining <= 0:
                return None
            
            cache_data = json.loads(redis_data)
            
            if "hit_count" in cache_data:
                cache_data["hit_count"] += 1
                cache_data["last_accessed"] = time.time()
                
                self.redis_client.setex(
                    redis_key, 
                    ttl_remaining, 
                    json.dumps(cache_data, default=str)
                )
            
            return cache_data.get("result", {})
            
        except (RedisError, json.JSONDecodeError) as e:
            # Re-raise pour gestion au niveau supérieur
            raise
    
    def _get_from_memory(self, key):
        """Récupère depuis cache mémoire local"""
        if key not in self._memory_cache:
            return None
        
        entry = self._memory_cache[key]
        
        if self._is_expired(entry):
            del self._memory_cache[key]
            return None
        
        self._memory_cache.move_to_end(key)
        entry.hit_count += 1
        entry.last_accessed = time.time()
        
        return entry.result.dict()
    
    def put(self, key, result, confidence, intent=None):
        """
        Met un élément en cache (Redis + mémoire avec TTL adaptatif)
        
        Raises:
            CacheError: Si erreur critique de cache
            CacheFullError: Si cache mémoire plein et éviction impossible
        """
        if not key or not isinstance(key, str):
            raise CacheError(
                f"Clé de cache invalide: {key}",
                cache_operation="put",
                cache_key=str(key)
            )
        
        if not isinstance(result, dict):
            raise CacheError(
                f"Résultat doit être un dictionnaire, reçu: {type(result)}",
                cache_operation="put",
                cache_key=key
            )
        
        with self._lock:
            if not self._should_cache(confidence, intent):
                return False
            
            ttl = self._calculate_adaptive_ttl(confidence)
            
            cache_data = {
                "query": result.get("query", ""),
                "result": result,
                "created_at": time.time(),
                "hit_count": 1,
                "last_accessed": time.time(),
                "confidence": confidence,
                "intent": intent.value if intent else None
            }
            
            success = False
            
            # Redis en premier
            if self.redis_available:
                try:
                    success |= self._put_to_redis(key, cache_data, ttl)
                except RedisConnectionError as e:  # ✅ UTILISÉ
                    self.logger.error(f"Redis connexion perdue lors du put: {e}")
                    self.redis_available = False
                    self._stats.redis_errors += 1
                except RedisError as e:
                    self.logger.warning(f"Redis put error: {e}")
                    self._stats.redis_errors += 1
            
            # Cache mémoire ensuite
            try:
                success |= self._put_to_memory(key, result, confidence)
            except CacheFullError as e:  # ✅ UTILISÉ
                self.logger.warning(f"Cache mémoire plein: {e}")
                # Tenter éviction forcée
                if self._force_evict_memory():
                    success |= self._put_to_memory(key, result, confidence)
                else:
                    raise  # Re-raise si éviction impossible
            
            if success:
                self._update_cache_stats(key, confidence)
            
            return success
    
    def _put_to_redis(self, key, cache_data, ttl):
        """
        Met en cache dans Redis
        
        Raises:
            RedisError: Pour erreurs Redis spécifiques
        """
        if not self.redis_available:
            return False
        
        redis_key = self._get_redis_key(key)
        
        try:
            redis_data = json.dumps(cache_data, default=str)
            self.redis_client.setex(redis_key, ttl, redis_data)
            self.logger.debug(f"Redis cache: {key[:50]}... (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            # Transform en RedisError pour cohérence
            raise RedisError(f"Erreur Redis put: {str(e)}")
    
    def _put_to_memory(self, key, result, confidence):
        """
        Met en cache en mémoire locale
        
        Raises:
            CacheFullError: Si cache plein et éviction impossible
        """
        try:
            # Vérification place disponible
            if len(self._memory_cache) >= self.max_size and key not in self._memory_cache:
                if not self._evict_memory_lru():
                    raise CacheFullError(
                        cache_size=len(self._memory_cache),
                        max_size=self.max_size
                    )  # ✅ UTILISÉ
            
            from conversation_service.models.intent import IntentResponse
            try:
                intent_response = IntentResponse(**result)
            except Exception as e:
                raise CacheError(
                    f"Erreur création IntentResponse pour cache mémoire: {e}",
                    cache_operation="memory_put",
                    cache_key=key
                )
            
            cache_entry = CacheEntry(
                query=result.get("query", ""),
                result=intent_response,
                created_at=time.time()
            )
            
            self._memory_cache[key] = cache_entry
            self._memory_cache.move_to_end(key)
            
            self.logger.debug(f"Memory cache: {key[:50]}...")
            return True
            
        except CacheFullError:
            # Re-raise les erreurs de cache plein
            raise
        except Exception as e:
            raise CacheError(
                f"Erreur critique cache mémoire put: {str(e)}",
                cache_operation="memory_put",
                cache_key=key
            )
    
    def _should_cache(self, confidence, intent):
        """Détermine si résultat doit être mis en cache selon stratégie"""
        min_confidence = self.strategy.get_min_confidence()
        
        if confidence < min_confidence:
            return False
        
        if self.strategy == CacheStrategy.SMART:
            if intent and intent in [IntentType.GREETING, IntentType.GOODBYE]:
                return confidence > 0.95
            
            if intent and hasattr(intent, 'is_financial') and intent.is_financial():
                return confidence > 0.6
        
        return True
    
    def _calculate_adaptive_ttl(self, confidence):
        """Calcule TTL adaptatif selon confiance"""
        if confidence >= 0.9:
            return self._ttl_by_confidence["very_high"]
        elif confidence >= 0.7:
            return self._ttl_by_confidence["high"]
        elif confidence >= 0.5:
            return self._ttl_by_confidence["medium"]
        else:
            return self._ttl_by_confidence["low"]
    
    def _is_expired(self, entry):
        """Vérifie si entrée cache mémoire est expirée"""
        ttl = self._calculate_adaptive_ttl(entry.result.confidence)
        age = time.time() - entry.created_at
        return age > ttl
    
    def _evict_memory_lru(self):
        """Éviction LRU mémoire standard"""
        if not self._memory_cache:
            return False
        
        lru_key, lru_entry = self._memory_cache.popitem(last=False)
        self._stats.cache_evictions += 1
        
        self.logger.debug(f"Memory LRU eviction: {lru_key[:50]}...")
        return True
    
    def _force_evict_memory(self):
        """
        Éviction forcée quand cache plein (évict multiple entries)
        
        Returns:
            bool: True si éviction réussie
        """
        if not self._memory_cache:
            return False
        
        # Éviction multiple si nécessaire
        evicted_count = 0
        target_size = int(self.max_size * 0.8)  # Réduire à 80% de la capacité
        
        while len(self._memory_cache) > target_size and self._memory_cache:
            if self._evict_memory_lru():
                evicted_count += 1
            else:
                break
        
        if evicted_count > 0:
            self.logger.info(f"Éviction forcée: {evicted_count} entrées supprimées")
            return True
        
        return False
    
    def _update_cache_stats(self, key, confidence):
        """Met à jour statistiques du cache"""
        self._stats.cache_size = len(self._memory_cache)
        
        if self._stats.cache_size > 0:
            total_confidence = self._stats.avg_confidence * (self._stats.cache_size - 1) + confidence
            self._stats.avg_confidence = total_confidence / self._stats.cache_size
        
        query_short = key[:50]
        if query_short not in self._stats.top_queries:
            self._stats.top_queries[query_short] = 0
        self._stats.top_queries[query_short] += 1
    
    def _update_hit_rate(self):
        """Met à jour taux de hit"""
        if self._stats.total_requests > 0:
            self._stats.hit_rate = self._stats.cache_hits / self._stats.total_requests
    
    def invalidate(self, key):
        """
        Invalide une entrée spécifique du cache (Redis + mémoire)
        
        Raises:
            CacheError: Si erreur critique d'invalidation
        """
        if not key:
            raise CacheError(
                "Clé d'invalidation ne peut pas être vide",
                cache_operation="invalidate"
            )
        
        with self._lock:
            invalidated = False
            
            # Redis invalidation
            if self.redis_available:
                try:
                    redis_key = self._get_redis_key(key)
                    result = self.redis_client.delete(redis_key)
                    if result > 0:
                        invalidated = True
                        self.logger.debug(f"Redis invalidated: {key[:50]}...")
                except RedisConnectionError as e:  # ✅ UTILISÉ
                    self.logger.error(f"Redis connexion perdue lors invalidation: {e}")
                    self.redis_available = False
                    self._stats.redis_errors += 1
                except RedisError as e:
                    self.logger.warning(f"Redis invalidation error: {e}")
                    self._stats.redis_errors += 1
            
            # Memory invalidation
            if key in self._memory_cache:
                del self._memory_cache[key]
                invalidated = True
                self.logger.debug(f"Memory invalidated: {key[:50]}...")
            
            return invalidated
    
    def invalidate_by_intent(self, intent):
        """Invalide toutes les entrées d'une intention (mémoire seulement)"""
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._memory_cache.items():
                if entry.result.intent == intent.value:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._memory_cache[key]
            
            if keys_to_remove:
                self.logger.info(f"Cache invalidé pour intention {intent.value}: {len(keys_to_remove)} entrées")
            
            return len(keys_to_remove)
    
    def clear(self):
        """
        Vide complètement le cache (Redis + mémoire)
        
        Raises:
            CacheError: Si erreur critique de clear
        """
        with self._lock:
            total_removed = 0
            
            # Clear Redis
            if self.redis_available:
                try:
                    pattern = f"{self.redis_config.prefix}:intent:*"
                    redis_keys = self.redis_client.keys(pattern)
                    if redis_keys:
                        removed = self.redis_client.delete(*redis_keys)
                        total_removed += removed
                        self.logger.info(f"Redis cleared: {removed} keys")
                except RedisConnectionError as e:  # ✅ UTILISÉ
                    self.logger.error(f"Redis connexion perdue lors clear: {e}")
                    self.redis_available = False
                    self._stats.redis_errors += 1
                except RedisError as e:
                    self.logger.warning(f"Redis clear error: {e}")
                    self._stats.redis_errors += 1
            
            # Clear Memory
            try:
                memory_size = len(self._memory_cache)
                self._memory_cache.clear()
                total_removed += memory_size
                
                self.logger.info(f"Cache vidé: {total_removed} entrées supprimées")
                return total_removed
                
            except Exception as e:
                raise CacheError(
                    f"Erreur critique clear cache mémoire: {str(e)}",
                    cache_operation="clear"
                )
    
    def cleanup_expired(self):
        """Nettoie les entrées expirées (mémoire seulement)"""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._memory_cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
            
            if expired_keys:
                self.logger.debug(f"Memory cleanup: {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def get_stats(self):
        """Retourne statistiques détaillées du cache hybride"""
        with self._lock:
            self.cleanup_expired()
            
            redis_info = {}
            if self.redis_available:
                try:
                    redis_info = {
                        "connected": True,
                        "url_masked": self.redis_config.url.split('@')[-1] if '@' in self.redis_config.url else 'localhost',
                        "db": self.redis_config.db,
                        "prefix": self.redis_config.prefix
                    }
                    
                    server_info = self.redis_client.info()
                    redis_info.update({
                        "redis_version": server_info.get("redis_version", "unknown"),
                        "used_memory_human": server_info.get("used_memory_human", "unknown"),
                        "connected_clients": server_info.get("connected_clients", 0)
                    })
                    
                except Exception as e:
                    redis_info = {"connected": False, "error": str(e)}
            else:
                redis_info = {"connected": False, "reason": "not_configured_or_unavailable"}
            
            top_queries = sorted(
                self._stats.top_queries.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            return {
                "cache_performance": {
                    "total_requests": self._stats.total_requests,
                    "cache_hits": self._stats.cache_hits,
                    "cache_misses": self._stats.cache_misses,
                    "hit_rate": round(self._stats.hit_rate, 3),
                    "miss_rate": round(1 - self._stats.hit_rate, 3),
                    "redis_hits": self._stats.redis_hits,
                    "memory_hits": self._stats.memory_hits,
                    "redis_errors": self._stats.redis_errors
                },
                "cache_distribution": {
                    "redis_hit_rate": round(self._stats.redis_hits / max(1, self._stats.cache_hits), 3),
                    "memory_hit_rate": round(self._stats.memory_hits / max(1, self._stats.cache_hits), 3),
                    "redis_reliability": round(1 - (self._stats.redis_errors / max(1, self._stats.total_requests)), 3)
                },
                "cache_management": {
                    "memory_size": len(self._memory_cache),
                    "max_memory_size": self.max_size,
                    "memory_usage_percent": round(len(self._memory_cache) / self.max_size * 100, 1),
                    "evictions": self._stats.cache_evictions,
                    "avg_confidence": round(self._stats.avg_confidence, 3)
                },
                "configuration": {
                    "strategy": self.strategy.value,
                    "default_ttl": self.default_ttl,
                    "ttl_by_confidence": self._ttl_by_confidence,
                    "redis_available": REDIS_AVAILABLE,  # ✅ UTILISÉ
                    "redis_config": self.redis_config.__dict__
                },
                "redis_info": redis_info,
                "top_queries": top_queries,
                "efficiency": {
                    "estimated_memory_kb": len(self._memory_cache) * 2,
                    "hybrid_advantage": "Redis + Memory fallback"
                }
            }
    
    def get_redis_connection_status(self):
        """
        Vérifie statut connexion Redis
        
        Raises:
            CacheError: Si erreur critique de vérification
        """
        if not REDIS_AVAILABLE:  # ✅ UTILISÉ
            return {"connected": False, "reason": "redis_package_not_available"}
        
        if not self.redis_available:
            return {"connected": False, "reason": "not_configured_or_disabled"}
        
        try:
            start_time = time.time()
            self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            return {
                "connected": True,
                "ping_time_ms": round(ping_time, 2),
                "config": {
                    "url": self.redis_config.url.split('@')[-1] if '@' in self.redis_config.url else 'localhost',
                    "db": self.redis_config.db,
                    "prefix": self.redis_config.prefix
                }
            }
            
        except RedisConnectionError as e:  # ✅ UTILISÉ
            self.redis_available = False
            return {
                "connected": False,
                "error": f"Connection error: {str(e)}",
                "last_error_time": time.time()
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "last_error_time": time.time()
            }
    
    def optimize_cache(self):
        """Optimisation du cache : mémoire seulement (Redis auto-expire)"""
        with self._lock:
            initial_size = len(self._memory_cache)
            
            expired_removed = self.cleanup_expired()
            
            low_usage_removed = 0
            if len(self._memory_cache) > self.max_size * 0.8:
                keys_to_remove = []
                current_time = time.time()
                
                for key, entry in self._memory_cache.items():
                    age = current_time - entry.created_at
                    if entry.hit_count <= 1 and age > 3600:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self._memory_cache[key]
                
                low_usage_removed = len(keys_to_remove)
            
            final_size = len(self._memory_cache)
            total_removed = initial_size - final_size
            
            self.logger.info(
                f"Cache optimization: {total_removed} entries removed "
                f"({expired_removed} expired + {low_usage_removed} low usage)"
            )
            
            return {
                "initial_size": initial_size,
                "final_size": final_size,
                "total_removed": total_removed,
                "expired_removed": expired_removed,
                "low_usage_removed": low_usage_removed,
                "redis_note": "Redis entries auto-expire",
                "redis_available": REDIS_AVAILABLE  # ✅ UTILISÉ
            }


# Alias pour rétrocompatibilité
IntelligentMemoryCache = HybridIntelligentCache

# Instance singleton du cache
_cache_instance = None


def get_memory_cache():
    """
    Factory function pour récupérer instance cache singleton
    
    Raises:
        CacheError: Si initialisation échoue
    """
    global _cache_instance
    if _cache_instance is None:
        try:
            _cache_instance = HybridIntelligentCache()
        except Exception as e:
            logger.error(f"Erreur initialisation cache singleton: {e}")
            raise CacheError(
                f"Impossible d'initialiser le cache: {str(e)}",
                cache_operation="singleton_initialization"
            )
    return _cache_instance


def create_cache_with_config(max_size=None, strategy=CacheStrategy.SMART):
    """
    Crée instance cache avec configuration spécifique
    
    Raises:
        CacheError: Si configuration invalide
    """
    try:
        return HybridIntelligentCache(
            max_size=max_size or config.performance.cache_max_size,
            strategy=strategy
        )
    except Exception as e:
        logger.error(f"Erreur création cache configuré: {e}")
        raise CacheError(
            f"Impossible de créer cache avec config: {str(e)}",
            cache_operation="configured_creation",
            details={"max_size": max_size, "strategy": strategy.value if strategy else None}
        )


def test_redis_connection():
    """
    Test de connexion Redis indépendant
    
    Returns:
        Dict avec statut de connexion
        
    Raises:
        CacheError: Si test échoue de manière critique
    """
    if not REDIS_AVAILABLE:  # ✅ UTILISÉ
        return {
            "available": False,
            "reason": "redis_package_not_installed",
            "recommendation": "pip install redis"
        }
    
    config_redis = RedisConfig()
    
    if not config_redis.enabled:
        return {
            "available": False,
            "reason": "redis_disabled_in_config",
            "recommendation": "Set REDIS_CACHE_ENABLED=true"
        }
    
    if not config_redis.url:
        return {
            "available": False,
            "reason": "redis_url_missing",
            "recommendation": "Set REDIS_URL environment variable"
        }
    
    try:
        test_client = redis.from_url(
            config_redis.url,
            db=config_redis.db,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            decode_responses=True
        )
        
        start_time = time.time()
        test_client.ping()
        ping_time = (time.time() - start_time) * 1000
        
        # Test d'écriture/lecture
        test_key = f"{config_redis.prefix}:test:{int(time.time())}"
        test_value = "connection_test"
        
        test_client.setex(test_key, 10, test_value)
        retrieved = test_client.get(test_key)
        test_client.delete(test_key)
        
        if retrieved != test_value:
            raise CacheError(
                "Redis read/write test failed",
                cache_operation="connection_test"
            )
        
        return {
            "available": True,
            "ping_time_ms": round(ping_time, 2),
            "url_masked": config_redis.url.split('@')[-1] if '@' in config_redis.url else 'localhost',
            "db": config_redis.db,
            "read_write_test": "passed"
        }
        
    except RedisConnectionError as e:  # ✅ UTILISÉ
        return {
            "available": False,
            "reason": "connection_failed",
            "error": str(e),
            "recommendation": "Check Redis server status and URL"
        }
    except Exception as e:
        raise CacheError(
            f"Redis connection test failed: {str(e)}",
            cache_operation="connection_test",
            details={"redis_url_masked": config_redis.url.split('@')[-1] if '@' in config_redis.url else 'localhost'}
        )


def create_test_cache_entry(query="test query", intent=IntentType.GREETING, confidence=0.9):
    """
    Crée une entrée de cache pour les tests
    
    Returns:
        Dict: Entrée de cache formatée
        
    Raises:
        CacheError: Si création échoue
    """
    try:
        from conversation_service.models.intent import IntentResponse
        from conversation_service.models.enums import DetectionMethod
        
        test_result = {
            "intent": intent.value,
            "intent_code": "TEST_CODE",
            "confidence": confidence,
            "processing_time_ms": 10.0,
            "method_used": DetectionMethod.RULES.value,
            "query": query,
            "entities": {},
            "suggestions": [],
            "cost_estimate": 0.0,
            "cached": False
        }
        
        # Validation via IntentResponse
        intent_response = IntentResponse(**test_result)
        
        return test_result
        
    except Exception as e:
        raise CacheError(
            f"Erreur création entrée de test: {str(e)}",
            cache_operation="test_entry_creation",
            details={"query": query, "intent": intent.value, "confidence": confidence}
        )


def benchmark_cache_performance(cache_instance=None, num_operations=1000):
    """
    Benchmark des performances du cache
    
    Returns:
        Dict avec résultats de performance
        
    Raises:
        CacheError: Si benchmark échoue
    """
    if cache_instance is None:
        cache_instance = get_memory_cache()
    
    try:
        import random
        
        results = {
            "operations": num_operations,
            "put_times": [],
            "get_times": [],
            "redis_available": REDIS_AVAILABLE and cache_instance.redis_available,  # ✅ UTILISÉ
            "memory_only": not (REDIS_AVAILABLE and cache_instance.redis_available)
        }
        
        # Génération données de test
        test_data = []
        for i in range(num_operations):
            test_entry = create_test_cache_entry(
                query=f"test query {i}",
                intent=random.choice(list(IntentType)),
                confidence=random.uniform(0.5, 1.0)
            )
            test_data.append((f"test_key_{i}", test_entry))
        
        # Benchmark PUT operations
        for key, result in test_data:
            start_time = time.time()
            cache_instance.put(key, result, result["confidence"], IntentType(result["intent"]))
            put_time = (time.time() - start_time) * 1000
            results["put_times"].append(put_time)
        
        # Benchmark GET operations
        for key, _ in test_data:
            start_time = time.time()
            cached_result = cache_instance.get(key)
            get_time = (time.time() - start_time) * 1000
            results["get_times"].append(get_time)
            
            if cached_result is None:
                logger.warning(f"Cache miss inattendu pour {key}")
        
        # Calcul statistiques
        results["avg_put_time_ms"] = sum(results["put_times"]) / len(results["put_times"])
        results["avg_get_time_ms"] = sum(results["get_times"]) / len(results["get_times"])
        results["max_put_time_ms"] = max(results["put_times"])
        results["max_get_time_ms"] = max(results["get_times"])
        results["cache_stats"] = cache_instance.get_stats()
        
        return results
        
    except Exception as e:
        raise CacheError(
            f"Erreur benchmark cache: {str(e)}",
            cache_operation="benchmark",
            details={"num_operations": num_operations}
        )


# Exports publics
__all__ = [
    "HybridIntelligentCache",
    "IntelligentMemoryCache",
    "CacheStats", 
    "RedisConfig",
    "get_memory_cache",
    "create_cache_with_config",
    "test_redis_connection",
    "create_test_cache_entry",
    "benchmark_cache_performance"
]