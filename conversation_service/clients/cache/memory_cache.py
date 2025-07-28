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
        self.redis_available = False
        
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
        
        # Initialisation Redis
        self._initialize_redis()
        
        self.logger.info(
            f"Cache hybride initialisé: Redis={self.redis_available}, "
            f"taille max mémoire={self.max_size}, TTL={default_ttl}s, stratégie={strategy.value}"
        )
    
    def _initialize_redis(self):
        """Initialise connexion Redis avec fallback gracieux"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis non disponible: package redis non installé")
            return
        
        if not self.redis_config.enabled:
            self.logger.info("Redis désactivé par configuration")
            return
        
        if not self.redis_config.url:
            self.logger.warning("Redis URL manquante dans configuration")
            return
        
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
            
            self.redis_client.ping()
            self.redis_available = True
            
            self.logger.info(f"✅ Redis connecté: {self.redis_config.url.split('@')[-1] if '@' in self.redis_config.url else 'localhost'}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Redis connexion échouée: {e}. Utilisation cache mémoire uniquement.")
            self.redis_client = None
            self.redis_available = False
    
    def _get_redis_key(self, key):
        """Génère clé Redis avec préfixe"""
        return f"{self.redis_config.prefix}:intent:{key}"
    
    def get(self, key):
        """Récupère un élément du cache (Redis puis mémoire)"""
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
                        
                except Exception as e:
                    self.logger.warning(f"Redis get error: {e}")
                    self._stats.redis_errors += 1
            
            # 2. Fallback cache mémoire
            memory_result = self._get_from_memory(key)
            if memory_result:
                self._stats.cache_hits += 1
                self._stats.memory_hits += 1
                self._update_hit_rate()
                self.logger.debug(f"Memory hit: {key[:50]}...")
                return memory_result
            
            # 3. Cache miss
            self._stats.cache_misses += 1
            return None
    
    def _get_from_redis(self, key):
        """Récupère depuis Redis avec gestion TTL"""
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
            self.logger.warning(f"Redis get parsing error: {e}")
            return None
    
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
        """Met un élément en cache (Redis + mémoire avec TTL adaptatif)"""
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
            
            if self.redis_available:
                success |= self._put_to_redis(key, cache_data, ttl)
            
            success |= self._put_to_memory(key, result, confidence)
            
            if success:
                self._update_cache_stats(key, confidence)
            
            return success
    
    def _put_to_redis(self, key, cache_data, ttl):
        """Met en cache dans Redis"""
        if not self.redis_available:
            return False
        
        redis_key = self._get_redis_key(key)
        
        try:
            redis_data = json.dumps(cache_data, default=str)
            self.redis_client.setex(redis_key, ttl, redis_data)
            self.logger.debug(f"Redis cache: {key[:50]}... (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            self.logger.warning(f"Redis put error: {e}")
            self._stats.redis_errors += 1
            return False
    
    def _put_to_memory(self, key, result, confidence):
        """Met en cache en mémoire locale"""
        try:
            if len(self._memory_cache) >= self.max_size and key not in self._memory_cache:
                if not self._evict_memory_lru():
                    return False
            
            from conversation_service.models.intent import IntentResponse
            try:
                intent_response = IntentResponse(**result)
            except Exception as e:
                self.logger.warning(f"Erreur création IntentResponse pour cache mémoire: {e}")
                return False
            
            cache_entry = CacheEntry(
                query=result.get("query", ""),
                result=intent_response,
                created_at=time.time()
            )
            
            self._memory_cache[key] = cache_entry
            self._memory_cache.move_to_end(key)
            
            self.logger.debug(f"Memory cache: {key[:50]}...")
            return True
            
        except Exception as e:
            self.logger.warning(f"Memory cache put error: {e}")
            return False
    
    def _should_cache(self, confidence, intent):
        """Détermine si résultat doit être mis en cache selon stratégie"""
        min_confidence = self.strategy.get_min_confidence()
        
        if confidence < min_confidence:
            return False
        
        if self.strategy == CacheStrategy.SMART:
            if intent and intent in [IntentType.GREETING, IntentType.GOODBYE]:
                return confidence > 0.95
            
            if intent and intent.is_financial(intent.value):
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
        """Éviction LRU mémoire"""
        if not self._memory_cache:
            return False
        
        lru_key, lru_entry = self._memory_cache.popitem(last=False)
        self._stats.cache_evictions += 1
        
        self.logger.debug(f"Memory LRU eviction: {lru_key[:50]}...")
        return True
    
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
        """Invalide une entrée spécifique du cache (Redis + mémoire)"""
        with self._lock:
            invalidated = False
            
            if self.redis_available:
                try:
                    redis_key = self._get_redis_key(key)
                    result = self.redis_client.delete(redis_key)
                    if result > 0:
                        invalidated = True
                        self.logger.debug(f"Redis invalidated: {key[:50]}...")
                except Exception as e:
                    self.logger.warning(f"Redis invalidation error: {e}")
            
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
        """Vide complètement le cache (Redis + mémoire)"""
        with self._lock:
            total_removed = 0
            
            if self.redis_available:
                try:
                    pattern = f"{self.redis_config.prefix}:intent:*"
                    redis_keys = self.redis_client.keys(pattern)
                    if redis_keys:
                        removed = self.redis_client.delete(*redis_keys)
                        total_removed += removed
                        self.logger.info(f"Redis cleared: {removed} keys")
                except Exception as e:
                    self.logger.warning(f"Redis clear error: {e}")
            
            memory_size = len(self._memory_cache)
            self._memory_cache.clear()
            total_removed += memory_size
            
            self.logger.info(f"Cache vidé: {total_removed} entrées supprimées")
            return total_removed
    
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
        """Vérifie statut connexion Redis"""
        if not self.redis_available:
            return {"connected": False, "reason": "not_available"}
        
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
                "redis_note": "Redis entries auto-expire"
            }


# Alias pour rétrocompatibilité
IntelligentMemoryCache = HybridIntelligentCache

# Instance singleton du cache
_cache_instance = None


def get_memory_cache():
    """Factory function pour récupérer instance cache singleton"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = HybridIntelligentCache()
    return _cache_instance


def create_cache_with_config(max_size=None, strategy=CacheStrategy.SMART):
    """Crée instance cache avec configuration spécifique"""
    return HybridIntelligentCache(
        max_size=max_size or config.performance.cache_max_size,
        strategy=strategy
    )


# Exports publics
__all__ = [
    "HybridIntelligentCache",
    "IntelligentMemoryCache",
    "CacheStats", 
    "RedisConfig",
    "get_memory_cache",
    "create_cache_with_config"
]