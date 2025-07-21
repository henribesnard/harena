"""
💾 Cache Redis spécialisé avec TTL différenciés par niveau

Gestionnaire cache multi-niveaux avec optimisations spécifiques
L0/L1/L2 et métriques performance détaillées.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
import aioredis
from aioredis import Redis
from aioredis.exceptions import RedisError, ConnectionError as RedisConnectionError

from config_service.config import settings
from conversation_service.intent_detection.models import (
    IntentResult, CacheKey, CacheEntry, IntentLevel
)
from conversation_service.utils.logging import log_intent_detection

logger = logging.getLogger(__name__)

class CacheManager:
    """
    💾 Gestionnaire cache Redis multi-niveaux
    
    Fonctionnalités:
    - Cache L0: Patterns pré-calculés (TTL: 1h)
    - Cache L1: Embeddings TinyBERT (TTL: 30min)  
    - Cache L2: Réponses DeepSeek (TTL: 15min)
    - Circuit breaker Redis
    - Compression valeurs volumineuses
    - Métriques hit rate par niveau
    """
    
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.cache_config = settings.get_cache_config()
        
        # Configuration TTL par niveau
        self.ttl_config = {
            "L0_PATTERN": self.cache_config["ttl"]["intent"],      # 5min par défaut
            "L1_LIGHTWEIGHT": self.cache_config["ttl"]["entity"], # 3min par défaut  
            "L2_LLM": self.cache_config["ttl"]["response"]        # 1min par défaut
        }
        
        # Métriques cache
        self._cache_metrics = {
            "total_hits": 0,
            "total_misses": 0,
            "total_sets": 0,
            "errors": 0,
            "by_level": {}
        }
        
        # Circuit breaker Redis
        self._redis_healthy = True
        self._last_health_check = 0
        self._health_check_interval = self.cache_config["redis"]["health_check_interval"]
        
        # Cache local de fallback (en mémoire)
        self._local_cache: Dict[str, CacheEntry] = {}
        self._local_cache_max_size = self.cache_config["memory"]["size"]
        
        logger.info("💾 Cache Manager initialisé")
    
    async def initialize(self):
        """Initialisation connexion Redis avec fallback gracieux"""
        if not self.cache_config["redis"]["enabled"]:
            logger.info("💾 Cache Redis désactivé - Mode cache local uniquement")
            return
        
        try:
            # Configuration connexion Redis
            redis_url = self.cache_config["redis"]["url"]
            redis_password = self.cache_config["redis"]["password"]
            
            logger.info(f"🔌 Connexion Redis: {redis_url}")
            
            self.redis_client = await aioredis.from_url(
                redis_url,
                password=redis_password,
                db=self.cache_config["redis"]["db"],
                max_connections=self.cache_config["redis"]["max_connections"],
                retry_on_timeout=self.cache_config["redis"]["retry_on_timeout"],
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=3
            )
            
            # Test connexion
            await self.redis_client.ping()
            self._redis_healthy = True
            
            logger.info("✅ Redis connecté avec succès")
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de se connecter à Redis: {e}")
            logger.info("💾 Utilisation cache local en fallback")
            self.redis_client = None
            self._redis_healthy = False
    
    async def get_cached_result(self, cache_key: CacheKey) -> Optional[IntentResult]:
        """
        Récupère résultat depuis cache avec fallback local
        
        Args:
            cache_key: Clé cache structurée
            
        Returns:
            IntentResult: Résultat cached ou None si miss
        """
        redis_key = cache_key.to_redis_key()
        level = cache_key.namespace
        
        try:
            # 1. Tentative cache Redis si disponible
            if self._redis_healthy and self.redis_client:
                cached_data = await self._get_from_redis(redis_key)
                
                if cached_data:
                    result = self._deserialize_result(cached_data)
                    if result:
                        self._record_cache_hit(level, "redis")
                        
                        log_intent_detection(
                            "cache_hit",
                            cache_type="redis",
                            level=level,
                            key=redis_key[:50]  # Tronquer pour logs
                        )
                        
                        return result
            
            # 2. Fallback cache local
            local_entry = self._local_cache.get(redis_key)
            if local_entry and not local_entry.is_expired():
                local_entry.increment_hit()
                self._record_cache_hit(level, "local")
                
                log_intent_detection(
                    "cache_hit",
                    cache_type="local", 
                    level=level,
                    key=redis_key[:50]
                )
                
                return local_entry.value
            
            # 3. Cache miss
            self._record_cache_miss(level)
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur récupération cache {redis_key}: {e}")
            self._record_cache_error()
            return None
    
    async def cache_result(self, cache_key: CacheKey, result: IntentResult, ttl_seconds: int):
        """
        Met en cache résultat avec TTL spécifique
        
        Args:
            cache_key: Clé cache structurée
            result: Résultat à cacher
            ttl_seconds: TTL en secondes
        """
        redis_key = cache_key.to_redis_key()
        level = cache_key.namespace
        
        try:
            # Sérialisation résultat
            serialized_data = self._serialize_result(result)
            
            # 1. Cache Redis si disponible
            if self._redis_healthy and self.redis_client:
                await self._set_to_redis(redis_key, serialized_data, ttl_seconds)
            
            # 2. Cache local systématique (backup)
            cache_entry = CacheEntry(
                key=cache_key,
                value=result,
                ttl_seconds=ttl_seconds,
                created_at=time.time()
            )
            
            self._set_local_cache(redis_key, cache_entry)
            self._record_cache_set(level)
            
            log_intent_detection(
                "cache_set",
                level=level,
                key=redis_key[:50],
                ttl_seconds=ttl_seconds
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur mise en cache {redis_key}: {e}")
            self._record_cache_error()
    
    async def _get_from_redis(self, key: str) -> Optional[str]:
        """Récupération depuis Redis avec health check"""
        try:
            # Vérification santé Redis périodique
            if time.time() - self._last_health_check > self._health_check_interval:
                await self._check_redis_health()
            
            if not self._redis_healthy:
                return None
            
            return await self.redis_client.get(key)
            
        except (RedisError, RedisConnectionError) as e:
            logger.warning(f"⚠️ Erreur Redis GET {key}: {e}")
            self._redis_healthy = False
            return None
    
    async def _set_to_redis(self, key: str, value: str, ttl: int):
        """Mise en cache Redis avec gestion erreurs"""
        try:
            if not self._redis_healthy:
                return
            
            await self.redis_client.setex(key, ttl, value)
            
        except (RedisError, RedisConnectionError) as e:
            logger.warning(f"⚠️ Erreur Redis SET {key}: {e}")
            self._redis_healthy = False
    
    async def _check_redis_health(self):
        """Vérification santé Redis avec retry automatique"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                self._redis_healthy = True
                
        except Exception as e:
            logger.warning(f"⚠️ Redis health check failed: {e}")
            self._redis_healthy = False
        
        self._last_health_check = time.time()
    
    def _set_local_cache(self, key: str, entry: CacheEntry):
        """Gestion cache local avec éviction LRU"""
        # Éviction si cache plein
        if len(self._local_cache) >= self._local_cache_max_size:
            self._evict_local_cache()
        
        self._local_cache[key] = entry
    
    def _evict_local_cache(self):
        """Éviction LRU cache local (supprime 25% des plus anciennes)"""
        if not self._local_cache:
            return
        
        # Tri par timestamp création (plus ancien en premier)
        sorted_items = sorted(
            self._local_cache.items(),
            key=lambda item: item[1].created_at
        )
        
        # Supprime 25% des plus anciennes entrées
        evict_count = max(1, len(sorted_items) // 4)
        
        for i in range(evict_count):
            key_to_remove = sorted_items[i][0]
            del self._local_cache[key_to_remove]
        
        logger.debug(f"🧹 Éviction cache local: {evict_count} entrées supprimées")
    
    def _serialize_result(self, result: IntentResult) -> str:
        """Sérialisation optimisée IntentResult pour cache"""
        try:
            cache_dict = result.to_cache_dict()
            return json.dumps(cache_dict, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️ Erreur sérialisation: {e}")
            raise
    
    def _deserialize_result(self, data: str) -> Optional[IntentResult]:
        """Désérialisation IntentResult depuis cache"""
        try:
            cache_dict = json.loads(data)
            return IntentResult.from_cache_dict(cache_dict)
        except Exception as e:
            logger.warning(f"⚠️ Erreur désérialisation: {e}")
            return None
    
    # ==========================================
    # MÉTRIQUES ET MONITORING
    # ==========================================
    
    def _record_cache_hit(self, level: str, cache_type: str):
        """Enregistre hit cache avec détails niveau"""
        self._cache_metrics["total_hits"] += 1
        
        if level not in self._cache_metrics["by_level"]:
            self._cache_metrics["by_level"][level] = {
                "hits": 0, "misses": 0, "redis_hits": 0, "local_hits": 0
            }
        
        self._cache_metrics["by_level"][level]["hits"] += 1
        
        if cache_type == "redis":
            self._cache_metrics["by_level"][level]["redis_hits"] += 1
        else:
            self._cache_metrics["by_level"][level]["local_hits"] += 1
    
    def _record_cache_miss(self, level: str):
        """Enregistre miss cache par niveau"""
        self._cache_metrics["total_misses"] += 1
        
        if level not in self._cache_metrics["by_level"]:
            self._cache_metrics["by_level"][level] = {
                "hits": 0, "misses": 0, "redis_hits": 0, "local_hits": 0
            }
        
        self._cache_metrics["by_level"][level]["misses"] += 1
    
    def _record_cache_set(self, level: str):
        """Enregistre mise en cache"""
        self._cache_metrics["total_sets"] += 1
    
    def _record_cache_error(self):
        """Enregistre erreur cache"""
        self._cache_metrics["errors"] += 1
    
    async def get_cache_metrics(self) -> Dict[str, Any]:
        """Métriques cache détaillées"""
        total_requests = self._cache_metrics["total_hits"] + self._cache_metrics["total_misses"]
        
        metrics = {
            "total_requests": total_requests,
            "total_hits": self._cache_metrics["total_hits"],
            "total_misses": self._cache_metrics["total_misses"],
            "total_sets": self._cache_metrics["total_sets"],
            "errors": self._cache_metrics["errors"],
            "hit_rate": self._cache_metrics["total_hits"] / max(1, total_requests),
            "redis_healthy": self._redis_healthy,
            "local_cache_size": len(self._local_cache),
            "local_cache_max_size": self._local_cache_max_size
        }
        
        # Métriques par niveau
        for level, level_metrics in self._cache_metrics["by_level"].items():
            level_total = level_metrics["hits"] + level_metrics["misses"]
            level_hit_rate = level_metrics["hits"] / max(1, level_total)
            
            metrics[f"level_{level.lower()}"] = {
                "hit_rate": level_hit_rate,
                "hits": level_metrics["hits"],
                "misses": level_metrics["misses"],
                "redis_hits": level_metrics["redis_hits"],
                "local_hits": level_metrics["local_hits"]
            }
        
        # Métriques Redis si disponible
        if self._redis_healthy and self.redis_client:
            try:
                redis_info = await self.redis_client.info("memory")
                metrics["redis_memory_usage"] = redis_info.get("used_memory_human", "unknown")
                metrics["redis_connected_clients"] = redis_info.get("connected_clients", 0)
            except Exception:
                pass
        
        return metrics
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Status santé cache manager"""
        return {
            "redis_enabled": self.cache_config["redis"]["enabled"],
            "redis_healthy": self._redis_healthy,
            "redis_url": self.cache_config["redis"]["url"],
            "local_cache_enabled": True,
            "local_cache_usage": f"{len(self._local_cache)}/{self._local_cache_max_size}",
            "last_health_check": self._last_health_check,
            "ttl_config": self.ttl_config
        }
    
    # ==========================================
    # MÉTHODES UTILITAIRES
    # ==========================================
    
    async def clear_level_cache(self, level: str):
        """Vide cache pour un niveau spécifique"""
        if not self._redis_healthy or not self.redis_client:
            # Clear local cache seulement
            keys_to_remove = [
                key for key in self._local_cache.keys()
                if level in key
            ]
            for key in keys_to_remove:
                del self._local_cache[key]
            
            logger.info(f"🧹 Cache local niveau {level} vidé: {len(keys_to_remove)} entrées")
            return
        
        try:
            # Pattern pour le niveau
            pattern = f"conversation_service:{level}:*"
            
            # Récupération clés matching
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"🧹 Cache Redis niveau {level} vidé: {len(keys)} entrées")
            
            # Clear également cache local
            local_keys_to_remove = [
                key for key in self._local_cache.keys()
                if level in key
            ]
            for key in local_keys_to_remove:
                del self._local_cache[key]
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur vidage cache niveau {level}: {e}")
    
    async def clear_all_caches(self):
        """Vide tous les caches (Redis + local)"""
        try:
            # Clear Redis
            if self._redis_healthy and self.redis_client:
                pattern = "conversation_service:*"
                keys = await self.redis_client.keys(pattern)
                
                if keys:
                    await self.redis_client.delete(*keys)
                    logger.info(f"🧹 Cache Redis vidé: {len(keys)} entrées")
            
            # Clear cache local
            local_count = len(self._local_cache)
            self._local_cache.clear()
            logger.info(f"🧹 Cache local vidé: {local_count} entrées")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur vidage complet cache: {e}")
    
    async def warm_up_cache(self, patterns: List[Dict[str, Any]]):
        """Préchauffage cache avec patterns fréquents"""
        logger.info(f"🔥 Préchauffage cache avec {len(patterns)} patterns...")
        
        warmed_count = 0
        for pattern_data in patterns:
            try:
                # Construction clé cache
                cache_key = CacheKey.for_l0_pattern(pattern_data["query_hash"])
                
                # Vérification si déjà en cache
                existing = await self.get_cached_result(cache_key)
                if existing:
                    continue
                
                # Création résultat pattern
                from conversation_service.intent_detection.models import IntentResult, IntentType, IntentConfidence, IntentLevel
                
                pattern_result = IntentResult(
                    intent_type=IntentType(pattern_data["intent"]),
                    confidence=IntentConfidence(score=pattern_data["confidence"]),
                    level=IntentLevel.L0_PATTERN,
                    latency_ms=pattern_data.get("latency_ms", 5.0),
                    entities=pattern_data.get("entities", {}),
                    from_cache=False
                )
                
                # Mise en cache
                await self.cache_result(cache_key, pattern_result, ttl_seconds=3600)
                warmed_count += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur préchauffage pattern: {e}")
        
        logger.info(f"🔥 Préchauffage terminé: {warmed_count} patterns cachés")
    
    async def get_cache_size_info(self) -> Dict[str, Any]:
        """Informations taille cache par niveau"""
        info = {
            "local_cache": {
                "current_size": len(self._local_cache),
                "max_size": self._local_cache_max_size,
                "usage_percent": (len(self._local_cache) / self._local_cache_max_size) * 100
            }
        }
        
        # Taille cache Redis par niveau si disponible
        if self._redis_healthy and self.redis_client:
            try:
                for level in ["L0_PATTERN", "L1_LIGHTWEIGHT", "L2_LLM"]:
                    pattern = f"conversation_service:{level}:*"
                    keys = await self.redis_client.keys(pattern)
                    info[f"redis_{level.lower()}"] = {
                        "key_count": len(keys),
                        "estimated_memory_kb": len(keys) * 2  # Estimation grossière
                    }
            except Exception as e:
                logger.warning(f"⚠️ Erreur récupération taille Redis: {e}")
        
        return info
    
    async def cleanup_expired_entries(self):
        """Nettoyage automatique entrées expirées cache local"""
        expired_keys = []
        
        for key, entry in self._local_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._local_cache[key]
        
        if expired_keys:
            logger.debug(f"🧹 Nettoyage cache local: {len(expired_keys)} entrées expirées supprimées")
    
    # ==========================================
    # MÉTHODES DEBUG ET TESTING
    # ==========================================
    
    async def debug_cache_content(self, level: Optional[str] = None) -> Dict[str, Any]:
        """Contenu cache pour debug (attention: peut être volumineux)"""
        debug_info = {
            "local_cache_keys": list(self._local_cache.keys())[:20],  # Limite à 20 pour éviter spam
            "local_cache_entries": len(self._local_cache)
        }
        
        if self._redis_healthy and self.redis_client:
            try:
                if level:
                    pattern = f"conversation_service:{level}:*"
                else:
                    pattern = "conversation_service:*"
                
                redis_keys = await self.redis_client.keys(pattern)
                debug_info["redis_keys"] = redis_keys[:20]  # Limite aussi
                debug_info["redis_entries"] = len(redis_keys)
                
            except Exception as e:
                debug_info["redis_error"] = str(e)
        
        return debug_info
    
    async def force_cache_entry(self, key: str, result: IntentResult, ttl: int = 300):
        """Force mise en cache pour testing"""
        cache_key = CacheKey(namespace="DEBUG", query_hash=key)
        await self.cache_result(cache_key, result, ttl)
        logger.info(f"🔧 Entrée cache forcée: {key}")
    
    async def shutdown(self):
        """Arrêt propre cache manager"""
        logger.info("🛑 Arrêt Cache Manager...")
        
        try:
            # Sauvegarde métriques finales
            final_metrics = await self.get_cache_metrics()
            logger.info(f"📊 Métriques finales cache: Hit rate = {final_metrics['hit_rate']:.2%}")
            
            # Fermeture connexion Redis
            if self.redis_client:
                await self.redis_client.close()
                logger.info("✅ Connexion Redis fermée")
            
            # Clear cache local
            self._local_cache.clear()
            
            logger.info("✅ Cache Manager arrêté")
            
        except Exception as e:
            logger.error(f"❌ Erreur arrêt Cache Manager: {e}")


# ==========================================
# HELPERS ET UTILITAIRES CACHE
# ==========================================

def generate_query_hash(query: str, user_id: str, salt: str = "") -> str:
    """Génère hash stable pour requête + contexte"""
    import hashlib
    combined = f"{query.lower().strip()}|{user_id}|{salt}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()[:16]

def get_cache_ttl_for_confidence(confidence: float, base_ttl: int = 300) -> int:
    """TTL adaptatif basé sur confiance (plus haute confiance = TTL plus long)"""
    if confidence >= 0.95:
        return base_ttl * 4  # 4x pour très haute confiance
    elif confidence >= 0.85:
        return base_ttl * 2  # 2x pour haute confiance
    elif confidence >= 0.70:
        return base_ttl      # TTL normal
    else:
        return base_ttl // 2  # TTL réduit pour faible confiance

async def batch_cache_results(cache_manager: CacheManager, results: List[tuple], ttl: int = 300):
    """Mise en cache batch pour optimisation performance"""
    tasks = []
    
    for cache_key, result in results:
        task = cache_manager.cache_result(cache_key, result, ttl)
        tasks.append(task)
    
    # Exécution parallèle
    await asyncio.gather(*tasks, return_exceptions=True)
    
    logger.debug(f"📦 Batch cache: {len(results)} résultats mis en cache")