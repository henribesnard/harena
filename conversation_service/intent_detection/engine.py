"""
🧠 Pipeline principal L0→L1→L2 avec routage intelligent

Orchestrateur principal Intent Detection Engine gérant le pipeline
de détection d'intentions avec fallbacks automatiques et métriques temps réel.
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, Any, Optional, List

from config_service.config import settings
from conversation_service.intent_detection.models import (
    IntentResult, IntentType, IntentLevel, IntentConfidence, CacheKey
)
from conversation_service.intent_detection.cache_manager import CacheManager
from conversation_service.intent_detection.pattern_matcher import PatternMatcher
from conversation_service.intent_detection.lightweight_classifier import LightweightClassifier
from conversation_service.intent_detection.llm_fallback import LLMFallback
from conversation_service.utils import record_intent_performance
from conversation_service.utils.logging import log_intent_detection

logger = logging.getLogger(__name__)

class IntentDetectionEngine:
    """
    🎯 Orchestrateur principal pipeline L0→L1→L2
    
    Pipeline intelligent:
    1. L0 - Pattern Matcher (<10ms, 85% hit rate)
    2. L1 - Lightweight Classifier (15-30ms, 12% usage)
    3. L2 - LLM Fallback (200-500ms, 3% usage)
    
    Fonctionnalités:
    - Routage basé seuils confiance
    - Cache multi-niveaux Redis
    - Métriques temps réel
    - Fallbacks gracieux
    """
    
    def __init__(self):
        self.cache_manager: Optional[CacheManager] = None
        self.pattern_matcher: Optional[PatternMatcher] = None
        self.lightweight_classifier: Optional[LightweightClassifier] = None
        self.llm_fallback: Optional[LLMFallback] = None
        
        # Seuils configuration
        self.confidence_threshold = settings.MIN_CONFIDENCE_THRESHOLD
        self.l0_confidence_threshold = 0.90  # Seuil élevé pour L0
        self.l1_confidence_threshold = 0.85  # Seuil moyen pour L1
        
        # État initialisation
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # Métriques temps réel
        self._total_requests = 0
        self._level_distribution = {"L0": 0, "L1": 0, "L2": 0}
        self._performance_history = []
        
        logger.info("🧠 Intent Detection Engine initialisé")
    
    async def initialize(self):
        """Initialisation asynchrone composants avec fallbacks"""
        if self._initialized:
            return
        
        async with self._initialization_lock:
            if self._initialized:
                return
            
            logger.info("⚙️ Initialisation Intent Detection Engine...")
            
            try:
                # 1. Cache Manager (critère)
                logger.info("💾 Initialisation Cache Manager...")
                self.cache_manager = CacheManager()
                await self.cache_manager.initialize()
                
                # 2. Pattern Matcher (critique pour L0)
                logger.info("⚡ Initialisation Pattern Matcher L0...")
                self.pattern_matcher = PatternMatcher(self.cache_manager)
                await self.pattern_matcher.initialize()
                
                # 3. Lightweight Classifier (important pour L1)
                logger.info("🧠 Initialisation Lightweight Classifier L1...")
                try:
                    self.lightweight_classifier = LightweightClassifier(self.cache_manager)
                    await self.lightweight_classifier.initialize()
                except Exception as e:
                    logger.warning(f"⚠️ Lightweight Classifier indisponible: {e}")
                    self.lightweight_classifier = None
                
                # 4. LLM Fallback (fallback pour L2)
                logger.info("🚀 Initialisation LLM Fallback L2...")
                try:
                    self.llm_fallback = LLMFallback(self.cache_manager)
                    await self.llm_fallback.initialize()
                except Exception as e:
                    logger.warning(f"⚠️ LLM Fallback indisponible: {e}")
                    self.llm_fallback = None
                
                self._initialized = True
                logger.info("✅ Intent Detection Engine initialisé avec succès")
                
            except Exception as e:
                logger.error(f"❌ Erreur initialisation Intent Detection Engine: {e}")
                raise
    
    async def detect_intent(self, user_query: str, user_id: str = "anonymous") -> IntentResult:
        """
        🎯 Méthode principale détection intention
        
        Pipeline L0→L1→L2 avec routage intelligent basé confiance:
        1. Essai L0 patterns (cache + regex optimisés)
        2. Si confiance < 0.90 → Essai L1 embeddings
        3. Si confiance < 0.85 → Fallback L2 LLM
        
        Args:
            user_query: Message utilisateur à analyser
            user_id: ID utilisateur pour personnalisation
            
        Returns:
            IntentResult: Résultat avec intention, confiance, entités
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        self._total_requests += 1
        
        # Normalisation requête
        normalized_query = self._normalize_query(user_query)
        query_hash = self._hash_query(normalized_query, user_id)
        
        log_intent_detection(
            "detection_start",
            user_id=user_id,
            query_hash=query_hash,
            query_length=len(normalized_query)
        )
        
        try:
            # ==========================================
            # ÉTAPE L0 - PATTERN MATCHING (<10ms)
            # ==========================================
            l0_result = await self._try_pattern_matching(normalized_query, user_id, query_hash)
            
            if l0_result and l0_result.confidence.score >= self.l0_confidence_threshold:
                # Succès L0 - Pattern trouvé avec haute confiance
                latency = (time.time() - start_time) * 1000
                l0_result.latency_ms = latency
                l0_result.user_id = user_id
                
                self._level_distribution["L0"] += 1
                await record_intent_performance("L0_PATTERN", latency, user_id, success=True)
                
                log_intent_detection(
                    "l0_success",
                    user_id=user_id,
                    intent=l0_result.intent_type.value,
                    confidence=l0_result.confidence.score,
                    latency_ms=latency,
                    from_cache=l0_result.from_cache
                )
                
                return l0_result
            
            # ==========================================
            # ÉTAPE L1 - LIGHTWEIGHT CLASSIFIER (15-30ms)
            # ==========================================
            if self.lightweight_classifier:
                l1_result = await self._try_lightweight_classification(normalized_query, user_id, query_hash)
                
                if l1_result and l1_result.confidence.score >= self.l1_confidence_threshold:
                    # Succès L1 - Classification embeddings suffisante
                    latency = (time.time() - start_time) * 1000
                    l1_result.latency_ms = latency
                    l1_result.user_id = user_id
                    
                    self._level_distribution["L1"] += 1
                    await record_intent_performance("L1_LIGHTWEIGHT", latency, user_id, success=True)
                    
                    log_intent_detection(
                        "l1_success",
                        user_id=user_id,
                        intent=l1_result.intent_type.value,
                        confidence=l1_result.confidence.score,
                        latency_ms=latency,
                        from_cache=l1_result.from_cache
                    )
                    
                    return l1_result
            
            # ==========================================
            # ÉTAPE L2 - LLM FALLBACK (200-500ms)
            # ==========================================
            if self.llm_fallback:
                l2_result = await self._try_llm_fallback(normalized_query, user_id, query_hash)
                
                if l2_result:
                    # Fallback L2 - Toujours accepté (dernière option)
                    latency = (time.time() - start_time) * 1000
                    l2_result.latency_ms = latency
                    l2_result.user_id = user_id
                    
                    self._level_distribution["L2"] += 1
                    await record_intent_performance("L2_LLM", latency, user_id, success=True)
                    
                    log_intent_detection(
                        "l2_success",
                        user_id=user_id,
                        intent=l2_result.intent_type.value,
                        confidence=l2_result.confidence.score,
                        latency_ms=latency,
                        from_cache=l2_result.from_cache
                    )
                    
                    return l2_result
            
            # ==========================================
            # FALLBACK ULTIME - INTENTION INCONNUE
            # ==========================================
            latency = (time.time() - start_time) * 1000
            
            fallback_result = IntentResult(
                intent_type=IntentType.UNKNOWN,
                confidence=IntentConfidence(score=0.0, reasoning="All levels failed"),
                level=IntentLevel.ERROR_FALLBACK,
                latency_ms=latency,
                user_id=user_id,
                processing_details={
                    "l0_available": self.pattern_matcher is not None,
                    "l1_available": self.lightweight_classifier is not None,
                    "l2_available": self.llm_fallback is not None,
                    "fallback_reason": "no_level_succeeded"
                }
            )
            
            await record_intent_performance("ERROR_FALLBACK", latency, user_id, success=False)
            
            log_intent_detection(
                "fallback_unknown",
                user_id=user_id,
                latency_ms=latency,
                reason="all_levels_failed"
            )
            
            return fallback_result
            
        except asyncio.TimeoutError:
            # Timeout global dépassé
            latency = (time.time() - start_time) * 1000
            
            timeout_result = IntentResult(
                intent_type=IntentType.UNKNOWN,
                confidence=IntentConfidence(score=0.0, reasoning="Global timeout exceeded"),
                level=IntentLevel.ERROR_TIMEOUT,
                latency_ms=latency,
                user_id=user_id
            )
            
            await record_intent_performance("ERROR_TIMEOUT", latency, user_id, success=False)
            
            log_intent_detection(
                "timeout_error",
                user_id=user_id,
                latency_ms=latency
            )
            
            return timeout_result
            
        except Exception as e:
            # Erreur système inattendue
            latency = (time.time() - start_time) * 1000
            
            error_result = IntentResult(
                intent_type=IntentType.UNKNOWN,
                confidence=IntentConfidence(score=0.0, reasoning=f"System error: {str(e)}"),
                level=IntentLevel.ERROR_FALLBACK,
                latency_ms=latency,
                user_id=user_id,
                processing_details={"error": str(e)}
            )
            
            await record_intent_performance("ERROR_SYSTEM", latency, user_id, success=False)
            
            log_intent_detection(
                "system_error",
                user_id=user_id,
                latency_ms=latency,
                error=str(e)
            )
            
            logger.error(f"💥 Erreur système détection intention: {e}", exc_info=True)
            return error_result
    
    async def _try_pattern_matching(self, query: str, user_id: str, query_hash: str) -> Optional[IntentResult]:
        """Tentative détection L0 avec pattern matching ultra-rapide"""
        if not self.pattern_matcher:
            return None
        
        try:
            # Vérification cache L0 d'abord
            cache_key = CacheKey.for_l0_pattern(query_hash)
            cached_result = await self.cache_manager.get_cached_result(cache_key)
            
            if cached_result:
                cached_result.from_cache = True
                return cached_result
            
            # Pattern matching sur requête normalisée
            start_time = time.time()
            pattern_result = await self.pattern_matcher.match_intent(query)
            processing_time = (time.time() - start_time) * 1000
            
            if pattern_result:
                pattern_result.latency_ms = processing_time
                
                # Cache résultat L0 si confiance élevée
                if pattern_result.confidence.score >= 0.85:
                    await self.cache_manager.cache_result(cache_key, pattern_result, ttl_seconds=3600)
                
                return pattern_result
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur pattern matching L0: {e}")
            return None
    
    async def _try_lightweight_classification(self, query: str, user_id: str, query_hash: str) -> Optional[IntentResult]:
        """Tentative détection L1 avec classification légère TinyBERT"""
        if not self.lightweight_classifier:
            return None
        
        try:
            # Vérification cache L1
            cache_key = CacheKey.for_l1_embedding(query_hash, user_id)
            cached_result = await self.cache_manager.get_cached_result(cache_key)
            
            if cached_result:
                cached_result.from_cache = True
                return cached_result
            
            # Classification embeddings
            start_time = time.time()
            l1_result = await self.lightweight_classifier.classify_intent(query, user_id)
            processing_time = (time.time() - start_time) * 1000
            
            if l1_result:
                l1_result.latency_ms = processing_time
                
                # Cache résultat L1 si confiance suffisante
                if l1_result.confidence.score >= 0.70:
                    await self.cache_manager.cache_result(cache_key, l1_result, ttl_seconds=1800)
                
                return l1_result
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lightweight classification L1: {e}")
            return None
    
    async def _try_llm_fallback(self, query: str, user_id: str, query_hash: str) -> Optional[IntentResult]:
        """Tentative détection L2 avec analyse LLM avancée"""
        if not self.llm_fallback:
            return None
        
        try:
            # Vérification cache L2
            cache_key = CacheKey.for_l2_llm(query_hash, user_id)
            cached_result = await self.cache_manager.get_cached_result(cache_key)
            
            if cached_result:
                cached_result.from_cache = True
                return cached_result
            
            # Analyse LLM complexe
            start_time = time.time()
            l2_result = await self.llm_fallback.analyze_complex_intent(query, user_id)
            processing_time = (time.time() - start_time) * 1000
            
            if l2_result:
                l2_result.latency_ms = processing_time
                
                # Cache résultat L2 (toujours - coût élevé)
                await self.cache_manager.cache_result(cache_key, l2_result, ttl_seconds=900)
                
                return l2_result
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur LLM fallback L2: {e}")
            return None
    
    def _normalize_query(self, query: str) -> str:
        """Normalisation requête utilisateur pour matching optimisé"""
        if not query:
            return ""
        
        # Nettoyage basique
        normalized = query.strip().lower()
        
        # Suppression caractères spéciaux en préservant accents français
        import re
        normalized = re.sub(r'[^\w\s\-\.àâäéèêëïîôöùûüÿñç]', ' ', normalized)
        
        # Normalisation espaces multiples
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _hash_query(self, query: str, user_id: str) -> str:
        """Génération hash stable pour cache basé sur requête + utilisateur"""
        combined = f"{query}|{user_id}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()[:16]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Status santé Intent Detection Engine avec métriques"""
        status = {
            "initialized": self._initialized,
            "total_requests": self._total_requests,
            "level_distribution": self._level_distribution.copy()
        }
        
        # Status composants
        if self.cache_manager:
            status["cache_manager"] = await self.cache_manager.get_health_status()
        
        if self.pattern_matcher:
            status["pattern_matcher"] = self.pattern_matcher.get_status()
        
        if self.lightweight_classifier:
            status["lightweight_classifier"] = await self.lightweight_classifier.get_status()
        
        if self.llm_fallback:
            status["llm_fallback"] = await self.llm_fallback.get_status()
        
        # Calcul distribution pourcentages
        if self._total_requests > 0:
            status["level_percentages"] = {
                level: round((count / self._total_requests) * 100, 1)
                for level, count in self._level_distribution.items()
            }
        
        return status
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Métriques performance détaillées pour monitoring"""
        metrics = {
            "total_requests": self._total_requests,
            "level_distribution": self._level_distribution.copy(),
            "configuration": {
                "l0_threshold": self.l0_confidence_threshold,
                "l1_threshold": self.l1_confidence_threshold,
                "global_threshold": self.confidence_threshold
            }
        }
        
        # Métriques composants si disponibles
        if self.cache_manager:
            metrics["cache_metrics"] = await self.cache_manager.get_cache_metrics()
        
        if self.pattern_matcher:
            metrics["l0_patterns"] = self.pattern_matcher.get_pattern_stats()
        
        return metrics
    
    async def shutdown(self):
        """Arrêt propre Intent Detection Engine"""
        logger.info("🛑 Arrêt Intent Detection Engine...")
        
        try:
            # Arrêt composants dans l'ordre inverse
            if self.llm_fallback:
                await self.llm_fallback.shutdown()
            
            if self.lightweight_classifier:
                await self.lightweight_classifier.shutdown()
            
            if self.pattern_matcher:
                await self.pattern_matcher.shutdown()
            
            if self.cache_manager:
                await self.cache_manager.shutdown()
            
            self._initialized = False
            logger.info("✅ Intent Detection Engine arrêté")
            
        except Exception as e:
            logger.error(f"❌ Erreur arrêt Intent Detection Engine: {e}")
    
    # ==========================================
    # MÉTHODES DEBUG ET TESTING
    # ==========================================
    
    async def force_level_detection(self, query: str, level: str, user_id: str = "test") -> IntentResult:
        """Force détection à un niveau spécifique (debug/testing)"""
        normalized_query = self._normalize_query(query)
        
        if level == "L0" and self.pattern_matcher:
            return await self._try_pattern_matching(normalized_query, user_id, "debug")
        elif level == "L1" and self.lightweight_classifier:
            return await self._try_lightweight_classification(normalized_query, user_id, "debug")
        elif level == "L2" and self.llm_fallback:
            return await self._try_llm_fallback(normalized_query, user_id, "debug")
        else:
            raise ValueError(f"Level {level} not available or invalid")
    
    async def clear_all_caches(self):
        """Vide tous les caches pour testing"""
        if self.cache_manager:
            await self.cache_manager.clear_all_caches()
            logger.info("🧹 Tous les caches vidés")
    
    def get_level_targets(self) -> Dict[str, Dict[str, Any]]:
        """Retourne cibles performance par niveau"""
        return {
            "L0_PATTERN": {
                "target_latency_ms": 10,
                "target_usage_percent": 85,
                "confidence_threshold": self.l0_confidence_threshold
            },
            "L1_LIGHTWEIGHT": {
                "target_latency_ms": 30,
                "target_usage_percent": 12,
                "confidence_threshold": self.l1_confidence_threshold
            },
            "L2_LLM": {
                "target_latency_ms": 500,
                "target_usage_percent": 3,
                "confidence_threshold": self.confidence_threshold
            }
        }