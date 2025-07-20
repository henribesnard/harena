"""
Moteur principal de détection d'intention hybride
Performance optimisée pour Heroku avec Redis Cloud
"""

import time
import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .models import IntentResult, IntentLevel, IntentConfidence
from .cache_manager import IntentCacheManager
from .pattern_matcher import PatternMatcher
from .lightweight_classifier import LightweightClassifier
from .llm_fallback import LLMFallback
from  conversation_service.config import settings
from  conversation_service.utils.metrics import IntentMetrics


@dataclass
class IntentDetectionEngine:
    """
    Moteur hybride de détection d'intention - Architecture 3 niveaux
    
    Performance ciblée:
    - L0 (Patterns): 5-10ms, 85% des requêtes
    - L1 (TinyBERT): 15-30ms, 12% des requêtes  
    - L2 (DeepSeek): 200-500ms, 3% des requêtes
    """
    
    def __init__(self):
        self.cache_manager = IntentCacheManager()
        self.pattern_matcher = PatternMatcher()
        self.lightweight_classifier = LightweightClassifier()
        self.llm_fallback = LLMFallback()
        self.metrics = IntentMetrics()
    
    async def detect_intent(
        self, 
        user_query: str, 
        user_id: int,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> IntentResult:
        """
        Pipeline principal de détection d'intention
        
        Args:
            user_query: Requête utilisateur brute
            user_id: ID utilisateur pour cache personnalisé
            conversation_context: Contexte conversationnel optionnel
            
        Returns:
            IntentResult avec intention détectée et métadonnées
        """
        start_time = time.time()
        
        try:
            # Étape 1: Normalisation et hachage (< 5ms)
            normalized_query = self._normalize_query(user_query)
            query_hash = self._hash_query(normalized_query, user_id)
            
            # Étape 2: L0 Cache - Patterns pré-calculés (5-10ms)
            l0_result = await self._try_l0_pattern_cache(
                normalized_query, query_hash, user_id
            )
            if l0_result and l0_result.confidence.score > 0.90:
                self.metrics.record_intent_detection(
                    level=IntentLevel.L0_PATTERN,
                    latency_ms=int((time.time() - start_time) * 1000),
                    cache_hit=True
                )
                return l0_result
            
            # Étape 3: L1 Classification - TinyBERT + Cache (15-30ms)
            l1_result = await self._try_l1_lightweight_classification(
                normalized_query, query_hash, user_id, conversation_context
            )
            if l1_result and l1_result.confidence.score > settings.INTENT_CONFIDENCE_THRESHOLD:
                self.metrics.record_intent_detection(
                    level=IntentLevel.L1_LIGHTWEIGHT,
                    latency_ms=int((time.time() - start_time) * 1000),
                    cache_hit=l1_result.metadata.get("cache_hit", False)
                )
                return l1_result
            
            # Étape 4: L2 Fallback - DeepSeek API (200-500ms)
            l2_result = await self._try_l2_llm_fallback(
                user_query, normalized_query, user_id, conversation_context
            )
            
            self.metrics.record_intent_detection(
                level=IntentLevel.L2_LLM,
                latency_ms=int((time.time() - start_time) * 1000),
                cache_hit=False
            )
            
            return l2_result
            
        except Exception as e:
            self.metrics.record_intent_error(str(e))
            # Fallback d'urgence avec intention générique
            return IntentResult(
                intent_type="general_query",
                entities={},
                confidence=IntentConfidence(score=0.5, level=IntentLevel.FALLBACK),
                level=IntentLevel.FALLBACK,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e), "fallback": True}
            )
    
    async def _try_l0_pattern_cache(
        self, 
        normalized_query: str, 
        query_hash: str, 
        user_id: int
    ) -> Optional[IntentResult]:
        """
        Niveau 0 - Cache patterns pré-calculés (5-10ms)
        """
        try:
            # Vérification cache pattern exact
            cached_result = await self.cache_manager.get_pattern_cache(normalized_query)
            if cached_result:
                return IntentResult.from_cache(
                    cached_result, 
                    level=IntentLevel.L0_PATTERN,
                    cache_hit=True
                )
            
            # Matching patterns financiers fréquents
            pattern_result = await self.pattern_matcher.match_financial_patterns(
                normalized_query
            )
            
            if pattern_result and pattern_result.confidence.score > 0.90:
                # Cache du résultat pour réutilisation
                await self.cache_manager.set_pattern_cache(
                    normalized_query, 
                    pattern_result,
                    ttl=settings.INTENT_CACHE_TTL_PATTERNS
                )
                return pattern_result
                
            return None
            
        except Exception as e:
            self.metrics.record_l0_error(str(e))
            return None
    
    async def _try_l1_lightweight_classification(
        self,
        normalized_query: str,
        query_hash: str, 
        user_id: int,
        conversation_context: Optional[Dict[str, Any]]
    ) -> Optional[IntentResult]:
        """
        Niveau 1 - Classification TinyBERT + Cache embeddings (15-30ms)
        """
        try:
            # Vérification cache embeddings utilisateur
            cached_embedding = await self.cache_manager.get_embedding_cache(
                query_hash, user_id
            )
            
            if not cached_embedding:
                # Génération embedding avec TinyBERT quantifié
                embedding = await self.lightweight_classifier.generate_embedding(
                    normalized_query, conversation_context
                )
                
                # Cache embedding pour réutilisation
                await self.cache_manager.set_embedding_cache(
                    query_hash, user_id, embedding,
                    ttl=settings.INTENT_CACHE_TTL_EMBEDDINGS
                )
            else:
                embedding = cached_embedding
            
            # Classification avec embeddings intentions préchargés
            classification_result = await self.lightweight_classifier.classify_intent(
                embedding, normalized_query
            )
            
            if classification_result and classification_result.confidence.score > settings.INTENT_CONFIDENCE_THRESHOLD:
                return classification_result
                
            return None
            
        except Exception as e:
            self.metrics.record_l1_error(str(e))
            return None
    
    async def _try_l2_llm_fallback(
        self,
        original_query: str,
        normalized_query: str, 
        user_id: int,
        conversation_context: Optional[Dict[str, Any]]
    ) -> IntentResult:
        """
        Niveau 2 - Fallback DeepSeek API (200-500ms)
        """
        try:
            # Vérification cache réponses LLM
            cache_key = f"llm:{user_id}:{self._hash_query(original_query, user_id)}"
            cached_llm_result = await self.cache_manager.get_llm_cache(cache_key)
            
            if cached_llm_result:
                return IntentResult.from_cache(
                    cached_llm_result,
                    level=IntentLevel.L2_LLM,
                    cache_hit=True
                )
            
            # Appel DeepSeek pour requêtes complexes/ambiguës
            llm_result = await self.llm_fallback.analyze_complex_intent(
                original_query, normalized_query, conversation_context
            )
            
            # Cache résultat LLM
            await self.cache_manager.set_llm_cache(
                cache_key, llm_result,
                ttl=settings.INTENT_CACHE_TTL_LLM_RESPONSES
            )
            
            return llm_result
            
        except Exception as e:
            self.metrics.record_l2_error(str(e))
            # Intention fallback garantie
            return IntentResult(
                intent_type="general_query",
                entities={"original_query": original_query},
                confidence=IntentConfidence(score=0.6, level=IntentLevel.L2_LLM),
                level=IntentLevel.L2_LLM,
                latency_ms=0,
                metadata={"llm_error": str(e), "fallback": True}
            )
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalisation rapide pour matching patterns et cache
        """
        # Normalisation basique optimisée performance
        normalized = query.lower().strip()
        
        # Remplacement patterns financiers communs
        replacements = {
            "€": "euros",
            "cb": "carte bancaire", 
            "virt": "virement",
            "resto": "restaurant",
            "comptes": "compte"
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def _hash_query(self, query: str, user_id: int) -> str:
        """
        Génération hash deterministe pour cache
        """
        content = f"{user_id}:{query}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Métriques de performance en temps réel
        """
        return await self.metrics.get_current_metrics()
    
    async def clear_user_cache(self, user_id: int) -> bool:
        """
        Nettoyage cache utilisateur spécifique
        """
        return await self.cache_manager.clear_user_cache(user_id)
    
    async def warm_up_cache(self) -> bool:
        """
        Préchauffage cache avec patterns fréquents
        """
        try:
            # Précharge patterns financiers les plus communs
            await self.pattern_matcher.preload_frequent_patterns()
            
            # Précharge embeddings intentions principales
            await self.lightweight_classifier.preload_intent_embeddings()
            
            return True
        except Exception as e:
            self.metrics.record_warmup_error(str(e))
            return False