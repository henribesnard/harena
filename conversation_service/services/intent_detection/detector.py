"""
🎯 Service Principal de Détection d'Intention

Orchestrateur principal reprenant la logique éprouvée du fichier original :
95% règles ultra-rapides, 5% fallback DeepSeek optionnel.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from conversation_service.models.enums import IntentType, DetectionMethod
from conversation_service.models.intent import IntentRequest, IntentResponse
from conversation_service.models.exceptions import IntentDetectionError, ServiceUnavailableError
from conversation_service.config import config
from .rule_engine import get_rule_engine
from ..entity_extraction.extractor import get_entity_extractor
from ..preprocessing.text_cleaner import get_text_cleaner
from conversation_service.clients.llm.deepseek_client import get_deepseek_client_sync

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Métriques complètes du service (reprises du fichier original)"""
    total_requests: int = 0
    rule_based_success: int = 0
    deepseek_fallback: int = 0
    avg_latency: float = 0.0
    total_cost: float = 0.0


class OptimizedIntentService:
    """
    Service optimisé règles + fallback optionnel
    
    Reprend exactement l'architecture qui fonctionne du fichier original :
    - 95% traitement par règles (ultra-rapide)
    - 5% fallback DeepSeek si nécessaire
    - Cache intelligent 
    - Métriques détaillées
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Services sous-jacents
        self.rule_engine = get_rule_engine()
        self.entity_extractor = get_entity_extractor()
        self.text_cleaner = get_text_cleaner()
        self.deepseek_client = None  # Lazy loading
        
        # Cache mémoire ultra-rapide (du fichier original)
        self.cache = {}
        self.cache_max_size = config.performance.cache_max_size
        
        # Métriques (structure identique fichier original)
        self.metrics = Metrics()
        
        self.logger.info("🚀 Service détection intention initialisé")
    
    async def initialize(self):
        """Initialisation asynchrone des composants"""
        self.logger.info("🚀 Initialisation service optimisé")
        
        # Client DeepSeek optionnel
        if config.service.enable_deepseek_fallback:
            try:
                self.deepseek_client = get_deepseek_client_sync()
                
                # Test connexion
                test_result = await self.deepseek_client.test_connection()
                if test_result["status"] == "connected":
                    self.logger.info("✅ DeepSeek fallback disponible")
                else:
                    self.logger.warning(f"⚠️ DeepSeek connexion échouée: {test_result.get('error')}")
                    
            except ServiceUnavailableError as e:
                self.logger.error(f"🚨 Service DeepSeek indisponible: {e}")
                self.deepseek_client = None
            except Exception as e:
                self.logger.warning(f"⚠️ DeepSeek fallback indisponible: {e}")
                self.deepseek_client = None
        else:
            self.logger.info("DeepSeek fallback désactivé par configuration")
    
    async def detect_intent(self, request: IntentRequest) -> IntentResponse:
        """
        Pipeline de détection optimisé (logique principale du fichier original)
        
        Args:
            request: Requête de détection d'intention
            
        Returns:
            Dict avec résultat détection complète
            
        Raises:
            IntentDetectionError: Si la détection échoue complètement
        """
        start_time = time.time()
        
        try:
            # Cache ultra-rapide (logique exacte fichier original)
            cache_key = request.query.lower().strip()
            if cache_key in self.cache and request.enable_cache:
                cached = self.cache[cache_key].copy()
                cached["processing_time_ms"] = (time.time() - start_time) * 1000
                cached["method_used"] = "cache"
                cached["cached"] = True
                return cached
            
            # Préprocessing avec gestion d'erreurs robuste
            try:
                preprocessed = self.text_cleaner.preprocess_query(request.query)
                clean_query = preprocessed["normalized_query"]
                query_hints = self.text_cleaner.extract_intent_hints(preprocessed)
            except Exception as e:
                self.logger.warning(f"Erreur préprocessing: {e}")
                clean_query = request.query.strip().lower()
                query_hints = {}
            
            # 1. RÈGLES INTELLIGENTES (ultra-rapide - 95% des cas)
            try:
                rule_intent, rule_confidence, rule_entities = self.rule_engine.detect_intent(clean_query)
            except Exception as e:
                self.logger.error(f"Erreur règles de détection: {e}")
                raise IntentDetectionError(
                    f"Échec détection par règles: {str(e)}",
                    query=request.query,
                    attempted_methods=[DetectionMethod.RULES]
                )
            
            # 2. Décision fallback (logique exacte fichier original)
            final_intent = rule_intent
            final_confidence = rule_confidence  
            final_entities = rule_entities
            method_used = DetectionMethod.RULES
            cost = 0.0
            
            # Fallback DeepSeek seulement si vraiment nécessaire
            should_use_deepseek = (
                request.use_deepseek_fallback and
                config.service.enable_deepseek_fallback and
                self.deepseek_client is not None and
                rule_confidence < config.performance.deepseek_threshold and
                rule_intent == IntentType.UNKNOWN
            )
            
            if should_use_deepseek:
                self.logger.info(f"🔄 Fallback DeepSeek (confiance règles: {rule_confidence:.3f})")
                try:
                    ds_result = await self._deepseek_fallback(request.query)
                    ds_intent, ds_confidence, ds_entities, ds_cost = ds_result
                    
                    if ds_confidence > rule_confidence:
                        final_intent = IntentType(ds_intent) if ds_intent in [e.value for e in IntentType] else IntentType.UNKNOWN
                        final_confidence = ds_confidence
                        final_entities = {**rule_entities, **ds_entities}  # Merge entities
                        method_used = DetectionMethod.LLM_FALLBACK
                        cost = ds_cost
                        self.metrics.deepseek_fallback += 1
                    else:
                        method_used = DetectionMethod.HYBRID
                        self.metrics.rule_based_success += 1
                        
                except ServiceUnavailableError as e:
                    self.logger.error(f"Service DeepSeek indisponible: {e}")
                    # Continuer avec résultat règles
                    self.metrics.rule_based_success += 1
                except Exception as e:
                    self.logger.error(f"Erreur fallback DeepSeek: {e}")
                    # Continuer avec résultat règles
                    self.metrics.rule_based_success += 1
            else:
                self.metrics.rule_based_success += 1
            
            # 3. Enrichissement entités si nécessaire
            if final_confidence > 0.5 and not final_entities:
                try:
                    entity_result = self.entity_extractor.extract_entities(
                        clean_query, 
                        intent_context=final_intent,
                        validate_entities=True
                    )
                    additional_entities = entity_result.get("entities", {})
                    final_entities.update(additional_entities)
                    
                except Exception as e:
                    self.logger.warning(f"Erreur enrichissement entités: {e}")
            
            # 4. Génération suggestions
            try:
                suggestions = self.rule_engine.get_suggestions(final_intent, final_entities)
            except Exception as e:
                self.logger.warning(f"Erreur génération suggestions: {e}")
                suggestions = []
            
            processing_time = (time.time() - start_time) * 1000
            
            # Résultat final (structure exacte fichier original)
            result = {
                "intent": final_intent.value,
                "intent_code": self.rule_engine.get_intent_to_search_code(final_intent),
                "confidence": final_confidence,
                "processing_time_ms": processing_time,
                "method_used": method_used.value,
                "query": request.query,
                "entities": final_entities,
                "suggestions": suggestions,
                "cost_estimate": cost,
                "cached": False
            }
            
            # Cache si confiance élevée (logique fichier original)
            if (final_confidence >= config.performance.cache_threshold and 
                len(self.cache) < self.cache_max_size and 
                request.enable_cache):
                cache_result = result.copy()
                cache_result.pop("processing_time_ms")
                cache_result.pop("cached")
                self.cache[cache_key] = cache_result
            
            # Métriques (exactement comme fichier original)
            self.metrics.total_requests += 1
            self.metrics.total_cost += cost
            self.metrics.avg_latency = (
                (self.metrics.avg_latency * (self.metrics.total_requests - 1) + processing_time) / 
                self.metrics.total_requests
            )
            
            # Validation et création de la réponse typée
            return IntentResponse(**result)
            
        except IntentDetectionError:
            # Re-raise les erreurs spécifiques
            raise
        except Exception as e:
            # Gestion des erreurs inattendues
            self.logger.error(f"Erreur inattendue détection intention: {e}", exc_info=True)
            raise IntentDetectionError(
                f"Erreur système lors de la détection: {str(e)}",
                query=request.query,
                attempted_methods=[DetectionMethod.RULES, DetectionMethod.LLM_FALLBACK]
            )
    
    async def _deepseek_fallback(self, query: str) -> Tuple[str, float, Dict[str, Any], float]:
        """
        Fallback DeepSeek optimisé (logique exacte fichier original)
        
        Returns:
            (intent, confidence, entities, cost)
            
        Raises:
            ServiceUnavailableError: Si le service DeepSeek est indisponible
        """
        if not self.deepseek_client:
            raise ServiceUnavailableError(
                "Client DeepSeek non initialisé",
                service_name="deepseek"
            )
        
        try:
            response = await self.deepseek_client.classify_intent(query)
            
            return (
                response.intent,
                response.confidence, 
                response.entities,
                response.cost_estimate
            )
            
        except Exception as e:
            self.logger.error(f"Erreur DeepSeek: {e}")
            # Transform generic error to specific service error
            raise ServiceUnavailableError(
                f"Service DeepSeek temporairement indisponible: {str(e)}",
                service_name="deepseek",
                retry_after=60
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Métriques complètes (structure exacte fichier original)
        """
        if self.metrics.total_requests == 0:
            return {"total_requests": 0, "service_ready": True}
        
        rule_success_rate = self.metrics.rule_based_success / self.metrics.total_requests
        deepseek_usage_rate = self.metrics.deepseek_fallback / self.metrics.total_requests
        
        return {
            "total_requests": self.metrics.total_requests,
            "avg_latency_ms": round(self.metrics.avg_latency, 2),
            "total_cost": round(self.metrics.total_cost, 4),
            "performance": {
                "rule_success_rate": round(rule_success_rate, 3),
                "deepseek_usage_rate": round(deepseek_usage_rate, 3),
                "meets_latency_target": self.metrics.avg_latency <= config.performance.target_latency_ms,
                "target_latency_ms": config.performance.target_latency_ms
            },
            "distribution": {
                "rules_success": self.metrics.rule_based_success,
                "deepseek_fallback": self.metrics.deepseek_fallback
            },
            "cache_size": len(self.cache),
            "efficiency": {
                "cost_per_request": round(
                    self.metrics.total_cost / self.metrics.total_requests, 6
                ) if self.metrics.total_requests > 0 else 0,
                "fast_responses_percent": round(rule_success_rate * 100, 1)
            },
            "component_metrics": {
                "rule_engine": self.rule_engine.get_rule_metrics(),
                "entity_extractor": self.entity_extractor.get_extraction_metrics(),
                "deepseek_client": self.deepseek_client.get_client_metrics() if self.deepseek_client else {}
            }
        }
    
    async def batch_detect_intent(
        self, 
        queries: List[str], 
        user_id: Optional[str] = None,
        use_deepseek_fallback: bool = True
    ) -> List[IntentResponse]:
        """
        Traitement batch de requêtes avec gestion d'erreurs robuste
        
        Raises:
            IntentDetectionError: Si le batch échoue complètement
        """
        if not queries:
            return []
        
        self.logger.info(f"Début batch détection: {len(queries)} requêtes")
        results = []
        errors_count = 0
        
        for query in queries:
            try:
                request = IntentRequest(
                    query=query,
                    user_id=user_id,
                    use_deepseek_fallback=use_deepseek_fallback
                )
                result = await self.detect_intent(request)
                results.append(result)
                
            except IntentDetectionError as e:
                self.logger.error(f"Erreur détection spécifique query '{query}': {e}")
                errors_count += 1
                # Créer une réponse d'erreur typée
                error_response = IntentResponse(
                    intent=IntentType.UNKNOWN,
                    intent_code="UNKNOWN",
                    confidence=0.0,
                    processing_time_ms=0.0,
                    method_used="error",
                    query=query,
                    entities={},
                    suggestions=[],
                    cost_estimate=0.0,
                    error=str(e),
                    error_type="IntentDetectionError"
                )
                results.append(error_response)
                
            except Exception as e:
                self.logger.error(f"Erreur inattendue query '{query}': {e}")
                errors_count += 1
                # Créer une réponse d'erreur système typée
                error_response = IntentResponse(
                    intent=IntentType.UNKNOWN,
                    intent_code="UNKNOWN",
                    confidence=0.0,
                    processing_time_ms=0.0,
                    method_used="error",
                    query=query,
                    entities={},
                    suggestions=[],
                    cost_estimate=0.0,
                    error=str(e),
                    error_type="SystemError"
                )
                results.append(error_response)
        
        # Si trop d'erreurs, lever une exception générale
        if errors_count > len(queries) * 0.5:  # Plus de 50% d'erreurs
            raise IntentDetectionError(
                f"Batch processing échoué: {errors_count}/{len(queries)} erreurs",
                attempted_methods=[DetectionMethod.BATCH]
            )
        
        return results
    
    def clear_cache(self):
        """Vide le cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        self.logger.info(f"Cache vidé: {cache_size} entrées supprimées")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques du cache"""
        return {
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_max_size,
            "cache_usage_percent": round(
                len(self.cache) / self.cache_max_size * 100, 1
            ) if self.cache_max_size > 0 else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification santé complète du service avec gestion d'erreurs
        """
        health_status = {
            "status": "healthy",
            "components": {
                "rule_engine": "operational",
                "entity_extractor": "operational", 
                "text_cleaner": "operational",
                "deepseek_client": "unknown",
                "cache": "operational"
            },
            "metrics": self.get_metrics(),
            "configuration": {
                "deepseek_enabled": config.service.enable_deepseek_fallback,
                "cache_enabled": True,
                "target_latency_ms": config.performance.target_latency_ms,
                "target_accuracy": config.performance.target_accuracy
            }
        }
        
        # Test DeepSeek si disponible
        if self.deepseek_client:
            try:
                test_result = await self.deepseek_client.test_connection()
                health_status["components"]["deepseek_client"] = test_result["status"]
            except ServiceUnavailableError as e:
                health_status["components"]["deepseek_client"] = f"unavailable: {e.message}"
            except Exception as e:
                health_status["components"]["deepseek_client"] = f"error: {str(e)}"
        
        # Détermination statut global
        failed_components = [
            name for name, status in health_status["components"].items() 
            if status.startswith("error") or status == "failed"
        ]
        
        if failed_components:
            if "deepseek_client" in failed_components and len(failed_components) == 1:
                health_status["status"] = "degraded"  # DeepSeek optionnel
            else:
                health_status["status"] = "unhealthy"
        
        return health_status
    
    def reset_metrics(self):
        """Remet à zéro toutes les métriques"""
        self.metrics = Metrics()
        self.rule_engine.reset_metrics()
        self.entity_extractor.reset_metrics()
        if self.deepseek_client:
            self.deepseek_client.reset_metrics()
        self.logger.info("Métriques remises à zéro")


# Instance singleton du service principal
_intent_service_instance = None

async def get_intent_service() -> OptimizedIntentService:
    """
    Factory function async pour récupérer instance service singleton
    
    Raises:
        ServiceUnavailableError: Si l'initialisation échoue
    """
    global _intent_service_instance
    if _intent_service_instance is None:
        try:
            _intent_service_instance = OptimizedIntentService()
            await _intent_service_instance.initialize()
        except Exception as e:
            logger.error(f"Erreur initialisation service: {e}")
            raise ServiceUnavailableError(
                f"Impossible d'initialiser le service de détection: {str(e)}",
                service_name="intent_detection"
            )
    return _intent_service_instance


def get_intent_service_sync() -> OptimizedIntentService:
    """
    Factory function synchrone pour récupérer instance service
    
    Raises:
        ServiceUnavailableError: Si l'instance n'est pas disponible
    """
    global _intent_service_instance
    if _intent_service_instance is None:
        try:
            _intent_service_instance = OptimizedIntentService()
        except Exception as e:
            logger.error(f"Erreur création service: {e}")
            raise ServiceUnavailableError(
                f"Impossible de créer le service de détection: {str(e)}",
                service_name="intent_detection"
            )
    return _intent_service_instance


# Exports publics
__all__ = [
    "OptimizedIntentService",
    "Metrics",
    "get_intent_service",
    "get_intent_service_sync"
]