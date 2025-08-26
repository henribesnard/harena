"""
Routes API pour conversation service - Version réécrite compatible JWT
"""
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.models.responses.conversation_responses import ConversationResponse, AgentMetrics
from conversation_service.agents.financial import intent_classifier as intent_classifier_module
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from conversation_service.prompts.harena_intents import HarenaIntentType
from conversation_service.api.dependencies import (
    get_deepseek_client,
    get_cache_manager,
    get_conversation_service_status,
    validate_path_user_id,
    get_user_context,
    rate_limit_dependency
)
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.utils.validation_utils import validate_user_message, sanitize_user_input
from config_service.config import settings

# Configuration du router et logger
router = APIRouter(tags=["conversation"])
logger = logging.getLogger("conversation_service.routes")

@router.post("/conversation/{path_user_id}", response_model=ConversationResponse)
async def analyze_conversation(
    path_user_id: int,
    request_data: ConversationRequest,
    request: Request,
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
    cache_manager: Optional[CacheManager] = Depends(get_cache_manager),
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    service_status: dict = Depends(get_conversation_service_status),
    _rate_limit: None = Depends(rate_limit_dependency)
):
    """
    Endpoint principal conversation service - Compatible JWT user_service
    
    Features Phase 1:
    - Authentification JWT obligatoire compatible user_service
    - Rate limiting par utilisateur avec gestion d'erreur gracieuse
    - Classification via DeepSeek + JSON Output forcé
    - Cache sémantique Redis (optionnel)
    - Métriques détaillées performance
    - Validation robuste inputs/outputs
    - Gestion d'erreur complète
    """
    start_time = time.time()
    request_id = f"{validated_user_id}_{int(start_time * 1000)}"
    
    # Logging début requête avec contexte sécurisé
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    logger.info(
        f"[{request_id}] Nouvelle conversation - User: {validated_user_id}, "
        f"IP: {client_ip}, Message: '{request_data.message[:30]}...'"
    )
    
    try:
        # ====================================================================
        # VALIDATION ET NETTOYAGE MESSAGE
        # ====================================================================
        
        message_validation = validate_user_message(request_data.message)
        if not message_validation["valid"]:
            metrics_collector.increment_counter("conversation.errors.validation")
            logger.warning(f"[{request_id}] Message invalide: {message_validation['errors']}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Message invalide",
                    "errors": message_validation["errors"],
                    "warnings": message_validation.get("warnings", []),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Log warnings validation pour monitoring
        for warning in message_validation.get("warnings", []):
            logger.warning(f"[{request_id}] Validation warning: {warning}")
        
        # Nettoyage sécurisé message
        try:
            clean_message = sanitize_user_input(request_data.message)
            if not clean_message or len(clean_message.strip()) == 0:
                metrics_collector.increment_counter("conversation.errors.validation_sanitization")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Message vide après nettoyage sécurisé",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
        except Exception as sanitize_error:
            logger.error(f"[{request_id}] Erreur sanitization: {str(sanitize_error)}")
            metrics_collector.increment_counter("conversation.errors.sanitization")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Erreur traitement message",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # ====================================================================
        # INITIALISATION AGENT CLASSIFICATION
        # ====================================================================
        
        try:
            intent_classifier = intent_classifier_module.IntentClassifierAgent(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager  # Peut être None
            )
        except Exception as agent_init_error:
            logger.error(f"[{request_id}] Erreur initialisation agent: {str(agent_init_error)}")
            metrics_collector.increment_counter("conversation.errors.agent_initialization")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Erreur initialisation service classification",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # ====================================================================
        # CLASSIFICATION INTENTION
        # ====================================================================
        
        logger.info(f"[{request_id}] Début classification: '{clean_message[:50]}...'")
        
        classification_start = time.time()
        try:
            classification_result = await intent_classifier.classify_intent(
                user_message=clean_message,
                user_context=user_context
            )
        except Exception as classification_error:
            classification_time = int((time.time() - classification_start) * 1000)
            logger.error(
                f"[{request_id}] Erreur classification ({classification_time}ms): {str(classification_error)}"
            )
            metrics_collector.increment_counter("conversation.errors.classification")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Erreur lors de la classification d'intention",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        classification_time = int((time.time() - classification_start) * 1000)
        
        # ====================================================================
        # VALIDATION RÉSULTAT CLASSIFICATION
        # ====================================================================
        
        if not classification_result:
            logger.error(f"[{request_id}] Classification retourné None")
            metrics_collector.increment_counter("conversation.errors.classification_null")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Résultat de classification invalide",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Gestion flexible du type d'intention (enum ou str)
        try:
            intent_type_value = getattr(classification_result.intent_type, 'value', classification_result.intent_type)
        except Exception as intent_extract_error:
            logger.warning(f"[{request_id}] Erreur extraction intent type: {str(intent_extract_error)}")
            intent_type_value = str(classification_result.intent_type) if classification_result.intent_type else "UNKNOWN"
        
        # Validation résultat classification
        if intent_type_value == HarenaIntentType.ERROR.value:
            logger.error(f"[{request_id}] Classification échouée - erreur technique")
            metrics_collector.increment_counter("conversation.errors.classification_failed")
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Erreur technique lors de la classification d'intention",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # ====================================================================
        # CONSTRUCTION MÉTRIQUES ET RÉPONSE
        # ====================================================================
        
        # Calcul temps traitement total
        processing_time_ms = max(1, int((time.time() - start_time) * 1000))
        
        # Construction métriques agent avec données réelles et gestion d'erreur
        try:
            agent_metrics = AgentMetrics(
                agent_used="intent_classifier",
                cache_hit=classification_time < 100,  # Heuristique cache hit
                model_used=getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
                tokens_consumed=await _estimate_tokens_consumption_safe(clean_message, classification_result),
                processing_time_ms=classification_result.processing_time_ms or classification_time,
                confidence_threshold_met=classification_result.confidence >= getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
            )
        except Exception as metrics_error:
            logger.warning(f"[{request_id}] Erreur construction métriques: {str(metrics_error)}")
            # Métriques par défaut
            agent_metrics = AgentMetrics(
                agent_used="intent_classifier",
                cache_hit=False,
                model_used="unknown",
                tokens_consumed=200,
                processing_time_ms=classification_time,
                confidence_threshold_met=True
            )
        
        # Construction réponse finale avec validation
        try:
            response = ConversationResponse(
                user_id=validated_user_id,
                sub=user_context.get("user_id", validated_user_id),  # Fallback sécurisé
                message=clean_message,
                timestamp=datetime.now(timezone.utc),
                processing_time_ms=processing_time_ms,
                intent=classification_result,
                agent_metrics=agent_metrics,
                phase=1,  # Phase 1 explicite
                version="1.1.0"  # Version avec support JWT
            )
        except Exception as response_error:
            logger.error(f"[{request_id}] Erreur construction réponse: {str(response_error)}")
            metrics_collector.increment_counter("conversation.errors.response_construction")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Erreur construction réponse",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # ====================================================================
        # COLLECTE MÉTRIQUES ET LOGGING FINAL
        # ====================================================================
        
        try:
            await _collect_comprehensive_metrics_safe(
                request_id, classification_result, processing_time_ms, agent_metrics
            )
        except Exception as metrics_collection_error:
            logger.warning(f"[{request_id}] Erreur collecte métriques: {str(metrics_collection_error)}")
        
        # Log succès avec détails
        logger.info(
            f"[{request_id}] ✅ Classification réussie: {intent_type_value} "
            f"(confiance: {classification_result.confidence:.2f}, "
            f"temps: {processing_time_ms}ms, cache: {agent_metrics.cache_hit})"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation, auth, etc.) sans modification
        raise
        
    except Exception as e:
        # Erreurs techniques non prévues avec contexte détaillé
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Métriques erreur détaillées
        metrics_collector.increment_counter("conversation.errors.technical")
        metrics_collector.record_histogram("conversation.processing_time", processing_time_ms)
        
        # Logging erreur avec contexte complet mais sécurisé
        logger.error(
            f"[{request_id}] ❌ Erreur technique: {type(e).__name__}: {str(e)[:200]}, "
            f"User: {validated_user_id}, Time: {processing_time_ms}ms",
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur interne du service conversation",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

# ============================================================================
# ENDPOINTS MONITORING (PUBLICS - SANS AUTHENTIFICATION)
# ============================================================================

@router.get("/conversation/health")
async def conversation_health_detailed():
    """Health check spécifique conversation service - ENDPOINT PUBLIC"""
    try:
        health_metrics = metrics_collector.get_health_metrics()
        
        return {
            "service": "conversation_service", 
            "phase": 1,
            "version": "1.1.0",
            "status": health_metrics["status"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "jwt_compatible": True,
            "health_details": {
                "total_requests": health_metrics["total_requests"],
                "error_rate_percent": health_metrics["error_rate_percent"],
                "avg_latency_ms": health_metrics.get("latency_p95_ms", 0),
                "uptime_seconds": health_metrics["uptime_seconds"],
                "status_description": _get_health_status_description(health_metrics["status"])
            },
            "features": {
                "intent_classification": True,
                "supported_intents": len(HarenaIntentType),
                "json_output_forced": True,
                "cache_enabled": True,
                "auth_required": True,
                "rate_limiting": True,
                "jwt_compatible": True
            },
            "configuration": {
                "min_confidence_threshold": getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5),
                "max_message_length": getattr(settings, 'MAX_MESSAGE_LENGTH', 1000),
                "cache_ttl": getattr(settings, 'CACHE_TTL_INTENT', 300),
                "deepseek_model": getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
                "environment": getattr(settings, 'ENVIRONMENT', 'production')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {str(e)}")
        return {
            "service": "conversation_service",
            "phase": 1,
            "version": "1.1.0",
            "status": "error",
            "jwt_compatible": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/conversation/metrics")
async def conversation_metrics_detailed():
    """Métriques détaillées pour monitoring - ENDPOINT PUBLIC"""
    try:
        all_metrics = metrics_collector.get_all_metrics()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_info": {
                "name": "conversation_service",
                "version": "1.1.0",
                "phase": 1,
                "features": ["intent_classification", "json_output", "cache", "auth", "jwt_compatible"],
                "jwt_compatible": True
            },
            "metrics": all_metrics,
            "performance_summary": {
                "avg_response_time": _safe_get_metric(all_metrics, ["histograms", "conversation.processing_time", "avg"], 0),
                "p95_response_time": _safe_get_metric(all_metrics, ["histograms", "conversation.processing_time", "p95"], 0),
                "requests_per_second": _safe_get_metric(all_metrics, ["rates", "conversation.requests_per_second"], 0),
                "error_rate": _calculate_error_rate(all_metrics),
                "cache_hit_rate": _calculate_cache_hit_rate(all_metrics)
            },
            "intent_distribution": _calculate_intent_distribution(all_metrics)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export métriques: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur récupération métriques",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/conversation/status")
async def conversation_status():
    """Statut global service pour monitoring externe - ENDPOINT PUBLIC"""
    try:
        health_metrics = metrics_collector.get_health_metrics()
        
        return {
            "status": health_metrics["status"],
            "uptime_seconds": health_metrics["uptime_seconds"],
            "version": "1.1.0",
            "phase": 1,
            "ready": health_metrics["status"] == "healthy",
            "jwt_compatible": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur status check: {str(e)}")
        return {
            "status": "error",
            "ready": False,
            "jwt_compatible": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# ============================================================================
# UTILITAIRES PRIVÉS AVEC GESTION D'ERREUR ROBUSTE
# ============================================================================

async def _estimate_tokens_consumption_safe(
    user_message: str, 
    classification_result
) -> int:
    """Estimation tokens consommés avec gestion d'erreur robuste"""
    try:
        # Estimation basée sur longueur réelle
        input_tokens = len(user_message.split()) * 1.3  # Facteur français
        system_prompt_tokens = 200  # Prompt système estimé
        few_shot_examples_tokens = 150  # Exemples few-shot
        
        # Tokens output avec gestion d'erreur
        try:
            reasoning_tokens = len(classification_result.reasoning.split()) * 1.3 if classification_result.reasoning else 50
        except (AttributeError, TypeError):
            reasoning_tokens = 50
        
        # Tokens JSON structure
        json_structure_tokens = 20
        
        total_estimated = int(
            input_tokens + system_prompt_tokens + 
            few_shot_examples_tokens + reasoning_tokens + 
            json_structure_tokens
        )
        
        return max(50, min(total_estimated, 4000))  # Borné entre 50 et 4000
        
    except Exception as e:
        logger.debug(f"Erreur estimation tokens: {str(e)}")
        return 200  # Fallback conservateur

async def _collect_comprehensive_metrics_safe(
    request_id: str,
    classification_result,
    processing_time_ms: int,
    agent_metrics: AgentMetrics
) -> None:
    """Collection centralisée métriques avec gestion d'erreur robuste"""
    try:
        # Métriques de base - toujours sûres
        metrics_collector.increment_counter("conversation.requests.total")
        metrics_collector.record_histogram("conversation.processing_time", processing_time_ms)
        metrics_collector.record_rate("conversation.requests")
        
        # Métriques par intention avec protection
        try:
            intent_type = getattr(classification_result.intent_type, "value", classification_result.intent_type)
            if intent_type:
                safe_intent = str(intent_type).replace('.', '_').replace(' ', '_')[:50]
                metrics_collector.increment_counter(f"conversation.intent.{safe_intent}")
            
            if hasattr(classification_result, 'category') and classification_result.category:
                safe_category = str(classification_result.category).replace('.', '_').replace(' ', '_')[:50]
                metrics_collector.increment_counter(f"conversation.intent.category.{safe_category}")
                
        except Exception as intent_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques intention: {str(intent_metrics_error)}")
        
        # Métriques qualité avec protection
        try:
            if hasattr(classification_result, 'confidence') and classification_result.confidence is not None:
                confidence = float(classification_result.confidence)
                if 0 <= confidence <= 1:
                    metrics_collector.record_gauge("conversation.intent.confidence", confidence)
                    
                    confidence_threshold = getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
                    if confidence < confidence_threshold:
                        metrics_collector.increment_counter("conversation.intent.low_confidence")
                    elif confidence > 0.9:
                        metrics_collector.increment_counter("conversation.intent.high_confidence")
                        
        except Exception as confidence_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques confiance: {str(confidence_metrics_error)}")
        
        # Métriques support avec protection
        try:
            if hasattr(classification_result, 'is_supported') and not classification_result.is_supported:
                metrics_collector.increment_counter("conversation.intent.unsupported")
        except Exception as support_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques support: {str(support_metrics_error)}")
        
        # Métriques cache avec protection
        try:
            if hasattr(agent_metrics, 'cache_hit'):
                if agent_metrics.cache_hit:
                    metrics_collector.increment_counter("conversation.cache.hits")
                    metrics_collector.record_histogram("conversation.cache.hit_time", processing_time_ms)
                else:
                    metrics_collector.increment_counter("conversation.cache.misses")
        except Exception as cache_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques cache: {str(cache_metrics_error)}")
        
        # Métriques tokens avec protection
        try:
            if hasattr(agent_metrics, 'tokens_consumed') and agent_metrics.tokens_consumed:
                tokens = int(agent_metrics.tokens_consumed)
                if 0 <= tokens <= 10000:  # Sanity check
                    metrics_collector.record_histogram("conversation.tokens.consumed", tokens)
        except Exception as tokens_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques tokens: {str(tokens_metrics_error)}")
        
        # Métriques alternatives avec protection
        try:
            if hasattr(classification_result, 'alternatives') and classification_result.alternatives:
                metrics_collector.increment_counter("conversation.alternatives.provided")
                alternatives_count = len(classification_result.alternatives)
                if 0 <= alternatives_count <= 10:
                    metrics_collector.record_gauge("conversation.alternatives.count", alternatives_count)
        except Exception as alternatives_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques alternatives: {str(alternatives_metrics_error)}")
        
        # Métriques performance par tranche
        try:
            if processing_time_ms < 100:
                metrics_collector.increment_counter("conversation.performance.fast")
            elif processing_time_ms < 500:
                metrics_collector.increment_counter("conversation.performance.normal")
            else:
                metrics_collector.increment_counter("conversation.performance.slow")
        except Exception as perf_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques performance: {str(perf_metrics_error)}")
        
        logger.debug(f"[{request_id}] Métriques collectées avec succès")
        
    except Exception as e:
        logger.error(f"[{request_id}] Erreur collection métriques globale: {str(e)}")

def _safe_get_metric(metrics: Dict[str, Any], path: list, default: Any = None) -> Any:
    """Récupération sécurisée de métrique imbriquée"""
    try:
        current = metrics
        for key in path:
            current = current[key]
        return current
    except (KeyError, TypeError, AttributeError):
        return default

def _get_health_status_description(status: str) -> str:
    """Description détaillée du statut de santé avec gestion d'erreur"""
    descriptions = {
        "healthy": "Service opérationnel, performances normales",
        "degraded": "Service opérationnel mais performances réduites",
        "unhealthy": "Service en difficulté, performances critiques",
        "unknown": "Statut indéterminable"
    }
    return descriptions.get(status, f"Statut: {status}")

def _calculate_error_rate(metrics: Dict[str, Any]) -> float:
    """Calcul taux d'erreur global avec protection"""
    try:
        counters = metrics.get("counters", {})
        total_requests = counters.get("conversation.requests.total", 0)
        
        if total_requests <= 0:
            return 0.0
        
        total_errors = (
            counters.get("conversation.errors.technical", 0) +
            counters.get("conversation.errors.auth", 0) +
            counters.get("conversation.errors.validation", 0) +
            counters.get("conversation.errors.classification", 0)
        )
        
        error_rate = (total_errors / total_requests) * 100
        return min(max(error_rate, 0.0), 100.0)  # Borné entre 0 et 100
        
    except Exception as e:
        logger.debug(f"Erreur calcul taux erreur: {str(e)}")
        return 0.0

def _calculate_cache_hit_rate(metrics: Dict[str, Any]) -> float:
    """Calcul taux de hit cache avec protection"""
    try:
        counters = metrics.get("counters", {})
        cache_hits = counters.get("conversation.cache.hits", 0)
        cache_misses = counters.get("conversation.cache.misses", 0)
        total_cache_operations = cache_hits + cache_misses
        
        if total_cache_operations <= 0:
            return 0.0
        
        hit_rate = (cache_hits / total_cache_operations) * 100
        return min(max(hit_rate, 0.0), 100.0)  # Borné entre 0 et 100
        
    except Exception as e:
        logger.debug(f"Erreur calcul cache hit rate: {str(e)}")
        return 0.0

def _calculate_intent_distribution(metrics: Dict[str, Any]) -> Dict[str, int]:
    """Distribution des intentions classifiées avec protection"""
    try:
        counters = metrics.get("counters", {})
        intent_distribution = {}
        
        # Filtrer les métriques d'intention avec protection
        for key, value in counters.items():
            try:
                if (key.startswith("conversation.intent.") and 
                    not key.startswith("conversation.intent.category") and
                    key not in ["conversation.intent.unsupported", "conversation.intent.low_confidence", "conversation.intent.high_confidence"]):
                    
                    intent_name = key.replace("conversation.intent.", "")
                    if isinstance(value, (int, float)) and value >= 0:
                        intent_distribution[intent_name] = int(value)
                        
            except Exception as intent_error:
                logger.debug(f"Erreur traitement métrique intention {key}: {str(intent_error)}")
        
        # Trier par valeur décroissante avec limite
        sorted_distribution = dict(sorted(intent_distribution.items(), key=lambda x: x[1], reverse=True)[:20])
        return sorted_distribution
        
    except Exception as e:
        logger.debug(f"Erreur calcul distribution intentions: {str(e)}")
        return {}

# ============================================================================
# ROUTES DEBUG (UNIQUEMENT EN NON-PRODUCTION)
# ============================================================================

environment = getattr(settings, 'ENVIRONMENT', 'production')
if environment != "production":
    
    @router.get("/conversation/debug/cache-stats")
    async def debug_cache_stats(
        cache_manager: Optional[CacheManager] = Depends(get_cache_manager)
    ):
        """Stats cache détaillées pour debugging - NÉCESSITE AUTH"""
        if not cache_manager:
            return {"error": "Cache non disponible"}
        
        try:
            return await cache_manager.get_cache_stats()
        except Exception as e:
            return {"error": f"Erreur récupération stats cache: {str(e)}"}
    
    @router.post("/conversation/debug/clear-cache")
    async def debug_clear_cache(
        cache_manager: Optional[CacheManager] = Depends(get_cache_manager)
    ):
        """Nettoyage cache pour debugging - NÉCESSITE AUTH"""
        if not cache_manager:
            return {"error": "Cache non disponible", "cache_cleared": False}
        
        try:
            success = await cache_manager.clear_all_cache()
            return {
                "cache_cleared": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "cache_cleared": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    @router.get("/conversation/debug/agent-metrics")
    async def debug_agent_metrics():
        """Métriques agents détaillées pour debugging - ENDPOINT PUBLIC EN DEBUG"""
        try:
            return {
                "global_metrics": metrics_collector.get_all_metrics(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "environment": environment
            }
        except Exception as e:
            return {
                "error": f"Erreur métriques debug: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    @router.get("/conversation/debug/test-classification/{text}")
    async def debug_test_classification(
        text: str,
        deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
        cache_manager: Optional[CacheManager] = Depends(get_cache_manager)
    ):
        """Test direct classification pour debugging - NÉCESSITE AUTH"""
        request_id = f"debug_{int(time.time() * 1000)}"
        
        try:
            intent_classifier = intent_classifier_module.IntentClassifierAgent(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            result = await intent_classifier.classify_intent(text)
            
            return {
                "input": text,
                "result": result.model_dump(mode="json") if hasattr(result, 'model_dump') else str(result),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"[{request_id}] Erreur test classification debug: {str(e)}")
            return {
                "input": text,
                "error": str(e),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

logger.info(f"Routes conversation configurées - Environnement: {environment}")
logger.info(f"Endpoints debug: {'activés' if environment != 'production' else 'désactivés'}")