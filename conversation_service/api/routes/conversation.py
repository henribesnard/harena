"""
Routes API pour conversation service - Phase 1 avec JSON Output forcé
"""
import logging
import time
from typing import Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.models.responses.conversation_responses import ConversationResponse, AgentMetrics
from conversation_service.agents.financial.intent_classifier import IntentClassifierAgent
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
    cache_manager: CacheManager = Depends(get_cache_manager),
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    service_status: dict = Depends(get_conversation_service_status),
    _rate_limit: None = Depends(rate_limit_dependency)
):
    """
    Endpoint principal Phase 1 : Classification d'intention avec JSON Output forcé
    
    Features:
    - Authentification JWT obligatoire
    - Rate limiting par utilisateur
    - Classification via DeepSeek + JSON Output forcé
    - Cache sémantique Redis
    - Métriques détaillées performance
    - Validation robuste inputs/outputs
    """
    start_time = time.time()
    request_id = f"{validated_user_id}_{int(start_time)}"
    
    # Logging début requête avec contexte
    logger.info(f"[{request_id}] Nouvelle requête conversation user {validated_user_id}")
    
    try:
        # Validation et nettoyage message utilisateur
        message_validation = validate_user_message(request_data.message)
        if not message_validation["valid"]:
            metrics_collector.increment_counter("conversation.errors.validation")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Message invalide",
                    "errors": message_validation["errors"],
                    "warnings": message_validation.get("warnings", [])
                }
            )
        
        # Log warnings validation pour monitoring
        for warning in message_validation.get("warnings", []):
            logger.warning(f"[{request_id}] Validation warning: {warning}")
        
        # Nettoyage sécurisé message
        clean_message = sanitize_user_input(request_data.message)
        if not clean_message:
            metrics_collector.increment_counter("conversation.errors.validation")
            raise HTTPException(
                status_code=400,
                detail="Message vide après nettoyage sécurisé"
            )
        
        # Initialisation agent intent classifier
        intent_classifier = IntentClassifierAgent(
            deepseek_client=deepseek_client,
            cache_manager=cache_manager
        )
        
        # Classification intention avec JSON Output forcé
        logger.info(f"[{request_id}] Classification intention: '{clean_message[:50]}...'")
        
        classification_start = time.time()
        classification_result = await intent_classifier.classify_intent(
            user_message=clean_message,
            user_context=user_context
        )
        classification_time = int((time.time() - classification_start) * 1000)

        # Validation résultat classification
        if classification_result.intent_type == HarenaIntentType.ERROR:
            logger.error(f"[{request_id}] Classification échouée - erreur technique")
            metrics_collector.increment_counter("conversation.errors.classification")
            raise HTTPException(
                status_code=500, 
                detail="Erreur technique lors de la classification d'intention"
            )

        # Calcul temps traitement total
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Construction métriques agent avec données réelles
        agent_metrics = AgentMetrics(
            agent_used="intent_classifier",
            cache_hit=classification_time < 100,  # Heuristique cache hit
            model_used=getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
            tokens_consumed=await _estimate_tokens_consumption(clean_message, classification_result),
            confidence_threshold_met=classification_result.confidence >= getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
        )
        
        # Construction réponse Phase 1 avec toutes les données
        response = ConversationResponse(
            user_id=validated_user_id,
            message=clean_message,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
            intent=classification_result,
            agent_metrics=agent_metrics
        )
        
        # Collecte métriques détaillées
        await _collect_comprehensive_metrics(
            request_id, classification_result, processing_time_ms, agent_metrics
        )
        
        # Log succès avec détails
        logger.info(
            f"[{request_id}] Classification réussie: {classification_result.intent_type.value} "
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
        
        # Logging erreur avec contexte complet
        logger.error(
            f"[{request_id}] Erreur technique: {str(e)}, "
            f"User: {validated_user_id}, Message: '{request_data.message[:50]}...', "
            f"Time: {processing_time_ms}ms",
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

@router.get("/conversation/health")
async def conversation_health_detailed():
    """Health check spécifique conversation service avec métriques détaillées"""
    try:
        health_metrics = metrics_collector.get_health_metrics()
        
        return {
            "service": "conversation_service", 
            "phase": 1,
            "status": health_metrics["status"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
                "rate_limiting": True
            },
            "configuration": {
                "min_confidence_threshold": getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5),
                "max_message_length": getattr(settings, 'MAX_MESSAGE_LENGTH', 1000),
                "cache_ttl": getattr(settings, 'CACHE_TTL_INTENT', 300),
                "deepseek_model": getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat')
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur health check: {str(e)}")
        return {
            "service": "conversation_service",
            "phase": 1,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/conversation/metrics")
async def conversation_metrics_detailed():
    """Métriques détaillées pour monitoring Prometheus compatible"""
    try:
        all_metrics = metrics_collector.get_all_metrics()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_info": {
                "name": "conversation_service",
                "version": "1.0.0",
                "phase": 1,
                "features": ["intent_classification", "json_output", "cache", "auth"]
            },
            "metrics": all_metrics,
            "performance_summary": {
                "avg_response_time": all_metrics.get("histograms", {}).get("conversation.processing_time", {}).get("avg", 0),
                "p95_response_time": all_metrics.get("histograms", {}).get("conversation.processing_time", {}).get("p95", 0),
                "requests_per_second": all_metrics.get("rates", {}).get("conversation.requests_per_second", 0),
                "error_rate": _calculate_error_rate(all_metrics),
                "cache_hit_rate": _calculate_cache_hit_rate(all_metrics)
            },
            "intent_distribution": _calculate_intent_distribution(all_metrics)
        }
        
    except Exception as e:
        logger.error(f"Erreur export métriques: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur récupération métriques",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/conversation/status")
async def conversation_status():
    """Statut global service pour monitoring externe"""
    try:
        health_metrics = metrics_collector.get_health_metrics()
        
        return {
            "status": health_metrics["status"],
            "uptime_seconds": health_metrics["uptime_seconds"],
            "version": "1.0.0",
            "phase": 1,
            "ready": health_metrics["status"] == "healthy"
        }
        
    except Exception as e:
        logger.error(f"Erreur status check: {str(e)}")
        return {
            "status": "error",
            "ready": False,
            "error": str(e)
        }

async def _estimate_tokens_consumption(
    user_message: str, 
    classification_result
) -> int:
    """Estimation tokens consommés avec facteurs réalistes"""
    try:
        # Estimation basée sur longueur réelle
        input_tokens = len(user_message.split()) * 1.3  # Facteur français
        system_prompt_tokens = 200  # Prompt système estimé
        few_shot_examples_tokens = 150  # Exemples few-shot
        output_tokens = len(classification_result.reasoning.split()) * 1.3
        
        # Tokens JSON structure
        json_structure_tokens = 20
        
        total_estimated = int(
            input_tokens + system_prompt_tokens + 
            few_shot_examples_tokens + output_tokens + 
            json_structure_tokens
        )
        
        return max(50, total_estimated)  # Minimum réaliste
        
    except Exception as e:
        logger.debug(f"Erreur estimation tokens: {str(e)}")
        return 200  # Fallback conservateur

async def _collect_comprehensive_metrics(
    request_id: str,
    classification_result,
    processing_time_ms: int,
    agent_metrics: AgentMetrics
) -> None:
    """Collection centralisée métriques avec labels détaillés"""
    try:
        # Métriques de base
        metrics_collector.increment_counter("conversation.requests.total")
        metrics_collector.record_histogram("conversation.processing_time", processing_time_ms)
        metrics_collector.record_rate("conversation.requests")
        
        # Métriques par intention avec détail
        intent_type = classification_result.intent_type.value
        metrics_collector.increment_counter(f"conversation.intent.{intent_type}")
        metrics_collector.increment_counter(f"conversation.intent.category.{classification_result.category}")
        
        # Métriques qualité fine
        metrics_collector.record_gauge("conversation.intent.confidence", classification_result.confidence)
        
        if not classification_result.is_supported:
            metrics_collector.increment_counter("conversation.intent.unsupported")
            metrics_collector.increment_counter(f"conversation.intent.unsupported.{intent_type}")
        
        # Métriques seuil confidence
        confidence_threshold = getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
        if classification_result.confidence < confidence_threshold:
            metrics_collector.increment_counter("conversation.intent.low_confidence")
        elif classification_result.confidence > 0.9:
            metrics_collector.increment_counter("conversation.intent.high_confidence")
        
        # Métriques cache détaillées
        if agent_metrics.cache_hit:
            metrics_collector.increment_counter("conversation.cache.hits")
            metrics_collector.record_histogram("conversation.cache.hit_time", processing_time_ms)
        else:
            metrics_collector.increment_counter("conversation.cache.misses")
        
        # Métriques tokens et coûts
        metrics_collector.record_histogram("conversation.tokens.consumed", agent_metrics.tokens_consumed)
        
        # Métriques alternatives
        if classification_result.alternatives:
            metrics_collector.increment_counter("conversation.alternatives.provided")
            metrics_collector.record_gauge("conversation.alternatives.count", len(classification_result.alternatives))
        
        # Métriques performance par tranche
        if processing_time_ms < 100:
            metrics_collector.increment_counter("conversation.performance.fast")
        elif processing_time_ms < 500:
            metrics_collector.increment_counter("conversation.performance.normal")
        else:
            metrics_collector.increment_counter("conversation.performance.slow")
        
        logger.debug(f"[{request_id}] Métriques collectées avec succès")
        
    except Exception as e:
        logger.error(f"[{request_id}] Erreur collection métriques: {str(e)}")

def _get_health_status_description(status: str) -> str:
    """Description détaillée du statut de santé"""
    descriptions = {
        "healthy": "Service opérationnel, performances normales",
        "degraded": "Service opérationnel mais performances réduites",
        "unhealthy": "Service en difficulté, performances critiques",
        "unknown": "Statut indéterminable"
    }
    return descriptions.get(status, "Statut inconnu")

def _calculate_error_rate(metrics: Dict[str, Any]) -> float:
    """Calcul taux d'erreur global"""
    try:
        counters = metrics.get("counters", {})
        total_requests = counters.get("conversation.requests.total", 0)
        total_errors = (
            counters.get("conversation.errors.technical", 0) +
            counters.get("conversation.errors.auth", 0) +
            counters.get("conversation.errors.validation", 0) +
            counters.get("conversation.errors.classification", 0)
        )
        
        return (total_errors / max(total_requests, 1)) * 100
        
    except Exception:
        return 0.0

def _calculate_cache_hit_rate(metrics: Dict[str, Any]) -> float:
    """Calcul taux de hit cache"""
    try:
        counters = metrics.get("counters", {})
        cache_hits = counters.get("conversation.cache.hits", 0)
        cache_misses = counters.get("conversation.cache.misses", 0)
        total_cache_operations = cache_hits + cache_misses
        
        return (cache_hits / max(total_cache_operations, 1)) * 100
        
    except Exception:
        return 0.0

def _calculate_intent_distribution(metrics: Dict[str, Any]) -> Dict[str, int]:
    """Distribution des intentions classifiées"""
    try:
        counters = metrics.get("counters", {})
        intent_distribution = {}
        
        for key, value in counters.items():
            if key.startswith("conversation.intent.") and not key.startswith("conversation.intent.category"):
                intent_name = key.replace("conversation.intent.", "")
                if intent_name not in ["unsupported", "low_confidence", "high_confidence"]:
                    intent_distribution[intent_name] = value
        
        return dict(sorted(intent_distribution.items(), key=lambda x: x[1], reverse=True))
        
    except Exception:
        return {}

# Routes additionnelles pour debugging (uniquement en développement)
if getattr(settings, 'ENVIRONMENT', 'production') != "production":
    
    @router.get("/conversation/debug/cache-stats")
    async def debug_cache_stats(
        cache_manager: CacheManager = Depends(get_cache_manager)
    ):
        """Stats cache détaillées pour debugging"""
        return await cache_manager.get_cache_stats()
    
    @router.post("/conversation/debug/clear-cache")
    async def debug_clear_cache(
        cache_manager: CacheManager = Depends(get_cache_manager)
    ):
        """Nettoyage cache pour debugging"""
        success = await cache_manager.clear_all_cache()
        return {
            "cache_cleared": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @router.get("/conversation/debug/agent-metrics")
    async def debug_agent_metrics():
        """Métriques agents détaillées pour debugging"""
        return {
            "global_metrics": metrics_collector.get_all_metrics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @router.get("/conversation/debug/test-classification/{text}")
    async def debug_test_classification(
        text: str,
        deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
        cache_manager: CacheManager = Depends(get_cache_manager)
    ):
        """Test direct classification pour debugging"""
        try:
            intent_classifier = IntentClassifierAgent(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            result = await intent_classifier.classify_intent(text)
            
            return {
                "input": text,
                "result": result.dict(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "input": text,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }