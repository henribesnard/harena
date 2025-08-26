"""
Routes API pour conversation service - Phase 1
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
from conversation_service.utils.validation_utils import validate_user_message
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
    Endpoint principal Phase 1 : Classification d'intention uniquement
    
    - Authentification JWT obligatoire
    - Rate limiting par utilisateur
    - Classification via DeepSeek + few-shots
    - Cache sémantique Redis
    - Métriques détaillées
    """
    start_time = time.time()
    request_id = f"{validated_user_id}_{int(start_time)}"
    
    # Logging début requête
    logger.info(f"[{request_id}] Nouvelle requête conversation user {validated_user_id}")
    
    try:
        # Validation message utilisateur
        message_validation = validate_user_message(request_data.message)
        if not message_validation["valid"]:
            metrics_collector.increment_counter("conversation.errors.validation")
            raise HTTPException(
                status_code=400,
                detail=f"Message invalide: {', '.join(message_validation['errors'])}"
            )
        
        # Log warnings validation
        for warning in message_validation.get("warnings", []):
            logger.warning(f"[{request_id}] Message validation warning: {warning}")
        
        # Initialisation agent intent classifier
        intent_classifier = IntentClassifierAgent(
            deepseek_client=deepseek_client,
            cache_manager=cache_manager
        )
        
        # Classification intention
        logger.info(f"[{request_id}] Classification intention pour: '{request_data.message[:50]}...'")
        
        classification_result = await intent_classifier.classify_intent(
            user_message=request_data.message,
            user_context=user_context
        )

        if classification_result.intent_type == HarenaIntentType.ERROR:
            raise HTTPException(status_code=500, detail="Erreur classification intention")

        # Calcul temps traitement total
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Construction métriques agent
        agent_metrics = AgentMetrics(
            agent_used="intent_classifier",
            cache_hit=classification_result.processing_time_ms < 50,  # Heuristique cache hit
            model_used="deepseek-chat",
            tokens_consumed=await _estimate_tokens_used(request_data.message, classification_result.reasoning),
            confidence_threshold_met=classification_result.confidence >= settings.MIN_CONFIDENCE_THRESHOLD
        )
        
        # Construction réponse Phase 1
        response = ConversationResponse(
            user_id=validated_user_id,
            message=request_data.message,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
            intent=classification_result,
            agent_metrics=agent_metrics
        )
        
        # Collecte métriques
        _collect_request_metrics(classification_result, processing_time_ms, agent_metrics)
        
        # Logging succès
        logger.info(
            f"[{request_id}] Classification réussie: {classification_result.intent_type.value} "
            f"(confiance: {classification_result.confidence:.2f}, temps: {processing_time_ms}ms)"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation, auth, etc.)
        raise
        
    except Exception as e:
        # Erreurs techniques non prévues
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Métriques erreur
        metrics_collector.increment_counter("conversation.errors.technical")
        metrics_collector.record_histogram("conversation.processing_time", processing_time_ms)
        
        logger.error(f"[{request_id}] Erreur technique: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail="Erreur interne du service conversation"
        )

@router.get("/conversation/health")
async def conversation_health():
    """Health check spécifique conversation service avec métriques"""
    try:
        health_metrics = metrics_collector.get_health_metrics()
        
        return {
            "service": "conversation_service", 
            "phase": 1,
            "status": health_metrics["status"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "total_requests": health_metrics["total_requests"],
                "error_rate_percent": health_metrics["error_rate_percent"],
                "avg_latency_ms": health_metrics.get("latency_p95_ms", 0),
                "uptime_seconds": health_metrics["uptime_seconds"]
            },
            "features": {
                "intent_classification": True,
                "supported_intents": 35,
                "cache_enabled": True,
                "auth_required": True
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur health check: {str(e)}")
        return {
            "service": "conversation_service",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/conversation/metrics")
async def conversation_metrics():
    """Métriques détaillées pour monitoring (Prometheus compatible)"""
    try:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics_collector.get_all_metrics(),
            "service_info": {
                "name": "conversation_service",
                "version": "1.0.0",
                "phase": 1
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur export métriques: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur récupération métriques"
        )

async def _estimate_tokens_used(user_message: str, ai_response: str) -> int:
    """Estimation grossière tokens utilisés"""
    try:
        # Approximation : 1 token ≈ 4 caractères pour le français
        input_tokens = len(user_message) // 4
        output_tokens = len(ai_response) // 4
        system_prompt_tokens = 200  # Estimation prompt système
        
        return input_tokens + output_tokens + system_prompt_tokens
        
    except Exception:
        return 0

def _collect_request_metrics(
    classification_result,
    processing_time_ms: int,
    agent_metrics: AgentMetrics
) -> None:
    """Collection centralisée métriques requête"""
    try:
        # Métriques de base
        metrics_collector.increment_counter("conversation.requests.total")
        metrics_collector.record_histogram("conversation.processing_time", processing_time_ms)
        metrics_collector.record_rate("conversation.requests")
        
        # Métriques par intention
        intent_type = classification_result.intent_type.value
        metrics_collector.increment_counter(f"conversation.intent.{intent_type}")
        
        # Métriques qualité
        metrics_collector.record_gauge("conversation.intent.confidence", classification_result.confidence)
        
        if not classification_result.is_supported:
            metrics_collector.increment_counter("conversation.intent.unsupported")
        
        if classification_result.confidence < settings.MIN_CONFIDENCE_THRESHOLD:
            metrics_collector.increment_counter("conversation.intent.low_confidence")
        
        # Métriques cache
        if agent_metrics.cache_hit:
            metrics_collector.increment_counter("conversation.cache.hits")
        else:
            metrics_collector.increment_counter("conversation.cache.misses")
        
        # Métriques tokens
        metrics_collector.record_histogram("conversation.tokens.used", agent_metrics.tokens_consumed)
        
    except Exception as e:
        logger.error(f"Erreur collection métriques: {str(e)}")

# Routes additionnelles pour debugging (uniquement en développement)
if settings.ENVIRONMENT != "production":
    
    @router.get("/conversation/debug/cache-stats")
    async def debug_cache_stats(
        cache_manager: CacheManager = Depends(get_cache_manager)
    ):
        """Stats cache pour debugging"""
        return await cache_manager.get_cache_stats()
    
    @router.post("/conversation/debug/clear-cache")
    async def debug_clear_cache(
        cache_manager: CacheManager = Depends(get_cache_manager)
    ):
        """Nettoyage cache pour debugging"""
        success = await cache_manager.clear_all_cache()
        return {"cache_cleared": success}
    
    @router.get("/conversation/debug/agent-metrics")
    async def debug_agent_metrics():
        """Métriques agents pour debugging"""
        return metrics_collector.get_all_metrics()