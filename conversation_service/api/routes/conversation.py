import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.models.responses.conversation_responses import ConversationResponse, AgentMetrics
from conversation_service.models.responses.autogen_conversation_response import (
    AutogenConversationResponse,
)
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
from conversation_service.api.dependencies import get_conversation_engine
from conversation_service.models.requests.conversation_requests import (
    ConversationRequest,
)
from conversation_service.models.responses.autogen_conversation_response import (
    AutogenConversationResponse,
)
from conversation_service.models.responses.conversation_responses import (
    AgentMetrics,
    ConversationResponse,
    IntentClassificationResult,
)
from conversation_service.prompts.harena_intents import HarenaIntentType
from conversation_service.utils.metrics_collector import metrics_collector
from config_service.config import settings


router = APIRouter(tags=["conversation"])
logger = logging.getLogger("conversation_service.routes")

@router.post("/conversation/{path_user_id}")

@router.post(
    "/conversation/{user_id}",
    response_model=Union[AutogenConversationResponse, ConversationResponse],
)
async def analyze_conversation(
    user_id: int,
    request: ConversationRequest,
    request_obj: Request,
    conversation_engine: Dict[str, Any] = Depends(get_conversation_engine),
):
    """Endpoint principal de classification d'intention."""

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
            intent_classifier = intent_classifier_module.IntentClassifierAgent()
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
        
        runtime = getattr(request.app.state, "autogen_runtime", None)
        autogen_enabled = (
            os.getenv("CONVERSATION_AUTOGEN_ENABLED", "true").lower() == "true"
        )
        response_payload: Dict[str, Any] | None = None
        if runtime and autogen_enabled:
            try:
                autogen_result = await runtime.process_message(
                    clean_message, user_context
                )
                response_payload = AutogenConversationResponse(
                    user_id=validated_user_id,
                    sub=user_context.get("user_id", validated_user_id),
                    message=clean_message,
                    timestamp=datetime.now(timezone.utc),
                    processing_time_ms=processing_time_ms,
                    intent=classification_result,
                    agent_metrics=agent_metrics,
                    entities=autogen_result.get("entities", {}),
                    autogen_metadata=autogen_result.get("metadata", {}),
                    phase=2,
                    version="2.0.0",
                ).model_dump()
            except Exception as autogen_error:
                logger.warning(
                    f"[{request_id}] Échec runtime AutoGen: {autogen_error}. Retour flux historique."
                )

        if response_payload is None:
            try:
                response_payload = ConversationResponse(
                    user_id=validated_user_id,
                    sub=user_context.get("user_id", validated_user_id),  # Fallback sécurisé
                    message=clean_message,
                    timestamp=datetime.now(timezone.utc),
                    processing_time_ms=processing_time_ms,
                    intent=classification_result,
                    agent_metrics=agent_metrics,
                    phase=1,  # Phase 1 explicite
                    version="1.1.0"  # Version avec support JWT
                ).model_dump()
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

        return JSONResponse(content=jsonable_encoder(response_payload))
        
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
async def conversation_health_detailed(request: Request):
    """Health check spécifique conversation service - ENDPOINT PUBLIC"""
    try:
        health_metrics = metrics_collector.get_health_metrics()
        autogen_available = bool(getattr(request.app.state, "autogen_runtime", None))

        return {
            "service": "conversation_service",
            "phase": 1,
            "version": "1.1.0",
            "status": health_metrics["status"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "jwt_compatible": True,
            "modes": {"legacy": True, "autogen": autogen_available},
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
    runtime = conversation_engine.get("runtime")
    legacy_agent = conversation_engine.get("legacy_agent")
    mode = conversation_engine.get("mode", "legacy")

    if mode == "autogen" and runtime is not None:
        try:
            return await _run_autogen_pipeline(user_id, request, runtime, start_time)
        except Exception as exc:
            logger.exception("Autogen pipeline failed, falling back to legacy: %s", exc)
            if legacy_agent is not None:
                return await _run_legacy_pipeline(user_id, request, legacy_agent, start_time)
            raise HTTPException(status_code=500, detail="Autogen pipeline failed")

    if legacy_agent is not None:
        return await _run_legacy_pipeline(user_id, request, legacy_agent, start_time)

    raise HTTPException(status_code=500, detail="No conversation engine available")


async def _run_autogen_pipeline(
    user_id: int,
    request: ConversationRequest,
    runtime: Any,
    start_time: float,
) -> AutogenConversationResponse:
    """Exécute le pipeline AutoGen."""

    team_cls = runtime.get_team("phase2")
    team = team_cls()
    result = await team.process_user_message(request.message, user_id)

    intent_dict = result.get("intent", {})
    intent = IntentClassificationResult(
        intent_type=HarenaIntentType(
            intent_dict.get("intent", HarenaIntentType.GENERAL_INQUIRY.value)
        ),
        confidence=float(intent_dict.get("confidence", 0.0)),
        reasoning=intent_dict.get("reasoning", ""),
        original_message=request.message,
        category=intent_dict.get("category", ""),
        is_supported=True,
    )

    processing_time_ms = int((time.time() - start_time) * 1000)
    agent_metrics = AgentMetrics(
        agent_used="autogen",
        model_used="deepseek-chat",
        tokens_consumed=0,
        processing_time_ms=processing_time_ms,
        confidence_threshold_met=True,
        cache_hit=False,
    )

    return AutogenConversationResponse(
        user_id=user_id,
        sub=user_id,
        message=request.message,
        timestamp=datetime.now(timezone.utc),
        processing_time_ms=processing_time_ms,
        intent=intent,
        agent_metrics=agent_metrics,
        entities=result.get("entities", []),
        autogen_metadata={"errors": result.get("errors", [])},
        phase=2,
    )


async def _run_legacy_pipeline(
    user_id: int,
    request: ConversationRequest,
    legacy_agent: Any,
    start_time: float,
) -> ConversationResponse:
    """Exécute le pipeline legacy."""

    classification = await legacy_agent.classify_for_team(request.message, user_id)

    intent = IntentClassificationResult(
        intent_type=HarenaIntentType(
            classification.get("intent", HarenaIntentType.GENERAL_INQUIRY.value)
        ),
        confidence=float(classification.get("confidence", 0.0)),
        reasoning=classification.get("reasoning", ""),
        original_message=request.message,
        category=classification.get("category", ""),
        is_supported=True,
    )

    processing_time_ms = int((time.time() - start_time) * 1000)
    agent_metrics = AgentMetrics(
        agent_used="legacy_intent_classifier",
        model_used="deepseek-chat",
        tokens_consumed=0,
        processing_time_ms=processing_time_ms,
        confidence_threshold_met=True,
        cache_hit=False,
    )

    return ConversationResponse(
        user_id=user_id,
        sub=user_id,
        message=request.message,
        timestamp=datetime.now(timezone.utc),
        processing_time_ms=processing_time_ms,
        intent=intent,
        agent_metrics=agent_metrics,
        phase=1,
    )


@router.get("/conversation/health")
async def conversation_health_detailed():
    """Health check du service."""

    metrics = metrics_collector.get_health_metrics()
    return {
        "service": "conversation_service",
        "phase": 1,
        "version": "1.0.0",
        "status": metrics.get("status", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jwt_compatible": True,
        "health_details": {
            "total_requests": metrics.get("total_requests", 0),
            "error_rate_percent": metrics.get("error_rate_percent", 0.0),
            "avg_latency_ms": metrics.get("latency_p95_ms", 0),
            "uptime_seconds": metrics.get("uptime_seconds", 0),
            "status_description": metrics.get("status", "unknown"),
        },
        "features": {
            "intent_classification": True,
            "supported_intents": len(HarenaIntentType),
            "json_output_forced": True,
            "cache_enabled": True,
            "auth_required": True,
            "rate_limiting": True,
            "jwt_compatible": True,
        },
    }

@router.get("/conversation/metrics")
async def conversation_metrics_detailed():
    """Retourne les métriques détaillées."""

    metrics = metrics_collector.get_all_metrics()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service_info": {
            "name": "conversation_service",
            "version": "1.0.0",
            "phase": 1,
            "features": [
                "intent_classification",
                "json_output",
                "cache",
                "auth",
                "jwt_compatible",
            ],
            "jwt_compatible": True,
        },
        "metrics": metrics,
        "performance_summary": {
            "avg_response_time": _safe_get_metric(
                metrics, ["histograms", "conversation.processing_time", "avg"], 0
            ),
            "p95_response_time": _safe_get_metric(
                metrics, ["histograms", "conversation.processing_time", "p95"], 0
            ),
            "requests_per_second": _safe_get_metric(
                metrics, ["rates", "conversation.requests_per_second"], 0
            ),
            "error_rate": _calculate_error_rate(metrics),
            "cache_hit_rate": _calculate_cache_hit_rate(metrics),
        },
        "intent_distribution": _calculate_intent_distribution(metrics),
    }


@router.get("/conversation/status")
async def conversation_status():
    """Statut global du service."""

    metrics = metrics_collector.get_health_metrics()
    status = metrics.get("status", "unknown")
    return {
        "status": status,
        "uptime_seconds": metrics.get("uptime_seconds", 0),
        "version": "1.0.0",
        "phase": 1,
        "ready": status == "healthy",
    }


def _safe_get_metric(metrics: Dict[str, Any], path: list, default: Any = None) -> Any:
    """Récupère une métrique imbriquée en toute sécurité."""

    try:
        current = metrics
        for key in path:
            current = current[key]
        return current
    except Exception:
        return default


def _calculate_error_rate(metrics: Dict[str, Any]) -> float:
    """Calcule le taux d'erreur global."""

    counters = metrics.get("counters", {})
    total_requests = counters.get("conversation.requests.total", 0)
    if total_requests <= 0:
        return 0.0
    total_errors = (
        counters.get("conversation.errors.technical", 0)
        + counters.get("conversation.errors.auth", 0)
        + counters.get("conversation.errors.validation", 0)
        + counters.get("conversation.errors.classification", 0)
    )
    return min(max((total_errors / total_requests) * 100, 0.0), 100.0)


def _calculate_cache_hit_rate(metrics: Dict[str, Any]) -> float:
    """Calcule le taux de hit du cache."""

    counters = metrics.get("counters", {})
    hits = counters.get("conversation.cache.hits", 0)
    misses = counters.get("conversation.cache.misses", 0)
    total = hits + misses
    if total <= 0:
        return 0.0
    return min(max((hits / total) * 100, 0.0), 100.0)


def _calculate_intent_distribution(metrics: Dict[str, Any]) -> Dict[str, int]:
    """Retourne la distribution des intentions classifiées."""

    counters = metrics.get("counters", {})
    distribution: Dict[str, int] = {}
    for key, value in counters.items():
        if key.startswith("conversation.intent.") and key.count(".") == 2:
            intent_name = key.split(".")[2]
            try:
                distribution[intent_name] = int(value)
            except Exception:
                continue
    return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:20])


logger.info(
    "Routes conversation configurées - Environnement: %s",
    getattr(settings, "ENVIRONMENT", "production"),
)

