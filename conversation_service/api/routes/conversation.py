import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, HTTPException, Request

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

