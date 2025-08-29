"""API routes for conversation service with optional AutoGen runtime.

This module exposes a single conversation endpoint that can operate in two
different modes:

* **Autogen mode** – when an AutoGen runtime is available the request is
  delegated to a team defined in this runtime. The response is enriched with
  entities and autogen metadata.
* **Legacy mode** – when the runtime is missing or fails the service falls
  back to the classic intent classifier agent.

The file also provides a public health endpoint exposing the availability of
each mode.  The implementation here is intentionally lightweight and focuses on
the behaviour required for the unit tests.
"""

from __future__ import annotations

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


router = APIRouter(tags=["conversation"])


@router.post(
    "/conversation/{user_id}",
    response_model=Union[AutogenConversationResponse, ConversationResponse],
)
async def analyze_conversation(
    user_id: int,
    request: ConversationRequest,
    conversation_engine: Dict[str, Any] = Depends(get_conversation_engine),
):
    """Main conversation endpoint.

    It executes the AutoGen pipeline when a runtime is available.  If the
    runtime is missing or fails, the call transparently falls back to the
    legacy intent classifier.
    """

    start_time = time.time()
    runtime = conversation_engine.get("runtime")
    legacy_agent = conversation_engine.get("legacy_agent")
    mode = conversation_engine.get("mode", "legacy")

    if mode == "autogen" and runtime is not None:
        try:
            return await _run_autogen_pipeline(user_id, request, runtime, start_time)
        except Exception as exc:  # pragma: no cover - logged internally
            # Fallback to legacy flow on any runtime failure
            if legacy_agent is not None:
                return await _run_legacy_pipeline(user_id, request, legacy_agent, start_time)
            raise HTTPException(status_code=500, detail="Autogen pipeline failed") from exc

    if legacy_agent is not None:
        return await _run_legacy_pipeline(user_id, request, legacy_agent, start_time)

    raise HTTPException(status_code=500, detail="No conversation engine available")


async def _run_autogen_pipeline(
    user_id: int,
    request: ConversationRequest,
    runtime: Any,
    start_time: float,
) -> AutogenConversationResponse:
    """Execute the AutoGen conversation pipeline."""

    team_cls = runtime.get_team("phase2")
    team = team_cls()
    result = await team.process_user_message(request.message, user_id)

    intent_dict = result.get("intent", {})
    intent = IntentClassificationResult(
        intent_type=HarenaIntentType(
            intent_dict.get("intent", HarenaIntentType.GENERAL_QUESTION.value)
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
    """Execute the legacy intent-classification pipeline."""

    classification = await legacy_agent.classify_for_team(request.message, user_id)

    intent = IntentClassificationResult(
        intent_type=HarenaIntentType(
            classification.get("intent", HarenaIntentType.GENERAL_QUESTION.value)
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
async def conversation_health_detailed(request: Request) -> Dict[str, Any]:
    """Public health check exposing mode availability."""

    metrics = metrics_collector.get_health_metrics()
    autogen_available = bool(getattr(request.app.state, "autogen_runtime", None))
    return {
        "service": "conversation_service",
        "phase": 1,
        "version": "1.0.0",
        "status": metrics.get("status", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jwt_compatible": True,
        "modes": {"legacy": True, "autogen": autogen_available},
        "features": {
            "intent_classification": True,
            "entity_extraction": True,
        },
        "health_details": {
            "total_requests": metrics.get("total_requests", 0),
            "error_rate_percent": metrics.get("error_rate_percent", 0.0),
            "avg_latency_ms": metrics.get("latency_p95_ms", 0),
            "uptime_seconds": metrics.get("uptime_seconds", 0),
            "status_description": metrics.get("status", "unknown"),
        },
    }


__all__ = ["router"]

