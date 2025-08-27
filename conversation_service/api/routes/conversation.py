"""Routes API pour conversation service - Phase 2 AutoGen"""
import logging
import time
from typing import Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.models.responses.conversation_responses import (
    ConversationResponsePhase2AutoGen,
    AutoGenTeamResponse,
    AutoGenMessage,
    ProcessingStatus,
)
from conversation_service.api.dependencies import (
    get_conversation_runtime,
    get_conversation_service_status,
    validate_path_user_id,
    get_user_context,
    rate_limit_dependency,
)
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.utils.validation_utils import (
    validate_user_message,
    sanitize_user_input,
)

router = APIRouter(tags=["conversation"])
logger = logging.getLogger("conversation_service.routes")


@router.post("/conversation/{path_user_id}", response_model=ConversationResponsePhase2AutoGen)
async def analyze_conversation(
    request_data: ConversationRequest,
    request: Request,
    runtime = Depends(get_conversation_runtime),
    user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    service_status: dict = Depends(get_conversation_service_status),
    _rate_limit: None = Depends(rate_limit_dependency),
):
    """Endpoint Phase 2 utilisant l'équipe AutoGen."""
    start_time = time.time()
    request_id = f"{user_id}_{int(start_time * 1000)}"

    client_ip = getattr(request.client, "host", "unknown") if request.client else "unknown"
    logger.info(
        f"[{request_id}] Phase2 conversation - User: {user_id}, "
        f"IP: {client_ip}, Message: '{request_data.message[:30]}...'"
    )

    message_validation = validate_user_message(request_data.message)
    if not message_validation["valid"]:
        metrics_collector.increment_counter("conversation.errors.validation")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Message invalide",
                "errors": message_validation["errors"],
                "warnings": message_validation.get("warnings", []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    clean_message = sanitize_user_input(request_data.message)

    team_result = await runtime.run_financial_team(clean_message, user_context)

    team_response = AutoGenTeamResponse(
        final_answer=team_result.get("final_answer", ""),
        steps=[
            AutoGenMessage(role=step.get("agent", ""), content=step.get("content", ""))
            for step in team_result.get("intermediate_steps", [])
        ],
        context=team_result.get("context", {}),
    )

    processing_time = int((time.time() - start_time) * 1000)

    return ConversationResponsePhase2AutoGen(
        user_id=user_id,
        message=clean_message,
        timestamp=datetime.now(timezone.utc),
        request_id=request_id,
        processing_time_ms=processing_time,
        team_response=team_response,
        status=ProcessingStatus.SUCCESS,
    )


@router.get("/conversation/status")
async def conversation_status():
    """Simple endpoint de disponibilité."""
    return {
        "service": "conversation_service",
        "status": "ready",
        "ready": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/conversation/health")
async def conversation_health():
    """Retourne un aperçu de l'état du service."""
    health_metrics = metrics_collector.get_health_metrics()
    return {
        "service": "conversation_service",
        "status": health_metrics.get("status", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "health_details": health_metrics,
        "features": [],
    }


@router.get("/conversation/metrics")
async def conversation_metrics():
    """Expose les métriques collectées."""
    metrics_data = metrics_collector.get_all_metrics()
    return {
        "timestamp": metrics_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "metrics": metrics_data,
        "service_info": {"name": "conversation_service", "version": "1.0.0", "phase": 1},
        "performance_summary": {},
    }
