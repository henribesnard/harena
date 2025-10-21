"""
Conversation endpoints for v3 API - Compatible with v1 format
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional, Dict, Any
from pydantic import BaseModel

from ....core.agent_orchestrator import AgentOrchestrator
from ....models import UserQuery, ConversationResponse
from ....config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/conversation", tags=["conversation"])

# Instance globale de l'orchestrateur
orchestrator: Optional[AgentOrchestrator] = None


class ClientInfo(BaseModel):
    """Client information"""
    platform: str = "web"
    version: str = "1.0.0"


class ConversationRequest(BaseModel):
    """Requête de conversation - Compatible v1 format"""
    client_info: Optional[ClientInfo] = None
    message: str
    message_type: str = "text"
    priority: str = "normal"
    conversation_id: Optional[int] = None


class ConversationResponseModel(BaseModel):
    """Modèle de réponse de conversation"""
    success: bool
    message: str
    total_results: Optional[int] = None
    aggregations_summary: Optional[str] = None
    metadata: dict = {}


def extract_jwt_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract JWT token from Authorization header"""
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "")
    return None


def get_orchestrator() -> AgentOrchestrator:
    """Dépendance pour obtenir l'orchestrateur"""
    global orchestrator
    if orchestrator is None:
        orchestrator = AgentOrchestrator(
            search_service_url=settings.SEARCH_SERVICE_URL,
            max_correction_attempts=settings.MAX_CORRECTION_ATTEMPTS,
            llm_model=settings.LLM_MODEL
        )
    return orchestrator


@router.post("/{user_id}", response_model=Dict[str, Any])
async def analyze_conversation(
    user_id: int,
    request: ConversationRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator),
    jwt_token: Optional[str] = Depends(extract_jwt_token)
) -> Dict[str, Any]:
    """
    Endpoint principal compatible v1 - POST /api/v1/conversation/{user_id}

    Le pipeline complet avec agents autonomes:
    1. Analyse la question (QueryAnalyzerAgent)
    2. Construit la query Elasticsearch (ElasticsearchBuilderAgent)
    3. Exécute la query sur search_service
    4. Auto-correction si échec
    5. Génère la réponse (ResponseGeneratorAgent)

    Args:
        user_id: ID de l'utilisateur (path parameter)
        request: Corps de la requête avec message, client_info, etc.
        jwt_token: Token JWT extrait de l'en-tête Authorization

    Returns:
        Réponse conversationnelle avec résultats et insights
    """
    try:
        logger.info(f"Received conversation request from user {user_id}")

        # Créer la requête utilisateur
        user_query = UserQuery(
            user_id=user_id,
            message=request.message,
            conversation_id=str(request.conversation_id) if request.conversation_id else None,
            context=[]
        )

        # Traiter la requête via l'orchestrateur
        response: ConversationResponse = await orch.process_query(
            user_query=user_query,
            jwt_token=jwt_token
        )

        # Format de réponse compatible v1
        return {
            "user_id": user_id,
            "message": request.message,
            "status": "completed" if response.success else "error",
            "response": {
                "message": response.message,
                "structured_data": {
                    "total_results": response.search_results.total if response.search_results else 0,
                    "aggregations_summary": response.aggregations_summary
                }
            },
            "search_summary": {
                "found_results": response.search_results is not None and response.search_results.total > 0,
                "total_results": response.search_results.total if response.search_results else 0
            },
            "metadata": response.metadata,
            "architecture": "v3_langchain_agents"
        }

    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/{user_id}/stream")
async def analyze_conversation_stream(
    user_id: int,
    request: ConversationRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator),
    jwt_token: Optional[str] = Depends(extract_jwt_token)
):
    """
    Endpoint streaming compatible v1 - POST /api/v1/conversation/{user_id}/stream

    Note: v3 ne supporte pas le streaming natif, renvoie une réponse complète
    """
    from fastapi.responses import StreamingResponse
    import json

    async def generate_stream():
        try:
            # Créer la requête utilisateur
            user_query = UserQuery(
                user_id=user_id,
                message=request.message,
                conversation_id=str(request.conversation_id) if request.conversation_id else None,
                context=[]
            )

            # Traiter la requête
            response: ConversationResponse = await orch.process_query(
                user_query=user_query,
                jwt_token=jwt_token
            )

            # Simuler un stream en envoyant la réponse complète
            yield f"data: {json.dumps({'type': 'response_start'})}\n\n"

            yield f"data: {json.dumps({'type': 'response_chunk', 'content': response.message})}\n\n"

            yield f"data: {json.dumps({'type': 'response_end', 'metadata': response.metadata})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get("/health")
async def conversation_health():
    """
    Health check compatible v1 - GET /api/v1/conversation/health
    """
    from datetime import datetime, timezone

    try:
        if orchestrator:
            health = await orchestrator.health_check()
            return {
                "service": "conversation_service_v3",
                "version": "3.0.0",
                "architecture": "langchain_agents",
                "status": "healthy" if health.get("healthy", False) else "degraded",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": health
            }
        else:
            return {
                "service": "conversation_service_v3",
                "version": "3.0.0",
                "status": "initializing",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "service": "conversation_service_v3",
            "version": "3.0.0",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/status")
async def conversation_status():
    """
    Status endpoint compatible v1 - GET /api/v1/conversation/status
    """
    from datetime import datetime, timezone

    try:
        if orchestrator:
            stats = orchestrator.get_stats()
            return {
                "status": "healthy",
                "version": "3.0.0",
                "architecture": "langchain_agents",
                "ready": True,
                "stats": stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "status": "initializing",
                "ready": False,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return {
            "status": "error",
            "ready": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/metrics")
async def conversation_metrics():
    """
    Metrics endpoint compatible v1 - GET /api/v1/conversation/metrics
    """
    from datetime import datetime, timezone

    try:
        if orchestrator:
            stats = orchestrator.get_stats()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": "v3_langchain_agents",
                "metrics": stats,
                "service_info": {
                    "name": "conversation_service_v3",
                    "version": "3.0.0",
                    "features": [
                        "autonomous_agents",
                        "auto_correction",
                        "elasticsearch_schema_aware",
                        "aggregations_support"
                    ]
                }
            }
        else:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "Orchestrator not initialized"
            }
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
