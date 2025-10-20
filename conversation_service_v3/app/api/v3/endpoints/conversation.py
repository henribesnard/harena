"""
Conversation endpoints for v3 API
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional
from pydantic import BaseModel

from ....core.agent_orchestrator import AgentOrchestrator
from ....models import UserQuery, ConversationResponse
from ....config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation", tags=["conversation"])

# Instance globale de l'orchestrateur
orchestrator: Optional[AgentOrchestrator] = None


class ConversationRequest(BaseModel):
    """Requête de conversation"""
    user_id: int
    message: str
    conversation_id: Optional[str] = None
    context: list = []


class ConversationResponseModel(BaseModel):
    """Modèle de réponse de conversation"""
    success: bool
    message: str
    total_results: Optional[int] = None
    aggregations_summary: Optional[str] = None
    metadata: dict = {}


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


@router.post("/ask", response_model=ConversationResponseModel)
async def ask_question(
    request: ConversationRequest,
    authorization: Optional[str] = Header(None),
    orch: AgentOrchestrator = Depends(get_orchestrator)
) -> ConversationResponseModel:
    """
    Pose une question sur les transactions financières

    Le pipeline complet avec agents autonomes:
    1. Analyse la question (QueryAnalyzerAgent)
    2. Construit la query Elasticsearch (ElasticsearchBuilderAgent)
    3. Exécute la query sur search_service
    4. Auto-correction si échec
    5. Génère la réponse (ResponseGeneratorAgent)

    Returns:
        Réponse conversationnelle avec résultats et insights
    """
    try:
        logger.info(f"Received conversation request from user {request.user_id}")

        # Extraire le token JWT si présent
        jwt_token = None
        if authorization and authorization.startswith("Bearer "):
            jwt_token = authorization.replace("Bearer ", "")

        # Créer la requête utilisateur
        user_query = UserQuery(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id,
            context=request.context
        )

        # Traiter la requête via l'orchestrateur
        response: ConversationResponse = await orch.process_query(
            user_query=user_query,
            jwt_token=jwt_token
        )

        # Convertir en modèle de réponse
        return ConversationResponseModel(
            success=response.success,
            message=response.message,
            total_results=response.search_results.total if response.search_results else None,
            aggregations_summary=response.aggregations_summary,
            metadata=response.metadata
        )

    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/stats")
async def get_stats(
    orch: AgentOrchestrator = Depends(get_orchestrator)
) -> dict:
    """
    Récupère les statistiques de l'orchestrateur et des agents

    Returns:
        Statistiques détaillées incluant:
        - Taux de succès
        - Taux de correction
        - Performance des agents
    """
    return orch.get_stats()


@router.get("/health")
async def health_check(
    orch: AgentOrchestrator = Depends(get_orchestrator)
) -> dict:
    """
    Health check de l'orchestrateur et des services

    Returns:
        Status de santé des agents et services externes
    """
    return await orch.health_check()
