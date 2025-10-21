"""Conversation endpoints for API v2."""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any

from ....auth.middleware import AuthMiddleware
from ....models.requests import ConversationRequest
from ....models.responses import ConversationResponse, ErrorResponse
from ....services.conversation_service import ConversationService

router = APIRouter(prefix="/api/v2", tags=["conversation-v2"])
auth = AuthMiddleware()


@router.post(
    "/conversation/{user_id}",
    response_model=ConversationResponse,
    responses={
        200: {"description": "Réponse générée avec succès"},
        401: {"description": "Non authentifié - Token invalide"},
        403: {"description": "Accès refusé - Pas vos données"},
        429: {"description": "Trop de requêtes - Rate limit dépassé"},
        500: {"description": "Erreur serveur interne"}
    },
    summary="Poser une question sur les transactions",
    description="""
    Endpoint principal pour poser une question en langage naturel sur les transactions.

    **Authentification requise**: JWT token dans header Authorization.

    **Sécurité**: L'utilisateur ne peut accéder qu'à ses propres données.

    **Exemples de questions**:
    - "Combien j'ai dépensé en restaurants ce mois-ci ?"
    - "Mes 5 commerces où je dépense le plus"
    - "Compare mes dépenses courses entre avril et mai"
    - "Mes transactions de plus de 100 euros"
    """
)
async def create_conversation(
    user_id: int,
    request: ConversationRequest,
    token_payload: dict = Depends(auth.verify_token)
) -> ConversationResponse:
    """
    Process a user question and return a natural language answer with insights.

    Args:
        user_id: ID de l'utilisateur (doit correspondre au token)
        request: Corps de la requête avec la question
        token_payload: Payload JWT vérifié (injecté automatiquement)

    Returns:
        ConversationResponse: Réponse avec insights et visualisations

    Example:
        ```bash
        curl -X POST http://localhost:3003/api/v2/conversation/12345 \\
          -H "Authorization: Bearer eyJhbGc..." \\
          -H "Content-Type: application/json" \\
          -d '{"query": "Combien dépensé en courses ce mois?"}'
        ```
    """
    # 1. Verify user access
    await auth.verify_user_access(user_id, token_payload)

    # 2. Process the conversation
    try:
        service = ConversationService()
        response = await service.process_conversation(
            user_id=user_id,
            query=request.query,
            context=request.context.model_dump() if request.context else None
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing conversation: {str(e)}"
        )


@router.get(
    "/conversations/{user_id}/history",
    summary="Récupérer l'historique des conversations",
    description="Récupère l'historique des conversations de l'utilisateur.",
    responses={
        200: {"description": "Historique récupéré avec succès"},
        401: {"description": "Non authentifié"},
        403: {"description": "Accès refusé"}
    }
)
async def get_conversation_history(
    user_id: int,
    limit: int = 20,
    offset: int = 0,
    token_payload: dict = Depends(auth.verify_token)
) -> Dict[str, Any]:
    """
    Retrieve conversation history for a user.

    Args:
        user_id: User ID
        limit: Maximum number of conversations to return
        offset: Offset for pagination
        token_payload: JWT payload

    Returns:
        dict: Conversation history
    """
    # Verify user access
    await auth.verify_user_access(user_id, token_payload)

    # TODO: Implement conversation history retrieval from database
    return {
        "user_id": user_id,
        "conversations": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }


@router.get(
    "/health",
    summary="Health check",
    description="Check the health status of the conversation service.",
    tags=["health"]
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "service": "conversation-service-v2",
        "version": "2.0.0"
    }
