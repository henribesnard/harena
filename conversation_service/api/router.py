"""
Configuration du routeur FastAPI.

Ce module configure le routeur FastAPI avec tous les endpoints
et leurs dépendances pour le service de conversation.
"""

from fastapi import APIRouter
from .endpoints import router as conversation_router

# Créer le routeur principal
router = APIRouter()

# Inclure les sous-routeurs
router.include_router(
    conversation_router,
    prefix="/conversations",
    tags=["conversations"]
)

# Points de terminaison de santé et informations
@router.get("/health", tags=["health"])
async def health_check():
    """
    Vérification de l'état de santé du service.
    
    Retourne un statut OK si le service est opérationnel.
    
    Returns:
        dict: État de santé du service
    """
    return {
        "status": "OK",
        "service": "conversation-service"
    }


@router.get("/info", tags=["health"])
async def service_info():
    """
    Informations sur le service.
    
    Retourne des informations sur la version et les capacités du service.
    
    Returns:
        dict: Informations sur le service
    """
    return {
        "name": "Harena Conversation Service",
        "version": "1.0.0",
        "description": "Service intelligent de conversation pour Harena"
    }