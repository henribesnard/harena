"""
Routes API pour le service de conversation.

Ce module définit les endpoints REST pour les conversations,
la gestion de l'historique et les statistiques.
"""
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json

from db_service.session import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from conversation_service.models import (
    ConversationRequest, ConversationResponse, ConversationSummary,
    ConversationDetail, ConversationStats, StreamChunk
)
from conversation_service.core.conversation_manager import conversation_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Instances globales (initialisées dans main.py)
deepseek_client = None
intent_detector = None
conversation_store = None
token_counter = None


@router.post("/chat", response_model=ConversationResponse)
async def chat(
    request: ConversationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Démarre ou continue une conversation avec l'assistant.
    
    Args:
        request: Requête de conversation
        current_user: Utilisateur authentifié
        
    Returns:
        ConversationResponse: Réponse complète de l'assistant
    """
    # Vérifier que l'utilisateur ne peut discuter que pour lui-même
    if request.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot chat for other users"
        )
    
    try:
        response = await conversation_manager.process_conversation(
            request, stream=False
        )
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors du chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ConversationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Démarre une conversation en mode streaming.
    
    Args:
        request: Requête de conversation
        current_user: Utilisateur authentifié
        
    Returns:
        StreamingResponse: Réponse en streaming
    """
    # Vérifier les permissions
    if request.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot chat for other users"
        )
    
    async def generate_stream():
        """Générateur pour le streaming des réponses."""
        try:
            async for chunk in conversation_manager.process_conversation_stream(request):
                # Formater en Server-Sent Events
                chunk_json = chunk.json()
                yield f"data: {chunk_json}\n\n"
            
            # Marquer la fin du stream
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Erreur lors du streaming: {e}", exc_info=True)
            error_chunk = StreamChunk(
                type="error",
                data={
                    "error_code": "STREAMING_ERROR",
                    "message": str(e)
                }
            )
            yield f"data: {error_chunk.json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@router.get("/conversations", response_model=List[ConversationSummary])
async def get_conversations(
    limit: int = Query(20, ge=1, le=100, description="Nombre de conversations"),
    offset: int = Query(0, ge=0, description="Décalage pour pagination"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère la liste des conversations de l'utilisateur.
    
    Args:
        limit: Nombre de conversations à retourner
        offset: Décalage pour la pagination
        current_user: Utilisateur authentifié
        
    Returns:
        List[ConversationSummary]: Liste des conversations
    """
    try:
        conversations = await conversation_store.get_user_conversations(
            current_user.id, limit=limit, offset=offset
        )
        
        return conversations
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère le détail d'une conversation spécifique.
    
    Args:
        conversation_id: ID de la conversation
        current_user: Utilisateur authentifié
        
    Returns:
        ConversationDetail: Détail de la conversation
    """
    try:
        conversation = await conversation_store.get_conversation_detail(
            conversation_id, current_user.id
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la conversation {conversation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )


@router.put("/conversations/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: str,
    title: str = Query(..., min_length=1, max_length=100),
    current_user: User = Depends(get_current_active_user)
):
    """
    Met à jour le titre d'une conversation.
    
    Args:
        conversation_id: ID de la conversation
        title: Nouveau titre
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Confirmation de la mise à jour
    """
    try:
        success = await conversation_store.update_conversation_title(
            conversation_id, current_user.id, title
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {
            "status": "success",
            "message": "Title updated successfully",
            "conversation_id": conversation_id,
            "new_title": title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du titre: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update title: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Supprime une conversation.
    
    Args:
        conversation_id: ID de la conversation
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Confirmation de la suppression
    """
    try:
        success = await conversation_store.delete_conversation(
            conversation_id, current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {
            "status": "success",
            "message": "Conversation deleted successfully",
            "conversation_id": conversation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@router.post("/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Archive une conversation.
    
    Args:
        conversation_id: ID de la conversation
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Confirmation de l'archivage
    """
    try:
        success = await conversation_store.archive_conversation(
            conversation_id, current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {
            "status": "success",
            "message": "Conversation archived successfully",
            "conversation_id": conversation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'archivage de la conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to archive conversation: {str(e)}"
        )


@router.get("/conversations/{conversation_id}/summary")
async def get_conversation_summary(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Génère un résumé d'une conversation.
    
    Args:
        conversation_id: ID de la conversation
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Résumé de la conversation
    """
    try:
        summary = await conversation_manager.get_conversation_summary(
            conversation_id, current_user.id
        )
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )


@router.get("/stats", response_model=ConversationStats)
async def get_conversation_stats(
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les statistiques de conversation de l'utilisateur.
    
    Args:
        current_user: Utilisateur authentifié
        
    Returns:
        ConversationStats: Statistiques de conversation
    """
    try:
        stats = await conversation_store.get_user_stats(current_user.id)
        return stats
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get("/intent/suggestions")
async def get_intent_suggestions(
    partial_message: str = Query(..., min_length=1, description="Message partiel"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Obtient des suggestions pour compléter un message.
    
    Args:
        partial_message: Début du message
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Suggestions de complétion
    """
    try:
        suggestions = intent_detector.get_intent_suggestions(partial_message)
        
        return {
            "partial_message": partial_message,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get suggestions: {str(e)}"
        )


@router.post("/feedback")
async def submit_feedback(
    conversation_id: str,
    message_id: str,
    rating: int = Query(..., ge=1, le=5, description="Note de 1 à 5"),
    feedback: str = Query(None, description="Commentaire optionnel"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Soumet un feedback sur une réponse de l'assistant.
    
    Args:
        conversation_id: ID de la conversation
        message_id: ID du message
        rating: Note de 1 à 5
        feedback: Commentaire optionnel
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Confirmation du feedback
    """
    try:
        # Enregistrer le feedback
        success = await conversation_store.save_feedback(
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=current_user.id,
            rating=rating,
            feedback=feedback
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation or message not found"
            )
        
        logger.info(
            f"Feedback reçu de l'utilisateur {current_user.id}: "
            f"conversation {conversation_id}, message {message_id}, "
            f"note {rating}/5"
        )
        
        return {
            "status": "success",
            "message": "Feedback enregistré avec succès",
            "conversation_id": conversation_id,
            "message_id": message_id,
            "rating": rating
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_old_conversations(
    days_threshold: int = Query(30, ge=1, le=365, description="Seuil en jours"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Nettoie les anciennes conversations de l'utilisateur.
    
    Args:
        days_threshold: Seuil en jours pour considérer une conversation comme ancienne
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Résultat du nettoyage
    """
    try:
        cleaned_count = await conversation_manager.cleanup_old_conversations(
            current_user.id, days_threshold
        )
        
        return {
            "status": "success",
            "message": f"Nettoyage terminé",
            "conversations_cleaned": cleaned_count,
            "days_threshold": days_threshold
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup conversations: {str(e)}"
        )


@router.get("/debug/info")
async def get_debug_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère des informations de debug (superuser uniquement).
    
    Args:
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Informations de debug
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        debug_info = {
            "conversation_manager": conversation_manager.get_stats(),
            "intent_detector": intent_detector.get_stats() if intent_detector else None,
            "token_counter": token_counter.get_stats() if token_counter else None,
            "deepseek_client": {
                "initialized": deepseek_client.is_initialized() if deepseek_client else False,
                "chat_model": deepseek_client.chat_model if deepseek_client else None,
                "reasoner_model": deepseek_client.reasoner_model if deepseek_client else None
            }
        }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos de debug: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get debug info: {str(e)}"
        )