"""
Endpoints FastAPI pour les conversations et messages.

Ce module définit les routes HTTP et WebSocket pour
les fonctionnalités de conversation du service.
"""

import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, Path, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..config.settings import settings
from ..config.constants import DEFAULT_CONVERSATIONS_LIMIT, MAX_CONVERSATIONS_LIMIT
from ..db.session import get_db
from ..models.conversation import (
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
    MessageCreate,
    MessageRead,
    ConversationResponse
)
from ..services.conversation_manager import ConversationManager
from ..repository.conversation_repository import ConversationRepository
from ..repository.message_repository import MessageRepository
from .dependencies import get_current_user, get_conversation_manager

router = APIRouter()


@router.post("/conversations", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """
    Crée une nouvelle conversation.
    
    Args:
        conversation: Données de la conversation à créer
        db: Session de base de données
        current_user: Informations de l'utilisateur authentifié
        conversation_manager: Gestionnaire de conversation
        
    Returns:
        Conversation créée
    """
    user_id = current_user["user_id"]
    
    # Assurer que l'utilisateur ne peut créer que ses propres conversations
    if conversation.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous ne pouvez pas créer une conversation pour un autre utilisateur"
        )
    
    # Créer la conversation
    return await conversation_manager.create_conversation(conversation)


@router.get("/conversations", response_model=List[ConversationRead])
async def get_conversations(
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_CONVERSATIONS_LIMIT, le=MAX_CONVERSATIONS_LIMIT),
    include_archived: bool = Query(False),
    include_deleted: bool = Query(False)
):
    """
    Récupère les conversations de l'utilisateur.
    
    Args:
        db: Session de base de données
        current_user: Informations de l'utilisateur authentifié
        skip: Nombre d'éléments à sauter
        limit: Nombre maximum d'éléments à récupérer
        include_archived: Inclure les conversations archivées
        include_deleted: Inclure les conversations supprimées
        
    Returns:
        Liste des conversations
    """
    user_id = current_user["user_id"]
    repo = ConversationRepository(db)
    
    # Récupérer les conversations
    return await repo.get_user_conversations(
        user_id=user_id,
        skip=skip,
        limit=limit,
        include_archived=include_archived,
        include_deleted=include_deleted
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Récupère une conversation spécifique.
    
    Args:
        conversation_id: ID de la conversation
        db: Session de base de données
        current_user: Informations de l'utilisateur authentifié
        
    Returns:
        Conversation demandée
    """
    user_id = current_user["user_id"]
    repo = ConversationRepository(db)
    
    # Récupérer la conversation
    conversation = await repo.get_conversation_by_id(conversation_id)
    
    # Vérifier que la conversation existe
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation non trouvée"
        )
    
    # Vérifier que l'utilisateur a accès à la conversation
    if conversation.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas accès à cette conversation"
        )
    
    return conversation


@router.put("/conversations/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: uuid.UUID,
    conversation_update: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Met à jour une conversation.
    
    Args:
        conversation_id: ID de la conversation
        conversation_update: Données de mise à jour
        db: Session de base de données
        current_user: Informations de l'utilisateur authentifié
        
    Returns:
        Conversation mise à jour
    """
    user_id = current_user["user_id"]
    repo = ConversationRepository(db)
    
    # Récupérer la conversation
    conversation = await repo.get_conversation_by_id(conversation_id)
    
    # Vérifier que la conversation existe
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation non trouvée"
        )
    
    # Vérifier que l'utilisateur a accès à la conversation
    if conversation.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas accès à cette conversation"
        )
    
    # Mettre à jour la conversation
    return await repo.update_conversation(conversation_id, conversation_update)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
    permanent: bool = Query(False)
):
    """
    Supprime une conversation.
    
    Args:
        conversation_id: ID de la conversation
        db: Session de base de données
        current_user: Informations de l'utilisateur authentifié
        permanent: Effectuer une suppression permanente
        
    Returns:
        Aucun contenu
    """
    user_id = current_user["user_id"]
    repo = ConversationRepository(db)
    
    # Récupérer la conversation
    conversation = await repo.get_conversation_by_id(conversation_id)
    
    # Vérifier que la conversation existe
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation non trouvée"
        )
    
    # Vérifier que l'utilisateur a accès à la conversation
    if conversation.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas accès à cette conversation"
        )
    
    # Supprimer la conversation
    if permanent:
        await repo.delete_conversation(conversation_id)
    else:
        await repo.mark_conversation_as_deleted(conversation_id)
    
    return None


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageRead])
async def get_conversation_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000)
):
    """
    Récupère les messages d'une conversation.
    
    Args:
        conversation_id: ID de la conversation
        db: Session de base de données
        current_user: Informations de l'utilisateur authentifié
        skip: Nombre d'éléments à sauter
        limit: Nombre maximum d'éléments à récupérer
        
    Returns:
        Liste des messages
    """
    user_id = current_user["user_id"]
    conv_repo = ConversationRepository(db)
    msg_repo = MessageRepository(db)
    
    # Récupérer la conversation
    conversation = await conv_repo.get_conversation_by_id(conversation_id)
    
    # Vérifier que la conversation existe
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation non trouvée"
        )
    
    # Vérifier que l'utilisateur a accès à la conversation
    if conversation.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas accès à cette conversation"
        )
    
    # Récupérer les messages
    return await msg_repo.get_conversation_messages(conversation_id, skip, limit)


@router.post("/conversations/{conversation_id}/messages", response_model=ConversationResponse)
async def add_message(
    conversation_id: uuid.UUID,
    message: MessageCreate,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """
    Ajoute un message à une conversation et obtient une réponse.
    
    Args:
        conversation_id: ID de la conversation
        message: Message à ajouter
        db: Session de base de données
        current_user: Informations de l'utilisateur authentifié
        conversation_manager: Gestionnaire de conversation
        
    Returns:
        Réponse à la conversation
    """
    user_id = current_user["user_id"]
    conv_repo = ConversationRepository(db)
    
    # Récupérer la conversation
    conversation = await conv_repo.get_conversation_by_id(conversation_id)
    
    # Vérifier que la conversation existe
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation non trouvée"
        )
    
    # Vérifier que l'utilisateur a accès à la conversation
    if conversation.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas accès à cette conversation"
        )
    
    # Vérifier que le message est pour cette conversation
    if str(message.conversation_id) != str(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="L'ID de la conversation ne correspond pas"
        )
    
    # Vérifier que l'utilisateur envoie son propre message
    if message.role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez envoyer que des messages avec le rôle 'user'"
        )
    
    # Ajouter le message et obtenir une réponse
    return await conversation_manager.process_message(conversation_id, message.content)


@router.post("/conversations/{conversation_id}/messages/stream")
async def add_message_stream(
    conversation_id: uuid.UUID,
    message: MessageCreate,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """
    Ajoute un message à une conversation et reçoit une réponse en streaming.
    
    Args:
        conversation_id: ID de la conversation
        message: Message à ajouter
        db: Session de base de données
        current_user: Informations de l'utilisateur authentifié
        conversation_manager: Gestionnaire de conversation
        
    Returns:
        StreamingResponse: Réponse en streaming
    """
    user_id = current_user["user_id"]
    conv_repo = ConversationRepository(db)
    
    # Récupérer la conversation
    conversation = await conv_repo.get_conversation_by_id(conversation_id)
    
    # Vérifier que la conversation existe
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation non trouvée"
        )
    
    # Vérifier que l'utilisateur a accès à la conversation
    if conversation.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas accès à cette conversation"
        )
    
    # Vérifier que le message est pour cette conversation
    if str(message.conversation_id) != str(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="L'ID de la conversation ne correspond pas"
        )
    
    # Vérifier que l'utilisateur envoie son propre message
    if message.role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez envoyer que des messages avec le rôle 'user'"
        )
    
    # Ajouter le message et obtenir une réponse en streaming
    return StreamingResponse(
        conversation_manager.process_message_stream(conversation_id, message.content),
        media_type="text/event-stream"
    )


@router.websocket("/ws/conversations/{conversation_id}")
async def websocket_conversation(
    websocket: WebSocket,
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """
    Point de terminaison WebSocket pour les conversations.
    
    Cette route permet une communication en temps réel pour les conversations.
    
    Args:
        websocket: Connexion WebSocket
        conversation_id: ID de la conversation
        db: Session de base de données
        conversation_manager: Gestionnaire de conversation
    """
    await websocket.accept()
    
    try:
        # NOTE: Dans un environnement de production, on devrait authentifier
        # l'utilisateur via le token JWT dans les requêtes WebSocket
        
        # Attente des messages
        while True:
            # Recevoir le message
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            # Traiter le message et obtenir une réponse en streaming
            async for response_chunk in conversation_manager.process_message_stream(
                conversation_id,
                message
            ):
                # Envoyer chaque morceau de la réponse
                await websocket.send_json({
                    "type": "chunk",
                    "content": response_chunk
                })
                
            # Envoyer un signal de fin de réponse
            await websocket.send_json({
                "type": "end",
                "conversation_id": str(conversation_id)
            })
    
    except WebSocketDisconnect:
        # Gérer la déconnexion
        pass
    
    except Exception as e:
        # Gérer les erreurs
        try:
            await websocket.send_json({
                "type": "error",
                "detail": str(e)
            })
        except:
            # La connexion est peut-être déjà fermée
            pass