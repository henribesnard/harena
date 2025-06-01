"""
Gestionnaire WebSocket pour le service de conversation.

Ce module gère les connexions WebSocket pour les conversations
temps réel et le streaming des réponses.
"""
import logging
import json
import asyncio
from typing import Dict, Any, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import OAuth2PasswordBearer

from conversation_service.models import ConversationRequest, StreamChunk
from conversation_service.core.conversation_manager import conversation_manager

logger = logging.getLogger(__name__)
websocket_router = APIRouter()

# Instances globales (initialisées dans main.py)
deepseek_client = None
intent_detector = None
conversation_store = None
token_counter = None


class ConnectionManager:
    """Gestionnaire des connexions WebSocket."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[int, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: int):
        """Accepte une nouvelle connexion WebSocket."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connecté: {connection_id} pour utilisateur {user_id}")
    
    def disconnect(self, connection_id: str, user_id: int):
        """Supprime une connexion."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket déconnecté: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message: dict):
        """Envoie un message à une connexion spécifique."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Erreur envoi WebSocket {connection_id}: {e}")
                # Nettoyer la connexion fermée
                user_id = None
                for uid, connections in self.user_connections.items():
                    if connection_id in connections:
                        user_id = uid
                        break
                if user_id:
                    self.disconnect(connection_id, user_id)
    
    async def send_to_user(self, user_id: int, message: dict):
        """Envoie un message à toutes les connexions d'un utilisateur."""
        if user_id in self.user_connections:
            connections = self.user_connections[user_id].copy()
            for connection_id in connections:
                await self.send_to_connection(connection_id, message)
    
    def get_user_connection_count(self, user_id: int) -> int:
        """Retourne le nombre de connexions actives pour un utilisateur."""
        return len(self.user_connections.get(user_id, set()))
    
    def get_total_connections(self) -> int:
        """Retourne le nombre total de connexions actives."""
        return len(self.active_connections)


# Instance globale du gestionnaire de connexions
connection_manager = ConnectionManager()


@websocket_router.websocket("/ws/chat/{user_id}/{connection_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    user_id: int,
    connection_id: str
):
    """
    Endpoint WebSocket principal pour les conversations.
    
    Args:
        websocket: Connexion WebSocket
        user_id: ID de l'utilisateur
        connection_id: ID unique de la connexion
    """
    await connection_manager.connect(websocket, connection_id, user_id)
    
    try:
        # Message de bienvenue
        welcome_message = {
            "type": "connected",
            "data": {
                "message": "Connexion établie avec Harena Assistant",
                "user_id": user_id,
                "connection_id": connection_id
            }
        }
        await connection_manager.send_to_connection(connection_id, welcome_message)
        
        # Boucle de traitement des messages
        while True:
            try:
                # Recevoir un message
                data = await websocket.receive_json()
                await handle_websocket_message(data, user_id, connection_id)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket {connection_id} déconnecté normalement")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Erreur JSON WebSocket {connection_id}: {e}")
                error_message = {
                    "type": "error",
                    "data": {
                        "error_code": "INVALID_JSON",
                        "message": "Format JSON invalide"
                    }
                }
                await connection_manager.send_to_connection(connection_id, error_message)
            except Exception as e:
                logger.error(f"Erreur WebSocket {connection_id}: {e}", exc_info=True)
                error_message = {
                    "type": "error",
                    "data": {
                        "error_code": "PROCESSING_ERROR",
                        "message": "Erreur lors du traitement du message"
                    }
                }
                await connection_manager.send_to_connection(connection_id, error_message)
    
    finally:
        connection_manager.disconnect(connection_id, user_id)


async def handle_websocket_message(
    data: Dict[str, Any],
    user_id: int,
    connection_id: str
):
    """
    Traite un message reçu via WebSocket.
    
    Args:
        data: Données du message
        user_id: ID de l'utilisateur
        connection_id: ID de la connexion
    """
    message_type = data.get("type")
    
    if message_type == "chat":
        await handle_chat_message(data, user_id, connection_id)
    elif message_type == "ping":
        await handle_ping_message(connection_id)
    elif message_type == "typing":
        await handle_typing_message(data, user_id, connection_id)
    else:
        logger.warning(f"Type de message WebSocket inconnu: {message_type}")
        error_message = {
            "type": "error",
            "data": {
                "error_code": "UNKNOWN_MESSAGE_TYPE",
                "message": f"Type de message non supporté: {message_type}"
            }
        }
        await connection_manager.send_to_connection(connection_id, error_message)


async def handle_chat_message(
    data: Dict[str, Any],
    user_id: int,
    connection_id: str
):
    """
    Traite un message de chat.
    
    Args:
        data: Données du message de chat
        user_id: ID de l'utilisateur
        connection_id: ID de la connexion
    """
    try:
        # Valider les données du message
        if "message" not in data:
            raise ValueError("Champ 'message' manquant")
        
        # Créer la requête de conversation
        conversation_request = ConversationRequest(
            user_id=user_id,
            message=data["message"],
            conversation_id=data.get("conversation_id"),
            context=data.get("context"),
            stream=True
        )
        
        # Confirmer la réception
        ack_message = {
            "type": "message_received",
            "data": {
                "message_id": data.get("message_id"),
                "status": "processing"
            }
        }
        await connection_manager.send_to_connection(connection_id, ack_message)
        
        # Traiter la conversation en streaming
        async for chunk in conversation_manager.process_conversation_stream(conversation_request):
            # Convertir le StreamChunk en message WebSocket
            ws_message = {
                "type": "stream_chunk",
                "data": {
                    "chunk_type": chunk.type,
                    "content": chunk.content,
                    "data": chunk.data,
                    "metadata": chunk.metadata
                }
            }
            
            await connection_manager.send_to_connection(connection_id, ws_message)
            
            # Petite pause pour éviter la surcharge
            await asyncio.sleep(0.01)
        
        # Message de fin
        end_message = {
            "type": "stream_complete",
            "data": {
                "message_id": data.get("message_id"),
                "status": "completed"
            }
        }
        await connection_manager.send_to_connection(connection_id, end_message)
        
    except ValueError as ve:
        error_message = {
            "type": "error",
            "data": {
                "error_code": "VALIDATION_ERROR",
                "message": str(ve)
            }
        }
        await connection_manager.send_to_connection(connection_id, error_message)
    except Exception as e:
        logger.error(f"Erreur traitement chat WebSocket: {e}", exc_info=True)
        error_message = {
            "type": "error", 
            "data": {
                "error_code": "CHAT_ERROR",
                "message": "Erreur lors du traitement du message de chat"
            }
        }
        await connection_manager.send_to_connection(connection_id, error_message)


async def handle_ping_message(connection_id: str):
    """
    Traite un message ping pour maintenir la connexion.
    
    Args:
        connection_id: ID de la connexion
    """
    pong_message = {
        "type": "pong",
        "data": {
            "timestamp": asyncio.get_event_loop().time()
        }
    }
    await connection_manager.send_to_connection(connection_id, pong_message)


async def handle_typing_message(
    data: Dict[str, Any],
    user_id: int,
    connection_id: str
):
    """
    Traite un indicateur de frappe (pour les conversations de groupe futures).
    
    Args:
        data: Données du message de frappe
        user_id: ID de l'utilisateur
        connection_id: ID de la connexion
    """
    # Pour l'instant, on log simplement l'événement
    logger.debug(f"Utilisateur {user_id} est en train de taper")
    
    # Dans le futur, on pourrait diffuser cette information
    # aux autres participants d'une conversation de groupe


@websocket_router.websocket("/ws/notifications/{user_id}")
async def websocket_notifications_endpoint(
    websocket: WebSocket,
    user_id: int
):
    """
    Endpoint WebSocket pour les notifications.
    
    Args:
        websocket: Connexion WebSocket
        user_id: ID de l'utilisateur
    """
    connection_id = f"notifications_{user_id}_{asyncio.get_event_loop().time()}"
    await connection_manager.connect(websocket, connection_id, user_id)
    
    try:
        # Message de bienvenue
        welcome_message = {
            "type": "notifications_connected",
            "data": {
                "message": "Connexion notifications établie",
                "user_id": user_id
            }
        }
        await connection_manager.send_to_connection(connection_id, welcome_message)
        
        # Maintenir la connexion active
        while True:
            try:
                # Attendre un message (principalement pour détecter les déconnexions)
                await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info(f"WebSocket notifications {connection_id} déconnecté")
                break
    
    finally:
        connection_manager.disconnect(connection_id, user_id)


# Fonction utilitaire pour envoyer des notifications
async def send_notification_to_user(
    user_id: int,
    notification_type: str,
    data: Dict[str, Any]
):
    """
    Envoie une notification à un utilisateur via WebSocket.
    
    Args:
        user_id: ID de l'utilisateur
        notification_type: Type de notification
        data: Données de la notification
    """
    notification_message = {
        "type": "notification",
        "data": {
            "notification_type": notification_type,
            "payload": data
        }
    }
    
    await connection_manager.send_to_user(user_id, notification_message)


# Fonction pour obtenir les statistiques des connexions
def get_websocket_stats() -> Dict[str, Any]:
    """
    Retourne les statistiques des connexions WebSocket.
    
    Returns:
        Dict: Statistiques des connexions
    """
    total_connections = connection_manager.get_total_connections()
    active_users = len(connection_manager.user_connections)
    
    # Statistiques par utilisateur
    user_stats = {}
    for user_id, connections in connection_manager.user_connections.items():
        user_stats[user_id] = len(connections)
    
    return {
        "total_connections": total_connections,
        "active_users": active_users,
        "connections_per_user": user_stats,
        "average_connections_per_user": total_connections / active_users if active_users > 0 else 0
    }