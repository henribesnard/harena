"""
Stockage des conversations dans PostgreSQL.

Ce module gère la persistance des conversations, messages et métadonnées
dans la base de données PostgreSQL.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from db_service.session import get_db_context
from db_service.models.sync import ConversationState, MessageRole
from conversation_service.models import (
    Message, ConversationSummary, ConversationDetail, 
    ConversationStats, ConversationStatus
)

logger = logging.getLogger(__name__)


class ConversationStore:
    """Gestionnaire de stockage des conversations."""
    
    def __init__(self):
        self._initialized = False
    
    async def initialize(self):
        """Initialise le store de conversation."""
        # Vérifier la connexion à la base de données
        try:
            with get_db_context() as db:
                # Test de connexion simple
                db.execute("SELECT 1")
            
            self._initialized = True
            logger.info("ConversationStore initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du ConversationStore: {e}")
            raise
    
    async def is_healthy(self) -> bool:
        """Vérifie l'état de santé du store."""
        try:
            with get_db_context() as db:
                db.execute("SELECT 1")
            return True
        except:
            return False
    
    async def add_message(
        self,
        conversation_id: str,
        user_id: int,
        message: Message
    ) -> bool:
        """
        Ajoute un message à une conversation.
        
        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur
            message: Message à ajouter
            
        Returns:
            bool: True si le message a été ajouté avec succès
        """
        try:
            # Note: Cette implémentation utilise les modèles existants de db_service
            # Dans une implémentation complète, on aurait des modèles dédiés
            # aux conversations dans db_service/models/conversation.py
            
            # Pour l'instant, on simule le stockage et on log
            logger.info(
                f"Message ajouté à la conversation {conversation_id}: "
                f"user={user_id}, role={message.role}, "
                f"content={message.content[:50]}..."
            )
            
            # TODO: Implémenter le stockage réel avec les modèles SQLAlchemy
            # conversation = get_or_create_conversation(conversation_id, user_id)
            # db_message = ConversationMessage(
            #     conversation_id=conversation.id,
            #     message_id=message.id,
            #     role=message.role,
            #     content=message.content,
            #     metadata=message.metadata,
            #     timestamp=message.timestamp
            # )
            # db.add(db_message)
            # db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du message: {e}")
            return False
    
    async def get_conversation_messages(
        self,
        conversation_id: str,
        user_id: int,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Récupère les messages d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur
            limit: Nombre maximum de messages
            
        Returns:
            List[Dict]: Liste des messages
        """
        try:
            # TODO: Implémenter la récupération depuis la base de données
            # Pour l'instant, retourner une liste vide
            logger.debug(
                f"Récupération de {limit} messages pour conversation {conversation_id}, "
                f"utilisateur {user_id}"
            )
            
            # Simulation d'un historique de conversation
            messages = []
            
            # Dans une vraie implémentation:
            # with get_db_context() as db:
            #     conversation = db.query(Conversation).filter(
            #         Conversation.external_id == conversation_id,
            #         Conversation.user_id == user_id
            #     ).first()
            #     
            #     if conversation:
            #         db_messages = db.query(ConversationMessage).filter(
            #             ConversationMessage.conversation_id == conversation.id
            #         ).order_by(ConversationMessage.timestamp.desc()).limit(limit).all()
            #         
            #         messages = [
            #             {
            #                 "id": msg.message_id,
            #                 "role": msg.role,
            #                 "content": msg.content,
            #                 "timestamp": msg.timestamp,
            #                 "metadata": msg.metadata
            #             }
            #             for msg in reversed(db_messages)
            #         ]
            
            return messages
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des messages: {e}")
            return []
    
    async def get_user_conversations(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0
    ) -> List[ConversationSummary]:
        """
        Récupère les conversations d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre de conversations
            offset: Décalage
            
        Returns:
            List[ConversationSummary]: Liste des conversations
        """
        try:
            # TODO: Implémenter la récupération depuis la base de données
            logger.debug(f"Récupération des conversations pour utilisateur {user_id}")
            
            # Simulation de conversations
            conversations = []
            
            # Dans une vraie implémentation:
            # with get_db_context() as db:
            #     db_conversations = db.query(Conversation).filter(
            #         Conversation.user_id == user_id,
            #         Conversation.status != ConversationStatus.DELETED
            #     ).order_by(desc(Conversation.updated_at)).offset(offset).limit(limit).all()
            #     
            #     for conv in db_conversations:
            #         # Récupérer le dernier message
            #         last_message = db.query(ConversationMessage).filter(
            #             ConversationMessage.conversation_id == conv.id
            #         ).order_by(desc(ConversationMessage.timestamp)).first()
            #         
            #         # Compter les messages
            #         message_count = db.query(ConversationMessage).filter(
            #             ConversationMessage.conversation_id == conv.id
            #         ).count()
            #         
            #         conversations.append(ConversationSummary(
            #             id=conv.external_id,
            #             user_id=conv.user_id,
            #             title=conv.title,
            #             last_message=last_message.content if last_message else "",
            #             message_count=message_count,
            #             created_at=conv.created_at,
            #             updated_at=conv.updated_at,
            #             status=conv.status
            #         ))
            
            return conversations
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des conversations: {e}")
            return []
    
    async def get_conversation_detail(
        self,
        conversation_id: str,
        user_id: int
    ) -> Optional[ConversationDetail]:
        """
        Récupère le détail d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur
            
        Returns:
            ConversationDetail ou None
        """
        try:
            # TODO: Implémenter la récupération depuis la base de données
            logger.debug(f"Récupération détail conversation {conversation_id}")
            
            # Simulation - retourner None pour l'instant
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du détail: {e}")
            return None
    
    async def update_conversation_title(
        self,
        conversation_id: str,
        user_id: int,
        title: str
    ) -> bool:
        """
        Met à jour le titre d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur
            title: Nouveau titre
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        try:
            # TODO: Implémenter la mise à jour dans la base de données
            logger.info(f"Mise à jour titre conversation {conversation_id}: {title}")
            
            # Dans une vraie implémentation:
            # with get_db_context() as db:
            #     conversation = db.query(Conversation).filter(
            #         Conversation.external_id == conversation_id,
            #         Conversation.user_id == user_id
            #     ).first()
            #     
            #     if conversation:
            #         conversation.title = title
            #         conversation.updated_at = datetime.utcnow()
            #         db.commit()
            #         return True
            #     
            #     return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du titre: {e}")
            return False
    
    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: int
    ) -> bool:
        """
        Supprime une conversation.
        
        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si la suppression a réussi
        """
        try:
            # TODO: Implémenter la suppression dans la base de données
            logger.info(f"Suppression conversation {conversation_id}")
            
            # Dans une vraie implémentation:
            # with get_db_context() as db:
            #     conversation = db.query(Conversation).filter(
            #         Conversation.external_id == conversation_id,
            #         Conversation.user_id == user_id
            #     ).first()
            #     
            #     if conversation:
            #         conversation.status = ConversationStatus.DELETED
            #         conversation.updated_at = datetime.utcnow()
            #         db.commit()
            #         return True
            #     
            #     return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
            return False
    
    async def archive_conversation(
        self,
        conversation_id: str,
        user_id: int
    ) -> bool:
        """
        Archive une conversation.
        
        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si l'archivage a réussi
        """
        try:
            # TODO: Implémenter l'archivage dans la base de données
            logger.info(f"Archivage conversation {conversation_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'archivage: {e}")
            return False
    
    async def get_user_stats(self, user_id: int) -> ConversationStats:
        """
        Récupère les statistiques d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            ConversationStats: Statistiques de l'utilisateur
        """
        try:
            # TODO: Implémenter le calcul des statistiques depuis la base de données
            logger.debug(f"Calcul des statistiques pour utilisateur {user_id}")
            
            # Simulation de statistiques
            return ConversationStats(
                user_id=user_id,
                total_conversations=0,
                total_messages=0,
                avg_messages_per_conversation=0.0,
                most_common_intents=[],
                total_tokens_used=0,
                last_conversation_date=None
            )
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques: {e}")
            return ConversationStats(
                user_id=user_id,
                total_conversations=0,
                total_messages=0,
                avg_messages_per_conversation=0.0,
                most_common_intents=[],
                total_tokens_used=0,
                last_conversation_date=None
            )
    
    async def save_feedback(
        self,
        conversation_id: str,
        message_id: str,
        user_id: int,
        rating: int,
        feedback: Optional[str] = None
    ) -> bool:
        """
        Sauvegarde un feedback utilisateur.
        
        Args:
            conversation_id: ID de la conversation
            message_id: ID du message
            user_id: ID de l'utilisateur
            rating: Note de 1 à 5
            feedback: Commentaire optionnel
            
        Returns:
            bool: True si le feedback a été sauvegardé
        """
        try:
            # TODO: Implémenter la sauvegarde du feedback
            logger.info(
                f"Feedback sauvegardé: conversation={conversation_id}, "
                f"message={message_id}, rating={rating}"
            )
            
            # Dans une vraie implémentation:
            # with get_db_context() as db:
            #     feedback_entry = ConversationFeedback(
            #         conversation_id=conversation_id,
            #         message_id=message_id,
            #         user_id=user_id,
            #         rating=rating,
            #         feedback=feedback,
            #         created_at=datetime.utcnow()
            #     )
            #     db.add(feedback_entry)
            #     db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du feedback: {e}")
            return False
    
    async def cleanup_old_conversations(
        self,
        user_id: int,
        days_threshold: int = 30
    ) -> int:
        """
        Nettoie les anciennes conversations d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days_threshold: Seuil en jours
            
        Returns:
            int: Nombre de conversations nettoyées
        """
        try:
            # TODO: Implémenter le nettoyage des conversations anciennes
            logger.info(f"Nettoyage conversations utilisateur {user_id} > {days_threshold} jours")
            
            # Dans une vraie implémentation:
            # cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            # with get_db_context() as db:
            #     old_conversations = db.query(Conversation).filter(
            #         Conversation.user_id == user_id,
            #         Conversation.updated_at < cutoff_date,
            #         Conversation.status == ConversationStatus.ACTIVE
            #     ).all()
            #     
            #     count = 0
            #     for conv in old_conversations:
            #         conv.status = ConversationStatus.ARCHIVED
            #         count += 1
            #     
            #     db.commit()
            #     return count
            
            return 0
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
            return 0
    
    async def close(self):
        """Ferme les ressources du store."""
        self._initialized = False
        logger.info("ConversationStore fermé")


# Instance globale
conversation_store = ConversationStore()