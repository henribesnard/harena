"""
Service de persistence des conversations en base de données.
Gère la sauvegarde et la récupération des conversations et messages.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from db_service.models.conversation import Conversation, ConversationTurn
from db_service.models.user import User
from conversation_service.models.responses.conversation_responses import ResponseContent

logger = logging.getLogger(__name__)


class ConversationPersistenceService:
    """Service de gestion de la persistence des conversations"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_conversation(
        self, 
        user_id: int, 
        title: Optional[str] = None,
        conversation_data: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Crée une nouvelle conversation pour un utilisateur"""
        try:
            conversation = Conversation(
                user_id=user_id,
                title=title or f"Conversation du {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                total_turns=0,
                last_activity_at=datetime.now(timezone.utc),
                data=conversation_data or {}
            )
            
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            
            logger.info(f"✅ Conversation créée - ID: {conversation.id}, User: {user_id}")
            return conversation
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création conversation - User: {user_id}, Error: {e}")
            raise
    
    def add_conversation_turn(
        self,
        conversation_id: int,
        user_message: str,
        assistant_response: str,
        turn_data: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """Ajoute un tour de conversation (message utilisateur + réponse IA)"""
        try:
            # Récupérer la conversation
            conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} introuvable")
            
            # Calculer le numéro du prochain tour
            next_turn_number = conversation.total_turns + 1
            
            # Créer le tour
            turn = ConversationTurn(
                conversation_id=conversation_id,
                turn_number=next_turn_number,
                user_message=user_message,
                assistant_response=assistant_response,
                data=turn_data or {}
            )
            
            self.db.add(turn)
            
            # Mettre à jour la conversation
            conversation.total_turns = next_turn_number
            conversation.last_activity_at = datetime.now(timezone.utc)
            
            self.db.commit()
            self.db.refresh(turn)
            
            logger.info(f"✅ Tour ajouté - Conversation: {conversation_id}, Tour: {next_turn_number}")
            return turn
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Erreur ajout tour - Conversation: {conversation_id}, Error: {e}")
            raise
    
    def get_user_conversations(
        self, 
        user_id: int, 
        limit: int = 20, 
        offset: int = 0
    ) -> List[Conversation]:
        """Récupère les conversations d'un utilisateur (les plus récentes en premier)"""
        try:
            conversations = (
                self.db.query(Conversation)
                .filter_by(user_id=user_id)
                .filter(Conversation.status == "active")
                .order_by(Conversation.last_activity_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            logger.debug(f"📂 Conversations récupérées - User: {user_id}, Count: {len(conversations)}")
            return conversations
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Erreur récupération conversations - User: {user_id}, Error: {e}")
            raise
    
    def get_conversation_with_turns(
        self, 
        conversation_id: int, 
        user_id: int
    ) -> Optional[Conversation]:
        """Récupère une conversation avec tous ses tours"""
        try:
            conversation = (
                self.db.query(Conversation)
                .filter_by(id=conversation_id, user_id=user_id)
                .first()
            )
            
            if conversation:
                # Les tours sont automatiquement chargés via la relation
                logger.debug(f"📖 Conversation récupérée - ID: {conversation_id}, Tours: {len(conversation.turns)}")
            
            return conversation
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Erreur récupération conversation - ID: {conversation_id}, Error: {e}")
            raise
    
    def update_conversation_title(
        self, 
        conversation_id: int, 
        user_id: int, 
        title: str
    ) -> bool:
        """Met à jour le titre d'une conversation"""
        try:
            result = (
                self.db.query(Conversation)
                .filter_by(id=conversation_id, user_id=user_id)
                .update({"title": title})
            )
            
            self.db.commit()
            
            if result > 0:
                logger.info(f"✅ Titre mis à jour - Conversation: {conversation_id}")
                return True
            else:
                logger.warning(f"⚠️ Conversation non trouvée - ID: {conversation_id}, User: {user_id}")
                return False
                
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Erreur mise à jour titre - Conversation: {conversation_id}, Error: {e}")
            raise
    
    def archive_conversation(
        self, 
        conversation_id: int, 
        user_id: int
    ) -> bool:
        """Archive une conversation (la marque comme inactive)"""
        try:
            result = (
                self.db.query(Conversation)
                .filter_by(id=conversation_id, user_id=user_id)
                .update({"status": "archived"})
            )
            
            self.db.commit()
            
            if result > 0:
                logger.info(f"📦 Conversation archivée - ID: {conversation_id}")
                return True
            else:
                logger.warning(f"⚠️ Conversation non trouvée - ID: {conversation_id}, User: {user_id}")
                return False
                
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"❌ Erreur archivage - Conversation: {conversation_id}, Error: {e}")
            raise
    
    def get_or_create_active_conversation(
        self, 
        user_id: int,
        conversation_title: Optional[str] = None
    ) -> Conversation:
        """Récupère la conversation active la plus récente ou en crée une nouvelle"""
        try:
            # Chercher la conversation active la plus récente
            active_conversation = (
                self.db.query(Conversation)
                .filter_by(user_id=user_id, status="active")
                .order_by(Conversation.last_activity_at.desc())
                .first()
            )
            
            if active_conversation:
                logger.debug(f"🔄 Conversation active trouvée - ID: {active_conversation.id}")
                return active_conversation
            else:
                # Créer une nouvelle conversation
                logger.info(f"➕ Création nouvelle conversation - User: {user_id}")
                return self.create_conversation(user_id, conversation_title)
                
        except SQLAlchemyError as e:
            logger.error(f"❌ Erreur récupération/création conversation - User: {user_id}, Error: {e}")
            raise


def create_conversation_data(
    request_id: str,
    intent_type: str,
    entities: Dict[str, Any],
    search_results_count: int = 0,
    processing_time_ms: int = 0
) -> Dict[str, Any]:
    """Crée les données metadata pour une conversation"""
    return {
        "request_id": request_id,
        "intent_type": intent_type,
        "entities": entities,
        "search_results_count": search_results_count,
        "processing_time_ms": processing_time_ms,
        "created_at": datetime.now(timezone.utc).isoformat()
    }


def create_turn_data(
    request_id: str,
    intent: Dict[str, Any],
    entities: Dict[str, Any],
    search_results_summary: Dict[str, Any],
    response_quality: Dict[str, Any],
    processing_time_ms: int
) -> Dict[str, Any]:
    """Crée les données metadata pour un tour de conversation"""
    return {
        "request_id": request_id,
        "intent": intent,
        "entities": entities,
        "search_results_summary": search_results_summary,
        "response_quality": response_quality,
        "processing_time_ms": processing_time_ms,
        "created_at": datetime.now(timezone.utc).isoformat()
    }