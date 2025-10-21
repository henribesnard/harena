"""
Service de persistence des conversations pour conversation_service_v3
Gère la sauvegarde et la récupération des conversations et messages en PostgreSQL
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from db_service.models.conversation import Conversation, ConversationTurn
from db_service.models.user import User

logger = logging.getLogger(__name__)


class ConversationPersistenceService:
    """
    Service de gestion de la persistence des conversations pour v3
    Compatible avec les modèles db_service existants
    """

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

            logger.info(f"✅ Tour ajouté - Conversation: {conversation_id}, Turn: {next_turn_number}")
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
        user_id: Optional[int] = None
    ) -> Optional[Conversation]:
        """
        Récupère une conversation avec tous ses tours

        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur (optionnel, pour vérification de sécurité)
        """
        try:
            query = self.db.query(Conversation).filter_by(id=conversation_id)

            # Ajouter le filtre user_id pour la sécurité si fourni
            if user_id is not None:
                query = query.filter_by(user_id=user_id)

            conversation = query.first()

            if conversation:
                # Les tours sont automatiquement chargés via la relation
                logger.debug(f"📖 Conversation récupérée - ID: {conversation_id}, Tours: {len(conversation.turns)}")
            else:
                logger.warning(f"⚠️ Conversation non trouvée - ID: {conversation_id}")

            return conversation

        except SQLAlchemyError as e:
            logger.error(f"❌ Erreur récupération conversation - ID: {conversation_id}, Error: {e}")
            raise

    def get_or_create_conversation(
        self,
        user_id: int,
        conversation_id: Optional[int] = None,
        conversation_title: Optional[str] = None
    ) -> Conversation:
        """
        Récupère une conversation existante ou en crée une nouvelle

        Args:
            user_id: ID de l'utilisateur
            conversation_id: ID de la conversation à récupérer (optionnel)
            conversation_title: Titre de la nouvelle conversation si créée

        Returns:
            Conversation existante ou nouvellement créée
        """
        try:
            # Si un conversation_id est fourni, essayer de le récupérer
            if conversation_id:
                conversation = self.get_conversation_with_turns(conversation_id, user_id)
                if conversation:
                    logger.debug(f"✅ Conversation existante trouvée - ID: {conversation_id}")
                    return conversation
                else:
                    logger.warning(f"⚠️ Conversation {conversation_id} introuvable, création d'une nouvelle")

            # Créer une nouvelle conversation
            logger.info(f"➕ Création nouvelle conversation - User: {user_id}")
            return self.create_conversation(user_id, conversation_title)

        except SQLAlchemyError as e:
            logger.error(f"❌ Erreur get_or_create_conversation - User: {user_id}, Error: {e}")
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


def create_turn_metadata_v3(
    user_query: str,
    query_analysis: Optional[Dict[str, Any]] = None,
    elasticsearch_query: Optional[Dict[str, Any]] = None,
    search_results_summary: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[int] = None,
    corrections_applied: int = 0
) -> Dict[str, Any]:
    """
    Crée les métadonnées pour un tour de conversation v3
    Adapté pour capturer les informations des agents LangChain

    Args:
        user_query: Message de l'utilisateur
        query_analysis: Analyse de la requête par QueryAnalyzerAgent
        elasticsearch_query: Query construite par ElasticsearchBuilderAgent
        search_results_summary: Résumé des résultats de recherche
        processing_time_ms: Temps de traitement en millisecondes
        corrections_applied: Nombre de corrections appliquées

    Returns:
        Dictionnaire de métadonnées
    """
    metadata = {
        "architecture": "v3_langchain_agents",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if query_analysis:
        metadata["query_analysis"] = query_analysis

    if elasticsearch_query:
        metadata["elasticsearch_query"] = elasticsearch_query

    if search_results_summary:
        metadata["search_results"] = search_results_summary

    if processing_time_ms is not None:
        metadata["processing_time_ms"] = processing_time_ms

    if corrections_applied > 0:
        metadata["corrections_applied"] = corrections_applied

    return metadata
