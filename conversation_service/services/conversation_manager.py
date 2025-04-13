"""
Orchestration du flux de conversation et coordination.

Ce module fournit le gestionnaire de conversation qui coordonne
les différents services pour traiter les messages des utilisateurs
et générer des réponses.
"""

import uuid
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from sqlalchemy.orm import Session

from ..config.settings import settings
from ..config.logging import get_logger
from ..db.models import ConversationState, MessageRole
from ..llm.llm_service import LLMService
from ..models.conversation import ConversationCreate, ConversationResponse
from ..models.intent import IntentType, IntentClassification
from ..repository.conversation_repository import ConversationRepository
from ..repository.message_repository import MessageRepository
from ..utils.token_counter import count_tokens
from .intent_classifier import IntentClassifier
from .query_builder import QueryBuilder
from .response_generator import ResponseGenerator

logger = get_logger(__name__)


class ConversationManager:
    """
    Gestionnaire central des conversations.
    
    Cette classe orchestre le flux de traitement des messages,
    de la réception à la génération de réponses, en passant
    par l'analyse d'intention et la requête des données.
    """
    
    def __init__(
        self,
        db: Session,
        llm_service: LLMService,
        intent_classifier: Optional[IntentClassifier] = None,
        query_builder: Optional[QueryBuilder] = None,
        response_generator: Optional[ResponseGenerator] = None
    ):
        """
        Initialise le gestionnaire de conversation.
        
        Args:
            db: Session de base de données
            llm_service: Service LLM
            intent_classifier: Classificateur d'intention
            query_builder: Constructeur de requête
            response_generator: Générateur de réponse
        """
        self.db = db
        self.llm_service = llm_service
        self.intent_classifier = intent_classifier or IntentClassifier(llm_service)
        self.query_builder = query_builder or QueryBuilder()
        self.response_generator = response_generator or ResponseGenerator(llm_service)
        
        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)
        
        logger.info("Gestionnaire de conversation initialisé")
    
    async def create_conversation(self, conversation_data: ConversationCreate) -> Any:
        """
        Crée une nouvelle conversation.
        
        Args:
            conversation_data: Données de la conversation à créer
            
        Returns:
            Conversation créée
        """
        logger.info(f"Création d'une nouvelle conversation pour l'utilisateur {conversation_data.user_id}")
        
        # Créer la conversation
        conversation = await self.conversation_repo.create_conversation(conversation_data)
        
        # Ajouter un message système initial si configuré
        if settings.DEFAULT_SYSTEM_PROMPT:
            await self.message_repo.create_system_message(
                conversation_id=conversation.id,
                content=settings.DEFAULT_SYSTEM_PROMPT,
                token_count=count_tokens(settings.DEFAULT_SYSTEM_PROMPT)
            )
            
            logger.info(f"Message système initial ajouté à la conversation {conversation.id}")
        
        return conversation
    
    async def process_message(
        self,
        conversation_id: uuid.UUID,
        message_content: str
    ) -> ConversationResponse:
        """
        Traite un message utilisateur et génère une réponse.
        
        Cette méthode orchestre le flux complet de traitement d'un message:
        1. Stockage du message utilisateur
        2. Classification de l'intention
        3. Requête des données nécessaires
        4. Génération de la réponse
        5. Stockage de la réponse
        
        Args:
            conversation_id: ID de la conversation
            message_content: Contenu du message utilisateur
            
        Returns:
            Réponse de la conversation
        """
        logger.info(f"Traitement d'un message dans la conversation {conversation_id}")
        
        # Mettre à jour l'activité de la conversation
        await self.conversation_repo.update_last_activity(conversation_id)
        
        # Stocker le message utilisateur
        user_token_count = count_tokens(message_content)
        user_message = await self.message_repo.create_user_message(
            conversation_id=conversation_id,
            content=message_content,
            token_count=user_token_count
        )
        
        # Récupérer le contexte de la conversation
        conversation_messages = await self.message_repo.get_conversation_context(
            conversation_id=conversation_id,
            max_messages=settings.MAX_CONVERSATION_HISTORY
        )
        
        # Classifier l'intention
        intent_classification = await self.intent_classifier.classify_intent(
            query=message_content,
            conversation_context=conversation_messages
        )
        
        intent_type = intent_classification.intent
        logger.info(f"Intention détectée: {intent_type}")
        
        # Construire la requête pour les données
        query_data = await self.query_builder.build_query(intent_classification, message_content)
        
        # Récupérer les données nécessaires
        transaction_data = {}
        account_data = {}
        
        # Exécuter les requêtes en fonction de l'intention
        if intent_type in [IntentType.SEARCH_TRANSACTION, IntentType.ANALYZE_SPENDING]:
            transaction_data = await self.query_builder.execute_transaction_query(query_data)
        
        if intent_type in [IntentType.CHECK_BALANCE, IntentType.ACCOUNT_INFO]:
            account_data = await self.query_builder.execute_account_query(query_data)
        
        # Préparer le contexte pour la génération de réponse
        response_context = {
            "intent": intent_type,
            "query": message_content,
            "intent_classification": intent_classification.dict(),
            "transaction_data": transaction_data,
            "account_data": account_data,
        }
        
        # Générer la réponse
        response_content = await self.response_generator.generate_response(
            query=message_content,
            conversation_context=conversation_messages,
            data_context=response_context
        )
        
        # Compter les tokens de la réponse
        response_token_count = count_tokens(response_content)
        
        # Stocker la réponse de l'assistant
        assistant_message = await self.message_repo.create_assistant_message(
            conversation_id=conversation_id,
            content=response_content,
            token_count=response_token_count,
            meta_data={
                "intent": intent_type,
                "intent_confidence": intent_classification.confidence,
                "entities": intent_classification.entities
            }
        )
        
        # Mettre à jour le titre de la conversation si c'est le premier message
        conversation = await self.conversation_repo.get_conversation_by_id(conversation_id)
        if conversation and not conversation.title:
            # Générer un titre basé sur le contenu du premier message
            title = message_content
            if len(title) > 50:
                title = title[:47] + "..."
            
            await self.conversation_repo.update_conversation(
                conversation_id=conversation_id,
                conversation_update={"title": title}
            )
        
        # Construire la réponse
        return ConversationResponse(
            conversation_id=conversation_id,
            message_id=assistant_message.id,
            content=response_content,
            token_count=response_token_count,
            created_at=assistant_message.created_at,
            intent=intent_type.value,
            meta_data={
                "user_message_id": user_message.id,
                "intent_confidence": intent_classification.confidence,
                "entities": intent_classification.entities
            },
            data={
                "has_transaction_data": bool(transaction_data),
                "has_account_data": bool(account_data)
            }
        )
    
    async def process_message_stream(
        self,
        conversation_id: uuid.UUID,
        message_content: str
    ) -> AsyncGenerator[str, None]:
        """
        Traite un message utilisateur et génère une réponse en streaming.
        
        Version streaming de process_message, qui retourne les morceaux
        de réponse au fur et à mesure qu'ils sont générés.
        
        Args:
            conversation_id: ID de la conversation
            message_content: Contenu du message utilisateur
            
        Yields:
            Morceaux de la réponse générée
        """
        logger.info(f"Traitement d'un message en streaming dans la conversation {conversation_id}")
        
        # Mettre à jour l'activité de la conversation
        await self.conversation_repo.update_last_activity(conversation_id)
        
        # Stocker le message utilisateur
        user_token_count = count_tokens(message_content)
        user_message = await self.message_repo.create_user_message(
            conversation_id=conversation_id,
            content=message_content,
            token_count=user_token_count
        )
        
        # Récupérer le contexte de la conversation
        conversation_messages = await self.message_repo.get_conversation_context(
            conversation_id=conversation_id,
            max_messages=settings.MAX_CONVERSATION_HISTORY
        )
        
        # Classifier l'intention
        intent_classification = await self.intent_classifier.classify_intent(
            query=message_content,
            conversation_context=conversation_messages
        )
        
        intent_type = intent_classification.intent
        logger.info(f"Intention détectée: {intent_type}")
        
        # Construire la requête pour les données
        query_data = await self.query_builder.build_query(intent_classification, message_content)
        
        # Récupérer les données nécessaires
        transaction_data = {}
        account_data = {}
        
        # Exécuter les requêtes en fonction de l'intention
        if intent_type in [IntentType.SEARCH_TRANSACTION, IntentType.ANALYZE_SPENDING]:
            transaction_data = await self.query_builder.execute_transaction_query(query_data)
        
        if intent_type in [IntentType.CHECK_BALANCE, IntentType.ACCOUNT_INFO]:
            account_data = await self.query_builder.execute_account_query(query_data)
        
        # Préparer le contexte pour la génération de réponse
        response_context = {
            "intent": intent_type,
            "query": message_content,
            "intent_classification": intent_classification.dict(),
            "transaction_data": transaction_data,
            "account_data": account_data,
        }
        
        # Créer un message assistant vide qui sera rempli progressivement
        assistant_message = await self.message_repo.create_assistant_message(
            conversation_id=conversation_id,
            content="",  # Contenu vide pour commencer
            token_count=0,
            meta_data={
                "intent": intent_type,
                "intent_confidence": intent_classification.confidence,
                "entities": intent_classification.entities,
                "streaming": True
            }
        )
        
        # Générer la réponse en streaming
        collected_content = ""
        async for content_chunk in self.response_generator.generate_response_stream(
            query=message_content,
            conversation_context=conversation_messages,
            data_context=response_context
        ):
            # Accumuler le contenu
            collected_content += content_chunk
            
            # Retourner le morceau au client
            yield content_chunk
        
        # Mettre à jour le message assistant avec le contenu complet
        response_token_count = count_tokens(collected_content)
        await self.message_repo.update_token_count(assistant_message.id, response_token_count)
        
        # Mettre à jour le contenu du message
        # Note: Dans une implémentation complète, il faudrait une méthode spécifique pour cela
        assistant_message.content = collected_content
        self.db.add(assistant_message)
        self.db.commit()
        
        # Mettre à jour le titre de la conversation si c'est le premier message
        conversation = await self.conversation_repo.get_conversation_by_id(conversation_id)
        if conversation and not conversation.title:
            # Générer un titre basé sur le contenu du premier message
            title = message_content
            if len(title) > 50:
                title = title[:47] + "..."
            
            await self.conversation_repo.update_conversation(
                conversation_id=conversation_id,
                conversation_update={"title": title}
            )