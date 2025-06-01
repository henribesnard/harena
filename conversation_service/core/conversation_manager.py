"""
Gestionnaire principal des conversations.

Ce module orchestre le flux complet de traitement d'une conversation :
détection d'intention, recherche, génération de réponse.
"""
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx

from conversation_service.models import (
    ConversationRequest, ConversationResponse, StreamChunk, 
    DetectedIntent, Message, MessageRole, IntentType
)
from conversation_service.core.deepseek_client import deepseek_client
from conversation_service.core.intent_detection import intent_detector
from conversation_service.core.query_formatter import query_formatter
from conversation_service.storage.conversation_store import conversation_store
from conversation_service.utils.token_counter import token_counter
from config_service.config import settings

logger = logging.getLogger(__name__)


class ConversationManager:
    """Gestionnaire principal des conversations."""
    
    def __init__(self):
        self.search_service_url = settings.SYNC_SERVICE_URL or "http://localhost:8004"
        self.max_search_retries = 3
        self.search_timeout = 30.0
        
    async def process_conversation(
        self,
        request: ConversationRequest,
        stream: bool = True
    ) -> ConversationResponse:
        """
        Traite une requête de conversation complète.
        
        Args:
            request: Requête de conversation
            stream: Mode streaming (pour compatibilité, mais retourne une réponse complète)
            
        Returns:
            ConversationResponse: Réponse complète
        """
        start_time = time.time()
        
        try:
            # 1. Obtenir ou créer la conversation
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # 2. Récupérer l'historique
            conversation_history = await conversation_store.get_conversation_messages(
                conversation_id, request.user_id, limit=10
            )
            
            # 3. Détecter l'intention
            intent = await intent_detector.detect_intent(
                request.message,
                conversation_history,
                request.context
            )
            
            # 4. Sauvegarder le message utilisateur
            user_message = Message(
                role=MessageRole.USER,
                content=request.message,
                metadata={"intent": intent.dict()}
            )
            
            await conversation_store.add_message(
                conversation_id, request.user_id, user_message
            )
            
            # 5. Effectuer une recherche si nécessaire
            search_results = None
            if self._needs_search(intent):
                search_results = await self._perform_search(
                    intent, request.user_id, request.context
                )
            
            # 6. Générer la réponse
            response_data = await deepseek_client.generate_contextual_response(
                intent.dict(),
                search_results,
                conversation_history,
                request.context
            )
            
            # 7. Sauvegarder la réponse
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response_data["content"],
                metadata={
                    "intent": intent.dict(),
                    "search_results": search_results,
                    "token_usage": response_data.get("usage", {})
                }
            )
            
            await conversation_store.add_message(
                conversation_id, request.user_id, assistant_message
            )
            
            # 8. Enregistrer les tokens
            if "usage" in response_data:
                token_counter.record_usage(
                    user_id=request.user_id,
                    input_tokens=response_data["usage"].get("prompt_tokens", 0),
                    output_tokens=response_data["usage"].get("completion_tokens", 0),
                    model=response_data.get("model", "unknown")
                )
            
            # 9. Construire la réponse
            processing_time = time.time() - start_time
            
            return ConversationResponse(
                conversation_id=conversation_id,
                message_id=assistant_message.id,
                content=response_data["content"],
                intent=intent,
                search_results=search_results,
                processing_time=processing_time,
                token_usage=response_data.get("usage"),
                metadata={
                    "model_used": response_data.get("model"),
                    "search_performed": search_results is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de conversation: {e}", exc_info=True)
            
            # Réponse d'erreur
            error_intent = DetectedIntent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                parameters={},
                reasoning=f"Erreur: {str(e)}"
            )
            
            return ConversationResponse(
                conversation_id=request.conversation_id or str(uuid.uuid4()),
                message_id=str(uuid.uuid4()),
                content="Je rencontre une difficulté pour traiter votre demande. Pouvez-vous reformuler votre question ?",
                intent=error_intent,
                search_results=None,
                processing_time=time.time() - start_time,
                token_usage=None,
                metadata={"error": str(e)}
            )
    
    async def process_conversation_stream(
        self,
        request: ConversationRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Traite une conversation en mode streaming.
        
        Args:
            request: Requête de conversation
            
        Yields:
            StreamChunk: Chunks de réponse en streaming
        """
        start_time = time.time()
        
        try:
            # 1. Obtenir ou créer la conversation
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            yield StreamChunk(
                type="metadata",
                data={"conversation_id": conversation_id, "status": "processing"}
            )
            
            # 2. Récupérer l'historique
            conversation_history = await conversation_store.get_conversation_messages(
                conversation_id, request.user_id, limit=10
            )
            
            # 3. Détecter l'intention
            yield StreamChunk(type="metadata", data={"status": "detecting_intent"})
            
            intent = await intent_detector.detect_intent(
                request.message,
                conversation_history,
                request.context
            )
            
            yield StreamChunk(
                type="intent",
                data=intent.dict(),
                metadata={"confidence": intent.confidence}
            )
            
            # 4. Sauvegarder le message utilisateur
            user_message = Message(
                role=MessageRole.USER,
                content=request.message,
                metadata={"intent": intent.dict()}
            )
            
            await conversation_store.add_message(
                conversation_id, request.user_id, user_message
            )
            
            # 5. Effectuer une recherche si nécessaire
            search_results = None
            if self._needs_search(intent):
                yield StreamChunk(type="metadata", data={"status": "searching"})
                
                search_results = await self._perform_search(
                    intent, request.user_id, request.context
                )
                
                if search_results:
                    yield StreamChunk(
                        type="search_results",
                        data={"results": search_results, "count": len(search_results)}
                    )
            
            # 6. Générer la réponse en streaming
            yield StreamChunk(type="metadata", data={"status": "generating_response"})
            
            response_content = ""
            token_count = 0
            
            async for content_chunk in deepseek_client.generate_contextual_response_stream(
                intent.dict(),
                search_results,
                conversation_history,
                request.context
            ):
                response_content += content_chunk
                token_count += len(content_chunk.split())  # Estimation simple
                
                yield StreamChunk(
                    type="content",
                    content=content_chunk,
                    metadata={"token_count": token_count}
                )
            
            # 7. Finaliser
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response_content,
                metadata={
                    "intent": intent.dict(),
                    "search_results": search_results,
                    "estimated_tokens": token_count
                }
            )
            
            await conversation_store.add_message(
                conversation_id, request.user_id, assistant_message
            )
            
            # 8. Chunk final
            processing_time = time.time() - start_time
            
            yield StreamChunk(
                type="done",
                data={
                    "conversation_id": conversation_id,
                    "message_id": assistant_message.id,
                    "processing_time": processing_time,
                    "total_tokens": token_count
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur lors du streaming de conversation: {e}", exc_info=True)
            
            yield StreamChunk(
                type="error",
                data={
                    "error_code": "PROCESSING_ERROR",
                    "message": "Une erreur est survenue lors du traitement",
                    "details": str(e)
                }
            )
    
    def _needs_search(self, intent: DetectedIntent) -> bool:
        """
        Détermine si une recherche est nécessaire selon l'intention.
        
        Args:
            intent: Intention détectée
            
        Returns:
            bool: True si une recherche est nécessaire
        """
        search_intents = {
            IntentType.SEARCH_TRANSACTIONS,
            IntentType.SPENDING_ANALYSIS,
            IntentType.CATEGORY_ANALYSIS,
            IntentType.MERCHANT_ANALYSIS,
            IntentType.TIME_ANALYSIS,
            IntentType.COMPARISON,
            IntentType.ACCOUNT_SUMMARY
        }
        
        return intent.intent_type in search_intents
    
    async def _perform_search(
        self,
        intent: DetectedIntent,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Effectue une recherche via le search_service.
        
        Args:
            intent: Intention détectée
            user_id: ID de l'utilisateur
            context: Contexte additionnel
            
        Returns:
            List: Résultats de recherche ou None
        """
        try:
            # Formater la requête de recherche
            search_query = query_formatter.format_search_query(intent, user_id, context)
            
            if not search_query:
                logger.warning("Impossible de formater la requête de recherche")
                return None
            
            # Appeler le search_service
            async with httpx.AsyncClient(timeout=self.search_timeout) as client:
                response = await client.post(
                    f"{self.search_service_url}/api/v1/search",
                    json=search_query,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    search_response = response.json()
                    return search_response.get("results", [])
                else:
                    logger.error(f"Erreur search_service: {response.status_code} - {response.text}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error("Timeout lors de l'appel au search_service")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return None
    
    async def get_conversation_summary(
        self,
        conversation_id: str,
        user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Génère un résumé d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Résumé de la conversation
        """
        try:
            # Récupérer tous les messages
            messages = await conversation_store.get_conversation_messages(
                conversation_id, user_id, limit=100
            )
            
            if not messages:
                return None
            
            # Extraire les intentions principales
            intents = []
            for msg in messages:
                if msg.get("metadata", {}).get("intent"):
                    intents.append(msg["metadata"]["intent"]["intent_type"])
            
            # Calculer les statistiques
            user_messages = [m for m in messages if m.get("role") == "user"]
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            
            # Générer un titre basé sur les intentions
            title = self._generate_conversation_title(intents, user_messages)
            
            return {
                "conversation_id": conversation_id,
                "title": title,
                "message_count": len(messages),
                "user_message_count": len(user_messages),
                "assistant_message_count": len(assistant_messages),
                "main_intents": list(set(intents)),
                "first_message": user_messages[0]["content"] if user_messages else "",
                "last_message": messages[-1]["content"] if messages else "",
                "created_at": messages[0].get("timestamp") if messages else None,
                "updated_at": messages[-1].get("timestamp") if messages else None
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé: {e}")
            return None
    
    def _generate_conversation_title(
        self,
        intents: List[str],
        user_messages: List[Dict[str, Any]]
    ) -> str:
        """
        Génère un titre pour la conversation basé sur les intentions.
        
        Args:
            intents: Liste des intentions détectées
            user_messages: Messages de l'utilisateur
            
        Returns:
            str: Titre généré
        """
        if not intents:
            return "Conversation générale"
        
        # Titres par intention principale
        intent_titles = {
            "search_transactions": "Recherche de transactions",
            "spending_analysis": "Analyse des dépenses",
            "account_summary": "Résumé des comptes",
            "category_analysis": "Analyse par catégorie",
            "merchant_analysis": "Analyse des marchands",
            "budget_inquiry": "Questions budget",
            "greeting": "Conversation générale",
            "help": "Aide et support"
        }
        
        # Prendre l'intention la plus fréquente
        from collections import Counter
        most_common_intent = Counter(intents).most_common(1)
        
        if most_common_intent:
            intent_type = most_common_intent[0][0]
            base_title = intent_titles.get(intent_type, "Conversation")
            
            # Ajouter du contexte si possible
            if user_messages and len(user_messages[0]["content"]) < 50:
                return f"{base_title}: {user_messages[0]['content'][:30]}..."
            
            return base_title
        
        return "Conversation"
    
    async def cleanup_old_conversations(
        self,
        user_id: int,
        days_threshold: int = 30
    ) -> int:
        """
        Nettoie les anciennes conversations d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days_threshold: Seuil en jours pour considérer une conversation comme ancienne
            
        Returns:
            int: Nombre de conversations nettoyées
        """
        try:
            return await conversation_store.cleanup_old_conversations(
                user_id, days_threshold
            )
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des conversations: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire."""
        return {
            "search_service_url": self.search_service_url,
            "max_search_retries": self.max_search_retries,
            "search_timeout": self.search_timeout
        }


# Instance globale
conversation_manager = ConversationManager()