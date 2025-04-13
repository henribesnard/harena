"""
Génération des réponses à partir des données financières.

Ce module est responsable de la génération de réponses en langage
naturel basées sur les données financières récupérées.
"""

import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, date

from ..config.settings import settings
from ..config.logging import get_logger
from ..config.constants import ERROR_MESSAGES
from ..llm.llm_service import LLMService
from ..llm.prompt_templates import get_response_generation_prompt

logger = get_logger(__name__)


class ResponseGenerator:
    """
    Générateur de réponses en langage naturel.
    
    Cette classe transforme les données financières en réponses
    naturelles et informatives pour l'utilisateur.
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialise le générateur de réponses.
        
        Args:
            llm_service: Service LLM pour la génération de texte
        """
        self.llm_service = llm_service
        logger.info("Générateur de réponses initialisé")
    
    async def generate_response(
        self,
        query: str,
        conversation_context: List[Dict[str, str]],
        data_context: Dict[str, Any]
    ) -> str:
        """
        Génère une réponse basée sur le contexte et les données.
        
        Args:
            query: Requête de l'utilisateur
            conversation_context: Contexte de la conversation
            data_context: Données financières et contexte
            
        Returns:
            Réponse générée
        """
        logger.info("Génération d'une réponse")
        
        try:
            # Vérifier si des erreurs se sont produites lors de la récupération des données
            if "error" in data_context:
                error_type = data_context.get("error", "generic_error")
                error_message = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["generic_error"])
                return error_message
            
            # Construire le prompt pour la génération de réponse
            response_prompt = get_response_generation_prompt(data_context)
            
            # Préparer les messages pour le LLM
            messages = conversation_context.copy()
            
            # Vérifier si le premier message est un message système
            has_system_message = messages and messages[0].get("role") == "system"
            
            # Si pas de message système, ajouter le prompt de réponse comme message système
            if not has_system_message:
                messages.insert(0, {"role": "system", "content": response_prompt})
            
            # Générer la réponse
            response_content = await self.llm_service.generate_response(
                messages=messages,
                temperature=0.7,
                max_tokens=settings.DEEPSEEK_MAX_TOKENS,
                stream=False
            )
            
            logger.info("Réponse générée avec succès")
            return response_content
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {str(e)}")
            return ERROR_MESSAGES["generic_error"]
    
    async def generate_response_stream(
        self,
        query: str,
        conversation_context: List[Dict[str, str]],
        data_context: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Génère une réponse en streaming basée sur le contexte et les données.
        
        Args:
            query: Requête de l'utilisateur
            conversation_context: Contexte de la conversation
            data_context: Données financières et contexte
            
        Yields:
            Morceaux de la réponse générée
        """
        logger.info("Génération d'une réponse en streaming")
        
        try:
            # Vérifier si des erreurs se sont produites lors de la récupération des données
            if "error" in data_context:
                error_type = data_context.get("error", "generic_error")
                error_message = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["generic_error"])
                yield error_message
                return
            
            # Construire le prompt pour la génération de réponse
            response_prompt = get_response_generation_prompt(data_context)
            
            # Préparer les messages pour le LLM
            messages = conversation_context.copy()
            
            # Vérifier si le premier message est un message système
            has_system_message = messages and messages[0].get("role") == "system"
            
            # Si pas de message système, ajouter le prompt de réponse comme message système
            if not has_system_message:
                messages.insert(0, {"role": "system", "content": response_prompt})
            
            # Générer la réponse en streaming
            async for content_chunk in self.llm_service.generate_response(
                messages=messages,
                temperature=0.7,
                max_tokens=settings.DEEPSEEK_MAX_TOKENS,
                stream=True
            ):
                yield content_chunk
            
            logger.info("Réponse en streaming générée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse en streaming: {str(e)}")
            yield ERROR_MESSAGES["generic_error"]