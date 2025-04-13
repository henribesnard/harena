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