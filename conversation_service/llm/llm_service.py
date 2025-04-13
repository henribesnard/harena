"""
Service d'interfaçage avec les modèles Deepseek.

Ce module fournit une interface unifiée pour interagir avec les
modèles de langage Deepseek, en gérant les appels API synchrones
et le streaming.
"""

import json
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, Union
from openai import AsyncOpenAI

from ..config.settings import settings
from ..config.logging import get_logger
from ..utils.token_counter import count_tokens
from .prompt_templates import load_system_prompt

logger = get_logger(__name__)


class LLMService:
    """
    Service pour interagir avec les modèles Deepseek.
    
    Cette classe fournit une interface commune pour appeler
    les modèles Deepseek avec différentes configurations et
    pour gérer les flux de streaming.
    """
    
    def __init__(self):
        """Initialise le service LLM avec la configuration de Deepseek."""
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        self.model = settings.DEEPSEEK_MODEL
        self.default_system_prompt = load_system_prompt()
        
        logger.info(f"Service LLM initialisé avec le modèle: {self.model}")
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        top_p: Optional[float] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Génère une réponse à partir d'une liste de messages.
        
        Args:
            messages: Liste des messages de la conversation
            temperature: Température de génération (0-2)
            max_tokens: Nombre maximal de tokens à générer
            system_prompt: Prompt système personnalisé
            stream: Activer le streaming des réponses
            top_p: Valeur top_p pour la génération (0-1)
            
        Returns:
            Réponse du modèle ou générateur asynchrone de morceaux
        """
        # Préparer les paramètres
        temperature = temperature if temperature is not None else settings.DEEPSEEK_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else settings.DEEPSEEK_MAX_TOKENS
        top_p = top_p if top_p is not None else settings.DEEPSEEK_TOP_P
        
        # Ajouter un message système si nécessaire
        full_messages = []
        
        # Vérifier si le premier message est déjà un message système
        has_system_message = messages and messages[0].get("role") == "system"
        
        # Ajouter un message système si nécessaire
        if not has_system_message:
            system_prompt = system_prompt or self.default_system_prompt
            full_messages.append({"role": "system", "content": system_prompt})
        
        # Ajouter les messages existants
        full_messages.extend(messages)
        
        try:
            # Compter les tokens (pour la surveillance)
            input_tokens = sum(count_tokens(msg.get("content", "")) for msg in full_messages)
            
            # Journaliser l'appel à l'API
            logger.info(
                f"Appel à l'API Deepseek: model={self.model}, temperature={temperature}, "
                f"input_tokens={input_tokens}, stream={stream}"
            )
            
            if stream:
                # Retourner un générateur asynchrone pour le streaming
                return self._stream_response(
                    full_messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
            else:
                # Appel synchrone
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=False
                )
                
                # Extraire le contenu de la réponse
                content = response.choices[0].message.content
                
                # Compter les tokens de sortie (pour la surveillance)
                output_tokens = count_tokens(content)
                
                # Journaliser la réponse
                logger.info(
                    f"Réponse de l'API Deepseek reçue: output_tokens={output_tokens}, "
                    f"request_id={response.id}"
                )
                
                return content
                
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à Deepseek: {str(e)}")
            raise
    
    async def _stream_response(
        self,
        full_messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> AsyncGenerator[str, None]:
        """
        Génère une réponse en streaming.
        
        Args:
            full_messages: Messages complets incluant le système
            temperature: Température de génération
            max_tokens: Nombre maximal de tokens
            top_p: Valeur top_p pour la génération
            
        Yields:
            Morceaux de la réponse au fur et à mesure qu'ils sont générés
        """
        try:
            # Appel à l'API en mode streaming
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True
            )
            
            # Variables pour accumuler la réponse
            collected_chunks = []
            collected_content = ""
            output_tokens = 0
            
            # Traiter chaque morceau de la réponse
            async for chunk in stream:
                # Extraire le contenu du morceau
                content_delta = chunk.choices[0].delta.content or ""
                collected_chunks.append(content_delta)
                collected_content += content_delta
                
                # Compter les tokens (approximativement)
                output_tokens = count_tokens(collected_content)
                
                # Fournir le morceau au client
                yield content_delta
            
            # Journaliser la fin du streaming
            logger.info(
                f"Streaming terminé: {len(collected_chunks)} morceaux, "
                f"output_tokens={output_tokens}"
            )
        
        except Exception as e:
            logger.error(f"Erreur lors du streaming Deepseek: {str(e)}")
            raise
    
    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classifie l'intention d'une requête utilisateur.
        
        Args:
            query: Requête utilisateur à classifier
            
        Returns:
            Dictionnaire contenant l'intention classifiée et les métadonnées
        """
        # Charger le prompt de classification d'intention
        system_prompt = """Vous êtes un assistant spécialisé dans l'analyse des requêtes financières.
Votre tâche est de classifier l'intention de l'utilisateur et d'extraire les entités clés.
Répondez au format JSON avec les champs suivants:
{
  "intent": "NOM_DE_L_INTENTION",
  "confidence": 0.XX,
  "entities": {
    "entity1": "valeur1",
    "entity2": "valeur2"
  }
}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classifiez l'intention suivante: {query}"}
        ]
        
        # Appeler le modèle avec une température basse pour des résultats plus déterministes
        response = await self.generate_response(
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
            stream=False
        )
        
        # Extraire le JSON de la réponse
        try:
            # Nettoyer la réponse pour extraire uniquement le JSON
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            
            json_str = json_str.strip()
            intent_data = json.loads(json_str)
            
            logger.info(f"Intention classifiée: {intent_data.get('intent')}")
            return intent_data
        
        except json.JSONDecodeError:
            logger.error(f"Erreur de décodage JSON de la classification: {response}")
            # Retourner une classification par défaut en cas d'erreur
            return {
                "intent": "GENERAL_QUERY",
                "confidence": 0.5,
                "entities": {}
            }