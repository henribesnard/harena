"""
Client DeepSeek pour la génération de réponses et le raisonnement.

Ce module gère les interactions avec l'API DeepSeek pour la génération
de réponses conversationnelles et la détection d'intention.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config_service.config import settings
from conversation_service.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Client pour interagir avec l'API DeepSeek."""
    
    def __init__(self):
        self.client = None
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.api_key = settings.DEEPSEEK_API_KEY
        self.chat_model = settings.DEEPSEEK_CHAT_MODEL
        self.reasoner_model = settings.DEEPSEEK_REASONER_MODEL
        self.max_tokens = settings.DEEPSEEK_MAX_TOKENS
        self.temperature = settings.DEEPSEEK_TEMPERATURE
        self.top_p = settings.DEEPSEEK_TOP_P
        self.timeout = settings.DEEPSEEK_TIMEOUT
        self._initialized = False
        
    async def initialize(self):
        """Initialise le client DeepSeek."""
        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY non définie")
            raise ValueError("DeepSeek API key is required")
        
        # Créer le client HTTP
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.timeout
        )
        
        # Tester la connexion
        try:
            await self._test_connection()
            self._initialized = True
            logger.info(f"DeepSeekClient initialisé avec succès")
            logger.info(f"Chat model: {self.chat_model}, Reasoner model: {self.reasoner_model}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation DeepSeek: {e}")
            await self.client.aclose()
            self.client = None
            raise
    
    async def _test_connection(self):
        """Teste la connexion à l'API DeepSeek."""
        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10,
                    "temperature": 0.1
                }
            )
            response.raise_for_status()
            logger.debug("Connexion DeepSeek testée avec succès")
        except Exception as e:
            logger.error(f"Test de connexion DeepSeek échoué: {e}")
            raise
    
    def is_initialized(self) -> bool:
        """Vérifie si le client est initialisé."""
        return self._initialized and self.client is not None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Génère une réponse via l'API DeepSeek Chat.
        
        Args:
            messages: Liste des messages de conversation
            model: Modèle à utiliser (défaut: chat_model)
            stream: Mode streaming
            **kwargs: Paramètres additionnels
            
        Returns:
            Dict: Réponse de l'API
        """
        if not self.client:
            raise ValueError("DeepSeekClient not initialized")
        
        # Paramètres par défaut
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": stream
        }
        
        # Ajouter les paramètres supplémentaires
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature", "top_p"]:
                payload[key] = value
        
        try:
            logger.debug(f"Requête DeepSeek: {len(messages)} messages, stream={stream}")
            
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            if stream:
                # Mode streaming non géré dans cette méthode
                raise ValueError("Use generate_response_stream for streaming")
            
            result = response.json()
            logger.debug(f"Réponse DeepSeek reçue: {result.get('usage', {})}")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP DeepSeek: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {e}")
            raise
    
    async def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Génère une réponse en streaming via l'API DeepSeek.
        
        Args:
            messages: Liste des messages de conversation
            model: Modèle à utiliser
            **kwargs: Paramètres additionnels
            
        Yields:
            Dict: Chunks de réponse en streaming
        """
        if not self.client:
            raise ValueError("DeepSeekClient not initialized")
        
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": True
        }
        
        # Ajouter les paramètres supplémentaires
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature", "top_p"]:
                payload[key] = value
        
        try:
            logger.debug(f"Streaming DeepSeek: {len(messages)} messages")
            
            async with self.client.stream(
                "POST", 
                "/chat/completions", 
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Supprimer "data: "
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = eval(data)  # Note: en production, utiliser json.loads
                            yield chunk
                        except Exception as e:
                            logger.warning(f"Erreur parsing chunk: {e}")
                            continue
                            
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP streaming DeepSeek: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du streaming: {e}")
            raise
    
    async def detect_intent(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Détecte l'intention d'un message utilisateur avec DeepSeek Reasoner.
        
        Args:
            user_message: Message de l'utilisateur
            conversation_history: Historique de conversation
            
        Returns:
            Dict: Intention détectée avec paramètres
        """
        # Construire le prompt de détection d'intention
        from conversation_service.prompts.intent_detection import get_intent_detection_prompt
        
        system_prompt = get_intent_detection_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Message utilisateur: {user_message}"}
        ]
        
        # Ajouter l'historique si disponible
        if conversation_history:
            context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in conversation_history[-5:]  # Derniers 5 messages
            ])
            messages.append({
                "role": "user", 
                "content": f"Contexte de conversation:\n{context}"
            })
        
        try:
            # Utiliser le modèle reasoner pour la détection d'intention
            response = await self.generate_response(
                messages=messages,
                model=self.reasoner_model,
                temperature=0.1,  # Plus déterministe pour la détection
                max_tokens=1000
            )
            
            # Extraire la réponse
            content = response["choices"][0]["message"]["content"]
            
            # Parser la réponse structurée
            return self._parse_intent_response(content)
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'intention: {e}")
            # Retourner une intention par défaut
            return {
                "intent_type": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "reasoning": f"Erreur lors de la détection: {str(e)}"
            }
    
    def _parse_intent_response(self, content: str) -> Dict[str, Any]:
        """
        Parse la réponse de détection d'intention.
        
        Args:
            content: Contenu de la réponse
            
        Returns:
            Dict: Intention parsée
        """
        try:
            # Rechercher les balises JSON dans la réponse
            import re
            import json
            
            # Chercher un bloc JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                intent_data = json.loads(json_str)
                
                # Valider les champs requis
                intent_type = intent_data.get("intent_type", "unknown")
                confidence = float(intent_data.get("confidence", 0.5))
                parameters = intent_data.get("parameters", {})
                reasoning = intent_data.get("reasoning", content)
                
                return {
                    "intent_type": intent_type,
                    "confidence": min(max(confidence, 0.0), 1.0),  # Clamp entre 0 et 1
                    "parameters": parameters,
                    "reasoning": reasoning
                }
            
            # Si pas de JSON, essayer de parser manuellement
            return self._manual_intent_parsing(content)
            
        except Exception as e:
            logger.warning(f"Erreur parsing intention: {e}")
            return {
                "intent_type": "unknown",
                "confidence": 0.5,
                "parameters": {},
                "reasoning": content
            }
    
    def _manual_intent_parsing(self, content: str) -> Dict[str, Any]:
        """Parse manuel pour les réponses non-JSON."""
        content_lower = content.lower()
        
        # Détection basique par mots-clés
        if any(word in content_lower for word in ["recherche", "trouver", "transaction"]):
            return {
                "intent_type": "search_transactions",
                "confidence": 0.7,
                "parameters": {},
                "reasoning": "Détection par mots-clés: recherche"
            }
        elif any(word in content_lower for word in ["dépense", "combien", "total"]):
            return {
                "intent_type": "spending_analysis",
                "confidence": 0.7,
                "parameters": {},
                "reasoning": "Détection par mots-clés: analyse des dépenses"
            }
        elif any(word in content_lower for word in ["bonjour", "salut", "hello"]):
            return {
                "intent_type": "greeting",
                "confidence": 0.9,
                "parameters": {},
                "reasoning": "Détection par mots-clés: salutation"
            }
        else:
            return {
                "intent_type": "general_question",
                "confidence": 0.5,
                "parameters": {},
                "reasoning": "Détection par défaut"
            }
    
    async def generate_contextual_response(
        self,
        intent: Dict[str, Any],
        search_results: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Génère une réponse contextuelle basée sur l'intention et les résultats.
        
        Args:
            intent: Intention détectée
            search_results: Résultats de recherche
            conversation_history: Historique de conversation
            user_context: Contexte utilisateur
            
        Returns:
            Dict: Réponse générée
        """
        from conversation_service.prompts.response_generation import get_response_prompt
        
        # Construire le prompt de génération
        system_prompt = get_response_prompt(intent["intent_type"])
        
        # Construire le message utilisateur avec contexte
        user_prompt_parts = [
            f"Intention: {intent['intent_type']}",
            f"Paramètres: {intent['parameters']}"
        ]
        
        if search_results:
            user_prompt_parts.append(f"Résultats de recherche: {search_results}")
        
        if user_context:
            user_prompt_parts.append(f"Contexte utilisateur: {user_context}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_parts)}
        ]
        
        # Ajouter l'historique récent
        if conversation_history:
            for msg in conversation_history[-3:]:  # 3 derniers messages
                messages.append(msg)
        
        try:
            response = await self.generate_response(
                messages=messages,
                temperature=0.7,  # Plus créatif pour les réponses
                max_tokens=2000
            )
            
            return {
                "content": response["choices"][0]["message"]["content"],
                "usage": response.get("usage", {}),
                "model": response.get("model", self.chat_model)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse contextuelle: {e}")
            return {
                "content": "Je rencontre une difficulté pour traiter votre demande. Pouvez-vous reformuler ?",
                "usage": {},
                "model": self.chat_model,
                "error": str(e)
            }
    
    async def generate_contextual_response_stream(
        self,
        intent: Dict[str, Any],
        search_results: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Génère une réponse contextuelle en streaming.
        
        Args:
            intent: Intention détectée
            search_results: Résultats de recherche
            conversation_history: Historique de conversation
            user_context: Contexte utilisateur
            
        Yields:
            str: Chunks de texte de la réponse
        """
        from conversation_service.prompts.response_generation import get_response_prompt
        
        # Construire le prompt (même logique que generate_contextual_response)
        system_prompt = get_response_prompt(intent["intent_type"])
        
        user_prompt_parts = [
            f"Intention: {intent['intent_type']}",
            f"Paramètres: {intent['parameters']}"
        ]
        
        if search_results:
            user_prompt_parts.append(f"Résultats de recherche: {search_results}")
        
        if user_context:
            user_prompt_parts.append(f"Contexte utilisateur: {user_context}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_parts)}
        ]
        
        if conversation_history:
            for msg in conversation_history[-3:]:
                messages.append(msg)
        
        try:
            async for chunk in self.generate_response_stream(
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            ):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
                        
        except Exception as e:
            logger.error(f"Erreur streaming réponse contextuelle: {e}")
            yield "Je rencontre une difficulté pour traiter votre demande."
    
    async def close(self):
        """Ferme le client DeepSeek."""
        if self.client:
            await self.client.aclose()
            self.client = None
            self._initialized = False
            logger.info("DeepSeekClient fermé")


# Instance globale
deepseek_client = DeepSeekClient()