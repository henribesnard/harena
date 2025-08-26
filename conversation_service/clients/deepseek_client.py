"""
Client DeepSeek optimisé pour conversation service - Phase 1 JSON Output Forcé
"""
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.deepseek")


class DeepSeekError(Exception):
    """Exception spécifique DeepSeek avec contexte détaillé"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class DeepSeekClient:
    """
    Client DeepSeek avec JSON Output forcé - Phase 1
    
    Features:
    - JSON Output obligatoire pour éviter parsing regex
    - Retry automatique avec backoff exponentiel 
    - Gestion fine des erreurs HTTP
    - Métriques détaillées performance
    - Session persistante optimisée
    - Health check intégré
    """
    
    def __init__(self):
        # Configuration API
        self.api_url = getattr(settings, 'DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self.api_key = settings.DEEPSEEK_API_KEY
        self.model_chat = getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat')
        self.max_tokens = getattr(settings, 'DEEPSEEK_MAX_TOKENS', 8192)
        self.temperature = getattr(settings, 'DEEPSEEK_TEMPERATURE', 1.0)
        self.timeout = getattr(settings, 'DEEPSEEK_TIMEOUT', 60)
        
        # État client
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self._request_count = 0
        self._total_tokens = 0
        
        logger.info(f"DeepSeek client configuré - Model: {self.model_chat}, Max tokens: {self.max_tokens}")
    
    async def initialize(self) -> None:
        """Initialisation client avec session aiohttp optimisée"""
        if self._initialized:
            return
        
        if not self.api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY non configuré")
        
        # Connector optimisé pour performances
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Harena-ConversationService/1.0"
            }
        )
        
        self._initialized = True
        logger.info("DeepSeek client initialisé avec succès")
    
    async def close(self) -> None:
        """Fermeture propre du client avec logging métriques"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        
        self._initialized = False
        
        logger.info(
            f"DeepSeek client fermé - Requests: {self._request_count}, "
            f"Total tokens: {self._total_tokens}"
        )
    
    async def health_check(self) -> bool:
        """
        Vérification santé API DeepSeek avec test JSON Output
        
        Returns:
            bool: True si API opérationnelle et JSON Output fonctionne
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Test minimal avec JSON Output forcé
            test_response = await self.chat_completion(
                messages=[{"role": "user", "content": "Test health check - respond with JSON"}],
                max_tokens=50,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Vérification que la réponse est bien du JSON
            content = test_response["choices"][0]["message"]["content"]
            import json
            json.loads(content)  # Validation JSON
            
            return True
            
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {str(e)}")
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Appel API chat completion avec support JSON output forcé
        
        Args:
            messages: Messages conversation format OpenAI
            max_tokens: Limite tokens réponse
            temperature: Température génération (0.0-2.0)
            model: Modèle DeepSeek à utiliser
            response_format: Format de réponse ({"type": "json_object"} pour JSON)
            
        Returns:
            Dict: Réponse API DeepSeek
            
        Raises:
            DeepSeekError: Erreurs API ou réseau
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._session:
            raise DeepSeekError("Session non initialisée")
        
        if not messages:
            raise DeepSeekError("Messages ne peuvent pas être vides")
        
        # Construction payload avec JSON Output par défaut
        payload = {
            "model": model or self.model_chat,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stream": False
        }
        
        # Force JSON Output si spécifié (recommandé pour Phase 1)
        if response_format:
            payload["response_format"] = response_format
            logger.debug(f"JSON output forcé: {response_format}")
        
        # Retry avec backoff exponentiel
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = datetime.now(timezone.utc)
                
                async with self._session.post(
                    f"{self.api_url}/chat/completions",
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Validation structure de base
                        if not result.get("choices") or len(result["choices"]) == 0:
                            raise DeepSeekError("Réponse API invalide: pas de choices")
                        
                        # Validation JSON Output si demandé
                        if response_format and response_format.get("type") == "json_object":
                            content = result["choices"][0]["message"]["content"]
                            if not self._validate_json_output(content):
                                raise DeepSeekError("JSON Output demandé mais réponse non-JSON reçue")
                        
                        # Métriques performance
                        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                        tokens_used = result.get("usage", {}).get("total_tokens", 0)
                        
                        self._request_count += 1
                        self._total_tokens += tokens_used
                        
                        logger.info(
                            f"DeepSeek success - Duration: {duration_ms}ms, "
                            f"Tokens: {tokens_used}, JSON: {bool(response_format)}, "
                            f"Attempt: {attempt + 1}/{max_retries}"
                        )
                        
                        return result
                    
                    # Gestion erreurs HTTP spécifiques
                    elif response.status == 400:
                        error_text = await response.text()
                        raise DeepSeekError(f"Requête invalide (400): {error_text}", status_code=400)
                    
                    elif response.status == 401:
                        raise DeepSeekError("API Key invalide (401)", status_code=401)
                    
                    elif response.status == 429:
                        # Rate limiting avec respect du Retry-After
                        retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                        logger.warning(f"Rate limited (429), retry après {retry_after}s")
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            raise DeepSeekError(
                                f"Rate limit persistant après {max_retries} tentatives",
                                status_code=429,
                                retry_after=retry_after
                            )
                    
                    elif response.status >= 500:
                        # Erreur serveur - retry automatique
                        error_text = await response.text()
                        logger.warning(f"Erreur serveur {response.status}: {error_text}, tentative {attempt + 1}")
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(base_delay * (2 ** attempt))
                            continue
                        else:
                            raise DeepSeekError(
                                f"Erreur serveur persistante ({response.status}): {error_text}",
                                status_code=response.status
                            )
                    
                    else:
                        # Autres erreurs HTTP
                        error_text = await response.text()
                        raise DeepSeekError(
                            f"Erreur HTTP {response.status}: {error_text}",
                            status_code=response.status
                        )
            
            except aiohttp.ClientError as e:
                logger.error(f"Erreur connexion DeepSeek (tentative {attempt + 1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise DeepSeekError(
                        f"Connexion échouée après {max_retries} tentatives: {str(e)}"
                    )
            
            except Exception as e:
                logger.error(f"Erreur inattendue DeepSeek API: {str(e)}")
                raise DeepSeekError(f"Erreur inattendue: {str(e)}")
        
        # Ne devrait jamais arriver
        raise DeepSeekError(f"Échec après {max_retries} tentatives")
    
    def _validate_json_output(self, content: str) -> bool:
        """Validation stricte que la réponse est du JSON valide"""
        try:
            import json
            json.loads(content.strip())
            return True
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.error(f"JSON Output invalide reçu: {content[:100]}...")
            return False
    
    async def estimate_tokens(self, text: str) -> int:
        """
        Estimation améliorée du nombre de tokens pour le français
        
        Args:
            text: Texte à analyser
            
        Returns:
            int: Estimation nombre de tokens
        """
        if not text:
            return 0
        
        # Estimation améliorée pour le français
        words = text.split()
        base_tokens = len(words)
        
        # Ajustements selon la complexité
        punctuation_tokens = text.count('.') + text.count('!') + text.count('?')
        special_chars = text.count('{') + text.count('}') + text.count('[') + text.count(']')
        
        estimated_tokens = base_tokens + punctuation_tokens + (special_chars // 2)
        
        # Facteur correctif pour le français (plus verbose que l'anglais)
        return int(estimated_tokens * 1.2)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Métriques client pour monitoring"""
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "avg_tokens_per_request": (
                self._total_tokens / self._request_count if self._request_count > 0 else 0
            ),
            "initialized": self._initialized,
            "session_active": bool(self._session and not self._session.closed),
            "model": self.model_chat,
            "api_url": self.api_url
        }
    
    def __del__(self):
        """Nettoyage automatique avec warning si session ouverte"""
        if hasattr(self, '_session') and self._session and not self._session.closed:
            logger.warning(
                f"DeepSeek session pas proprement fermée - "
                f"Requests: {self._request_count}, Tokens: {self._total_tokens}"
            )
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit avec logging exception"""
        if exc_type:
            logger.error(f"Exception dans contexte DeepSeek: {exc_type.__name__}: {exc_val}")
        await self.close()