"""
Client DeepSeek optimisé pour conversation service
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
    """Exception spécifique DeepSeek"""
    pass

class DeepSeekClient:
    """Client DeepSeek avec optimisations performances et coûts"""
    
    def __init__(self):
        self.api_url = settings.DEEPSEEK_API_URL
        self.api_key = settings.DEEPSEEK_API_KEY
        self.model_chat = settings.DEEPSEEK_MODEL_CHAT
        self.max_tokens = settings.DEEPSEEK_MAX_TOKENS
        self.temperature = settings.DEEPSEEK_TEMPERATURE
        self.timeout = settings.DEEPSEEK_TIMEOUT
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialisation client avec session aiohttp"""
        if self._initialized:
            return
        
        if not self.api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY non configuré")
        
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        self._initialized = True
        logger.info("DeepSeek client initialisé")
    
    async def close(self) -> None:
        """Fermeture propre du client"""
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False
    
    async def health_check(self) -> bool:
        """Vérification santé API DeepSeek"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Test simple avec message minimal
            test_response = await self.chat_completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            
            return bool(test_response and test_response.get("choices"))
            
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {str(e)}")
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Appel API chat completion avec retry automatique"""
        
        if not self._initialized:
            await self.initialize()
        
        if not self._session:
            raise DeepSeekError("Session non initialisée")
        
        payload = {
            "model": model or self.model_chat,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stream": False
        }
        
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
                        
                        # Log métriques
                        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                        tokens_used = result.get("usage", {}).get("total_tokens", 0)
                        
                        logger.info(
                            f"DeepSeek API call successful - Duration: {duration_ms}ms, "
                            f"Tokens: {tokens_used}, Attempt: {attempt + 1}"
                        )
                        
                        return result
                    
                    # Gestion erreurs HTTP
                    elif response.status == 429:
                        # Rate limiting
                        retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                        logger.warning(f"Rate limited, retry after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    elif response.status >= 500:
                        # Erreur serveur - retry
                        logger.warning(f"Server error {response.status}, retry attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(base_delay * (2 ** attempt))
                            continue
                    
                    # Autres erreurs HTTP
                    error_text = await response.text()
                    raise DeepSeekError(f"HTTP {response.status}: {error_text}")
            
            except aiohttp.ClientError as e:
                logger.error(f"DeepSeek API connection error: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                raise DeepSeekError(f"Connection failed after {max_retries} attempts: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected DeepSeek API error: {str(e)}")
                raise DeepSeekError(f"Unexpected error: {str(e)}")
        
        raise DeepSeekError(f"Failed after {max_retries} attempts")
    
    async def count_tokens(self, text: str) -> int:
        """Estimation grossière du nombre de tokens"""
        # Approximation : 1 token ≈ 4 caractères pour le français
        return len(text) // 4
    
    def __del__(self):
        """Nettoyage automatique"""
        if self._session and not self._session.closed:
            logger.warning("DeepSeek session not properly closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()