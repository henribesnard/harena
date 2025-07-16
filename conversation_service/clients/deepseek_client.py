import httpx
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config.settings import settings
from ..models.conversation import DeepSeekError

logger = logging.getLogger(__name__)

class DeepSeekResponse:
    """Réponse DeepSeek formatée"""
    def __init__(self, content: str, usage: Dict[str, Any], response_time: float):
        self.content = content
        self.usage = usage
        self.response_time = response_time
        self.timestamp = datetime.utcnow()

class DeepSeekClient:
    """Client HTTP asynchrone pour l'API DeepSeek"""
    
    def __init__(self):
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.api_key = settings.DEEPSEEK_API_KEY
        self.chat_model = settings.DEEPSEEK_CHAT_MODEL
        self.timeout = settings.DEEPSEEK_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
        
        # Cache simple en mémoire
        self._cache = {}
        self._cache_ttl = settings.CLASSIFICATION_CACHE_TTL
        self._cache_size = settings.CACHE_SIZE
        
        # Métriques
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "last_request_time": None
        }
        
        # Headers HTTP
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "HarenaConversationService/1.0"
        }
        
        logger.info(f"Client DeepSeek initialisé - Model: {self.chat_model}, Timeout: {self.timeout}s")
    
    def _generate_cache_key(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """Génère une clé de cache pour la requête"""
        # Utilise le contenu des messages + température comme clé
        content = json.dumps(messages, sort_keys=True) + str(temperature)
        return str(hash(content))
    
    def _get_from_cache(self, cache_key: str) -> Optional[DeepSeekResponse]:
        """Récupère une réponse du cache si elle est valide"""
        if cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if datetime.utcnow() - cached_item["timestamp"] < timedelta(seconds=self._cache_ttl):
                self._metrics["cache_hits"] += 1
                logger.debug(f"Cache hit pour la clé: {cache_key[:10]}...")
                return cached_item["response"]
            else:
                # Expiration du cache
                del self._cache[cache_key]
        return None
    
    def _save_to_cache(self, cache_key: str, response: DeepSeekResponse):
        """Sauvegarde une réponse dans le cache"""
        # Nettoyage du cache si trop plein
        if len(self._cache) >= self._cache_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
        
        self._cache[cache_key] = {
            "response": response,
            "timestamp": datetime.utcnow()
        }
        logger.debug(f"Réponse mise en cache: {cache_key[:10]}...")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Effectue la requête HTTP avec retry automatique"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Erreur HTTP DeepSeek: {e.response.status_code} - {e.response.text}")
                raise DeepSeekError(
                    message=f"Erreur HTTP {e.response.status_code}: {e.response.text}",
                    details={"status_code": e.response.status_code, "response": e.response.text}
                )
            except httpx.RequestError as e:
                logger.error(f"Erreur de requête DeepSeek: {str(e)}")
                raise DeepSeekError(
                    message=f"Erreur de requête: {str(e)}",
                    details={"error": str(e)}
                )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True
    ) -> DeepSeekResponse:
        """
        Effectue une requête de chat completion à DeepSeek
        
        Args:
            messages: Liste des messages de conversation
            temperature: Température pour la génération (optionnel)
            max_tokens: Nombre max de tokens (optionnel)
            use_cache: Utiliser le cache ou non
            
        Returns:
            DeepSeekResponse: Réponse formatée
        """
        start_time = time.time()
        
        # Paramètres par défaut
        if temperature is None:
            temperature = settings.DEEPSEEK_TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.DEEPSEEK_MAX_TOKENS
        
        # Vérification du cache
        cache_key = self._generate_cache_key(messages, temperature) if use_cache else None
        if cache_key:
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Payload de la requête
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": settings.DEEPSEEK_TOP_P,
            "stream": False
        }
        
        try:
            # Effectuer la requête
            response_data = await self._make_request(payload)
            
            # Mesurer le temps de réponse
            response_time = time.time() - start_time
            
            # Extraire le contenu et usage
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage", {})
            
            # Créer la réponse
            deepseek_response = DeepSeekResponse(
                content=content,
                usage=usage,
                response_time=response_time
            )
            
            # Mise en cache
            if cache_key:
                self._save_to_cache(cache_key, deepseek_response)
            
            # Mise à jour des métriques
            self._update_metrics(True, response_time, usage.get("total_tokens", 0))
            
            logger.debug(f"Requête DeepSeek réussie - Temps: {response_time:.2f}s, Tokens: {usage.get('total_tokens', 0)}")
            
            return deepseek_response
            
        except Exception as e:
            self._update_metrics(False, time.time() - start_time, 0)
            logger.error(f"Erreur lors de la requête DeepSeek: {str(e)}")
            raise
    
    def _update_metrics(self, success: bool, response_time: float, tokens: int):
        """Met à jour les métriques internes"""
        self._metrics["total_requests"] += 1
        self._metrics["last_request_time"] = datetime.utcnow()
        
        if success:
            self._metrics["successful_requests"] += 1
            self._metrics["total_tokens"] += tokens
            
            # Calcul moyenne mobile temps de réponse
            total_successful = self._metrics["successful_requests"]
            current_avg = self._metrics["avg_response_time"]
            self._metrics["avg_response_time"] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        else:
            self._metrics["failed_requests"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du client"""
        total_requests = self._metrics["total_requests"]
        cache_hits = self._metrics["cache_hits"]
        
        return {
            "total_requests": total_requests,
            "successful_requests": self._metrics["successful_requests"],
            "failed_requests": self._metrics["failed_requests"],
            "success_rate": self._metrics["successful_requests"] / total_requests if total_requests > 0 else 0,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total_requests if total_requests > 0 else 0,
            "total_tokens": self._metrics["total_tokens"],
            "avg_response_time": self._metrics["avg_response_time"],
            "last_request_time": self._metrics["last_request_time"],
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        """Vide le cache"""
        self._cache.clear()
        logger.info("Cache DeepSeek vidé")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Informations sur le cache"""
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
            "ttl_seconds": self._cache_ttl,
            "hit_rate": self._metrics["cache_hits"] / self._metrics["total_requests"] if self._metrics["total_requests"] > 0 else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé de la connexion DeepSeek"""
        try:
            test_messages = [
                {"role": "user", "content": "test"}
            ]
            
            start_time = time.time()
            await self.chat_completion(test_messages, use_cache=False)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }

# Instance globale du client
deepseek_client = DeepSeekClient()