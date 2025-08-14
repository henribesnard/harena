"""
Client DeepSeek optimisé pour Conversation Service MVP.

Ce module fournit un client DeepSeek haute performance avec :
- Cache intelligent multi-niveaux
- Circuit breaker pour la résilience
- Rate limiting respectueux des quotas
- Optimisations spécifiques DeepSeek V3/R1
- Monitoring détaillé coûts et performance

Performance cibles :
- <100ms pour réponses cachées
- <2s pour appels API DeepSeek
- 90% réduction coûts via cache
- 99.9% disponibilité via circuit breaker

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - CORRIGÉ ASYNC
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import os
import hashlib

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

# Imports locaux
from ..utils.cache import MultiLevelCache, generate_cache_key
from ..utils.metrics import get_default_metrics_collector

logger = logging.getLogger(__name__)


class DeepSeekError(Exception):
    """Exception de base pour les erreurs DeepSeek."""
    pass


class DeepSeekTimeoutError(DeepSeekError):
    """Exception pour les timeouts DeepSeek."""
    pass


class DeepSeekRateLimitError(DeepSeekError):
    """Exception pour les erreurs de rate limit."""
    pass


class DeepSeekModelType(str, Enum):
    """Types de modèles DeepSeek disponibles."""
    
    CHAT = "deepseek-chat"
    REASONER = "deepseek-reasoner"


@dataclass
class DeepSeekUsageStats:
    """Statistiques d'utilisation DeepSeek."""
    
    # Compteurs de requêtes
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0
    
    # Métriques de performance
    total_response_time_ms: float = 0.0
    avg_response_time_ms: float = field(init=False)
    
    # Utilisation tokens
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    
    # Circuit breaker
    circuit_breaker_trips: int = 0
    rate_limit_hits: int = 0
    
    def __post_init__(self):
        """Calcule les métriques dérivées."""
        self.avg_response_time_ms = (
            self.total_response_time_ms / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
    
    def add_request(
        self, 
        response_time_ms: float, 
        input_tokens: int = 0, 
        output_tokens: int = 0,
        from_cache: bool = False,
        success: bool = True
    ):
        """Ajoute une requête aux statistiques."""
        self.total_requests += 1
        self.total_response_time_ms += response_time_ms
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        if from_cache:
            self.cached_requests += 1
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calcul coût (prix depuis env vars)
        input_cost = float(os.getenv('COST_PER_1K_INPUT_TOKENS', '0.0005'))
        output_cost = float(os.getenv('COST_PER_1K_OUTPUT_TOKENS', '0.0015'))
        
        self.total_cost_usd += (input_tokens * input_cost / 1000) + (output_tokens * output_cost / 1000)
        
        # Recalcul moyenne
        self.avg_response_time_ms = self.total_response_time_ms / self.total_requests


@dataclass 
class CircuitBreakerState:
    """État du circuit breaker."""
    
    failures: int = 0
    last_failure_time: Optional[float] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self):
        """Enregistre un succès."""
        self.failures = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Enregistre un échec."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        failure_threshold = int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5'))
        if self.failures >= failure_threshold:
            self.state = "OPEN"
    
    def can_attempt_request(self) -> bool:
        """Vérifie si une requête peut être tentée."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            recovery_timeout = int(os.getenv('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '60'))
            if self.last_failure_time and time.time() - self.last_failure_time > recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        
        # HALF_OPEN state
        return True


@dataclass
class DeepSeekResponse:
    """Représente une réponse simplifiée de DeepSeek."""

    content: str
    raw: ChatCompletion


class DeepSeekOptimizer:
    """Optimiseur statique pour prompts et requêtes DeepSeek."""
    
    @staticmethod
    def optimize_prompt(prompt: str, model_type: DeepSeekModelType = DeepSeekModelType.CHAT) -> str:
        """
        Optimise un prompt pour DeepSeek.
        
        Args:
            prompt: Prompt original
            model_type: Type de modèle DeepSeek
            
        Returns:
            Prompt optimisé
        """
        if not prompt or not prompt.strip():
            return prompt
        
        optimized = prompt.strip()

        max_chars = int(os.getenv("DEEPSEEK_MAX_PROMPT_CHARS", "2000"))
        
        # Optimisations spécifiques DeepSeek Chat
        if model_type == DeepSeekModelType.CHAT:
            # DeepSeek préfère les instructions claires et structurées
            if not optimized.startswith(("Vous", "Tu", "Votre", "Je")):
                optimized = f"Vous devez {optimized.lower()}"
            
            # Encourage les réponses structurées pour faciliter le parsing
            if "format" not in optimized.lower():
                optimized += "\n\nRépondez de manière structurée et précise."
        
        # Optimisations spécifiques DeepSeek Reasoner
        elif model_type == DeepSeekModelType.REASONER:
            # Reasoner bénéficie d'instructions explicites sur le raisonnement
            if "étapes" not in optimized.lower() and "raisonnement" not in optimized.lower():
                optimized = f"Analysez étape par étape :\n\n{optimized}"

        if len(optimized) > max_chars:
            optimized = optimized[:max_chars] + "..."

        return optimized
    
    @staticmethod
    def optimize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Optimise une liste de messages pour DeepSeek.
        
        Args:
            messages: Messages originaux
            
        Returns:
            Messages optimisés
        """
        if not messages:
            return messages
        
        optimized = []
        
        for msg in messages:
            content = msg.get("content", "")
            optimized_content = DeepSeekOptimizer.optimize_prompt(content)
            optimized.append({
                "role": msg.get("role"),
                "content": optimized_content
            })
        
        return optimized
    
    @staticmethod
    def batch_requests(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prépare des requêtes pour traitement en lot.
        
        Args:
            requests: Liste de requêtes
            
        Returns:
            Requêtes optimisées pour le batch
        """
        if not requests:
            return requests
        
        # Pour l'instant, retourne les requêtes telles quelles
        # Future optimisation : grouper par paramètres similaires
        return requests


class DeepSeekClient:
    """
    Client DeepSeek optimisé avec cache, circuit breaker et monitoring.
    
    Features:
    - Cache multi-niveaux pour réduire les coûts
    - Circuit breaker pour la résilience
    - Rate limiting respectueux
    - Métriques détaillées
    - Support Chat + Reasoner
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_enabled: bool = None,
        timeout: int = None,
        max_retries: int = None
    ):
        """
        Initialise le client DeepSeek.
        
        Args:
            api_key: Clé API DeepSeek (défaut: DEEPSEEK_API_KEY env var)
            base_url: URL de base (défaut: DEEPSEEK_BASE_URL env var)
            cache_enabled: Active le cache (défaut: LLM_CACHE_ENABLED env var)
            timeout: Timeout en secondes (défaut: LLM_TIMEOUT env var)
            max_retries: Nombre maximum de tentatives (défaut: MAX_RETRIES env var)
        """
        # Configuration depuis env vars
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url or os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self.timeout = timeout or int(os.getenv('LLM_TIMEOUT', os.getenv('DEEPSEEK_TIMEOUT', '15')))
        self.max_retries = max_retries or int(os.getenv('MAX_RETRIES', '3'))
        self.expected_latency_ms = int(os.getenv('DEEPSEEK_EXPECTED_LATENCY_MS', '1500'))
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        
        # Client AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        # Cache et état
        cache_enabled_env = os.getenv('LLM_CACHE_ENABLED', 'True').lower() == 'true'
        self.cache_enabled = cache_enabled if cache_enabled is not None else cache_enabled_env

        if self.cache_enabled:
            self.cache = MultiLevelCache(
                l1_size=int(os.getenv('LLM_CACHE_MAX_SIZE', '1000')),
                l1_ttl=int(os.getenv('LLM_CACHE_TTL', '300')),
                l2_ttl=int(os.getenv('CACHE_TTL', '3600'))
            )
        else:
            self.cache = None
        
        # Circuit breaker
        cb_enabled = os.getenv('CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true'
        self.circuit_breaker_enabled = cb_enabled
        self.circuit_breaker = CircuitBreakerState()
        
        # Statistiques
        self.stats = DeepSeekUsageStats()
        
        # Rate limiting
        self.rate_limit_enabled = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
        self.requests_per_minute = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', '60'))
        self.request_timestamps: List[float] = []
        
        logger.info(f"DeepSeekClient initialized: cache={self.cache_enabled}, cb={self.circuit_breaker_enabled}")
    
    def _generate_cache_key(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Génère une clé de cache pour la requête."""
        # Inclut les messages et paramètres importants
        cache_data = {
            "messages": messages,
            "model": kwargs.get("model", "deepseek-chat"),
            "temperature": kwargs.get("temperature", 1.0),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 0.95)
        }
        
        return generate_cache_key("deepseek_completion", **cache_data)
    
    def _check_rate_limit(self) -> bool:
        """Vérifie si on peut faire une requête (rate limiting)."""
        if not self.rate_limit_enabled:
            return True
        
        now = time.time()
        
        # Nettoie les timestamps anciens (> 1 minute)
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < 60
        ]
        
        # Vérifie la limite
        if len(self.request_timestamps) >= self.requests_per_minute:
            self.stats.rate_limit_hits += 1
            return False
        
        return True
    
    def _record_request(self):
        """Enregistre une requête pour le rate limiting."""
        if self.rate_limit_enabled:
            self.request_timestamps.append(time.time())
    
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        use_cache: bool = True,
        **kwargs
    ) -> ChatCompletion:
        """
        Crée une completion DeepSeek avec cache et optimisations.
        
        Args:
            messages: Messages de la conversation
            model: Modèle DeepSeek à utiliser
            temperature: Température (0.0-2.0)
            max_tokens: Nombre max de tokens de sortie
            top_p: Top-p sampling
            use_cache: Utilise le cache si disponible
            **kwargs: Autres paramètres OpenAI
            
        Returns:
            ChatCompletion response
            
        Raises:
            DeepSeekError: Erreur générale
            DeepSeekTimeoutError: Timeout
            DeepSeekRateLimitError: Rate limit atteint
        """
        metrics = get_default_metrics_collector()

        # Paramètres par défaut depuis env vars
        model = model or os.getenv('DEEPSEEK-CHAT-MODEL', 'deepseek-chat')
        temperature = temperature if temperature is not None else float(os.getenv('DEEPSEEK_TEMPERATURE', '1.0'))
        max_allowed = int(os.getenv('DEEPSEEK_MAX_TOKENS', '1024'))
        max_tokens = max_tokens or max_allowed
        max_tokens = min(max_tokens, max_allowed)
        top_p = top_p if top_p is not None else float(os.getenv('DEEPSEEK_TOP_P', '0.95'))

        # Optimisation des messages
        optimized_messages = DeepSeekOptimizer.optimize_messages(messages)

        # Vérification cache
        cache_key = None
        if self.cache_enabled and use_cache and self.cache:
            cache_key = self._generate_cache_key(
                optimized_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

            start_time = time.time()
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                response_time = (time.time() - start_time) * 1000
                self.stats.add_request(
                    response_time_ms=response_time,
                    from_cache=True,
                    success=True
                )
                metrics.record_deepseek_usage(
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    duration_ms=response_time,
                    cost_usd=0.0,
                    success=True,
                )
                logger.debug(f"Cache hit for DeepSeek request: {response_time:.1f}ms")
                return cached_response

        for attempt in range(self.max_retries):
            if self.circuit_breaker_enabled and not self.circuit_breaker.can_attempt_request():
                self.stats.circuit_breaker_trips += 1
                raise DeepSeekError("Circuit breaker is OPEN - too many failures")

            if not self._check_rate_limit():
                raise DeepSeekRateLimitError("Rate limit exceeded")

            start_time = time.time()
            try:
                # Enregistre la requête
                self._record_request()

                # Appel API DeepSeek
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=optimized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs
                )

                if self.circuit_breaker_enabled:
                    self.circuit_breaker.record_success()

                if self.cache_enabled and cache_key and self.cache:
                    cache_ttl = int(os.getenv('LLM_CACHE_TTL', '300'))
                    if temperature < 0.3:
                        cache_ttl *= 3
                    await self.cache.set(cache_key, response, ttl=cache_ttl)

                response_time = (time.time() - start_time) * 1000
                usage = response.usage
                self.stats.add_request(
                    response_time_ms=response_time,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    success=True
                )
                input_cost = float(os.getenv('COST_PER_1K_INPUT_TOKENS', '0.0005'))
                output_cost = float(os.getenv('COST_PER_1K_OUTPUT_TOKENS', '0.0015'))
                cost_usd = 0.0
                if usage:
                    cost_usd = (
                        usage.prompt_tokens * input_cost / 1000
                        + usage.completion_tokens * output_cost / 1000
                    )
                metrics.record_deepseek_usage(
                    model=model,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    duration_ms=response_time,
                    cost_usd=cost_usd,
                    success=True,
                )

                logger.debug(
                    f"DeepSeek API call successful: {response_time:.1f}ms, {usage.total_tokens if usage else 0} tokens"
                )

                return response

            except asyncio.TimeoutError:
                if self.circuit_breaker_enabled:
                    self.circuit_breaker.record_failure()

                response_time = (time.time() - start_time) * 1000
                self.stats.add_request(response_time_ms=response_time, success=False)
                metrics.record_deepseek_usage(
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    duration_ms=response_time,
                    cost_usd=0.0,
                    success=False,
                )

                if attempt == self.max_retries - 1:
                    raise DeepSeekTimeoutError("DeepSeek API call timed out")

                await asyncio.sleep(1)

            except Exception as e:
                if self.circuit_breaker_enabled:
                    self.circuit_breaker.record_failure()

                response_time = (time.time() - start_time) * 1000
                self.stats.add_request(response_time_ms=response_time, success=False)
                metrics.record_deepseek_usage(
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    duration_ms=response_time,
                    cost_usd=0.0,
                    success=False,
                )

                logger.error(f"DeepSeek API error: {str(e)}")

                if attempt == self.max_retries - 1:
                    raise DeepSeekError(f"DeepSeek API error: {str(e)}")

                await asyncio.sleep(1)
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ChatCompletion:
        """
        Alias pour create_completion avec modèle Chat par défaut.
        
        Args:
            messages: Messages de conversation
            **kwargs: Paramètres de completion
            
        Returns:
            ChatCompletion response
        """
        return await self.create_completion(
            messages=messages,
            model=os.getenv('DEEPSEEK-CHAT-MODEL', 'deepseek-chat'),
            **kwargs
        )

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> DeepSeekResponse:
        """Generate a response using DeepSeek chat completion."""

        completion = await self.create_chat_completion(messages=messages, **kwargs)

        content = ""
        if completion.choices:
            content = completion.choices[0].message.content

        return DeepSeekResponse(content=content, raw=completion)
    
    async def create_reasoning_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ChatCompletion:
        """
        Crée une completion avec le modèle Reasoner.
        
        Args:
            messages: Messages de conversation
            **kwargs: Paramètres de completion
            
        Returns:
            ChatCompletion response
        """
        # Paramètres optimisés pour Reasoner
        reasoner_params = {
            "model": os.getenv('DEEPSEEK-REASONER-MODEL', 'deepseek-reasoner'),
            "temperature": 0.1,  # Plus déterministe pour le raisonnement
            "max_tokens": 2000,  # Reasoner génère plus de texte
            **kwargs
        }
        
        return await self.create_completion(messages=messages, **reasoner_params)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'utilisation détaillées.
        
        Returns:
            Dictionnaire avec métriques complètes
        """
        base_stats = {
            "requests": {
                "total": self.stats.total_requests,
                "successful": self.stats.successful_requests,
                "failed": self.stats.failed_requests,
                "cached": self.stats.cached_requests,
                "success_rate": self.stats.successful_requests / self.stats.total_requests if self.stats.total_requests > 0 else 0.0,
                "cache_hit_rate": self.stats.cached_requests / self.stats.total_requests if self.stats.total_requests > 0 else 0.0
            },
            "performance": {
                "avg_response_time_ms": self.stats.avg_response_time_ms,
                "total_response_time_ms": self.stats.total_response_time_ms
            },
            "tokens": {
                "total_input": self.stats.total_input_tokens,
                "total_output": self.stats.total_output_tokens,
                "total_cost_usd": round(self.stats.total_cost_usd, 4)
            },
            "reliability": {
                "circuit_breaker_trips": self.stats.circuit_breaker_trips,
                "circuit_breaker_state": self.circuit_breaker.state,
                "rate_limit_hits": self.stats.rate_limit_hits
            }
        }
        
        # CORRECTION: Gestion async de cache stats
        if self.cache_enabled and self.cache:
            try:
                # Si get_stats est synchrone (pour L1 seulement)
                if hasattr(self.cache, 'l1_cache') and hasattr(self.cache.l1_cache, 'get_stats'):
                    base_stats["cache"] = self.cache.l1_cache.get_stats()
                else:
                    base_stats["cache"] = {"note": "Cache stats require async call"}
            except Exception:
                base_stats["cache"] = {"error": "Cache stats unavailable"}
        
        return base_stats
    
    async def get_usage_stats_async(self) -> Dict[str, Any]:
        """
        Version async de get_usage_stats avec stats cache complètes.
        
        Returns:
            Dictionnaire avec métriques complètes incluant cache async
        """
        base_stats = self.get_usage_stats()
        
        # Ajout des stats cache complètes
        if self.cache_enabled and self.cache:
            try:
                cache_stats = await self.cache.get_stats()
                base_stats["cache"] = cache_stats
            except Exception as e:
                base_stats["cache"] = {"error": f"Cache async stats error: {str(e)}"}
        
        return base_stats
    
    async def clear_cache(self) -> None:
        """Vide le cache (version async)."""
        if self.cache_enabled and self.cache:
            await self.cache.clear()
            logger.info("DeepSeek cache cleared")
    
    def clear_cache_sync(self) -> None:
        """Vide le cache (version synchrone pour compatibilité)."""
        if self.cache_enabled and self.cache:
            # Vide seulement le cache L1 en mode synchrone
            if hasattr(self.cache, 'l1_cache'):
                self.cache.l1_cache.clear()
            logger.info("DeepSeek L1 cache cleared (sync)")
    
    def reset_stats(self) -> None:
        """Remet à zéro les statistiques."""
        self.stats = DeepSeekUsageStats()
        self.circuit_breaker = CircuitBreakerState()
        self.request_timestamps.clear()
        logger.info("DeepSeek stats reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie la santé du client DeepSeek.
        
        Returns:
            Status de santé avec métriques
        """
        try:
            # Test simple avec cache désactivé
            test_messages = [
                {"role": "system", "content": "Répondez uniquement 'OK'"},
                {"role": "user", "content": "Test"}
            ]
            
            start_time = time.time()
            response = await self.create_completion(
                messages=test_messages,
                max_tokens=10,
                temperature=0.0,
                use_cache=False
            )
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 1),
                "circuit_breaker_state": self.circuit_breaker.state,
                "api_accessible": True,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state,
                "api_accessible": False,
                "last_check": datetime.now().isoformat()
            }
    
    async def close(self) -> None:
        """Ferme le client proprement."""
        if hasattr(self.client, 'close'):
            await self.client.close()
        
        # CORRECTION: Ferme aussi le cache
        if self.cache_enabled and self.cache:
            await self.cache.close()
        
        logger.info("DeepSeek client closed")


# Factory functions
def create_deepseek_client(**kwargs) -> DeepSeekClient:
    """
    Factory pour créer un client DeepSeek.
    
    Args:
        **kwargs: Arguments pour DeepSeekClient
        
    Returns:
        Instance DeepSeekClient configurée
    """
    kwargs.setdefault("cache_enabled", True)
    return DeepSeekClient(**kwargs)


# Instance globale optionnelle
_default_client: Optional[DeepSeekClient] = None

async def get_default_client() -> DeepSeekClient:
    """Retourne le client DeepSeek par défaut."""
    global _default_client
    if _default_client is None:
        _default_client = DeepSeekClient(cache_enabled=True)
    return _default_client
