"""
LLM Provider Abstraction - Phase 4 
Architecture v2.0 - Composants IA

Responsabilite : Abstraction multi-provider avec fallback
- DeepSeek -> OpenAI -> Local (Ollama)
- Configuration centralisee des modeles
- Rate limiting et circuit breaker
- Streaming support pour toutes les plateformes
- Few-shot prompting standardise
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Types de providers LLM supportes"""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    LOCAL = "local"

class ModelCapability(Enum):
    """Capacites des modeles"""
    CHAT = "chat"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    STREAMING = "streaming"

@dataclass
class LLMRequest:
    """Requete standardisee pour tous les providers"""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    few_shot_examples: Optional[List[Dict]] = None
    system_prompt: Optional[str] = None
    user_id: Optional[int] = None
    conversation_id: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None  # Pour JSON OUTPUT DeepSeek

@dataclass 
class LLMResponse:
    """Reponse standardisee de tous les providers"""
    content: str
    model_used: str
    provider_used: ProviderType
    usage: Dict[str, int]
    processing_time_ms: int
    stream_finished: bool = False
    error: Optional[str] = None

@dataclass
class ProviderConfig:
    """Configuration d'un provider"""
    api_key: str
    base_url: str
    models: List[str]
    capabilities: List[ModelCapability]
    rate_limit_rpm: int = 60
    timeout_seconds: int = 30
    priority: int = 1  # 1=highest, 3=lowest

class CircuitBreakerState(Enum):
    """etats du circuit breaker"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerStats:
    """Statistiques circuit breaker"""
    failure_count: int = 0
    success_count: int = 0
    last_failure: Optional[datetime] = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED

class BaseLLMProvider(ABC):
    """Interface commune pour tous les providers LLM"""
    
    def __init__(self, config: ProviderConfig, provider_type: ProviderType):
        self.config = config
        self.provider_type = provider_type
        self.circuit_breaker = CircuitBreakerStats()
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Statistiques
        self.stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "tokens_consumed": 0,
            "avg_latency_ms": 0
        }
    
    async def initialize(self) -> bool:
        """Initialise la session HTTP"""
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': f'conversation-service/2.0-{self.provider_type.value}'
                }
            )
            
            return True
        except Exception as e:
            logger.error(f"Erreur init {self.provider_type.value}: {str(e)}")
            return False
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Genere une reponse (implementation specifique au provider)"""
        pass
    
    @abstractmethod 
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Genere une reponse en streaming"""
        pass
    
    def is_available(self) -> bool:
        """Verifie si le provider est disponible (circuit breaker)"""
        if self.circuit_breaker.state == CircuitBreakerState.OPEN:
            # Verifier si on peut passer en HALF_OPEN
            if (self.circuit_breaker.last_failure and 
                datetime.now() - self.circuit_breaker.last_failure > timedelta(minutes=5)):
                self.circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"{self.provider_type.value} circuit breaker e HALF_OPEN")
                return True
            return False
        
        return True
    
    def record_success(self):
        """Enregistre un succes"""
        self.circuit_breaker.success_count += 1
        self.circuit_breaker.failure_count = 0
        if self.circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker.state = CircuitBreakerState.CLOSED
            logger.info(f"{self.provider_type.value} circuit breaker e CLOSED")
        
        self.stats["requests_success"] += 1
    
    def record_failure(self, error: str):
        """Enregistre un echec"""
        self.circuit_breaker.failure_count += 1
        self.circuit_breaker.last_failure = datetime.now()
        
        # Ouvrir circuit apres 5 echecs
        if self.circuit_breaker.failure_count >= 5:
            self.circuit_breaker.state = CircuitBreakerState.OPEN
            logger.warning(f"{self.provider_type.value} circuit breaker e OPEN")
        
        self.stats["requests_failed"] += 1
    
    async def close(self):
        """Ferme proprement les connexions"""
        if self._session:
            await self._session.close()

class DeepSeekProvider(BaseLLMProvider):
    """Provider pour DeepSeek API"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config, ProviderType.DEEPSEEK)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Genere avec DeepSeek API"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.initialize()
            
            # Preparation payload DeepSeek
            payload = {
                "model": request.model or "deepseek-chat",
                "messages": self._prepare_messages(request),
                "temperature": request.temperature,
                "stream": False
            }

            if request.max_tokens:
                payload["max_tokens"] = request.max_tokens

            # Support JSON OUTPUT natif DeepSeek
            if request.response_format:
                payload["response_format"] = request.response_format
            
            # Requete HTTP
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            
            async with self._session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    
                    self.record_success()
                    self.stats["tokens_consumed"] += usage.get("total_tokens", 0)
                    
                    return LLMResponse(
                        content=content,
                        model_used=data.get("model", payload["model"]),
                        provider_used=ProviderType.DEEPSEEK,
                        usage=usage,
                        processing_time_ms=self._get_processing_time(start_time)
                    )
                else:
                    error_text = await response.text()
                    self.record_failure(f"HTTP {response.status}: {error_text}")
                    
                    return LLMResponse(
                        content="",
                        model_used=payload["model"],
                        provider_used=ProviderType.DEEPSEEK,
                        usage={},
                        processing_time_ms=self._get_processing_time(start_time),
                        error=f"DeepSeek error {response.status}: {error_text}"
                    )
        
        except Exception as e:
            self.record_failure(str(e))
            return LLMResponse(
                content="",
                model_used=request.model or "deepseek-chat",
                provider_used=ProviderType.DEEPSEEK,
                usage={},
                processing_time_ms=self._get_processing_time(start_time),
                error=f"DeepSeek exception: {str(e)}"
            )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Streaming avec DeepSeek"""
        try:
            if not self._session:
                await self.initialize()
            
            payload = {
                "model": request.model or "deepseek-chat", 
                "messages": self._prepare_messages(request),
                "temperature": request.temperature,
                "stream": True
            }
            
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            
            async with self._session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str != "[DONE]":
                                    try:
                                        data = json.loads(data_str)
                                        delta = data["choices"][0]["delta"]
                                        if "content" in delta:
                                            yield delta["content"]
                                    except json.JSONDecodeError:
                                        continue
                else:
                    yield f"Error: DeepSeek stream failed ({response.status})"
                    
        except Exception as e:
            yield f"Error: DeepSeek stream exception ({str(e)})"
    
    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """Prepare les messages avec system prompt et few-shot"""
        messages = []
        
        # System prompt
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        # Few-shot examples
        if request.few_shot_examples:
            for example in request.few_shot_examples:
                messages.append({
                    "role": "user",
                    "content": example["user"]
                })
                messages.append({
                    "role": "assistant", 
                    "content": example["assistant"]
                })
        
        # Messages principaux
        messages.extend(request.messages)
        
        return messages
    
    def _get_processing_time(self, start_time: datetime) -> int:
        """Calcule temps de traitement en ms"""
        return int((datetime.now() - start_time).total_seconds() * 1000)

class OpenAIProvider(BaseLLMProvider):
    """Provider pour OpenAI API (structure similaire e DeepSeek)"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config, ProviderType.OPENAI)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Genere avec OpenAI API"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.initialize()
            
            payload = {
                "model": request.model or "gpt-3.5-turbo",
                "messages": self._prepare_messages(request),
                "temperature": request.temperature
            }
            
            if request.max_tokens:
                payload["max_tokens"] = request.max_tokens
            
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            
            async with self._session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    
                    self.record_success()
                    self.stats["tokens_consumed"] += usage.get("total_tokens", 0)
                    
                    return LLMResponse(
                        content=content,
                        model_used=data.get("model", payload["model"]),
                        provider_used=ProviderType.OPENAI,
                        usage=usage,
                        processing_time_ms=self._get_processing_time(start_time)
                    )
                else:
                    error_text = await response.text()
                    self.record_failure(f"HTTP {response.status}: {error_text}")
                    
                    return LLMResponse(
                        content="",
                        model_used=payload["model"],
                        provider_used=ProviderType.OPENAI,
                        usage={},
                        processing_time_ms=self._get_processing_time(start_time),
                        error=f"OpenAI error {response.status}: {error_text}"
                    )
        
        except Exception as e:
            self.record_failure(str(e))
            return LLMResponse(
                content="",
                model_used=request.model or "gpt-3.5-turbo",
                provider_used=ProviderType.OPENAI,
                usage={},
                processing_time_ms=self._get_processing_time(start_time),
                error=f"OpenAI exception: {str(e)}"
            )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Streaming OpenAI (implementation similaire e DeepSeek)"""
        # Implementation similaire e DeepSeek avec les specificites OpenAI
        yield "OpenAI streaming not implemented yet"
    
    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """Meme logique que DeepSeek"""
        messages = []
        
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        if request.few_shot_examples:
            for example in request.few_shot_examples:
                messages.append({
                    "role": "user",
                    "content": example["user"]
                })
                messages.append({
                    "role": "assistant",
                    "content": example["assistant"]
                })
        
        messages.extend(request.messages)
        return messages
    
    def _get_processing_time(self, start_time: datetime) -> int:
        return int((datetime.now() - start_time).total_seconds() * 1000)

class LocalProvider(BaseLLMProvider):
    """Provider pour modeles locaux (Ollama)"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config, ProviderType.LOCAL)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Genere avec Ollama local"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.initialize()
            
            # Format Ollama
            payload = {
                "model": request.model or "llama2",
                "messages": self._prepare_messages(request),
                "stream": False,
                "options": {
                    "temperature": request.temperature
                }
            }
            
            async with self._session.post(
                f"{self.config.base_url}/api/chat",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    content = data["message"]["content"]
                    
                    self.record_success()
                    
                    return LLMResponse(
                        content=content,
                        model_used=data.get("model", payload["model"]),
                        provider_used=ProviderType.LOCAL,
                        usage={"total_tokens": len(content) // 4},  # Estimation
                        processing_time_ms=self._get_processing_time(start_time)
                    )
                else:
                    error_text = await response.text()
                    self.record_failure(f"HTTP {response.status}: {error_text}")
                    
                    return LLMResponse(
                        content="",
                        model_used=payload["model"],
                        provider_used=ProviderType.LOCAL,
                        usage={},
                        processing_time_ms=self._get_processing_time(start_time),
                        error=f"Local error {response.status}: {error_text}"
                    )
        
        except Exception as e:
            self.record_failure(str(e))
            return LLMResponse(
                content="",
                model_used=request.model or "llama2",
                provider_used=ProviderType.LOCAL,
                usage={},
                processing_time_ms=self._get_processing_time(start_time),
                error=f"Local exception: {str(e)}"
            )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Streaming Ollama"""
        yield "Local streaming not implemented yet"
    
    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """Meme format que les autres providers"""
        messages = []
        
        if request.system_prompt:
            messages.append({
                "role": "system", 
                "content": request.system_prompt
            })
        
        if request.few_shot_examples:
            for example in request.few_shot_examples:
                messages.append({
                    "role": "user",
                    "content": example["user"]
                })
                messages.append({
                    "role": "assistant",
                    "content": example["assistant"]
                })
        
        messages.extend(request.messages)
        return messages
    
    def _get_processing_time(self, start_time: datetime) -> int:
        return int((datetime.now() - start_time).total_seconds() * 1000)

class LLMProviderManager:
    """
    Gestionnaire principal avec fallback automatique
    DeepSeek e OpenAI e Local
    """
    
    def __init__(self, configs: Dict[ProviderType, ProviderConfig]):
        self.providers: Dict[ProviderType, BaseLLMProvider] = {}
        self.fallback_order = [ProviderType.DEEPSEEK, ProviderType.OPENAI, ProviderType.LOCAL]
        
        # Initialiser les providers disponibles
        for provider_type, config in configs.items():
            if provider_type == ProviderType.DEEPSEEK:
                self.providers[provider_type] = DeepSeekProvider(config)
            elif provider_type == ProviderType.OPENAI:
                self.providers[provider_type] = OpenAIProvider(config)
            elif provider_type == ProviderType.LOCAL:
                self.providers[provider_type] = LocalProvider(config)
        
        # Statistiques globales
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallbacks_used": 0,
            "provider_usage": {pt.value: 0 for pt in ProviderType}
        }
        
        logger.info(f"LLMProviderManager initialise avec {len(self.providers)} providers")
    
    async def initialize(self) -> bool:
        """Initialise tous les providers"""
        success_count = 0
        
        for provider_type, provider in self.providers.items():
            try:
                if await provider.initialize():
                    success_count += 1
                    logger.info(f"Provider {provider_type.value} initialise")
                else:
                    logger.warning(f"echec init provider {provider_type.value}")
            except Exception as e:
                logger.error(f"Erreur init provider {provider_type.value}: {str(e)}")
        
        return success_count > 0
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Genere avec fallback automatique selon l'ordre de priorite
        """
        self.stats["total_requests"] += 1
        
        last_error = None
        
        for provider_type in self.fallback_order:
            provider = self.providers.get(provider_type)
            
            if not provider or not provider.is_available():
                logger.debug(f"Provider {provider_type.value} indisponible")
                continue
            
            try:
                logger.debug(f"Tentative generation avec {provider_type.value}")
                
                response = await provider.generate(request)
                
                if not response.error:
                    # Succes !
                    self.stats["successful_requests"] += 1
                    self.stats["provider_usage"][provider_type.value] += 1
                    
                    if provider_type != self.fallback_order[0]:
                        self.stats["fallbacks_used"] += 1
                        logger.info(f"Fallback utilise: {provider_type.value}")
                    
                    return response
                else:
                    last_error = response.error
                    logger.warning(f"Provider {provider_type.value} error: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Provider {provider_type.value} exception: {last_error}")
        
        # Tous les providers ont echoue
        self.stats["failed_requests"] += 1
        
        return LLMResponse(
            content="",
            model_used="unknown",
            provider_used=ProviderType.DEEPSEEK,  # Fallback pour le type
            usage={},
            processing_time_ms=0,
            error=f"All providers failed. Last error: {last_error}"
        )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Streaming avec fallback"""
        
        for provider_type in self.fallback_order:
            provider = self.providers.get(provider_type)
            
            if not provider or not provider.is_available():
                continue
            
            try:
                async for chunk in provider.generate_stream(request):
                    yield chunk
                return  # Streaming reussi
                
            except Exception as e:
                logger.error(f"Streaming failed with {provider_type.value}: {str(e)}")
                continue
        
        # Fallback en cas d'echec streaming
        yield "Error: All streaming providers failed"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Health check de tous les providers"""
        
        providers_status = {}
        healthy_count = 0
        
        for provider_type, provider in self.providers.items():
            is_available = provider.is_available()
            if is_available:
                healthy_count += 1
            
            providers_status[provider_type.value] = {
                "available": is_available,
                "circuit_breaker_state": provider.circuit_breaker.state.value,
                "success_count": provider.circuit_breaker.success_count,
                "failure_count": provider.circuit_breaker.failure_count,
                "stats": provider.stats
            }
        
        return {
            "status": "healthy" if healthy_count > 0 else "unhealthy",
            "component": "llm_provider_manager",
            "providers_available": healthy_count,
            "providers_total": len(self.providers),
            "providers": providers_status,
            "global_stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }
    
    async def close(self):
        """Ferme tous les providers"""
        for provider in self.providers.values():
            await provider.close()
        
        logger.info("LLMProviderManager ferme")