import logging
from typing import List, Dict, Optional, Any

from openai import AsyncOpenAI

from config_service.config import settings
from .cache_client import CacheClient


logger = logging.getLogger(__name__)


class OpenAIClient:
    """Simple wrapper around the official OpenAI client.

    This helper centralises configuration (API key, base URL, model names)
    and provides a light caching layer to avoid repeated API calls during
    development and tests.  Only the minimal features needed by the project
    are implemented for now.
    """

    def __init__(self, cache: Optional[CacheClient] = None) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            timeout=settings.OPENAI_TIMEOUT,
        )
        self._default_model = settings.OPENAI_CHAT_MODEL
        self._embedding_model = settings.OPENAI_EMBEDDING_MODEL
        self._cache = cache

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Return the text content of a chat completion.

        If a ``cache_key`` is supplied and a cache client is available, the
        response will be stored using this key.  Subsequent calls with the same
        key will return the cached value.
        """

        chosen_model = model or self._default_model

        if cache_key and self._cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

        response = await self._client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", settings.OPENAI_MAX_TOKENS),
            temperature=kwargs.get("temperature", settings.OPENAI_TEMPERATURE),
            top_p=kwargs.get("top_p", settings.OPENAI_TOP_P),
        )
        content = response.choices[0].message["content"]

        if cache_key and self._cache:
            await self._cache.set(cache_key, content, ttl=settings.LLM_CACHE_TTL)

        return content

    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Return the embedding vector for the provided text."""
        chosen_model = model or self._embedding_model
        response = await self._client.embeddings.create(model=chosen_model, input=text)
        return response.data[0].embedding
