"""Intent classification utilities with Redis-backed cache."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

try:  # pragma: no cover - optional runtime dependency
    from .intent_classifier import IntentClassifierAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    IntentClassifierAgent = None  # type: ignore

from ..clients.cache_client import CacheClient
from ..core.cache_manager import CacheManager
from ..models.core_models import IntentResult

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class IntentClassificationCache:
    """Cache for intent classification results using Redis when available."""

    def __init__(self, ttl: int = 600) -> None:
        client = CacheClient(REDIS_URL)
        self._cache = CacheManager(cache_client=client, default_ttl=ttl)
        self.hits = 0
        # Tests do not provide user identifiers; a fixed value isolates keys.
        self._user_id = "test"

    def get(self, message: str) -> Optional[IntentResult]:
        result = asyncio.run(self._cache.get(message, self._user_id))
        if result is not None:
            self.hits += 1
        return result

    def set(self, message: str, result: IntentResult) -> None:
        asyncio.run(self._cache.set(message, result, user_id=self._user_id))

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0


__all__ = ["IntentClassifierAgent", "IntentClassificationCache"]
