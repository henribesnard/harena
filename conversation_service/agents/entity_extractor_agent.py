"""Entity extraction utilities with Redis-backed cache."""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - optional runtime dependency
    from .entity_extractor import EntityExtractorAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    EntityExtractorAgent = None  # type: ignore

from ..clients.cache_client import CacheClient
from ..core.cache_manager import CacheManager
from ..models.core_models import FinancialEntity

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class EntityExtractionCache:
    """Cache extracted entities keyed by message and intent."""

    def __init__(self, ttl: int = 900) -> None:
        client = CacheClient(REDIS_URL)
        self._cache = CacheManager(cache_client=client, default_ttl=ttl)
        self.hits = 0
        self._user_id = "test"

    @staticmethod
    def _make_key(message: str, intent: str) -> str:
        return f"{message}:{intent}"

    def get(self, message: str, intent: str) -> Optional[Dict[str, object]]:
        key = self._make_key(message, intent)
        entities = asyncio.run(self._cache.get(key, self._user_id))
        if entities is None:
            return None
        self.hits += 1
        return {"entities": entities, "cached": True}

    def set(self, message: str, intent: str, entities: List[FinancialEntity]) -> None:
        key = self._make_key(message, intent)
        asyncio.run(self._cache.set(key, entities, user_id=self._user_id))

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0


__all__ = ["EntityExtractorAgent", "EntityExtractionCache"]
