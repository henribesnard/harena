"""Entity extraction utilities with a lightweight in-memory cache."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

# Default cache time-to-live for entity extraction results (seconds)
DEFAULT_TTL = 180

try:  # pragma: no cover - optional runtime dependency
    from .entity_extractor import EntityExtractorAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    EntityExtractorAgent = None  # type: ignore

from ..models.core_models import FinancialEntity


class EntityExtractionCache:
    """Simple in-memory cache for extracted entities.

    Cache keys include the ``user_id``, ``intent`` and original ``message``
    separated by colons (``"{user_id}:{intent}:{message}"``).  Cached
    entries expire after ``DEFAULT_TTL`` seconds unless a different
    ``ttl`` is specified when setting or retrieving.
    """

    def __init__(self) -> None:
        # Cache extracted entities along with the timestamp they were cached
        # to allow TTL-based invalidation.
        self._store: Dict[str, Tuple[List[FinancialEntity], float]] = {}
        self.hits: int = 0

    @staticmethod
    def _make_key(user_id: str, message: str, intent: str) -> str:
        # Include intent before the message to avoid collisions when the
        # same message is used for different intents.
        return f"{user_id}:{intent}:{message}"

    def get(
        self, user_id: str, message: str, intent: str, ttl: int = DEFAULT_TTL
    ) -> Optional[Dict[str, object]]:
        key = self._make_key(user_id, message, intent)
        entry = self._store.get(key)
        if entry is None:
            return None
        entities, timestamp = entry
        if time.time() - timestamp > ttl:
            self._store.pop(key, None)
            return None
        self.hits += 1
        return {"entities": entities, "cached": True}

    def set(
        self,
        user_id: str,
        message: str,
        intent: str,
        entities: List[FinancialEntity],
        ttl: int = DEFAULT_TTL,
    ) -> None:
        key = self._make_key(user_id, message, intent)
        self._store[key] = (entities, time.time())

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


__all__ = ["EntityExtractorAgent", "EntityExtractionCache"]
