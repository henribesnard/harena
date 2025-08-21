"""Entity extraction utilities with a lightweight in-memory cache."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - optional runtime dependency
    from .entity_extractor import EntityExtractorAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    EntityExtractorAgent = None  # type: ignore

from ..models.core_models import FinancialEntity


class EntityExtractionCache:
    """Simple in-memory cache for extracted entities."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, List[FinancialEntity]]] = {}
        self.hits: int = 0

    @staticmethod
    def _make_key(user_id: str, message: str, intent: str) -> str:
        return f"{user_id}:{message}:{intent}"

    def get(
        self, user_id: str, message: str, intent: str, ttl: int = 180
    ) -> Optional[Dict[str, object]]:
        key = self._make_key(user_id, message, intent)
        entry = self._store.get(key)
        if entry is None:
            return None
        inserted_at, entities = entry
        if time.time() - inserted_at > ttl:
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
        ttl: int = 180,
    ) -> None:
        key = self._make_key(user_id, message, intent)
        self._store[key] = (time.time(), entities)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


__all__ = ["EntityExtractorAgent", "EntityExtractionCache"]
