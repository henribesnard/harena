"""Entity extraction utilities with a lightweight in-memory cache."""

from __future__ import annotations

from typing import Dict, List, Optional

try:  # pragma: no cover - optional runtime dependency
    from .entity_extractor import EntityExtractorAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    EntityExtractorAgent = None  # type: ignore

from ..models.core_models import FinancialEntity


class EntityExtractionCache:
    """Simple in-memory cache for extracted entities."""

    def __init__(self) -> None:
        self._store: Dict[str, List[FinancialEntity]] = {}
        self.hits: int = 0

    @staticmethod
    def _make_key(message: str, intent: str) -> str:
        return f"{message}:{intent}"

    def get(self, message: str, intent: str) -> Optional[Dict[str, object]]:
        key = self._make_key(message, intent)
        entities = self._store.get(key)
        if entities is None:
            return None
        self.hits += 1
        return {"entities": entities, "cached": True}

    def set(self, message: str, intent: str, entities: List[FinancialEntity]) -> None:
        key = self._make_key(message, intent)
        self._store[key] = entities

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


__all__ = ["EntityExtractorAgent", "EntityExtractionCache"]
