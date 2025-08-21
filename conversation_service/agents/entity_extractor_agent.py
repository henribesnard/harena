"""Wrapper module for entity extraction utilities.

This file re-exports :class:`EntityExtractorAgent` from the existing
``entity_extractor`` module and defines :class:`EntityExtractionCache` used in
unit tests.  The cache stores lists of :class:`FinancialEntity` objects keyed by
both the user's prompt and the intent type.
"""

from typing import Dict, List, Optional, Tuple

# Importing the full ``EntityExtractorAgent`` would require optional runtime
# dependencies.  We attempt to import it lazily and fall back to ``None`` when
# those dependencies are unavailable.
try:  # pragma: no cover - defensive import
    from .entity_extractor import EntityExtractorAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    EntityExtractorAgent = None  # type: ignore

from ..models.core_models import FinancialEntity


class EntityExtractionCache:
    """In-memory cache of extracted entities."""

    def __init__(self) -> None:
        # Keyed by (message, intent)
        self._store: Dict[Tuple[str, str], List[FinancialEntity]] = {}
        self.hits: int = 0

    @staticmethod
    def _make_key(message: str, intent: str) -> Tuple[str, str]:
        return message, str(intent)

    def get(self, message: str, intent: str) -> Optional[Dict[str, object]]:
        """Return cached entities for ``message`` and ``intent`` if available."""
        key = self._make_key(message, intent)
        entities = self._store.get(key)
        if entities is None:
            return None
        self.hits += 1
        return {"entities": entities, "cached": True}

    def set(
        self, message: str, intent: str, entities: List[FinancialEntity]
    ) -> None:
        """Store ``entities`` for the given ``message`` and ``intent``."""
        key = self._make_key(message, intent)
        self._store[key] = entities

    def clear(self) -> None:
        """Clear the cache and reset hit counter."""
        self._store.clear()
        self.hits = 0


__all__ = ["EntityExtractorAgent", "EntityExtractionCache"]
