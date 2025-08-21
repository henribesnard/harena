"""Caches for entity extraction agents.

The :class:`EntityExtractionCache` mirrors the API of
:class:`~conversation_service.agents.intent_classifier_agent.IntentClassificationCache`.
It stores lists of `FinancialEntity` instances keyed by both the user query
and the type of analysis performed.  Each cache hit increments a counter so
that callers can monitor effectiveness.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    """Simple in-memory cache for entity extraction results."""

    def __init__(self) -> None:
        # Keyed by (query, analysis_type)
        self._store: Dict[Tuple[str, str], List[FinancialEntity]] = {}
        self.hits: int = 0

    def set(self, query: str, analysis_type: str, entities: List[FinancialEntity]) -> None:
        """Store extracted entities for a query/analysis pair."""
        self._store[(query, analysis_type)] = entities

    def get(self, query: str, analysis_type: str) -> Optional[Dict[str, object]]:
        """Retrieve cached entities if present.

        Returns a dictionary containing the cached entities with a flag
        indicating the result was served from cache.  The hit counter is only
        incremented when a cached value is found.
        """
        entities = self._store.get((query, analysis_type))
        if entities is None:
            return None
        self.hits += 1
        return {"cached": True, "entities": entities}
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
