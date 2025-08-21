"""Caches for entity extraction agents.

The :class:`EntityExtractionCache` mirrors the API of
:class:`~conversation_service.agents.intent_classifier_agent.IntentClassificationCache`.
It stores lists of `FinancialEntity` instances keyed by both the user query
and the type of analysis performed.  Each cache hit increments a counter so
that callers can monitor effectiveness.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
