"""Utility classes for intent classification agents.

This module provides a tiny in-memory cache used in tests.  The cache
stores `IntentResult` objects keyed by the original user query and keeps
track of cache hit statistics.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..models.core_models import IntentResult


class IntentClassificationCache:
    """Cache for intent classification results.

    The cache exposes two simple operations:

    - :meth:`set` to store a result for a given query
    - :meth:`get` to retrieve a cached result

    Each successful retrieval increments the :attr:`hits` counter so tests
    can assert cache effectiveness.
    """

    def __init__(self) -> None:
        self._store: Dict[str, IntentResult] = {}
        self.hits: int = 0

    def set(self, query: str, result: IntentResult) -> None:
        """Store an intent classification result.

        Parameters
        ----------
        query:
            Original user query used as cache key.
        result:
            The :class:`IntentResult` produced by the classifier.
        """
        self._store[query] = result

    def get(self, query: str) -> Optional[IntentResult]:
        """Retrieve a cached result if present.

        Each successful lookup increments the :attr:`hits` counter.  When the
        query is not found ``None`` is returned and the counter is unchanged.
        """
        result = self._store.get(query)
        if result is not None:
            self.hits += 1
        return result
