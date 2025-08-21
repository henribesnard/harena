"""Wrapper module exposing intent classification utilities.

This wrapper re-exports :class:`IntentClassifierAgent` from the existing
``intent_classifier`` module and provides a lightweight in-memory cache used in
unit tests.  The cache stores :class:`IntentResult` instances keyed by the
user's prompt.
"""

from typing import Dict, Optional

# The real ``IntentClassifierAgent`` pulls in optional runtime dependencies
# (OpenAI clients, HTTP libraries, etc.).  Importing it eagerly would cause the
# test environment to require those extras.  We therefore attempt to import the
# class lazily and fall back to ``None`` when those dependencies are missing.
try:  # pragma: no cover - defensive import
    from .intent_classifier import IntentClassifierAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    IntentClassifierAgent = None  # type: ignore

from ..models.core_models import IntentResult


class IntentClassificationCache:
    """Simple in-memory cache for intent classification results.

    The cache is intentionally minimal – it supports storing and retrieving
    :class:`IntentResult` objects by the original user message and tracks the
    number of cache hits for testing purposes.
    """

    def __init__(self) -> None:
        self._store: Dict[str, IntentResult] = {}
        self.hits: int = 0

    def get(self, message: str) -> Optional[IntentResult]:
        """Retrieve a cached result for ``message`` if available."""
        result = self._store.get(message)
        if result is not None:
            self.hits += 1
        return result

    def set(self, message: str, result: IntentResult) -> None:
        """Store ``result`` for ``message`` in the cache."""
        self._store[message] = result

    def clear(self) -> None:
        """Clear all cached entries and reset hit counter."""
        self._store.clear()
        self.hits = 0


__all__ = ["IntentClassifierAgent", "IntentClassificationCache"]
