"""Utilities for intent classification agents used in tests."""

from __future__ import annotations

from typing import Dict, Optional

try:  # pragma: no cover - optional heavy dependency
    from .intent_classifier import IntentClassifierAgent  # type: ignore
except Exception:  # pragma: no cover
    IntentClassifierAgent = None  # type: ignore

from ..models.core_models import IntentResult


class IntentClassificationCache:
    """Simple in-memory cache for intent classification results."""

    def __init__(self) -> None:
        self._store: Dict[str, IntentResult] = {}
        self.hits: int = 0

    def set(self, message: str, result: IntentResult) -> None:
        """Store ``result`` for ``message`` in the cache."""
        self._store[message] = result

    def get(self, message: str) -> Optional[IntentResult]:
        """Retrieve a cached result for ``message`` if available."""
        result = self._store.get(message)
        if result is not None:
            self.hits += 1
        return result

    def clear(self) -> None:
        """Clear all cached entries and reset hit counter."""
        self._store.clear()
        self.hits = 0


__all__ = ["IntentClassifierAgent", "IntentClassificationCache"]
