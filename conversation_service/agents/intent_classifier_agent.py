"""Utilities for intent classification agents used in tests."""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

# Default cache time-to-live for intent classification results (seconds)
DEFAULT_TTL = 300

try:  # pragma: no cover - optional dependency
    from .intent_classifier import IntentClassifierAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    IntentClassifierAgent = None  # type: ignore

from ..models.core_models import IntentResult


class IntentClassificationCache:
    def __init__(self) -> None:
        # Store the cached result alongside the time it was inserted so
        # we can easily expire entries based on a TTL.
        self._store: Dict[str, Tuple[IntentResult, float]] = {}
        self.hits: int = 0

    @staticmethod
    def _make_key(user_id: str, message: str) -> str:
        return f"{user_id}:{message}"

    def get(
        self, user_id: str, message: str, ttl: int = DEFAULT_TTL
    ) -> Optional[IntentResult]:
        key = self._make_key(user_id, message)
        entry = self._store.get(key)
        if entry is None:
            return None
        result, timestamp = entry
        if time.time() - timestamp > ttl:
            # Remove expired entry
            self._store.pop(key, None)
            return None
        self.hits += 1
        return result

    def set(
        self, user_id: str, message: str, result: IntentResult, ttl: int = DEFAULT_TTL
    ) -> None:
        key = self._make_key(user_id, message)
        self._store[key] = (result, time.time())

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


__all__ = ["IntentClassifierAgent", "IntentClassificationCache"]
