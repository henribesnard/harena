"""Utilities for intent classification agents used in tests."""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from .intent_classifier import IntentClassifierAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    IntentClassifierAgent = None  # type: ignore

from ..models.core_models import IntentResult


class IntentClassificationCache:
    """Simple in-memory cache for intent classification results."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, IntentResult]] = {}
        self.hits: int = 0

    @staticmethod
    def _make_key(user_id: str, message: str) -> str:
        return f"{user_id}:{message}"

    def get(self, user_id: str, message: str, ttl: int = 300) -> Optional[IntentResult]:
        key = self._make_key(user_id, message)
        entry = self._store.get(key)
        if entry is None:
            return None
        inserted_at, result = entry
        if time.time() - inserted_at > ttl:
            self._store.pop(key, None)
            return None
        self.hits += 1
        return result

    def set(self, user_id: str, message: str, result: IntentResult, ttl: int = 300) -> None:
        key = self._make_key(user_id, message)
        self._store[key] = (time.time(), result)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


__all__ = ["IntentClassifierAgent", "IntentClassificationCache"]
