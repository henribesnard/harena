"""Utilities for intent classification agents used in tests."""
from __future__ import annotations

from typing import Dict, Optional

try:  # pragma: no cover - optional dependency
    from .intent_classifier import IntentClassifierAgent  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    IntentClassifierAgent = None  # type: ignore

from ..models.core_models import IntentResult


class IntentClassificationCache:
    """Simple in-memory cache for intent classification results."""

    def __init__(self) -> None:
        self._store: Dict[str, IntentResult] = {}
        self.hits: int = 0

    def get(self, message: str) -> Optional[IntentResult]:
        result = self._store.get(message)
        if result is not None:
            self.hits += 1
        return result

    def set(self, message: str, result: IntentResult) -> None:
        self._store[message] = result

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


__all__ = ["IntentClassifierAgent", "IntentClassificationCache"]
