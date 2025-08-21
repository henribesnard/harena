"""Intent classification cache utilities for tests."""
from typing import Dict, Optional

from ..models.conversation_models import IntentResult


class IntentClassificationCache:
    """Simple in-memory cache for intent classification results."""

    def __init__(self) -> None:
        self._store: Dict[str, IntentResult] = {}
        self.hits = 0

    def set(self, message: str, result: IntentResult) -> None:
        self._store[message] = result

    def get(self, message: str) -> Optional[IntentResult]:
        result = self._store.get(message)
        if result is not None:
            self.hits += 1
        return result
