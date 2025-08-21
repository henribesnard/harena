"""Entity extraction cache utilities for tests."""
from typing import Dict, List, Tuple, Optional

from ..models.financial_models import FinancialEntity


class EntityExtractionCache:
    """Simple in-memory cache storing extracted entities per message and intent."""

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], List[FinancialEntity]] = {}
        self.hits = 0

    def set(self, message: str, intent: str, entities: List[FinancialEntity]) -> None:
        self._store[(message, intent)] = entities

    def get(self, message: str, intent: str) -> Optional[Dict[str, List[FinancialEntity] | bool]]:
        key = (message, intent)
        if key in self._store:
            self.hits += 1
            return {"entities": self._store[key], "cached": True}
        return None
