from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class IntentType(str, Enum):
    """Supported high level user intents."""

    BALANCE_INQUIRY = "BALANCE_INQUIRY"
    MERCHANT_ANALYSIS = "MERCHANT_ANALYSIS"
    UNKNOWN = "UNKNOWN"


class EntityType(str, Enum):
    """Types of financial entities extracted from user messages."""

    MERCHANT = "MERCHANT"
    BENEFICIARY = "BENEFICIARY"
    AMOUNT = "AMOUNT"


@dataclass
class FinancialEntity:
    """Structured representation of an entity extracted from text."""

    entity_type: EntityType
    raw_value: str
    normalized_value: str
    confidence: float

    def is_action_related(self) -> bool:
        """Return ``True`` if this entity is related to an actionable item."""

        return self.entity_type in {EntityType.BENEFICIARY, EntityType.MERCHANT}

    def to_search_filter(self) -> Dict[str, Any]:
        """Convert the entity into a dictionary usable as a search filter."""

        if self.entity_type == EntityType.AMOUNT:
            amount = float(self.normalized_value)
            return {
                "range": {"amount_abs": {"gte": amount * 0.9, "lte": amount * 1.1}}
            }
        if self.entity_type == EntityType.MERCHANT:
            return {
                "bool": {
                    "should": [{"match": {"merchant_name": self.normalized_value}}]
                }
            }
        return {"match": {self.entity_type.value.lower(): self.normalized_value}}


@dataclass
class IntentResult:
    """Result returned by the intent classification agent."""

    intent: IntentType
    confidence: float
    reasoning: Optional[str] = None


@dataclass
class AgentResponse:
    """Standard response object returned by conversation agents."""

    agent_name: str
    success: bool
    result: Optional[Any] = None
    processing_time_ms: int = 0
    error_message: Optional[str] = None
    tokens_used: int = 0
    cached: bool = False


__all__ = [
    "FinancialEntity",
    "IntentResult",
    "IntentType",
    "AgentResponse",
    "EntityType",
]
