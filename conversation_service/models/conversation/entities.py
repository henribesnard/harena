"""Entity models used in the conversation service."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AmountEntity(BaseModel):
    """Represents a monetary amount mentioned in a conversation."""

    value: float
    currency: str


class MerchantEntity(BaseModel):
    """Represents a merchant extracted from a conversation."""

    name: str


class DateEntity(BaseModel):
    """Represents a date extracted from a conversation."""

    date: date


class CategoryEntity(BaseModel):
    """Represents a transaction category extracted from a conversation."""

    name: str


class TransactionTypeEntity(BaseModel):
    """Represents a transaction type such as credit or debit."""

    transaction_type: str


class EntitiesExtractionResult(BaseModel):
    """Aggregates all entities extracted from a conversation."""

    amounts: List[AmountEntity] = Field(default_factory=list)
    merchants: List[MerchantEntity] = Field(default_factory=list)
    dates: List[DateEntity] = Field(default_factory=list)
    categories: List[CategoryEntity] = Field(default_factory=list)
    transaction_types: List[TransactionTypeEntity] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AmountEntity",
    "MerchantEntity",
    "DateEntity",
    "CategoryEntity",
    "TransactionTypeEntity",
    "EntitiesExtractionResult",
]
