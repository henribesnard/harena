"""Models for entities extracted from user conversations.

These models represent amounts, merchants, dates and optional metadata
that can be detected in a user request. They are built with Pydantic to
benefit from validation and serialization utilities.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AmountEntity(BaseModel):
    """Represents a monetary amount mentioned by the user."""

    model_config = ConfigDict(validate_assignment=True)

    value: float
    currency: str = "EUR"

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Amount value must be positive")
        return v

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        if not v or len(v) != 3:
            raise ValueError("Currency must be a 3-letter code")
        return v.upper()


class MerchantEntity(BaseModel):
    """Represents a merchant or vendor name."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    name: str
    category: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Merchant name cannot be empty")
        return v.strip()


class DateEntity(BaseModel):
    """Represents a date or date range."""

    model_config = ConfigDict(validate_assignment=True)

    start_date: date
    end_date: Optional[date] = None

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, v: Optional[date], info):
        start = info.data.get("start_date")
        if v and start and v < start:
            raise ValueError("end_date cannot be before start_date")
        return v


class ConversationEntities(BaseModel):
    """Container for all entities extracted from a message."""

    model_config = ConfigDict(validate_assignment=True)

    amounts: List[AmountEntity] = Field(default_factory=list)
    merchants: List[MerchantEntity] = Field(default_factory=list)
    dates: List[DateEntity] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

