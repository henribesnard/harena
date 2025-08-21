"""Custom Pydantic validators for the Harena conversation service."""

from __future__ import annotations

import re
from pydantic import field_validator

__all__ = ["CurrencyAmountValidators"]


class CurrencyAmountValidators:
    """Reusable mixin providing currency and amount validators."""

    @field_validator("currency")
    @classmethod
    def validate_currency_code(cls, value: str) -> str:
        """Ensure currency codes follow ISO 4217."""
        if not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("currency must be a 3-letter ISO code")
        return value

    @field_validator("amount")
    @classmethod
    def validate_positive_amount(cls, value: float) -> float:
        """Verify that monetary amounts are positive."""
        if value < 0:
            raise ValueError("amount must be positive")
        return value
