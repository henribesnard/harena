"""Financial domain data models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .enums import Currency
from core.validators import non_empty_str, positive_number


class Transaction(BaseModel):
    """A financial transaction."""

    amount: float = Field(..., gt=0)
    currency: Currency = Field(...)
    description: Optional[str] = Field(default=None, max_length=500)

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, v: float) -> float:
        return float(positive_number(v))

    @field_validator("description")
    @classmethod
    def _validate_desc(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return non_empty_str(v)
        return v


class AccountBalance(BaseModel):
    """Balance information for an account."""

    account_id: str = Field(...)
    balance: float = Field(..., ge=0)
    currency: Currency = Field(default=Currency.USD)

    @field_validator("account_id")
    @classmethod
    def _validate_account_id(cls, v: str) -> str:
        return non_empty_str(v)

    @field_validator("balance")
    @classmethod
    def _validate_balance(cls, v: float) -> float:
        return float(positive_number(v))


__all__ = ["Transaction", "AccountBalance"]
