"""Financial taxonomy definitions for Harena conversation service."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List

__all__ = ["FinancialCategory", "FINANCIAL_TAXONOMY", "get_subcategories"]


class FinancialCategory(str, Enum):
    """High level financial categories used in Harena."""

    INCOME = "INCOME"
    EXPENSE = "EXPENSE"
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"


FINANCIAL_TAXONOMY: Dict[FinancialCategory, List[str]] = {
    FinancialCategory.INCOME: ["salary", "bonus", "interest", "dividend"],
    FinancialCategory.EXPENSE: [
        "groceries",
        "transport",
        "housing",
        "utilities",
        "entertainment",
        "health",
    ],
    FinancialCategory.ASSET: ["cash", "checking_account", "savings_account", "investment"],
    FinancialCategory.LIABILITY: ["mortgage", "loan", "credit_card", "overdraft"],
}


def get_subcategories(category: FinancialCategory) -> List[str]:
    """Return subcategories for a financial category."""
    return FINANCIAL_TAXONOMY.get(category, [])
