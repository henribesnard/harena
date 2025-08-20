"""
Utility definitions for intent taxonomies.

This module centralises mapping between intent types, their high level
categories and suggested actions.  The data is deliberately lightweight and
can be expanded easily as new intents are introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..models.financial_models import IntentCategory


@dataclass(frozen=True)
class IntentTaxonomy:
    """Mapping of an intent to its category and suggested actions."""

    intent_type: str
    category: IntentCategory
    suggested_actions: List[str]


# Minimal taxonomy used by the MVP.  Additional intents can be appended to this
# list without modifying calling code.
TAXONOMY: List[IntentTaxonomy] = [
    IntentTaxonomy("TRANSACTION_SEARCH", IntentCategory.FINANCIAL_QUERY, ["list_transactions"]),
    IntentTaxonomy(
        "SPENDING_ANALYSIS",
        IntentCategory.SPENDING_ANALYSIS,
        ["calculate_total", "spending_breakdown"],
    ),
    IntentTaxonomy("BALANCE_INQUIRY", IntentCategory.BALANCE_INQUIRY, ["get_current_balance"]),
]


def get_taxonomy(intent_type: str) -> Optional[IntentTaxonomy]:
    """Return taxonomy information for a given intent type."""

    intent_type = intent_type.upper()
    for tax in TAXONOMY:
        if tax.intent_type == intent_type:
            return tax
    return None
