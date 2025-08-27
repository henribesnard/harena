"""Utility functions to validate coherence between entities and intents."""
from __future__ import annotations

from typing import List

from conversation_service.prompts.harena_intents import HarenaIntentType

from .entities import ConversationEntities

# Intents that require specific entity types
INTENTS_REQUIRING_AMOUNT = {
    HarenaIntentType.SEARCH_BY_AMOUNT,
    HarenaIntentType.SEARCH_BY_AMOUNT_AND_DATE,
}

INTENTS_REQUIRING_MERCHANT = {
    HarenaIntentType.SEARCH_BY_MERCHANT,
    HarenaIntentType.MERCHANT_INQUIRY,
}

INTENTS_REQUIRING_DATE = {
    HarenaIntentType.SEARCH_BY_DATE,
    HarenaIntentType.SEARCH_BY_AMOUNT_AND_DATE,
}


def validate_entities_intent(intent: HarenaIntentType, entities: ConversationEntities) -> List[str]:
    """Validate that provided entities match the expected intent requirements.

    Args:
        intent: Predicted user intent.
        entities: Entities extracted from the user message.

    Returns:
        A list of issues found. The list is empty when validation succeeds.
    """
    issues: List[str] = []

    if intent in INTENTS_REQUIRING_AMOUNT and not entities.amounts:
        issues.append("amounts required for this intent")

    if intent in INTENTS_REQUIRING_MERCHANT and not entities.merchants:
        issues.append("merchants required for this intent")

    if intent in INTENTS_REQUIRING_DATE and not entities.dates:
        issues.append("dates required for this intent")

    return issues

