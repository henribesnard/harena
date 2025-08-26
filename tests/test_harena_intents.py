import pytest
from conversation_service.prompts.harena_intents import HarenaIntentType, INTENT_DESCRIPTIONS, INTENT_CATEGORIES

ALL_INTENTS = [
    HarenaIntentType.TRANSACTION_SEARCH,
    HarenaIntentType.SEARCH_BY_DATE,
    HarenaIntentType.SEARCH_BY_AMOUNT,
    HarenaIntentType.SEARCH_BY_MERCHANT,
    HarenaIntentType.SEARCH_BY_CATEGORY,
    HarenaIntentType.SEARCH_BY_AMOUNT_AND_DATE,
    HarenaIntentType.SEARCH_BY_OPERATION_TYPE,
    HarenaIntentType.SEARCH_BY_TEXT,
    HarenaIntentType.COUNT_TRANSACTIONS,
    HarenaIntentType.MERCHANT_INQUIRY,
    HarenaIntentType.FILTER_REQUEST,
    HarenaIntentType.SPENDING_ANALYSIS,
    HarenaIntentType.SPENDING_ANALYSIS_BY_CATEGORY,
    HarenaIntentType.SPENDING_ANALYSIS_BY_PERIOD,
    HarenaIntentType.SPENDING_COMPARISON,
    HarenaIntentType.TREND_ANALYSIS,
    HarenaIntentType.CATEGORY_ANALYSIS,
    HarenaIntentType.COMPARISON_QUERY,
    HarenaIntentType.BALANCE_INQUIRY,
    HarenaIntentType.ACCOUNT_BALANCE_SPECIFIC,
    HarenaIntentType.BALANCE_EVOLUTION,
    HarenaIntentType.GREETING,
    HarenaIntentType.CONFIRMATION,
    HarenaIntentType.CLARIFICATION,
    HarenaIntentType.GENERAL_QUESTION,
    HarenaIntentType.TRANSFER_REQUEST,
    HarenaIntentType.PAYMENT_REQUEST,
    HarenaIntentType.CARD_BLOCK,
    HarenaIntentType.BUDGET_INQUIRY,
    HarenaIntentType.GOAL_TRACKING,
    HarenaIntentType.EXPORT_REQUEST,
    HarenaIntentType.OUT_OF_SCOPE,
    HarenaIntentType.UNCLEAR_INTENT,
    HarenaIntentType.UNKNOWN,
    HarenaIntentType.TEST_INTENT,
    HarenaIntentType.ERROR,
]


@pytest.mark.parametrize("intent", ALL_INTENTS)
def test_intent_has_description(intent):
    """Every intent should have a textual description."""
    assert intent in INTENT_DESCRIPTIONS


@pytest.mark.parametrize("intent", ALL_INTENTS)
def test_intent_in_categories(intent):
    """Every intent should belong to one of the intent categories."""
    assert any(
        intent in intent_list for intent_list in INTENT_CATEGORIES.values()
    )
