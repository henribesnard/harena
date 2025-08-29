import pytest
from datetime import datetime, timezone, date

from conversation_service.models.conversation import (
    ExtractedAmount,
    ExtractedMerchant,
    ExtractedDate,
    EntityExtractionResult,
)
from conversation_service.models.responses.autogen_conversation_response import (
    AutogenConversationResponse,
)
from conversation_service.models.responses.conversation_responses import (
    IntentClassificationResult,
    AgentMetrics,
)
from conversation_service.prompts.harena_intents import HarenaIntentType


def _make_intent() -> IntentClassificationResult:
    return IntentClassificationResult(
        intent_type=HarenaIntentType.GREETING,
        confidence=0.9,
        reasoning="salutation",
        original_message="salut",
        category="TEST",
        is_supported=True,
    )


def _make_metrics() -> AgentMetrics:
    return AgentMetrics(
        agent_used="test_agent",
        model_used="test_model",
        tokens_consumed=10,
        processing_time_ms=1,
        confidence_threshold_met=True,
        cache_hit=False,
    )


# --- Entity validations ----------------------------------------------------

def test_extracted_amount_valid_and_invalid_currency():
    amt = ExtractedAmount(value=10.5, currency="eur")
    assert amt.currency == "EUR"

    with pytest.raises(ValueError):
        ExtractedAmount(value=5, currency="EURO")


def test_extracted_merchant_validation():
    merchant = ExtractedMerchant(name="Amazon")
    assert merchant.name == "Amazon"

    with pytest.raises(ValueError):
        ExtractedMerchant(name="   ")


def test_extracted_date_parsing():
    d = ExtractedDate(date="2024-05-20")
    assert d.date == date(2024, 5, 20)

    with pytest.raises(ValueError):
        ExtractedDate(date="not-a-date")


# --- EntityExtractionResult helpers ---------------------------------------

def test_from_llm_response_builds_entities():
    payload = {
        "entities": [
            {"type": "amount", "value": 100, "currency": "EUR"},
            {"type": "merchant", "value": "Amazon"},
            {"type": "date", "value": "2024-01-01"},
            {"type": "category", "value": "Shopping"},
            {"type": "transaction_type", "value": "DEBIT"},
        ],
        "extraction_metadata": {"model": "test"},
        "team_context": {"intent": "TRANSACTION_SEARCH"},
        "global_confidence": 0.75,
    }

    result = EntityExtractionResult.from_llm_response(payload)

    assert [a.value for a in result.amounts] == [100.0]
    assert result.merchants[0].name == "Amazon"
    assert result.dates[0].date == date(2024, 1, 1)
    assert result.categories[0].name == "Shopping"
    assert result.transaction_types[0].transaction_type == "DEBIT"
    assert result.extraction_metadata["model"] == "test"
    assert result.team_context["intent"] == "TRANSACTION_SEARCH"
    assert result.global_confidence == 0.75


def test_create_fallback_result():
    fb = EntityExtractionResult.create_fallback_result("timeout", {"foo": "bar"})

    assert fb["extraction_success"] is False
    assert isinstance(fb["entities"], EntityExtractionResult)
    assert fb["entities"].extraction_metadata["error"] == "timeout"
    assert fb["team_context"] == {"foo": "bar"}


# --- AutogenConversationResponse integration -------------------------------

def test_autogen_conversation_response_accepts_entities():
    response = AutogenConversationResponse(
        user_id=1,
        message="hello",
        timestamp=datetime.now(timezone.utc),
        intent=_make_intent(),
        agent_metrics=_make_metrics(),
        processing_time_ms=1,
    )

    assert response.user_id == 1
    assert response.entities is None

    ent = EntityExtractionResult.from_llm_response({"entities": []})
    response.entities = ent

    assert response.entities is ent
    assert response.intent.intent_type == HarenaIntentType.GREETING
