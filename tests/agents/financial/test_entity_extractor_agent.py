import json
from datetime import date
from types import SimpleNamespace

import pytest

from conversation_service.agents.financial.entity_extractor import (
    EntityExtractorAgent,
)
from conversation_service.agents.financial import entity_extractor as ee
from conversation_service.agents.financial.intent_classifier import (
    IntentClassifierAgent,
)


@pytest.fixture(autouse=True)
def patch_entity_models(monkeypatch):
    """Provide lightweight stand-ins for Pydantic models used in extraction."""

    class AmountEntity(SimpleNamespace):
        def __init__(self, value: float, currency: str):
            super().__init__(value=float(value), currency=currency)

    class MerchantEntity(SimpleNamespace):
        def __init__(self, name: str):
            super().__init__(name=name)

    class DateEntity(SimpleNamespace):
        def __init__(self, date):
            super().__init__(date=date)

    class CategoryEntity(SimpleNamespace):
        def __init__(self, name: str):
            super().__init__(name=name)

    class TransactionTypeEntity(SimpleNamespace):
        def __init__(self, transaction_type: str):
            super().__init__(transaction_type=transaction_type)

    class EntitiesExtractionResult:
        def __init__(self, extraction_metadata=None):
            self.amounts = []
            self.merchants = []
            self.dates = []
            self.categories = []
            self.transaction_types = []
            self.extraction_metadata = extraction_metadata or {}

    monkeypatch.setattr(ee, "AmountEntity", AmountEntity)
    monkeypatch.setattr(ee, "MerchantEntity", MerchantEntity)
    monkeypatch.setattr(ee, "DateEntity", DateEntity)
    monkeypatch.setattr(ee, "CategoryEntity", CategoryEntity)
    monkeypatch.setattr(ee, "TransactionTypeEntity", TransactionTypeEntity)
    monkeypatch.setattr(ee, "EntitiesExtractionResult", EntitiesExtractionResult)


@pytest.mark.asyncio
async def test_amount_extraction(monkeypatch):
    """LLM reply yields multiple amount entities with normalization."""
    agent = EntityExtractorAgent()

    async def fake_reply(prompt: str):
        return json.dumps(
            {
                "extraction_success": True,
                "entities": [
                    {"type": "amount", "value": 150, "currency": "EUR"},
                    {"type": "amount", "value": 2000, "currency": "EUR"},
                    {"type": "amount", "value": 50, "currency": "EUR"},
                ],
            }
        )

    monkeypatch.setattr(agent, "a_generate_reply", fake_reply, raising=False)

    team_ctx = {
        "original_message": "150€ + 2k + environ 50",
        "intent": "TRANSACTION_SEARCH",
    }
    result = await agent.extract_entities_from_team_context(team_ctx)

    amounts = [a.value for a in result["entities"].amounts]
    assert amounts == [150.0, 2000.0, 50.0]


@pytest.mark.asyncio
async def test_relative_date_extraction(monkeypatch):
    """Relative dates are converted to ISO format dates."""
    agent = EntityExtractorAgent()
    today = date.today()
    this_month = today.replace(day=1)
    last_january_year = today.year - 1 if today.month <= 1 else today.year
    last_january = date(last_january_year, 1, 1)

    async def fake_reply(prompt: str):
        return json.dumps(
            {
                "extraction_success": True,
                "entities": [
                    {"type": "date", "value": this_month.isoformat()},
                    {"type": "date", "value": last_january.isoformat()},
                ],
            }
        )

    monkeypatch.setattr(agent, "a_generate_reply", fake_reply, raising=False)
    team_ctx = {"original_message": "ce mois et janvier dernier", "intent": "TRANSACTION_SEARCH"}
    result = await agent.extract_entities_from_team_context(team_ctx)

    extracted = [d.date for d in result["entities"].dates]
    assert extracted == [this_month, last_january]


@pytest.mark.asyncio
async def test_merchant_normalization(monkeypatch):
    """Merchant names are normalized from colloquial forms."""
    agent = EntityExtractorAgent()

    async def fake_reply(prompt: str):
        return json.dumps(
            {
                "extraction_success": True,
                "entities": [{"type": "merchant", "value": "McDonald's"}],
            }
        )

    monkeypatch.setattr(agent, "a_generate_reply", fake_reply, raising=False)
    team_ctx = {"original_message": "McDo", "intent": "TRANSACTION_SEARCH"}
    result = await agent.extract_entities_from_team_context(team_ctx)

    merchants = [m.name for m in result["entities"].merchants]
    assert merchants == ["McDonald's"]


@pytest.mark.asyncio
async def test_malformed_json_fallback(monkeypatch):
    """Malformed JSON should trigger a fallback response."""
    agent = EntityExtractorAgent()

    async def bad_reply(prompt: str):
        return "{not-json}"

    monkeypatch.setattr(agent, "a_generate_reply", bad_reply, raising=False)
    team_ctx = {"original_message": "bad", "intent": "TRANSACTION_SEARCH"}
    result = await agent.extract_entities_from_team_context(team_ctx)

    assert result["extraction_success"] is False
    assert result["entities"].amounts == []


@pytest.mark.asyncio
async def test_no_entities_returns_empty_but_valid(monkeypatch):
    """A message with no entities should return an empty but valid result."""
    agent = EntityExtractorAgent()

    async def empty_reply(prompt: str):
        return json.dumps({"extraction_success": True, "entities": []})

    monkeypatch.setattr(agent, "a_generate_reply", empty_reply, raising=False)
    team_ctx = {"original_message": "bonjour", "intent": "GREETING"}
    result = await agent.extract_entities_from_team_context(team_ctx)

    assert result["extraction_success"] is True
    assert result["entities"].amounts == []
    assert result["entities"].merchants == []


@pytest.mark.asyncio
async def test_integration_with_intent_classifier(monkeypatch):
    """EntityExtractor integrates with IntentClassifier via team_context."""
    classifier = IntentClassifierAgent()

    async def fake_classify(message: str):
        return json.dumps({"intent": "TRANSACTION_SEARCH", "confidence": 0.9})

    monkeypatch.setattr(classifier, "a_generate_reply", fake_classify, raising=False)
    classification = await classifier.classify_for_team("where did I spend?", user_id=1)
    team_ctx = classification["team_context"]

    extractor = EntityExtractorAgent()

    async def fake_extract(prompt: str):
        return json.dumps(
            {
                "extraction_success": True,
                "entities": [{"type": "merchant", "value": "McDonald's"}],
            }
        )

    monkeypatch.setattr(extractor, "a_generate_reply", fake_extract, raising=False)
    result = await extractor.extract_entities_from_team_context(team_ctx)

    assert result["team_context"]["original_message"] == team_ctx["original_message"]
    assert result["entities"].merchants[0].name == "McDonald's"
