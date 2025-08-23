pytest_plugins = ("pytest_asyncio",)

import json
from types import SimpleNamespace
from typing import Dict

import pytest

from conversation_service.agents import (
    EntityExtractor,
    IntentClassifier,
    QueryGenerator,
    ResponseGenerator,
)
from conversation_service.models.agent_models import DynamicFinancialEntity
from conversation_service.models.enums import EntityType, IntentType


# ---------------------------------------------------------------------------
# Fake OpenAI client
# ---------------------------------------------------------------------------

INTENT_EXAMPLES: Dict[str, IntentType] = {
    "show all transactions": IntentType.TRANSACTION_SEARCH,
    "transactions on may 5": IntentType.SEARCH_BY_DATE,
    "transactions over 50 euros": IntentType.SEARCH_BY_AMOUNT,
    "transactions at amazon": IntentType.SEARCH_BY_MERCHANT,
    "transactions for groceries": IntentType.SEARCH_BY_CATEGORY,
    "over 100 euros in march": IntentType.SEARCH_BY_AMOUNT_AND_DATE,
    "only debit operations": IntentType.SEARCH_BY_OPERATION_TYPE,
    "search 'subscription'": IntentType.SEARCH_BY_TEXT,
    "count my transactions": IntentType.COUNT_TRANSACTIONS,
    "amazon spending analysis": IntentType.MERCHANT_INQUIRY,
    "add debit filter": IntentType.FILTER_REQUEST,
    "analyze my spending": IntentType.SPENDING_ANALYSIS,
    "spending by category": IntentType.SPENDING_ANALYSIS_BY_CATEGORY,
    "spending in april": IntentType.SPENDING_ANALYSIS_BY_PERIOD,
    "compare january and february": IntentType.SPENDING_COMPARISON,
    "spending trend": IntentType.TREND_ANALYSIS,
    "category breakdown": IntentType.CATEGORY_ANALYSIS,
    "restaurants vs groceries": IntentType.COMPARISON_QUERY,
    "what is my balance": IntentType.BALANCE_INQUIRY,
    "balance of savings account": IntentType.ACCOUNT_BALANCE_SPECIFIC,
    "balance evolution": IntentType.BALANCE_EVOLUTION,
}


class FakeOpenAIClient:
    async def chat_completion(self, *, model, messages, agent_name, **kwargs):  # type: ignore[override]
        user_content = messages[-1]["content"]
        if agent_name == "intent-classifier":
            if user_content == "bad json":
                content = "not json"
            else:
                intent = next(
                    (v for k, v in INTENT_EXAMPLES.items() if k in user_content),
                    IntentType.UNKNOWN,
                )
                content = json.dumps({"intent_type": intent, "confidence_score": 0.9})
        elif agent_name == "entity-extractor":
            entities = []
            if "amazon" in user_content.lower():
                entities.append(
                    {
                        "entity_type": EntityType.MERCHANT,
                        "value": "Amazon",
                        "confidence_score": 0.8,
                    }
                )
            content = json.dumps({"entities": entities})
        elif agent_name == "query-generator":
            payload = json.loads(user_content)
            query = f"{payload['intent']['intent_type']} with {len(payload['entities'])} entities"
            content = json.dumps({"query": query})
        elif agent_name == "response-generator":
            payload = json.loads(user_content)
            content = json.dumps(
                {
                    "response": f"Result for {payload['query']}",
                    "intent": payload["intent"],
                    "entities": payload["entities"],
                    "confidence_score": 0.99,
                }
            )
        else:
            content = "{}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intent_classifier_handles_many_intents():
    client = FakeOpenAIClient()
    classifier = IntentClassifier(client)
    for text, expected in INTENT_EXAMPLES.items():
        result = await classifier.classify(text)
        assert result.intent_type == expected
        assert 0.0 <= result.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_invalid_json_raises():
    client = FakeOpenAIClient()
    classifier = IntentClassifier(client)
    with pytest.raises(ValueError):
        await classifier.classify("bad json")


class Pipeline:
    def __init__(self, client):
        self.intent = IntentClassifier(client)
        self.entities = EntityExtractor(client)
        self.query = QueryGenerator(client)
        self.response = ResponseGenerator(client)

    async def run(self, text: str):
        intent = await self.intent.classify(text)
        entities = await self.entities.extract(text)
        query = await self.query.generate(intent, entities)
        return await self.response.respond(intent, entities, query)


@pytest.mark.asyncio
async def test_agent_pipeline_end_to_end():
    client = FakeOpenAIClient()
    pipeline = Pipeline(client)
    result = await pipeline.run("show all transactions at amazon")
    assert result.intent.intent_type == IntentType.TRANSACTION_SEARCH
    assert result.entities and result.entities[0].value == "Amazon"
    assert result.response.startswith("Result for")
