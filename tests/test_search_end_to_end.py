import asyncio

from conversation_service.agents import base_financial_agent

# Ensure BaseFinancialAgent does not require AutoGen during tests
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.search_query_agent import SearchQueryAgent
from conversation_service.models.financial_models import (
    FinancialEntity,
    EntityType,
    IntentResult,
    IntentCategory,
    DetectionMethod,
)
import pytest
try:
    from search_service.core.search_engine import SearchEngine
    from search_service.models.request import SearchRequest
except Exception:  # pragma: no cover - skip if deps missing
    SearchEngine = None
    SearchRequest = None


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"


class DummyElasticsearchClient:
    async def search(self, index, body, size, from_):
        return {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_score": 1.0,
                        "_source": {
                            "transaction_id": "t1",
                            "user_id": 1,
                            "amount": -15.99,
                            "amount_abs": 15.99,
                            "currency_code": "EUR",
                            "transaction_type": "debit",
                            "date": "2025-02-01",
                            "primary_description": "Netflix abonnement",
                            "merchant_name": "Netflix",
                            "category_name": "Streaming",
                            "operation_type": "card",
                        },
                    }
                ],
            }
        }


class DummyElasticsearchClientNoMerchant:
    async def search(self, index, body, size, from_):
        return {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_score": 1.0,
                        "_source": {
                            "transaction_id": "t1",
                            "user_id": 1,
                            "amount": -15.99,
                            "amount_abs": 15.99,
                            "currency_code": "EUR",
                            "transaction_type": "debit",
                            "date": "2025-02-01",
                            "primary_description": "Netflix abonnement",
                            "category_name": "Streaming",
                            "operation_type": "card",
                        },
                    }
                ],
            }
        }


class DummyElasticsearchClientCount:
    async def count(self, index, body):
        return {"count": 5}


class DummyElasticsearchClientAmountAbs:
    async def search(self, index, body, size, from_):
        has_filter = False
        for clause in body.get("query", {}).get("bool", {}).get("must", []):
            if "range" in clause and "amount_abs" in clause["range"]:
                gte = clause["range"]["amount_abs"].get("gte", 0)
                if gte <= 150:
                    has_filter = True
        if has_filter:
            return {
                "hits": {
                    "total": {"value": 1},
                    "hits": [
                        {
                            "_score": 1.0,
                            "_source": {
                                "transaction_id": "t1",
                                "user_id": 1,
                                "amount": -150.0,
                                "amount_abs": 150.0,
                                "currency_code": "EUR",
                                "transaction_type": "debit",
                                "date": "2025-02-01",
                                "primary_description": "Paiement carte",
                                "category_name": "Divers",
                                "operation_type": "card",
                            },
                        }
                    ],
                }
            }
        return {"hits": {"total": {"value": 0}, "hits": []}}

@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_netflix_month_question_returns_transactions():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[
            FinancialEntity(
                entity_type=EntityType.MERCHANT,
                raw_value="Netflix",
                normalized_value="netflix",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )
    user_message = "Combien j’ai dépensé pour Netflix ce mois ?"
    search_contract = asyncio.run(
        agent._generate_search_contract(intent_result, user_message, user_id=1)
    )
    request_dict = search_contract.to_search_request()
    assert request_dict["query"] == "netflix"

    engine = SearchEngine(elasticsearch_client=DummyElasticsearchClient())
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert response["results"] and response["results"][0]["merchant_name"] == "Netflix"


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_text_search_returns_results_without_merchant_name():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[
            FinancialEntity(
                entity_type=EntityType.MERCHANT,
                raw_value="Netflix",
                normalized_value="netflix",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )
    user_message = "Combien j’ai dépensé pour Netflix ce mois ?"
    search_contract = asyncio.run(
        agent._generate_search_contract(intent_result, user_message, user_id=1)
    )
    request_dict = search_contract.to_search_request()
    assert request_dict["query"] == "netflix"
    assert "merchant_name" not in request_dict["filters"]

    engine = SearchEngine(elasticsearch_client=DummyElasticsearchClientNoMerchant())
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert response["results"]
    assert response["results"][0]["merchant_name"] is None
    assert "netflix" in response["results"][0]["primary_description"].lower()


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_count_transactions_returns_correct_count():
    engine = SearchEngine(elasticsearch_client=DummyElasticsearchClientCount())
    request = SearchRequest(user_id=1, query="", filters={})
    count = asyncio.run(engine.count(request))
    assert count == 5


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_amount_abs_filter_matches_negative_amount():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[
            FinancialEntity(
                entity_type=EntityType.AMOUNT,
                raw_value="100",
                normalized_value=100,
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
        suggested_actions=["filter_by_amount_greater"],
    )
    user_message = "transactions supérieures à 100€"
    search_contract = asyncio.run(
        agent._generate_search_contract(intent_result, user_message, user_id=1)
    )
    request_dict = search_contract.to_search_request()
    assert request_dict["filters"].get("amount_abs", {}).get("gte") == 100
    engine = SearchEngine(elasticsearch_client=DummyElasticsearchClientAmountAbs())
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert response["results"] and response["results"][0]["amount"] == -150.0

