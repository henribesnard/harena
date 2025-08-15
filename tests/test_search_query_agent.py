from conversation_service.agents import base_financial_agent

# Ensure the base agent does not require AutoGen during tests
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.search_query_agent import SearchQueryAgent
from conversation_service.models.financial_models import (
    FinancialEntity,
    EntityType,
    IntentResult,
    IntentCategory,
    DetectionMethod,
)
from conversation_service.models.service_contracts import (
    SearchServiceQuery,
    QueryMetadata,
    SearchParameters,
    SearchFilters,
)
import asyncio
import pytest
from datetime import datetime, timedelta

try:
    from search_service.core.search_engine import SearchEngine
    from search_service.models.request import SearchRequest
except Exception:  # pragma: no cover - skip if deps missing
    SearchEngine = None
    SearchRequest = None


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"


class DummyHTTPResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class DummyHTTPClient:
    def __init__(self, data):
        self._data = data

    async def post(self, url, json, headers):
        return DummyHTTPResponse(self._data)


def test_prepare_entity_context_with_string_entity_type():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Starbucks",
        normalized_value="Starbucks",
        confidence=0.9,
    )
    # Simulate an entity where entity_type is a plain string
    entity.entity_type = "MERCHANT"
    context = agent._prepare_entity_extraction_context("message", [entity])
    assert "MERCHANT" in context


def test_generate_search_contract_deduplicates_terms():
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
                raw_value="Carrefour",
                normalized_value="carrefour",
                confidence=0.8,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "Carrefour", user_id=1)
    )

    request = search_query.to_search_request()
    assert request["query"].split().count("carrefour") == 1
    assert "merchants" not in request["filters"]
    assert "merchant_name" not in request["filters"]
    assert "user_id" not in request["filters"]
    assert request["user_id"] == 1


def test_relative_date_current_month():
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
                entity_type=EntityType.RELATIVE_DATE,
                raw_value="ce mois",
                normalized_value="current_month",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "dépenses ce mois", user_id=1)
    )
    request = search_query.to_search_request()
    date_filter = request["filters"].get("date")

    now = datetime.utcnow()
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = next_month - timedelta(days=1)

    assert date_filter["gte"] == start.strftime("%Y-%m-%d")
    assert date_filter["lte"] == end.strftime("%Y-%m-%d")


def test_execute_search_query_converts_fields():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    if not hasattr(agent, "name"):
        agent.name = agent._name

    response_data = {
        "response_metadata": {
            "query_id": "q1",
            "processing_time_ms": 1.0,
            "total_results": 1,
            "returned_results": 1,
            "has_more_results": False,
            "search_strategy_used": "semantic",
        },
        "results": [
            {
                "transaction_id": "t1",
                "date": "2024-01-01T00:00:00Z",
                "amount": 20.5,
                "currency_code": "EUR",
                "primary_description": "Coffee shop",
                "merchant_name": "Starbucks",
                "category_name": "Food",
                "account_id": 987,
                "transaction_type": "debit",
            }
        ],
        "success": True,
    }

    agent.http_client = DummyHTTPClient(response_data)

    query = SearchServiceQuery(
        query_metadata=QueryMetadata(
            conversation_id="conv1", user_id=1, intent_type="TEST_INTENT"
        ),
        search_parameters=SearchParameters(),
        filters=SearchFilters(),
    )

    response = asyncio.run(agent._execute_search_query(query))
    result = response.results[0]
    assert result["currency"] == "EUR"
    assert result["description"] == "Coffee shop"
    assert result["merchant"] == "Starbucks"
    assert result["category"] == "Food"
    assert result["account_id"] == "987"


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


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_netflix_search_returns_results_without_merchant_name():
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
