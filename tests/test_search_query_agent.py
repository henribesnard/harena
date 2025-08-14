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
import asyncio
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
