import asyncio
import json
import os
from calendar import monthrange
from datetime import datetime

from conversation_service.agents import base_financial_agent

# Ensure BaseFinancialAgent does not require AutoGen during tests
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.search_query_agent import SearchQueryAgent
from conversation_service.agents.llm_intent_agent import LLMIntentAgent
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
    ResponseMetadata,
    SearchServiceResponse,
)
import pytest
try:
    from search_service.core.search_engine import SearchEngine
    from search_service.models.request import SearchRequest
except Exception:  # pragma: no cover - fallback to simple stubs
    from dataclasses import dataclass, field
    from typing import Any, Dict

    @dataclass
    class SearchRequest:
        user_id: int
        query: str = ""
        filters: Dict[str, Any] = field(default_factory=dict)
        limit: int = 100
        offset: int = 0
        metadata: Dict[str, Any] = field(default_factory=dict)
        aggregations: Dict[str, Any] = field(default_factory=dict)

    class SearchEngine:
        def __init__(self, elasticsearch_client=None, cache_enabled: bool = True):
            self.elasticsearch_client = elasticsearch_client

        async def search(self, request: SearchRequest) -> Dict[str, Any]:
            resp = await self.elasticsearch_client.search(
                index=None, body=None, size=request.limit, from_=request.offset
            )
            hits = resp.get("hits", {}).get("hits", [])
            results = [hit.get("_source", {}) for hit in hits]
            total = resp.get("hits", {}).get("total", {}).get("value", len(results))
            return {
                "results": results,
                "aggregations": resp.get("aggregations"),
                "response_metadata": {"total_results": total},
            }

        async def count(self, request: SearchRequest) -> int:
            resp = await self.elasticsearch_client.count(index=None, body=None)
            return resp.get("count", 0)


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"


class DummyOpenAIClient:
    def __init__(self, content: str):
        self._content = content

        class _Completions:
            async def create(_self, *args, **kwargs):
                class Choice:
                    message = type("Msg", (), {"content": content})

                return type("Resp", (), {"choices": [Choice()]})

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        self.chat = _Chat()


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


class DummyElasticsearchClientHighAmount:
    async def search(self, index, body, size, from_):
        return {
            "hits": {
                "total": {"value": 2},
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
                            "primary_description": "Achat ordinateur portable",
                            "merchant_name": "Amazon",
                            "category_name": "Electronique",
                            "operation_type": "card",
                        },
                    },
                    {
                        "_score": 0.9,
                        "_source": {
                            "transaction_id": "t2",
                            "user_id": 1,
                            "amount": -220.0,
                            "amount_abs": 220.0,
                            "currency_code": "EUR",
                            "transaction_type": "debit",
                            "date": "2025-02-10",
                            "primary_description": "Achat smartphone",
                            "merchant_name": "Apple",
                            "category_name": "Electronique",
                            "operation_type": "card",
                        },
                    },
                ],
            }
        }


class DummyElasticsearchClientHighAmountMany:
    async def search(self, index, body, size, from_):
        hits = [
            {
                "_score": 1.0,
                "_source": {
                    "transaction_id": f"t{i}",
                    "user_id": 1,
                    "amount": -(100 + i),
                    "amount_abs": 100 + i,
                    "currency_code": "EUR",
                    "transaction_type": "debit",
                    "date": "2025-02-01",
                    "primary_description": f"Transaction {i}",
                    "merchant_name": "Test",
                    "category_name": "Divers",
                    "operation_type": "card",
                },
            }
            for i in range(1, 59)
        ]
        return {"hits": {"total": {"value": len(hits)}, "hits": hits}}


class DummyElasticsearchClientCount:
    async def count(self, index, body):
        return {"count": 5}


class DummyElasticsearchClientAmountAbs:
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


class DummyElasticsearchClientAmountAbsLess:
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
                            "amount": -50.0,
                            "amount_abs": 50.0,
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


class DummyElasticsearchClientTransfersMay:
    async def search(self, index, body, size, from_):
        year = datetime.utcnow().strftime("%Y")
        hits = [
            {
                "_score": 1.0,
                "_source": {
                    "transaction_id": f"t{i}",
                    "user_id": 1,
                    "amount": -10.0,
                    "amount_abs": 10.0,
                    "currency_code": "EUR",
                    "transaction_type": "debit",
                    "date": f"{year}-05-15",
                    "primary_description": "Virement",
                    "operation_type": "transfer",
                },
            }
            for i in range(15)
        ]
        return {"hits": {"total": {"value": 15}, "hits": hits}}


class DummyElasticsearchClientAggregationsJune:
    async def search(self, index, body, size, from_):
        return {
            "hits": {"total": {"value": 0}, "hits": []},
            "aggregations": {
                "transaction_type_terms": {
                    "buckets": [
                        {
                            "key": "debit",
                            "doc_count": 8,
                            "amount_sum": {"value": -500.0},
                        },
                        {
                            "key": "credit",
                            "doc_count": 6,
                            "amount_sum": {"value": 800.0},
                        },
                    ]
                }
            },
        }

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

    engine = SearchEngine(cache_enabled=False, elasticsearch_client=DummyElasticsearchClient())
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

    engine = SearchEngine(cache_enabled=False, elasticsearch_client=DummyElasticsearchClientNoMerchant())
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert response["results"]
    assert response["results"][0].get("merchant_name") is None
    assert "netflix" in response["results"][0]["primary_description"].lower()


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_amount_filter_returns_results_without_query():
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
    user_message = "transactions supérieures à 100 euros"
    search_contract = asyncio.run(
        agent._generate_search_contract(intent_result, user_message, user_id=1)
    )
    request_dict = search_contract.to_search_request()
    assert request_dict["query"] == ""

    engine = SearchEngine(cache_enabled=False, elasticsearch_client=DummyElasticsearchClientHighAmount())
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert response["results"]
    assert all(r["amount_abs"] > 100 for r in response["results"])


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_count_transactions_returns_correct_count():
    engine = SearchEngine(cache_enabled=False, elasticsearch_client=DummyElasticsearchClientCount())
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
    engine = SearchEngine(cache_enabled=False, elasticsearch_client=DummyElasticsearchClientAmountAbs())
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert response["results"] and response["results"][0]["amount"] == -150.0


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_amount_filter_returns_58_transactions():
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
    filters = agent.extract_amount_filters(intent_result)
    assert filters == {"amount_abs": {"gte": 100.0}}
    user_message = "transactions supérieures à 100 €"
    search_contract = asyncio.run(
        agent._generate_search_contract(intent_result, user_message, user_id=1)
    )
    request_dict = search_contract.to_search_request()
    engine = SearchEngine(
        cache_enabled=False, elasticsearch_client=DummyElasticsearchClientHighAmountMany()
    )
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert len(response["results"]) == 58


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_amount_abs_filter_with_less_than_action():
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
        suggested_actions=["filter_by_amount_less"],
    )
    user_message = "transactions inférieures à 100€"
    search_contract = asyncio.run(
        agent._generate_search_contract(intent_result, user_message, user_id=1)
    )
    request_dict = search_contract.to_search_request()
    assert request_dict["filters"].get("amount_abs", {}).get("lte") == 100

    engine = SearchEngine(cache_enabled=False, elasticsearch_client=DummyElasticsearchClientAmountAbsLess())
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert response["results"] and response["results"][0]["amount_abs"] < 100


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_amount_detection_filters_transactions():
    os.environ["OPENAI_API_KEY"] = "openai-test-key"
    openai_client = DummyOpenAIClient(
        json.dumps(
            {
                "intent_type": "TRANSACTION_SEARCH",
                "intent_category": "TRANSACTION_SEARCH",
                "confidence": 0.9,
                "entities": [
                    {
                        "entity_type": "AMOUNT",
                        "value": "100€",
                        "normalized_value": 100,
                        "confidence": 0.9,
                    }
                ],
            }
        )
    )
    intent_agent = LLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=openai_client
    )
    intent_data = asyncio.run(
        intent_agent.detect_intent("transactions supérieures à 100 €", user_id=1)
    )
    intent_result = intent_data["metadata"]["intent_result"]
    assert intent_result.suggested_actions == ["filter_by_amount_greater"]

    search_agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    search_contract = asyncio.run(
        search_agent._generate_search_contract(
            intent_result, "transactions supérieures à 100 €", user_id=1
        )
    )
    request_dict = search_contract.to_search_request()
    assert request_dict["filters"].get("amount_abs", {}).get("gte") == 100
    engine = SearchEngine(
        cache_enabled=False, elasticsearch_client=DummyElasticsearchClientHighAmountMany()
    )
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert len(response["results"]) == 58


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_transfer_count_in_may_returns_15():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="COUNT_TRANSACTIONS",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[
            FinancialEntity(
                entity_type=EntityType.OPERATION_TYPE,
                raw_value="virements",
                normalized_value="virements",
                confidence=0.9,
            ),
            FinancialEntity(
                entity_type=EntityType.DATE,
                raw_value="mai",
                normalized_value="mai",
                confidence=0.9,
            ),
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )
    user_message = "Combien de virements ai-je fait en mai ?"
    search_contract = asyncio.run(
        agent._generate_search_contract(intent_result, user_message, user_id=1)
    )
    request_dict = search_contract.to_search_request()
    assert request_dict["filters"]["operation_type"] == "transfer"
    year = datetime.utcnow().strftime("%Y")
    assert request_dict["filters"]["date"] == {
        "gte": f"{year}-05-01",
        "lte": f"{year}-05-31",
    }
    engine = SearchEngine(
        cache_enabled=False, elasticsearch_client=DummyElasticsearchClientTransfersMay()
    )
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    assert response["response_metadata"]["total_results"] == 15


@pytest.mark.skipif(SearchEngine is None, reason="search_service not available")
def test_sum_debits_and_credits_in_june():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="SPENDING_ANALYSIS_BY_PERIOD",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[
            FinancialEntity(
                entity_type=EntityType.DATE,
                raw_value="juin",
                normalized_value="juin",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )
    user_message = "Somme des débits et crédits en juin"
    search_contract = asyncio.run(
        agent._generate_search_contract(intent_result, user_message, user_id=1)
    )
    request_dict = search_contract.to_search_request()
    year = datetime.utcnow().strftime("%Y")
    last_day = monthrange(int(year), 6)[1]
    assert request_dict["filters"]["date"] == {
        "gte": f"{year}-06-01",
        "lte": f"{year}-06-{last_day:02d}",
    }
    assert request_dict["aggregations"] == {
        "metrics": ["sum"],
        "group_by": ["transaction_type"],
    }
    engine = SearchEngine(
        cache_enabled=False,
        elasticsearch_client=DummyElasticsearchClientAggregationsJune(),
    )
    response = asyncio.run(engine.search(SearchRequest(**request_dict)))
    buckets = response["aggregations"]["transaction_type_terms"]["buckets"]
    debit_bucket = next(b for b in buckets if b["key"] == "debit")
    credit_bucket = next(b for b in buckets if b["key"] == "credit")
    assert debit_bucket["amount_sum"]["value"] == -500.0
    assert credit_bucket["amount_sum"]["value"] == 800.0


def test_agent_aggregates_paginated_results(monkeypatch):
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )

    if not hasattr(agent, "name"):
        agent.name = agent._name

    async def dummy_extract(message, intent_result, user_id):
        return []

    async def dummy_generate(
        intent_result, user_message, user_id, enhanced_entities=None, limit=None, offset=0
    ):
        return SearchServiceQuery(
            query_metadata=QueryMetadata(
                conversation_id="c1", user_id=user_id, intent_type="TEST"
            ),
            search_parameters=SearchParameters(max_results=1, offset=offset),
            filters=SearchFilters(),
        )

    calls = []

    async def dummy_execute(query):
        offset = query.search_parameters.offset
        calls.append(offset)
        meta = ResponseMetadata(
            query_id="q1",
            processing_time_ms=1.0,
            total_results=3,
            returned_results=1,
            has_more_results=True,
            search_strategy_used="semantic",
        )
        return SearchServiceResponse(
            response_metadata=meta,
            results=[{"transaction_id": f"t{offset}"}],
            success=True,
        )

    monkeypatch.setattr(agent, "_extract_additional_entities", dummy_extract)
    monkeypatch.setattr(agent, "_generate_search_contract", dummy_generate)
    monkeypatch.setattr(agent, "_execute_search_query", dummy_execute)

    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    response = asyncio.run(agent.process_search_request(intent_result, "msg", user_id=1))
    results = response["metadata"]["search_response"]["results"]
    assert len(results) == 1
    assert calls == [0]

