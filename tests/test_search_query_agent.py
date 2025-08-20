from conversation_service.agents import base_financial_agent

# Ensure the base agent does not require AutoGen during tests
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.search_query_agent import SearchQueryAgent, QueryOptimizer
from conversation_service.agents.llm_intent_agent import LLMIntentAgent
from conversation_service.models.financial_models import (
    FinancialEntity,
    EntityType,
    IntentResult,
    IntentCategory,
    DetectionMethod,
)
from conversation_service.constants import TRANSACTION_TYPES
from conversation_service.models.service_contracts import (
    SearchServiceQuery,
    QueryMetadata,
    SearchParameters,
    SearchFilters,
    ResponseMetadata,
    SearchServiceResponse,
)
import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
import calendar

try:
    from search_service.core.search_engine import SearchEngine
except Exception:  # pragma: no cover - skip if deps missing
    SearchEngine = None

try:
    from search_service.models.request import SearchRequest
    from search_service.core.query_builder import QueryBuilder
except Exception:  # pragma: no cover - skip if deps missing
    SearchRequest = None
    QueryBuilder = None


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


def make_amount_intent(normalized_value, actions=None):
    return IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[
            FinancialEntity(
                entity_type=EntityType.AMOUNT,
                raw_value="amount",
                normalized_value=normalized_value,
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        suggested_actions=actions,
        processing_time_ms=1.0,
    )


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


def test_transaction_type_filter_uses_allowed_values():
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
                entity_type=EntityType.TRANSACTION_TYPE,
                raw_value="Débit",
                normalized_value="debit",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "Débit", user_id=1)
    )
    request = search_query.to_search_request()
    assert request["filters"]["transaction_types"] == ["debit"]
    assert request["filters"]["transaction_types"][0] in TRANSACTION_TYPES


def test_operation_type_synonym_applied_without_transaction_types():
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
                entity_type=EntityType.OPERATION_TYPE,
                raw_value="virements",
                normalized_value="virements",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "virements", user_id=1)
    )
    request = search_query.to_search_request()
    assert request["filters"]["operation_type"] == "transfer"
    assert "transaction_types" not in request["filters"]


def test_default_limit_capped_to_100(monkeypatch):
    monkeypatch.setenv("SEARCH_QUERY_DEFAULT_LIMIT", "150")
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )
    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "", user_id=1)
    )
    assert search_query.search_parameters.max_results == 100
    assert agent.default_limit == 100


def test_explicit_limit_capped_to_100():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )
    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "", user_id=1, limit=150)
    )
    assert search_query.search_parameters.max_results == 100


def test_no_date_filter_without_time_constraint():
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
                raw_value="Amazon",
                normalized_value="amazon",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(
            intent_result, "dépenses chez Amazon", user_id=1
        )
    )
    request = search_query.to_search_request()
    assert "date" not in request["filters"]


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


def test_date_filter_with_french_month_name():
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
                entity_type=EntityType.DATE,
                raw_value="mai",
                normalized_value="mai",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "dépenses en mai", user_id=1)
    )
    request = search_query.to_search_request()
    date_filter = request["filters"].get("date")

    year = datetime.utcnow().strftime("%Y")
    assert date_filter["gte"] == f"{year}-05-01"
    assert date_filter["lte"] == f"{year}-05-31"


@pytest.mark.parametrize(
    "french_month, month_num",
    [("juin", "06"), ("février", "02")],
)
def test_date_filter_handles_month_end_days(french_month, month_num):
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
                entity_type=EntityType.DATE,
                raw_value=french_month,
                normalized_value=french_month,
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(
            intent_result, f"dépenses en {french_month}", user_id=1
        )
    )
    request = search_query.to_search_request()
    date_filter = request["filters"].get("date")

    year = datetime.utcnow().strftime("%Y")
    max_day = calendar.monthrange(int(year), int(month_num))[1]
    assert date_filter["gte"] == f"{year}-{month_num}-01"
    assert date_filter["lte"] == f"{year}-{month_num}-{max_day:02d}"


def test_date_filter_with_month_and_year_and_accent():
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
                entity_type=EntityType.DATE,
                raw_value="février 2024",
                normalized_value="février 2024",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(
            intent_result, "dépenses en février 2024", user_id=1
        )
    )
    request = search_query.to_search_request()
    date_filter = request["filters"].get("date")

    assert date_filter["gte"] == "2024-02-01"
    assert date_filter["lte"] == "2024-02-29"


def test_amount_filter_without_date():
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

    search_query = asyncio.run(
        agent._generate_search_contract(
            intent_result, "transactions supérieures à 100 euros", user_id=1
        )
    )
    request = search_query.to_search_request()
    assert "amount_abs" in request["filters"]
    assert "date" not in request["filters"]
    assert request["query"] == ""


def test_amount_filter_with_string_value():
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
                normalized_value="100",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
        suggested_actions=["filter_by_amount_greater"],
    )

    search_query = asyncio.run(
        agent._generate_search_contract(
            intent_result, "transactions supérieures à 100 euros", user_id=1
        )
    )
    request = search_query.to_search_request()
    assert request["filters"].get("amount_abs") == {"gte": 100.0}


def test_no_multimatch_for_amount_phrase():
    if QueryBuilder is None or SearchRequest is None:
        pytest.skip("search_service not available")

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

    search_query = asyncio.run(
        agent._generate_search_contract(
            intent_result, "transactions supérieures à 100 euros", user_id=1
        )
    )
    request = search_query.to_search_request()

    qb = QueryBuilder()
    es_query = qb.build_query(SearchRequest(**request))

    assert all("multi_match" not in clause for clause in es_query["query"]["bool"]["must"])

def test_extract_amount_filters_gte_only():
    intent_result = make_amount_intent({"gte": 50})
    filters = QueryOptimizer.extract_amount_filters(intent_result)
    assert filters == {"amount": {"gte": 50.0}}


def test_extract_amount_filters_lte_only():
    intent_result = make_amount_intent({"lte": 100})
    filters = QueryOptimizer.extract_amount_filters(intent_result)
    assert filters == {"amount": {"lte": 100.0}}


def test_extract_amount_filters_range():
    intent_result = make_amount_intent({"gte": 50, "lte": 100})
    filters = QueryOptimizer.extract_amount_filters(intent_result)
    assert filters == {"amount": {"gte": 50.0, "lte": 100.0}}


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"gte": 50}, {"amount": {"gte": 50.0}}),
        ({"lte": 100}, {"amount": {"lte": 100.0}}),
        ({"gte": 50, "lte": 100}, {"amount": {"gte": 50.0, "lte": 100.0}}),
    ],
)
def test_generate_search_contract_amount_filters(value, expected):
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = make_amount_intent(value)
    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "", user_id=1)
    )
    request = search_query.to_search_request()
    assert request["user_id"] == 1
    assert request["filters"] == expected


def test_extract_amount_filters_absolute_comparison():
    intent_result = make_amount_intent(100, actions=["filter_by_amount_greater"])
    filters = QueryOptimizer.extract_amount_filters(intent_result)
    assert filters == {"amount_abs": {"gte": 100.0}}


def test_extract_amount_filters_absolute_comparison_string():
    intent_result = make_amount_intent("100", actions=["filter_by_amount_greater"])
    filters = QueryOptimizer.extract_amount_filters(intent_result)
    assert filters == {"amount_abs": {"gte": 100.0}}


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


def test_search_query_single_call(monkeypatch):
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
                conversation_id="conv1", user_id=user_id, intent_type="TEST_INTENT"
            ),
            search_parameters=SearchParameters(max_results=1, offset=offset),
            filters=SearchFilters(),
        )

    calls = []

    async def dummy_execute(query):
        calls.append(query.search_parameters.offset)
        meta = ResponseMetadata(
            query_id="q1",
            processing_time_ms=1.0,
            total_results=2,
            returned_results=1,
            has_more_results=True,
            search_strategy_used="semantic",
        )
        return SearchServiceResponse(
            response_metadata=meta,
            results=[{"transaction_id": f"t{query.search_parameters.offset}"}],
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

    asyncio.run(agent.process_search_request(intent_result, "msg", user_id=1))
    assert calls == [0]


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


def test_amount_filter_sent_to_search_service():
    """Ensure amount filters are forwarded to the Search Service."""

    # LLM intent agent returns amount entity with euro units
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

    content = (
        '{"intent_type": "TRANSACTION_SEARCH", "intent_category": "TRANSACTION_SEARCH", '
        '"confidence": 0.9, "entities": [{"entity_type": "AMOUNT", "value": "100 euros", '
        '"normalized_value": "100 euros", "confidence": 0.9}], '
        '"suggested_actions": ["filter_by_amount_greater"]}'
    )
    intent_agent = LLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=DummyOpenAIClient(content)
    )
    intent_result = asyncio.run(
        intent_agent.detect_intent("transactions supérieures à 100 euros", user_id=1)
    )["metadata"]["intent_result"]

    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )

    async def dummy_extract(message, intent_result, user_id):
        return []

    agent._extract_additional_entities = dummy_extract

    captured: Dict[str, Any] = {}

    class CaptureHTTPClient:
        async def post(self, url, json, headers):
            captured["payload"] = json
            return DummyHTTPResponse({"results": [], "response_metadata": {}, "success": True})

    agent.http_client = CaptureHTTPClient()

    asyncio.run(
        agent.process_search_request(
            intent_result, "transactions supérieures à 100 euros", user_id=1
        )
    )

    assert captured["payload"]["filters"]["amount_abs"] == {"gte": 100.0}


@pytest.mark.parametrize("tx_type", TRANSACTION_TYPES)
def test_transaction_type_filter_included(tx_type):
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
                entity_type=EntityType.TRANSACTION_TYPE,
                raw_value=tx_type,
                normalized_value=tx_type,
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "", user_id=1)
    )
    request = search_query.to_search_request()
    assert request["filters"]["transaction_types"] == [tx_type]
    assert request["filters"]["transaction_types"][0] in TRANSACTION_TYPES


def test_operation_type_synonym_conversion():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="SEARCH_BY_OPERATION_TYPE",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[
            FinancialEntity(
                entity_type=EntityType.OPERATION_TYPE,
                raw_value="virements",
                normalized_value="virements",
                confidence=0.9,
            )
        ],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "", user_id=1)
    )
    request = search_query.to_search_request()
    assert request["filters"]["operation_type"] == "transfer"


@pytest.mark.parametrize(
    "intent_type",
    [
        "SPENDING_ANALYSIS",
        "SPENDING_ANALYSIS_BY_PERIOD",
        "COUNT_TRANSACTIONS",
        "SPENDING_COMPARISON",
    ],
)
def test_aggregation_request_added(intent_type):
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type=intent_type,
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )
    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "", user_id=1)
    )
    assert search_query.aggregations is not None
    assert search_query.aggregations.metrics == ["sum"]
    assert search_query.aggregations.group_by == ["transaction_type"]
    request = search_query.to_search_request()
    assert request["aggregations"] == {
        "metrics": ["sum"],
        "group_by": ["transaction_type"],
    }


def _minimal_query(user_id: int = 1) -> SearchServiceQuery:
    return SearchServiceQuery(
        query_metadata=QueryMetadata(
            conversation_id="c1",
            user_id=user_id,
            intent_type="TRANSACTION_SEARCH",
            language="fr",
            priority="normal",
            source_agent="search_query_agent",
        ),
        search_parameters=SearchParameters(
            search_text="test",
            max_results=10,
            offset=0,
        ),
        filters=SearchFilters(user_id=user_id),
    )


def _minimal_response() -> SearchServiceResponse:
    return SearchServiceResponse(
        response_metadata=ResponseMetadata(
            query_id="q1",
            processing_time_ms=1.0,
            total_results=0,
            returned_results=0,
            has_more_results=False,
            search_strategy_used="lexical",
        ),
        results=[],
        success=True,
    )


def test_process_search_request_uses_llm(monkeypatch):
    calls = {"llm": 0}

    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
        use_llm_query=True,
    )

    async def dummy_extract(msg, intent, user_id):
        return []

    async def dummy_llm(*args, **kwargs):
        calls["llm"] += 1
        return _minimal_query()

    async def dummy_execute(query):
        return _minimal_response()

    async def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback should not be used")

    monkeypatch.setattr(agent, "_extract_additional_entities", dummy_extract)
    monkeypatch.setattr(agent, "generate_query_with_llm", dummy_llm)
    monkeypatch.setattr(agent, "_execute_search_query", dummy_execute)
    monkeypatch.setattr(agent, "_generate_search_contract", fail_fallback)

    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    asyncio.run(agent.process_search_request(intent_result, "msg", user_id=1))
    assert calls["llm"] == 1


def test_process_search_request_fallback_on_llm_error(monkeypatch):
    calls = {"llm": 0, "fallback": 0}

    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
        use_llm_query=True,
    )

    async def dummy_extract(msg, intent, user_id):
        return []

    async def failing_llm(*args, **kwargs):
        calls["llm"] += 1
        raise ValueError("bad")

    async def fallback_generate(*args, **kwargs):
        calls["fallback"] += 1
        return _minimal_query()

    async def dummy_execute(query):
        return _minimal_response()

    monkeypatch.setattr(agent, "_extract_additional_entities", dummy_extract)
    monkeypatch.setattr(agent, "generate_query_with_llm", failing_llm)
    monkeypatch.setattr(agent, "_generate_search_contract", fallback_generate)
    monkeypatch.setattr(agent, "_execute_search_query", dummy_execute)

    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=1.0,
    )

    asyncio.run(agent.process_search_request(intent_result, "msg", user_id=1))
    assert calls["llm"] == 1
    assert calls["fallback"] == 1
