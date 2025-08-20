import asyncio
import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from datetime import datetime

from conversation_service.agents.response_agent import ResponseAgent
from conversation_service.models.service_contracts import (
    SearchServiceResponse,
    ResponseMetadata,
    TransactionResult,
)
from conversation_service.constants import TRANSACTION_TYPES


class DummyOpenAIClient:
    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        class Raw:
            usage = type("Usage", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()
        return type("Resp", (), {"content": messages[-1]["content"], "raw": Raw()})()


def test_response_agent_handles_full_search_results():
    agent = ResponseAgent(llm_client=DummyOpenAIClient())

    search_response = SearchServiceResponse(
        response_metadata=ResponseMetadata(
            query_id="q1",
            processing_time_ms=1.0,
            total_results=1,
            returned_results=1,
            has_more_results=False,
            search_strategy_used="semantic",
        ),
        results=[
            TransactionResult(
                transaction_id="t1",
                date="2024-01-01T00:00:00Z",
                amount=20.5,
                currency="EUR",
                description="Coffee shop",
                merchant="Starbucks",
                category="Food",
                account_id="acc1",
                transaction_type="debit",
            )
        ],
        success=True,
    )
    assert search_response.results[0].transaction_type in TRANSACTION_TYPES

    search_results = {"metadata": {"search_response": search_response}}

    result = asyncio.run(
        agent.generate_response("Quels sont mes derniers achats ?", search_results, user_id=1)
    )

    assert "content" in result
    assert result["metadata"]["search_stats"]["total_results"] == 1


def test_response_agent_handles_no_transactions_current_month():
    agent = ResponseAgent(llm_client=DummyOpenAIClient())

    search_response = SearchServiceResponse(
        response_metadata=ResponseMetadata(
            query_id="q2",
            processing_time_ms=1.0,
            total_results=0,
            returned_results=0,
            has_more_results=False,
            search_strategy_used="semantic",
        ),
        results=[],
        success=True,
    )

    search_results = {
        "metadata": {
            "search_response": search_response,
            "search_query": {"filters": {"date": {"gte": "2025-08-01", "lte": "2025-08-31"}}},
        }
    }

    result = asyncio.run(
        agent.generate_response("Mes transactions ce mois-ci ?", search_results, user_id=1)
    )

    today = datetime.utcnow().strftime("%d/%m/%Y")
    assert "Aucune transaction" in result["content"]
    assert "période spécifiée" in result["content"]
    assert today in result["content"]


def test_response_agent_summarizes_large_result_set():
    agent = ResponseAgent(llm_client=DummyOpenAIClient())

    results = [
        TransactionResult(
            transaction_id=f"t{i}",
            date=f"2024-01-{i+1:02d}T00:00:00Z",
            amount=10.0 + i,
            currency="EUR",
            description=f"Transaction {i}",
            merchant="Shop",
            category="Misc",
            account_id="acc1",
            transaction_type="debit",
        )
        for i in range(25)
    ]
    assert all(r.transaction_type in TRANSACTION_TYPES for r in results)

    search_response = SearchServiceResponse(
        response_metadata=ResponseMetadata(
            query_id="q3",
            processing_time_ms=1.0,
            total_results=25,
            returned_results=25,
            has_more_results=False,
            search_strategy_used="semantic",
        ),
        results=results,
        success=True,
    )

    search_results = {"metadata": {"search_response": search_response}}

    result = asyncio.run(
        agent.generate_response("Résumé de mes transactions", search_results, user_id=1)
    )

    formatted = result["metadata"]["formatted_results"]
    assert "... et 20 autres transactions (total 25)" in formatted
    assert result["metadata"]["search_stats"]["total_results"] == 25


def test_response_agent_displays_aggregated_amounts():
    agent = ResponseAgent(llm_client=DummyOpenAIClient())

    search_response = SearchServiceResponse(
        response_metadata=ResponseMetadata(
            query_id="q4",
            processing_time_ms=1.0,
            total_results=0,
            returned_results=0,
            has_more_results=False,
            search_strategy_used="semantic",
        ),
        results=[],
        aggregations={
            "transaction_type_terms": {
                "buckets": [
                    {"key": "debit", "doc_count": 2, "amount_sum": {"value": -30.0}},
                    {"key": "credit", "doc_count": 1, "amount_sum": {"value": 100.0}},
                ]
            }
        },
        success=True,
    )
    for bucket in search_response.aggregations["transaction_type_terms"]["buckets"]:
        assert bucket["key"] in TRANSACTION_TYPES

    search_results = {"metadata": {"search_response": search_response}}

    result = asyncio.run(
        agent.generate_response("Analyse des dépenses", search_results, user_id=1)
    )

    formatted = result["metadata"]["formatted_results"]
    assert "Montants par type de transaction" in formatted
    assert "debit" in formatted
    assert "credit" in formatted
    assert "Total:" in formatted

