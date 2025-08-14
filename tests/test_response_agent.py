import asyncio
import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.response_agent import ResponseAgent
from conversation_service.models.service_contracts import (
    SearchServiceResponse,
    ResponseMetadata,
    TransactionResult,
)


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"

    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        class Raw:
            usage = type("Usage", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()
        return type("Resp", (), {"content": "Réponse test", "raw": Raw()})()


def test_response_agent_handles_full_search_results():
    agent = ResponseAgent(deepseek_client=DummyDeepSeekClient())

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

    search_results = {"metadata": {"search_response": search_response}}

    result = asyncio.run(
        agent.generate_response("Quels sont mes derniers achats ?", search_results, user_id=1)
    )

    assert "content" in result
    assert result["metadata"]["search_stats"]["total_results"] == 1

