import asyncio
import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.llm_intent_agent import LLMIntentAgent
from conversation_service.models.financial_models import EntityType


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"

    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        class Response:
            content = (
                '{"intent": "SEARCH_BY_MERCHANT", '
                '"entities": [{"type": "MERCHANT", "value": "Netflix"}]}'
            )

        return Response()


def test_llm_intent_agent_parses_output_correctly():
    agent = LLMIntentAgent(deepseek_client=DummyDeepSeekClient())
    result = asyncio.run(
        agent.detect_intent("Combien j’ai dépensé pour Netflix ce mois ?", user_id=1)
    )
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_type == "SEARCH_BY_MERCHANT"
    merchant = next(
        e for e in intent_result.entities if e.entity_type == EntityType.MERCHANT
    )
    assert merchant.normalized_value == "Netflix"
