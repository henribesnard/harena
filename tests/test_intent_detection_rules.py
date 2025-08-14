import asyncio
import pytest
import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.llm_intent_agent import LLMIntentAgent


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"
    mapping = {
        "recherche pizza": "SEARCH_BY_TEXT",
        "combien d'opérations ce mois": "COUNT_TRANSACTIONS",
        "nombre de mouvements ce mois": "COUNT_TRANSACTIONS",
        "tendance budget 2025": "ANALYZE_TRENDS",
        "bonjour": "GREETING",
    }

    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        text = messages[-1]["content"]
        intent = self.mapping.get(text, "OUT_OF_SCOPE")
        class Response:
            content = f'{{"intent": "{intent}", "entities": []}}'
        return Response()


agent = LLMIntentAgent(deepseek_client=DummyDeepSeekClient())


@pytest.mark.parametrize(
    "text, expected_intent",
    [
        ("recherche pizza", "SEARCH_BY_TEXT"),
        ("combien d'opérations ce mois", "COUNT_TRANSACTIONS"),
        ("nombre de mouvements ce mois", "COUNT_TRANSACTIONS"),
        ("tendance budget 2025", "ANALYZE_TRENDS"),
        ("bonjour", "GREETING"),
    ],
)
def test_llm_agent_detects_intents(text: str, expected_intent: str) -> None:
    result = asyncio.run(agent.detect_intent(text, user_id=1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_type == expected_intent
