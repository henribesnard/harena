import pytest

from conversation_service.agents.llm_intent_agent import LLMIntentAgent


class DummyDeepSeek:
    api_key = "test-key"
    base_url = "http://api.example.com"


@pytest.mark.asyncio
async def test_intent_agent_caches_results(openai_mock):
    agent = LLMIntentAgent(deepseek_client=DummyDeepSeek(), openai_client=openai_mock)
    first = await agent.detect_intent("bonjour", user_id=1)
    second = await agent.detect_intent("bonjour", user_id=1)
    assert "cache_hit" not in first["metadata"]
    assert second["metadata"]["cache_hit"] is True
    assert first["content"] == second["content"]
