import asyncio
import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.advanced_llm_intent_agent import AdvancedLLMIntentAgent
from conversation_service.utils.cache import MultiLevelCache


class CountingClient:
    def __init__(self):
        self.calls = 0
        self.api_key = "test-key"
        self.base_url = "http://api"

    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        self.calls += 1

        class Response:
            content = '{"intent": "GREETING", "confidence": 0.9, "entities": []}'

        return Response()


def test_system_prompt_contains_examples():
    client = CountingClient()
    agent = AdvancedLLMIntentAgent(deepseek_client=client)
    assert "Exemples" in agent.config.system_message


def test_caching_reduces_llm_calls():
    client = CountingClient()
    cache = MultiLevelCache()
    agent = AdvancedLLMIntentAgent(deepseek_client=client, cache=cache)

    msg = "Bonjour"
    first = asyncio.run(agent.detect_intent(msg, user_id=1))
    second = asyncio.run(agent.detect_intent(msg, user_id=1))

    assert first == second
    assert client.calls == 1


class FlakyClient:
    def __init__(self):
        self.calls = 0
        self.api_key = "test-key"
        self.base_url = "http://api"

    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        self.calls += 1
        if self.calls < 2:
            raise RuntimeError("temporary failure")

        class Response:
            content = '{"intent": "GREETING", "confidence": 0.9, "entities": []}'

        return Response()


def test_retry_with_backoff_succeeds():
    client = FlakyClient()
    cache = MultiLevelCache()
    agent = AdvancedLLMIntentAgent(
        deepseek_client=client, cache=cache, max_retries=3, backoff_factor=0
    )

    result = asyncio.run(agent.detect_intent("Salut", user_id=1))
    assert client.calls == 2
    assert result["metadata"]["intent_result"].intent_type == "GREETING"


class FailThenSucceedClient:
    def __init__(self):
        self.calls = 0
        self.api_key = "test-key"
        self.base_url = "http://api"

    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("failure")

        class Response:
            content = '{"intent": "GREETING", "confidence": 0.9, "entities": []}'

        return Response()


def test_fallback_to_base_agent():
    client = FailThenSucceedClient()
    cache = MultiLevelCache()
    agent = AdvancedLLMIntentAgent(
        deepseek_client=client, cache=cache, max_retries=1, backoff_factor=0
    )

    result = asyncio.run(agent.detect_intent("Salut", user_id=1))
    assert client.calls == 2
    assert result["metadata"]["intent_result"].intent_type == "GREETING"
