import asyncio
from types import SimpleNamespace
import types
import sys


class _DummyClient:
    def __init__(self, *args, **kwargs):
        pass


sys.modules["openai"] = types.SimpleNamespace(
    OpenAI=_DummyClient, AsyncOpenAI=_DummyClient
)

from quick_intent_test import HarenaIntentAgent, IntentCategory


class DummyUsage:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0


class DummyResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = DummyUsage()


def make_dummy_client(response_json: str):
    async def create(**kwargs):
        return DummyResponse(response_json)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_action_request_returns_unsupported():
    agent = HarenaIntentAgent(api_key="test")
    response_json = (
        '{"intent_type": "UNSUPPORTED", "intent_category": "UNCLEAR_INTENT", '
        '"confidence": 0.5, "entities": []}'
    )
    agent.async_client = make_dummy_client(response_json)
    result = asyncio.run(agent.detect_intent_async("Effectue un virement de 50€"))

    assert result.intent_type == "UNSUPPORTED"
    assert result.intent_category == IntentCategory.UNCLEAR_INTENT
    assert result.validation_errors == ["Intent not supported."]
    assert result.search_required is False

