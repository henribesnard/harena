import sys
import types
import asyncio

# Stub external modules required by DeepSeek client and models
openai_module = types.ModuleType("openai")
openai_module.AsyncOpenAI = type("AsyncOpenAI", (), {})
openai_types = types.ModuleType("openai.types")
openai_chat = types.ModuleType("openai.types.chat")
openai_chat.ChatCompletion = type("ChatCompletion", (), {})
openai_types.chat = openai_chat
sys.modules.setdefault("openai", openai_module)
sys.modules.setdefault("openai.types", openai_types)
sys.modules.setdefault("openai.types.chat", openai_chat)

httpx_module = types.ModuleType("httpx")
sys.modules.setdefault("httpx", httpx_module)

# Minimal pydantic stub
pydantic_stub = types.ModuleType("pydantic")


class BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    def model_dump(self):
        return self.__dict__


def Field(default=None, *args, **kwargs):
    return default


def field_validator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def model_validator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


pydantic_stub.BaseModel = BaseModel
pydantic_stub.Field = Field
pydantic_stub.field_validator = field_validator
pydantic_stub.model_validator = model_validator
pydantic_stub.ValidationError = type("ValidationError", (Exception,), {})
sys.modules["pydantic"] = pydantic_stub

from types import SimpleNamespace

# Ensure we import the real HybridIntentAgent (not stubbed by other tests)
sys.modules.pop("conversation_service.agents.hybrid_intent_agent", None)
from conversation_service.agents.hybrid_intent_agent import HybridIntentAgent
import conversation_service.agents.base_financial_agent as base_financial_agent

base_financial_agent.AUTOGEN_AVAILABLE = True


class DummyDeepSeekClient:
    api_key = "test"
    base_url = "http://test"

    async def generate_response(self, *args, **kwargs):
        return SimpleNamespace(content="Intention: GREETING\nConfiance: 0.9\nEntités: []")


def test_rule_based_no_search_needed_flag():
    agent = HybridIntentAgent(DummyDeepSeekClient())
    result = asyncio.run(agent._try_rule_based_detection("bonjour"))
    assert result is not None
    assert result.search_required is False


def test_ai_fallback_respects_no_search_needed():
    agent = HybridIntentAgent(DummyDeepSeekClient())
    agent.rule_confidence_threshold = 1.0  # force fallback to AI
    output = asyncio.run(agent.detect_intent("bonjour"))
    intent_result = output["metadata"]["intent_result"]
    assert intent_result.search_required is False
