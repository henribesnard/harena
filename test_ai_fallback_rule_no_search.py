import sys
import types
import asyncio

# Stub external dependencies not available during tests
openai_module = types.ModuleType("openai")
openai_module.AsyncOpenAI = type("AsyncOpenAI", (), {})
openai_types = types.ModuleType("openai.types")
openai_chat = types.ModuleType("openai.types.chat")
openai_chat.ChatCompletion = type("ChatCompletion", (), {})
openai_types.chat = openai_chat
sys.modules["openai"] = openai_module
sys.modules["openai.types"] = openai_types
sys.modules["openai.types.chat"] = openai_chat
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

pydantic_stub = types.ModuleType("pydantic")


class BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    def model_dump(self, *args, **kwargs):
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
sys.modules["pydantic"] = pydantic_stub

import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True
base_financial_agent.AssistantAgent.__init__ = lambda self, *args, **kwargs: setattr(self, "name", kwargs.get("name"))


class DeepSeekClient:
    def __init__(self):
        self.api_key = "test"
        self.base_url = "http://test"

    async def generate_response(self, *args, **kwargs):
        return types.SimpleNamespace(content="Intention: GENERAL\nConfiance: 0.65\n")


from conversation_service.agents.hybrid_intent_agent import HybridIntentAgent
from conversation_service.models.financial_models import DetectionMethod


def test_low_confidence_rule_no_search():
    agent = HybridIntentAgent(DeepSeekClient())
    agent.rule_confidence_threshold = 0.95
    result = asyncio.run(agent.detect_intent("merci"))
    intent_result = result["metadata"]["intent_result"]
    suggestions = agent.rule_engine.all_rules["GRATITUDE"].suggested_responses

    assert result["metadata"]["detection_method"] == DetectionMethod.AI_FALLBACK
    assert result["metadata"]["rule_backup"]["intent_type"] == "GRATITUDE"
    assert intent_result.search_required is False
    assert intent_result.suggested_actions == suggestions

