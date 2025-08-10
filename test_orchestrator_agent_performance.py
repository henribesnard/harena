import asyncio
import sys
import types
import pytest

# Stub external dependencies that may be missing
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

openai_module = types.ModuleType("openai")
openai_module.AsyncOpenAI = type("AsyncOpenAI", (), {})
openai_types = types.ModuleType("openai.types")
openai_chat = types.ModuleType("openai.types.chat")
openai_chat.ChatCompletion = type("ChatCompletion", (), {})
openai_types.chat = openai_chat
sys.modules["openai"] = openai_module
sys.modules["openai.types"] = openai_types
sys.modules["openai.types.chat"] = openai_chat

pydantic_module = types.ModuleType("pydantic")
class BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

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

class ValidationError(Exception):
    pass

pydantic_module.BaseModel = BaseModel
pydantic_module.Field = Field
pydantic_module.field_validator = field_validator
pydantic_module.model_validator = model_validator
pydantic_module.ValidationError = ValidationError
sys.modules["pydantic"] = pydantic_module

# Stub internal modules
for mod_name in [
    "conversation_service.agents.hybrid_intent_agent",
    "conversation_service.agents.search_query_agent",
    "conversation_service.agents.response_agent",
]:
    mod = types.ModuleType(mod_name)
    cls_name = mod_name.split('.')[-1].title().replace('_', '')
    setattr(mod, cls_name, type(cls_name, (), {}))
    sys.modules[mod_name] = mod

models_conv = types.ModuleType("conversation_service.models.conversation_models")
class ConversationContext:
    pass
class ConversationTurn:
    pass
class ConversationRequest:
    pass
class ConversationResponse:
    pass
models_conv.ConversationContext = ConversationContext
models_conv.ConversationTurn = ConversationTurn
models_conv.ConversationRequest = ConversationRequest
models_conv.ConversationResponse = ConversationResponse
sys.modules["conversation_service.models.conversation_models"] = models_conv

from conversation_service.agents.orchestrator_agent import OrchestratorAgent
import conversation_service.agents.base_financial_agent as base_financial_agent

# Ensure BaseFinancialAgent uses stubbed features
base_financial_agent.AUTOGEN_AVAILABLE = True

class DummyDeepSeekClient:
    api_key = "test"
    base_url = "http://test"

class DummyIntentAgent:
    name = "intent"
    deepseek_client = DummyDeepSeekClient()

    async def execute_with_metrics(self, data):
        intent = types.SimpleNamespace(intent_type="GENERAL", search_required=True)
        return types.SimpleNamespace(success=True, metadata={"intent_result": intent})

class DummySearchAgent:
    name = "search"
    deepseek_client = DummyDeepSeekClient()

    async def execute_with_metrics(self, data):
        return types.SimpleNamespace(success=True, metadata={})

class DummyResponseAgent:
    name = "response"
    deepseek_client = DummyDeepSeekClient()

    async def execute_with_metrics(self, data):
        return types.SimpleNamespace(success=True, content="ok", metadata={})


@pytest.mark.slow
def test_workflow_completes_under_threshold():
    agent = OrchestratorAgent(
        DummyIntentAgent(),
        DummySearchAgent(),
        DummyResponseAgent(),
        performance_threshold_ms=500,
    )
    result = asyncio.run(agent.process_conversation("hello", "conv1"))
    duration = result["metadata"]["execution_details"]["total_duration_ms"]
    assert duration <= agent.performance_threshold_ms
