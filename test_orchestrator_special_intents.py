import sys
import types
import asyncio
import pytest

# Stub external dependencies
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

# Stub internal modules required for import
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

contracts = types.ModuleType("conversation_service.models.service_contracts")
for name in [
    "SearchServiceQuery",
    "SearchServiceResponse",
    "QueryMetadata",
    "SearchParameters",
    "SearchFilters",
    "ResponseMetadata",
    "TransactionResult",
    "AggregationRequest",
    "AggregationResult",
    "validate_search_query_contract",
    "validate_search_response_contract",
    "create_minimal_query",
    "create_error_response",
]:
    setattr(contracts, name, type(name, (), {}) if name[0].isupper() else lambda *a, **k: None)
sys.modules["conversation_service.models.service_contracts"] = contracts

core_ds = types.ModuleType("conversation_service.core.deepseek_client")
class DeepSeekClient:
    pass
core_ds.DeepSeekClient = DeepSeekClient
sys.modules["conversation_service.core.deepseek_client"] = core_ds

from conversation_service.agents import base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.orchestrator_agent import OrchestratorAgent
from conversation_service.models.financial_models import (
    IntentResult,
    IntentCategory,
    DetectionMethod,
)


class DummyDeepSeekClient:
    api_key = "test"
    base_url = "http://test"


class DummySearchAgent:
    name = "search_agent"
    deepseek_client = DummyDeepSeekClient()

    async def execute_with_metrics(self, data):
        return types.SimpleNamespace(success=True, metadata={})


class DummyResponseAgent:
    name = "response_agent"
    deepseek_client = DummyDeepSeekClient()

    async def execute_with_metrics(self, data):
        return types.SimpleNamespace(success=True, content="response", metadata={})


def build_agent(intent_result):
    class DummyIntentAgent:
        name = "intent_agent"
        deepseek_client = DummyDeepSeekClient()

        async def execute_with_metrics(self, data):
            return types.SimpleNamespace(success=True, metadata={"intent_result": intent_result})

    return OrchestratorAgent(DummyIntentAgent(), DummySearchAgent(), DummyResponseAgent())


@pytest.mark.parametrize(
    "intent_type,search_required",
    [
        ("GREETING", False),
        ("HELP", False),
        ("GOODBYE", False),
        ("GRATITUDE", False),
        ("BALANCE_QUERY", True),
    ],
)
def test_orchestrator_special_intents(intent_type, search_required):
    suggested = ["action_suggestion"] if not search_required else None
    category = (
        IntentCategory.BALANCE_INQUIRY if search_required else IntentCategory.GENERAL_QUESTION
    )
    intent_result = IntentResult(
        intent_type=intent_type,
        intent_category=category,
        confidence=0.9,
        entities=[],
        method=DetectionMethod.RULE_BASED,
        processing_time_ms=1.0,
        suggested_actions=suggested,
        search_required=search_required,
    )
    agent = build_agent(intent_result)
    result = asyncio.run(agent.process_conversation("hi", "conv"))

    steps = {step["name"]: step["status"] for step in result["metadata"]["execution_details"]["steps"]}

    if not search_required:
        assert steps["search_query"] == "skipped"
        assert steps["response_generation"] == "skipped"
        assert result["content"] == suggested[0]
    else:
        assert steps["search_query"] == "completed"
        assert steps["response_generation"] == "completed"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
