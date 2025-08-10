import sys
import types
import asyncio

# Stub pydantic since it's not installed
pydantic_stub = types.ModuleType("pydantic")

class BaseModel:
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)

def Field(*args, **kwargs):
    return None

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
sys.modules.setdefault("pydantic", pydantic_stub)

# Stub BaseFinancialAgent and related models
base_module = types.ModuleType("conversation_service.agents.base_financial_agent")
class BaseFinancialAgent:
    def __init__(self, *args, **kwargs):
        pass
base_module.BaseFinancialAgent = BaseFinancialAgent
sys.modules["conversation_service.agents.base_financial_agent"] = base_module

models_agent = types.ModuleType("conversation_service.models.agent_models")
class AgentConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
models_agent.AgentConfig = AgentConfig
class AgentResponse: ...
class TeamWorkflow: ...
models_agent.AgentResponse = AgentResponse
models_agent.TeamWorkflow = TeamWorkflow
sys.modules["conversation_service.models.agent_models"] = models_agent

models_conv = types.ModuleType("conversation_service.models.conversation_models")
class ConversationContext: ...
class ConversationTurn: ...
class ConversationRequest: ...
class ConversationResponse: ...
models_conv.ConversationContext = ConversationContext
models_conv.ConversationTurn = ConversationTurn
models_conv.ConversationRequest = ConversationRequest
models_conv.ConversationResponse = ConversationResponse
sys.modules["conversation_service.models.conversation_models"] = models_conv

core_ds = types.ModuleType("conversation_service.core.deepseek_client")
class DeepSeekClient:
    def __init__(self, api_key="test", base_url="http://test"):
        self.api_key = api_key
        self.base_url = base_url
core_ds.DeepSeekClient = DeepSeekClient
sys.modules["conversation_service.core.deepseek_client"] = core_ds

from conversation_service.agents.hybrid_intent_agent import HybridIntentAgent
from conversation_service.models.financial_models import IntentResult


def test_greeting_rule_sets_search_required_and_suggested_response():
    agent = HybridIntentAgent(DeepSeekClient())
    result = asyncio.run(agent._try_rule_based_detection("bonjour"))
    assert isinstance(result, IntentResult)
    assert result.intent_type == "GREETING"
    assert result.search_required is False
    assert result.suggested_actions
    assert any("Bonjour" in s or "Salut" in s or "Hello" in s for s in result.suggested_actions)
