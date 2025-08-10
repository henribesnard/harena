import sys
import types
import asyncio
import pytest

# Stub external dependencies to allow importing WorkflowExecutor
pydantic_stub = types.ModuleType("pydantic")
class BaseModel: ...
def Field(*args, **kwargs): return None

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

# Stub httpx dependency
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

# Stub internal modules required for import
base_module = types.ModuleType("conversation_service.agents.base_financial_agent")
class BaseFinancialAgent: ...
base_module.BaseFinancialAgent = BaseFinancialAgent
sys.modules["conversation_service.agents.base_financial_agent"] = base_module

for mod_name in [
    "conversation_service.agents.hybrid_intent_agent",
    "conversation_service.agents.search_query_agent",
    "conversation_service.agents.response_agent",
]:
    mod = types.ModuleType(mod_name)
    cls_name = mod_name.split('.')[-1].title().replace('_', '')
    setattr(mod, cls_name, type(cls_name, (), {}))
    sys.modules[mod_name] = mod

models_agent = types.ModuleType("conversation_service.models.agent_models")
class AgentConfig: ...
class AgentResponse: ...
models_agent.AgentConfig = AgentConfig
models_agent.AgentResponse = AgentResponse
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
class DeepSeekClient: ...
core_ds.DeepSeekClient = DeepSeekClient
sys.modules["conversation_service.core.deepseek_client"] = core_ds

from conversation_service.agents.orchestrator_agent import WorkflowExecutor


class DummyIntentAgent:
    name = "intent_agent"

    async def execute_with_metrics(self, data):
        raise RuntimeError("intent failure")


class DummySearchAgent:
    name = "search_agent"

    async def execute_with_metrics(self, data):
        return type("Response", (), {"success": True, "metadata": {}, "error_message": None})()


class DummyResponseAgent:
    name = "response_agent"

    async def execute_with_metrics(self, data):
        return type("Response", (), {"success": True, "content": "ok", "error_message": None})()


def test_performance_summary_in_failure(monkeypatch):
    """Workflow should return performance_summary even on catastrophic failure."""
    executor = WorkflowExecutor(DummyIntentAgent(), DummySearchAgent(), DummyResponseAgent())

    def failing_fallback_intent(self):
        raise RuntimeError("no fallback")

    monkeypatch.setattr(WorkflowExecutor, "_create_fallback_intent", failing_fallback_intent)

    result = asyncio.run(executor.execute_workflow("hello", "conv1"))

    assert "performance_summary" in result
    summary = result["performance_summary"]
    assert {"completed_steps", "failed_steps", "total_steps"} <= summary.keys()


if __name__ == "__main__":
    import pytest, sys

    sys.exit(pytest.main([__file__]))
