import asyncio
import sys
import types

# Stub external dependencies that are not available in the test environment
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

# Stub internal agent modules to avoid loading full implementations
for mod_name in [
    "conversation_service.agents.hybrid_intent_agent",
    "conversation_service.agents.search_query_agent",
    "conversation_service.agents.response_agent",
]:
    mod = types.ModuleType(mod_name)
    cls_name = mod_name.split('.')[-1].title().replace('_', '')
    setattr(mod, cls_name, type(cls_name, (), {}))
    sys.modules[mod_name] = mod

from conversation_service.agents.orchestrator_agent import OrchestratorAgent
import conversation_service.agents.base_financial_agent as base_financial_agent

# Ensure BaseFinancialAgent does not require the real AutoGen package
base_financial_agent.AUTOGEN_AVAILABLE = True


class DummyDeepSeekClient:
    api_key = "test"
    base_url = "http://test"


class DummyAgent:
    def __init__(self, name: str):
        self.name = name
        self.deepseek_client = DummyDeepSeekClient()


class AsyncDummyAgent(DummyAgent):
    async def execute_with_metrics(self, data):
        return type("Resp", (), {"success": True, "metadata": {}, "content": "ok", "error_message": None})()


class FailingIntentAgent(DummyAgent):
    async def execute_with_metrics(self, data):
        raise RuntimeError("intent failure")


class FailingOrchestratorAgent(OrchestratorAgent):
    async def _execute_workflow(self, user_message: str, conversation_id: str):
        # Simulate a workflow result missing performance_summary and execution_details
        return {
            "success": False,
            "final_response": "Workflow failed",
        }


def test_failed_workflow_does_not_raise_performance_summary_error():
    agent = FailingOrchestratorAgent(
        DummyAgent("intent"), DummyAgent("search"), DummyAgent("response")
    )
    result = asyncio.run(agent.process_conversation("hello", "conv1"))

    assert result["content"] == "Workflow failed"
    assert "performance_summary" not in result["content"]
    assert result["metadata"]["performance_summary"] == {}
    assert result["metadata"]["execution_details"] == {}
    assert result["metadata"]["workflow_success"] is False


def test_workflow_executor_exception_returns_summary(monkeypatch):
    agent = OrchestratorAgent(
        FailingIntentAgent("intent"),
        AsyncDummyAgent("search"),
        AsyncDummyAgent("response"),
    )

    def failing_fallback(self):
        raise RuntimeError("no fallback")

    monkeypatch.setattr(agent.workflow_executor, "_create_fallback_intent", failing_fallback)

    result = asyncio.run(agent.process_conversation("hello", "conv1"))

    summary = result["metadata"]["performance_summary"]
    assert result["metadata"]["workflow_success"] is False
    assert {"completed_steps", "failed_steps", "total_steps"} <= summary.keys()


if __name__ == "__main__":
    import pytest, sys

    sys.exit(pytest.main([__file__]))
