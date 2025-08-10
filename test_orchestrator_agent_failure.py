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
