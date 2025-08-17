import asyncio
import conversation_service.agents.orchestrator_agent as oa
from conversation_service.agents.orchestrator_agent import WorkflowExecutor
from conversation_service.agents.response_agent import ResponseAgent, SEARCH_ERROR_MESSAGE
from conversation_service.models.agent_models import AgentResponse
from conversation_service.models.financial_models import DetectionMethod
from types import SimpleNamespace


oa.INTENT_TIMEOUT_SECONDS = 0.01
oa.INTENT_MAX_RETRIES = 2
oa.INTENT_BACKOFF_BASE = 0


class FailingIntentAgent:
    name = "intent_agent"

    def __init__(self):
        self.calls = 0

    async def execute_with_metrics(self, input_data, user_id):
        self.calls += 1
        raise Exception("fail")


class DummySearchAgent:
    name = "search_agent"

    async def execute_with_metrics(self, input_data, user_id):
        return AgentResponse(
            agent_name=self.name,
            content="search",
            metadata={"search_results_count": 0},
            execution_time_ms=0,
            success=True,
        )


class DummyResponseAgent:
    name = "response_agent"

    async def execute_with_metrics(self, input_data, user_id):
        return AgentResponse(
            agent_name=self.name,
            content="ok",
            metadata={},
            execution_time_ms=0,
            success=True,
        )


def test_fallback_intent_keeps_workflow_running():
    intent_agent = FailingIntentAgent()
    executor = WorkflowExecutor(intent_agent, DummySearchAgent(), DummyResponseAgent())

    result = asyncio.run(executor.execute_workflow("hello", "c1", 1))

    assert intent_agent.calls == 2
    assert result["workflow_data"]["intent_result"].method == DetectionMethod.FALLBACK

    search_step = next(
        step for step in result["execution_details"]["steps"] if step["name"] == "search_query"
    )
    assert search_step["status"] == "completed"
    assert result["final_response"] == "ok"


class DummyIntentAgent:
    name = "intent_agent"

    async def execute_with_metrics(self, input_data, user_id):
        intent_result = SimpleNamespace(search_required=True, suggested_actions=None)
        return AgentResponse(
            agent_name=self.name,
            content="intent",
            metadata={"intent_result": intent_result},
            execution_time_ms=0,
            success=True,
        )


class FailingSearchAgent:
    name = "search_agent"

    async def execute_with_metrics(self, input_data, user_id):
        raise Exception("search failed")


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"

    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        class Raw:
            usage = type("Usage", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()
        return type("Resp", (), {"content": messages[-1]["content"], "raw": Raw()})()


def test_search_error_informs_user():
    intent_agent = DummyIntentAgent()
    search_agent = FailingSearchAgent()
    response_agent = ResponseAgent(deepseek_client=DummyDeepSeekClient())
    executor = WorkflowExecutor(intent_agent, search_agent, response_agent)

    result = asyncio.run(executor.execute_workflow("hello", "c2", 1))

    assert result["workflow_data"]["search_error"] is True
    assert result["final_response"] == SEARCH_ERROR_MESSAGE
    response_step = next(
        step for step in result["execution_details"]["steps"] if step["name"] == "response_generation"
    )
    assert response_step["status"] == "failed"
