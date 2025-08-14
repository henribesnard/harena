import asyncio

import conversation_service.agents.orchestrator_agent as oa
from conversation_service.agents.orchestrator_agent import WorkflowExecutor
from conversation_service.models.agent_models import AgentResponse
from conversation_service.models.financial_models import DetectionMethod


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
