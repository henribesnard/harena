import asyncio
from types import SimpleNamespace

from conversation_service.agents.orchestrator_agent import OrchestratorAgent
from conversation_service.models.agent_models import AgentResponse


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"


class DummyIntentAgent:
    name = "intent_agent"

    def __init__(self):
        self.deepseek_client = DummyDeepSeekClient()

    async def execute_with_metrics(self, input_data, user_id):
        intent_result = SimpleNamespace(search_required=True, suggested_actions=None)
        return AgentResponse(
            agent_name=self.name,
            content="intent",
            metadata={"intent_result": intent_result},
            execution_time_ms=0,
            success=True,
        )


class DummySearchAgent:
    name = "search_agent"

    def __init__(self, count: int):
        self.count = count

    async def execute_with_metrics(self, input_data, user_id):
        results = [{"id": i} for i in range(self.count)]
        metadata = {
            "search_query": {},
            "search_response": {
                "results": results,
                "response_metadata": {"returned_results": self.count},
            },
            "enhanced_entities": [],
            "execution_time_ms": 0,
            "search_results_count": self.count,
        }
        return AgentResponse(
            agent_name=self.name,
            content=f"Search completed: {self.count} results",
            metadata=metadata,
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


def test_workflow_metadata_search_results_count_matches_list_length():
    orchestrator = OrchestratorAgent(
        DummyIntentAgent(), DummySearchAgent(3), DummyResponseAgent()
    )
    result = asyncio.run(
        orchestrator.process_conversation("hello", "c1", 1)
    )

    workflow_data = result["metadata"]["workflow_data"]

    assert workflow_data["search_results_count"] == len(
        workflow_data["search_results"]
    )

