import asyncio
from conversation_service.core.conversation_manager import MemoryStore
from conversation_service.models.conversation_models import ConversationContext, ConversationTurn
from conversation_service.core.mvp_team_manager import MVPTeamManager
from conversation_service.models.agent_models import AgentResponse


def test_memory_store_crud_operations():
    async def run():
        store = MemoryStore()
        conv_id = "conv-test"
        context = ConversationContext(
            conversation_id=conv_id,
            user_id=1,
            turns=[],
            current_turn=0,
            status="active",
            language="fr",
        )
        await store.save_context(context)
        fetched = await store.get_context(conv_id)
        assert fetched is not None
        turn = ConversationTurn(
            user_message="hello",
            assistant_response="hi",
            turn_number=1,
            processing_time_ms=0.5,
        )
        await store.add_turn(conv_id, turn)
        fetched_after = await store.get_context(conv_id)
        assert fetched_after.current_turn == 1
        await store.clear_context(conv_id)
        assert await store.get_context(conv_id) is None
    asyncio.run(run())


def test_team_manager_process_user_message_with_mocked_agents():
    manager = MVPTeamManager()
    manager.is_initialized = True
    manager._update_team_stats = lambda success, exec_time: None
    manager._trigger_initial_health_check_if_needed = lambda: None

    async def fake_health_check():
        pass

    manager._perform_health_check = fake_health_check

    class FakeConversationManager:
        def __init__(self):
            self.calls = []

        async def update_user_context(self, conversation_id, user_id, user_message):
            self.calls.append((conversation_id, user_id, user_message))

    manager.conversation_manager = FakeConversationManager()

    class FakeOrchestrator:
        async def execute_with_metrics(self, payload, user_id):
            return AgentResponse(
                agent_name="mock_orchestrator",
                content="mock reply",
                metadata={"agent_chain": ["mock_agent"]},
                execution_time_ms=1.0,
            )

    manager.orchestrator = FakeOrchestrator()

    response = asyncio.run(
        manager.process_user_message("bonjour", user_id=7, conversation_id="c-42")
    )
    assert response.content == "mock reply"
    assert manager.conversation_manager.calls == [("c-42", 7, "bonjour")]
