import pytest

from conversation_service.core.mvp_team_manager import MVPTeamManager


class DummyResponse:
    def __init__(self, content: str):
        self.content = content
        self.success = True
        self.confidence_score = 0.9
        self.error_message = None
        self.metadata = {
            "intent_result": None,
            "agent_chain": [],
            "search_results_count": 0,
        }


async def dummy_process(user_message: str, user_id: int, conversation_id: str):
    return DummyResponse(f"echo:{user_message}")


@pytest.mark.asyncio
async def test_process_user_message_with_metadata():
    manager = MVPTeamManager()
    manager.process_user_message = dummy_process  # type: ignore[attr-defined]
    result = await manager.process_user_message_with_metadata("hello", 1, "c1")
    assert result["content"] == "echo:hello"
    assert result["success"] is True
    assert result["metadata"]["agent_chain"] == []
