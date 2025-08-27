import pytest
from unittest.mock import AsyncMock

from conversation_service.autogen_core.agent_runtime import ConversationServiceRuntime
from conversation_service.autogen_core.conversation_state import ConversationState
from conversation_service.prompts.autogen.entity_extraction_prompts import (
    ENTITY_EXTRACTION_SYSTEM_MESSAGE,
)


class DummyDeepSeek:
    def __init__(self):
        self.chat = AsyncMock(side_effect=Exception("boom"))


class DummyFactory:
    def __init__(self):
        self.create_user_proxy = AsyncMock(return_value=object())
        self.create_assistant = AsyncMock(return_value=object())
        self.deepseek = DummyDeepSeek()


@pytest.mark.asyncio
async def test_process_conversation_deepseek_failure():
    factory = DummyFactory()
    runtime = ConversationServiceRuntime(factory=factory)
    state = ConversationState()

    result = await runtime.process_conversation(state, "hello")

    assert result == {"error": "llm_failure"}
    factory.create_assistant.assert_awaited_once_with(
        "assistant", ENTITY_EXTRACTION_SYSTEM_MESSAGE
    )
    # Failure should not add assistant message to state
    assert len(state.turns) == 1
