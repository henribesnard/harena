import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so that the `conversation_service`
# package can be imported when tests are executed directly from this
# subdirectory.
sys.path.append(str(Path(__file__).resolve().parents[3]))

# Import shared fixtures from the top-level tests package
TESTS_DIR = Path(__file__).resolve().parents[3] / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))
from tests.conftest import *  # noqa: F401,F403

try:  # pragma: no cover - executed during test setup
    from conversation_service.agents.financial.intent_classifier import (  # type: ignore
        IntentClassifierAgent,
    )
    from conversation_service.agents.financial.entity_extractor import (  # type: ignore
        EntityExtractorAgent,
    )
except Exception:  # pragma: no cover - autogen library missing
    import types

    class _DummyAssistantAgent:
        def __init__(self, name="agent", **_):
            self.name = name

        def add_capability(self, *_args, **_kwargs):
            pass

        async def a_initiate_chat(self, *_args, **_kwargs):
            pass

        async def a_generate_reply(self, *_args, **_kwargs):
            return ""

    class _DummyGroupChat:
        def __init__(self, *_, **__):
            self.messages = []
            self.agents = []

    class _DummyGroupChatManager:
        def __init__(self, *_, **__):
            self.groupchat = _DummyGroupChat()

    autogen_stub = types.ModuleType("autogen")
    autogen_stub.AssistantAgent = _DummyAssistantAgent
    autogen_stub.GroupChat = _DummyGroupChat
    autogen_stub.GroupChatManager = _DummyGroupChatManager
    sys.modules["autogen"] = autogen_stub

    from conversation_service.agents.financial.intent_classifier import (  # type: ignore
        IntentClassifierAgent,
    )
    from conversation_service.agents.financial.entity_extractor import (  # type: ignore
        EntityExtractorAgent,
    )


@pytest.fixture
def mock_llm_config():
    """Return a minimal LLM configuration used for testing."""
    return {
        "config_list": [
            {
                "model": "deepseek-chat",
                "temperature": 0.0,
                "max_tokens": 256,
                "cache_seed": 42,
            }
        ]
    }


@pytest.fixture
def mock_deepseek_client():
    """Provide a minimal asynchronous DeepSeek client stub."""

    class _MockClient:
        async def chat_completion(self, *args, **kwargs):
            return {
                "choices": [
                    {"message": {"content": "{\"status\": \"ok\"}"}}
                ],
                "usage": {"total_tokens": 1},
            }

    return _MockClient()


@pytest.fixture
def intent_classifier_agent(mock_llm_config):
    """Intent classifier agent with a stubbed LLM reply."""
    agent = IntentClassifierAgent()
    agent.llm_config = mock_llm_config

    async def _fake_reply(message: str):
        return '{"intent": "GENERAL_INQUIRY", "confidence": 0.9}'

    agent.a_generate_reply = _fake_reply
    return agent


@pytest.fixture
def entity_extractor_agent(mock_llm_config):
    """Entity extractor agent with stubbed LLM reply."""
    agent = EntityExtractorAgent(intent_context={})
    agent.llm_config = mock_llm_config

    async def _fake_reply(message: str):
        return '{"extraction_success": true, "entities": [], "team_context": {}}'

    agent.a_generate_reply = _fake_reply
    return agent


@pytest.fixture
def sample_team_context():
    """Sample team context dictionary."""
    return {"user_message": "hello", "intent": "GREETING", "user_id": 1}


@pytest.fixture
def mock_team_results():
    """Mocked team results for conversation processing."""
    return {"team": "test", "status": "ok"}
