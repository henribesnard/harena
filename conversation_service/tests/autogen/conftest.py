import pytest

from conversation_service.agents.financial.intent_classifier import IntentClassifierAgent
from conversation_service.agents.financial.entity_extractor import EntityExtractorAgent


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
