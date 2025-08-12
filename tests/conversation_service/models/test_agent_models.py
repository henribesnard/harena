import pytest
from datetime import datetime

from conversation_service.models.agent_models import AgentConfig, AgentResponse


def test_agent_config_defaults_and_dump():
    config = AgentConfig(
        name="test_agent",
        model_client_config={
            "model": "deepseek-chat",
            "api_key": "sk",
            "base_url": "https://api.deepseek.com",
        },
        system_message="You are a test agent.",
    )

    assert config.max_consecutive_auto_reply == 3
    assert config.description is None
    assert config.agent_type == "AssistantAgent"
    assert config.temperature == 0.1
    assert config.max_tokens == 1000
    assert config.timeout_seconds == 30
    assert config.retry_attempts == 2

    data = config.model_dump()
    assert data["name"] == "test_agent"
    assert data["model_client_config"]["model"] == "deepseek-chat"
    assert data["system_message"] == "You are a test agent."


def test_agent_response_success():
    resp = AgentResponse(
        agent_name="agent",
        content="ok",
        execution_time_ms=10.5,
    )

    assert resp.success is True
    assert resp.error_message is None
    assert resp.metadata == {}
    assert isinstance(resp.timestamp, datetime)

    data = resp.model_dump()
    assert data["agent_name"] == "agent"
    assert data["content"] == "ok"


def test_agent_response_error_requires_message():
    with pytest.raises(ValueError):
        AgentResponse(
            agent_name="agent",
            content="fail",
            execution_time_ms=1.0,
            success=False,
        )
