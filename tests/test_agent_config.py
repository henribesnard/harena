import pytest

from conversation_service.models.agent_models import AgentConfig


def test_agent_config_accepts_deepseek_model():
    config = AgentConfig(
        name="test-deepseek",
        model_client_config={
            "model": "deepseek-chat",
            "api_key": "sk-test",
            "base_url": "https://api.deepseek.com",
        },
        system_message="You are a test agent for DeepSeek models.",
    )

    assert config.model_client_config["model"] == "deepseek-chat"


def test_agent_config_accepts_openai_gpt_model():
    """Ensure that models with the ``gpt-`` prefix are considered valid."""
    config = AgentConfig(
        name="test-gpt",
        model_client_config={
            "model": "gpt-4o-mini",
            "api_key": "sk-test",
            "base_url": "https://api.openai.com/v1",
        },
        system_message="You are a test agent for GPT models.",
    )

    assert config.model_client_config["model"].startswith("gpt-")

