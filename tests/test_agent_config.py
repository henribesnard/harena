import pytest

from dataclasses import dataclass
from typing import Dict


@dataclass
class AgentConfig:
    name: str
    model_client_config: Dict[str, str]
    system_message: str


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

