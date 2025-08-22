"""Configuration helpers for AutoGen based agents."""

from dataclasses import dataclass
from typing import Any, Dict

from pydantic_settings import BaseSettings, SettingsConfigDict

from .openai_config import OpenAIConfig


@dataclass
class AgentParameters:
    """Minimal set of parameters required to build an AutoGen agent."""

    name: str
    system_message: str
    model_client_config: Dict[str, Any]
    max_consecutive_auto_reply: int = 1
    description: str | None = None

    model_config = SettingsConfigDict(extra="ignore")


class AutoGenConfig(BaseSettings):
    """Configuration for AutoGen agents.

    Values are loaded from environment variables prefixed with ``AUTOGEN_``.
    The configuration intentionally exposes only a small subset of settings
    but can easily be extended as the project grows.
    """

    agent_name: str = "harena-agent"
    max_consecutive_auto_reply: int = 1
    description: str = "Harena financial assistant"

    model_config = SettingsConfigDict(
        extra="ignore", env_prefix="AUTOGEN_", env_file=".env"
    )

    def build_agent_params(
        self,
        system_message: str,
        *,
        openai: OpenAIConfig | None = None,
    ) -> AgentParameters:
        """Return :class:`AgentParameters` for an agent using OpenAI settings."""

        openai = openai or OpenAIConfig()
        model_config_dict = {
            "model": openai.chat_model,
            "api_key": openai.api_key,
            "base_url": openai.base_url,
            "temperature": openai.temperature,
            "max_tokens": openai.max_tokens,
            "top_p": openai.top_p,
            "timeout": openai.timeout,
        }
        return AgentParameters(
            name=self.agent_name,
            system_message=system_message,
            model_client_config=model_config_dict,
            max_consecutive_auto_reply=self.max_consecutive_auto_reply,
            description=self.description,
        )


# ``get_autogen_config`` in :mod:`config` provides a cached instance of this
# class.  The class itself remains importable for testing purposes.

