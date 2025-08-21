"""OpenAI configuration utilities.

This module exposes a small ``OpenAIConfig`` class that gathers all
parameters required to interact with the OpenAI API.  Values are read
from environment variables prefixed with ``OPENAI_`` allowing different
settings per deployment environment.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseSettings):
    """Configuration container for OpenAI clients."""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    reasoner_model: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 0.95
    timeout: int = 30

    # Read variables from the environment using the OPENAI_ prefix
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env")


# Export a ready-to-use instance
openai_config = OpenAIConfig()
