"""OpenAI configuration utilities."""

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

    # Environment variables are prefixed with ``OPENAI_``
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore")


# ``get_openai_config`` in :mod:`config` should be used instead of creating
# instances directly.  The class is re-exported for type checking and tests.

