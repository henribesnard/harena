"""Application settings for the conversation service.

The settings rely on environment variables and can be reloaded at
runtime by calling :func:`reload_settings`.  Only a minimal subset of
variables required by the service are defined here.
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic settings for environment‑driven configuration."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "[%(asctime)s] %(levelname)s in %(name)s: %(message)s"


settings = Settings()
"""Instance global de configuration."""


def reload_settings() -> Settings:
    """Reload settings from the environment and return the new instance."""

    global settings
    settings = Settings()  # type: ignore[assignment]
    return settings
