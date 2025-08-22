"""Application settings for the conversation service.

The settings rely on environment variables and can be reloaded at
runtime by calling :func:`reload_settings`.  Only a minimal subset of
variables required by the service are defined here.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic settings for environment‑driven configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # Search service
    search_service_url: str = Field(
        "http://localhost:8000/api/v1/search", env="SEARCH_SERVICE_URL"
    )

    # Redis
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    redis_health_check_interval: int = Field(30, env="REDIS_HEALTH_CHECK_INTERVAL")
    redis_retry_on_timeout: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    redis_cache_prefix: str = Field("conversation_service", env="REDIS_CACHE_PREFIX")

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "[%(asctime)s] %(levelname)s in %(name)s: %(message)s"


settings = Settings()
"""Instance globale de configuration."""


def reload_settings() -> Settings:
    """Reload settings from the environment and return the new instance."""

    global settings
    settings = Settings()  # type: ignore[assignment]
    return settings
