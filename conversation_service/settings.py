"""Configuration for the conversation service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-based settings."""

    search_service_url: str = Field(
        "http://localhost:8000/api/v1/search", env="SEARCH_SERVICE_URL"
    )
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    redis_password: str | None = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    redis_health_check_interval: int = Field(30, env="REDIS_HEALTH_CHECK_INTERVAL")
    redis_retry_on_timeout: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    redis_cache_prefix: str = Field("conversation_service", env="REDIS_CACHE_PREFIX")

    class Config:
        env_file = ".env"
        extra = "ignore"
        case_sensitive = False


settings = Settings()
