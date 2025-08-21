from pydantic_settings import BaseSettings, SettingsConfigDict

from config_service.config import settings as global_settings


class AutogenConfig(BaseSettings):
    """Central configuration for AutoGen powered components."""

    model_config = SettingsConfigDict(env_prefix="AUTOGEN_")

    OPENAI_API_KEY: str = global_settings.OPENAI_API_KEY
    OPENAI_MODEL: str = global_settings.OPENAI_CHAT_MODEL
    CACHE_ENABLED: bool = global_settings.LLM_CACHE_ENABLED
    CACHE_TTL: int = global_settings.LLM_CACHE_TTL


autogen_settings = AutogenConfig()
