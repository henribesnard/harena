from .openai_config import OpenAISettings
from .autogen_config import AutoGenSettings
from .database_config import DatabaseSettings


class Settings(DatabaseSettings, AutoGenSettings, OpenAISettings):
    """Paramètres globaux de l'application."""


settings = Settings()

__all__ = ["settings"]
