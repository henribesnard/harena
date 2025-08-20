from .openai_config import OpenAISettings
from .autogen_config import AutoGenSettings


class Settings(AutoGenSettings, OpenAISettings):
    """Paramètres globaux de l'application."""


settings = Settings()

__all__ = ["settings"]
