from .openai_config import OpenAISettings
from .autogen_config import AutoGenSettings


class Settings(AutoGenSettings, OpenAISettings):
    """Paramètres globaux de l'application."""


settings = Settings()
from config_service.config import GlobalSettings, settings as _settings

# Re-export settings for the new configuration module
settings = _settings

__all__ = ["GlobalSettings", "settings"]
