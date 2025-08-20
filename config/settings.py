from config_service.config import GlobalSettings, settings as _settings

# Re-export settings for the new configuration module
settings = _settings

__all__ = ["GlobalSettings", "settings"]
