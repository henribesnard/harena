"""Generic application settings used by multiple modules."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application wide settings loaded from environment variables."""

    environment: str = "production"
    base_dir: str = str(Path(__file__).resolve().parent.parent)

    # A subset of frequently used configuration values
    CORS_ORIGINS: str = "http://localhost"
    BRIDGE_CLIENT_ID: str = ""
    BRIDGE_CLIENT_SECRET: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# The ``get_settings`` accessor in :mod:`config` should be preferred over
# instantiating this class directly.  It is re-exported for backwards
# compatibility in a few places (mainly the unit tests).

