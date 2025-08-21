"""Application-wide configuration settings.

This module centralizes access to environment variables and common
filesystem paths used across the codebase.  Values are loaded using
``pydantic_settings.BaseSettings`` so they can be overridden through the
environment or a ``.env`` file during local development.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """General application settings.

    Attributes
    ----------
    environment:
        Deployment environment name (e.g. ``production``, ``development``).
    base_dir:
        Absolute path to the project's root directory.  This is helpful for
        constructing other path based settings.
    """

    environment: str = "production"
    base_dir: str = str(Path(__file__).resolve().parent)

    model_config = SettingsConfigDict(env_file=".env")


# Singleton instance used throughout the project
settings = Settings()
