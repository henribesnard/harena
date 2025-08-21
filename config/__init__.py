"""Central configuration utilities for the Harena project.

This package exposes small helper functions that return configured
instances of the different settings classes used across the codebase.
Each accessor is wrapped in :func:`functools.lru_cache` so the same
instance is reused every time ``get_*`` is called.  This keeps the
objects singletons without relying on module level globals.
"""

from functools import lru_cache

from .settings import Settings
from .openai_config import OpenAIConfig
from .autogen_config import AutoGenConfig


@lru_cache()
def get_settings() -> Settings:
    """Return application settings loaded from the environment."""

    return Settings()


@lru_cache()
def get_openai_config() -> OpenAIConfig:
    """Return the OpenAI configuration settings."""

    return OpenAIConfig()


@lru_cache()
def get_autogen_config() -> AutoGenConfig:
    """Return configuration for AutoGen agents."""

    return AutoGenConfig()


__all__ = [
    "get_settings",
    "get_openai_config",
    "get_autogen_config",
    "Settings",
    "OpenAIConfig",
    "AutoGenConfig",
]
