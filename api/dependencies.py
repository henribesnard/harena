"""Reusable dependency callables for FastAPI routes."""

from config import (
    get_autogen_config as _get_autogen_config,
    get_openai_config as _get_openai_config,
    get_settings as _get_settings,
)


def get_settings():
    """Return global application settings."""

    return _get_settings()


def get_openai_config():
    """Return the OpenAI configuration."""

    return _get_openai_config()


def get_autogen_config():
    """Return configuration for AutoGen agents."""

    return _get_autogen_config()


__all__ = [
    "get_settings",
    "get_openai_config",
    "get_autogen_config",
]

