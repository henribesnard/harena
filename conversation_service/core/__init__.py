"""Core utilities for the conversation service.

This package exposes the main components required by the conversation
service and provides helper functions for validating the runtime
environment using a generic OpenAI configuration via
:class:`~config_service.config.GlobalSettings`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from config_service.config import settings

logger = logging.getLogger(__name__)

# Optional conversation manager dependency
try:  # pragma: no cover - optional dependency
    from .conversation_manager import ConversationManager
    CONVERSATION_MANAGER_AVAILABLE = True
except Exception as e:  # pragma: no cover - graceful fallback
    logger.warning(f"ConversationManager not available: {e}")
    ConversationManager = None  # type: ignore
    CONVERSATION_MANAGER_AVAILABLE = False

MVPTeamManager: Any | None = None
TeamConfiguration: Any | None = None
TEAM_MANAGER_AVAILABLE = False


def load_team_manager() -> tuple[Any | None, Any | None]:
    """Lazily import the team manager components."""
    global MVPTeamManager, TeamConfiguration, TEAM_MANAGER_AVAILABLE

    if TEAM_MANAGER_AVAILABLE and MVPTeamManager and TeamConfiguration:
        return MVPTeamManager, TeamConfiguration

    try:  # pragma: no cover - optional dependency
        from .mvp_team_manager import (
            MVPTeamManager as _MVPTeamManager,
            TeamConfiguration as _TeamConfiguration,
        )

        MVPTeamManager = _MVPTeamManager
        TeamConfiguration = _TeamConfiguration
        TEAM_MANAGER_AVAILABLE = True
    except Exception as e:  # pragma: no cover - graceful fallback
        logger.warning(f"MVPTeamManager not available: {e}")
        MVPTeamManager = None
        TeamConfiguration = None
        TEAM_MANAGER_AVAILABLE = False

    return MVPTeamManager, TeamConfiguration


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .mvp_team_manager import MVPTeamManager, TeamConfiguration
    from .conversation_manager import ConversationManager

__version__ = "1.0.0"
__author__ = "Conversation Service Team"

__all__ = [
    "ConversationManager",
    "MVPTeamManager",
    "TeamConfiguration",
    "load_team_manager",
    "check_core_dependencies",
    "get_available_components",
    "get_core_config",
    "run_core_validation",
]

# Default configuration unrelated to model provider
DEFAULT_CORE_CONFIG = {
    "cache_enabled": True,
    "metrics_enabled": True,
    "rate_limit_enabled": True,
    "circuit_breaker_enabled": True,
    "conversation_max_turns": 20,
    "conversation_memory_backend": "memory",
    "team_workflow_timeout": 60,
    "team_health_check_interval": 300,
    "auto_recovery_enabled": True,
}


def check_core_dependencies() -> dict:
    """Return availability of optional core components."""
    load_team_manager()

    return {
        "conversation_manager": CONVERSATION_MANAGER_AVAILABLE,
        "team_manager": TEAM_MANAGER_AVAILABLE,
        "all_available": all(
            [CONVERSATION_MANAGER_AVAILABLE, TEAM_MANAGER_AVAILABLE]
        ),
    }


def get_available_components() -> list[str]:
    """List the names of available core components."""
    load_team_manager()
    available: list[str] = []

    if CONVERSATION_MANAGER_AVAILABLE:
        available.append("ConversationManager")

    if TEAM_MANAGER_AVAILABLE:
        available.extend(["MVPTeamManager", "TeamConfiguration"])

    return available


def get_core_config() -> dict:
    """Return core configuration with essential OpenAI settings."""
    openai = settings
    config = DEFAULT_CORE_CONFIG.copy()
    config.update(
        {
            "OPENAI_API_KEY": openai.OPENAI_API_KEY,
            "OPENAI_BASE_URL": openai.OPENAI_BASE_URL,
            "OPENAI_CHAT_MODEL": openai.OPENAI_CHAT_MODEL,
        }
    )
    return config


def validate_core_setup() -> dict:
    """Validate component availability and OpenAI configuration."""
    load_team_manager()

    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": [],
    }

    deps = check_core_dependencies()
    if not deps["conversation_manager"]:
        results["errors"].append(
            "ConversationManager not available - no conversation context"
        )
        results["valid"] = False

    if not deps["team_manager"]:
        results["errors"].append(
            "MVPTeamManager not available - no team orchestration"
        )
        results["valid"] = False

    # Validate OpenAI settings
    openai_settings = settings
    if not openai_settings.OPENAI_API_KEY:
        results["errors"].append("OpenAI API key is not configured")
        results["valid"] = False

    config = get_core_config()
    if config["conversation_max_turns"] < 5:
        results["warnings"].append(
            "Max conversation turns very low - limited conversation capability"
        )
    if config["team_workflow_timeout"] < 30:
        results["warnings"].append("Team workflow timeout low - complex workflows may fail")

    required_env_vars = ["OPENAI_API_KEY"]
    for var in required_env_vars:
        if not getattr(settings, var, None):
            results["errors"].append(f"Required environment variable missing: {var}")
            results["valid"] = False

    # Info messages

    
    # Info messages
    results["info"].append(f"Core package version: {__version__}")
    results["info"].append(
        f"Available components: {len(get_available_components())}"
    )
    results["info"].append(
        f"Configuration loaded with {len(config)} parameters"
    )

    return results


logger.info(f"Core package initialized - version {__version__}")


def run_core_validation() -> dict:
    """Execute core setup validation with logging."""
    validation = validate_core_setup()

    if not validation["valid"]:
        logger.error(f"Core setup validation failed: {validation['errors']}")
    if validation["warnings"]:
        logger.warning(f"Core setup warnings: {validation['warnings']}")

    logger.info("Core package ready for use")
    return validation

