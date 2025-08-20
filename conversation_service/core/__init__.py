"""
Core package for Conversation Service MVP.

This module provides the essential core components for the AutoGen-based
conversation service, including DeepSeek client integration, conversation
management, and team orchestration.

Exports:
    Client Components:
        - DeepSeekClient: Optimized DeepSeek API client with caching
        - DeepSeekOptimizer: Static optimization utilities for DeepSeek
        - DeepSeekError: Base exception for DeepSeek errors
        - DeepSeekTimeoutError: Timeout-specific exception
        - DeepSeekRateLimitError: Rate limit exception
    
    Conversation Management:
        - ConversationManager: Multi-turn conversation context management
    
    Team Management:
        - MVPTeamManager: Complete 4-agent team orchestration
        - TeamConfiguration: Team configuration data class

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Complete Core Package
"""

import logging
from config.settings import settings
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Import guards for optional dependencies
try:
    from .deepseek_client import (
        DeepSeekClient,
        DeepSeekOptimizer,
        DeepSeekError,
        DeepSeekTimeoutError,
        DeepSeekRateLimitError
    )
    DEEPSEEK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DeepSeek client not available: {e}")
    DEEPSEEK_AVAILABLE = False
    # Fallback classes
    DeepSeekClient = None
    DeepSeekOptimizer = None
    DeepSeekError = Exception
    DeepSeekTimeoutError = Exception
    DeepSeekRateLimitError = Exception

try:
    from .conversation_manager import ConversationManager
    CONVERSATION_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ConversationManager not available: {e}")
    CONVERSATION_MANAGER_AVAILABLE = False
    ConversationManager = None

MVPTeamManager = None
TeamConfiguration = None
TEAM_MANAGER_AVAILABLE = False


def load_team_manager():
    """Lazily load the team manager components."""
    global MVPTeamManager, TeamConfiguration, TEAM_MANAGER_AVAILABLE

    if TEAM_MANAGER_AVAILABLE and MVPTeamManager and TeamConfiguration:
        return MVPTeamManager, TeamConfiguration

    try:
        from .mvp_team_manager import (
            MVPTeamManager as _MVPTeamManager,
            TeamConfiguration as _TeamConfiguration,
        )
        MVPTeamManager = _MVPTeamManager
        TeamConfiguration = _TeamConfiguration
        TEAM_MANAGER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"MVPTeamManager not available: {e}")
        MVPTeamManager = None
        TeamConfiguration = None
        TEAM_MANAGER_AVAILABLE = False

    return MVPTeamManager, TeamConfiguration

# Conditional imports for type checking
if TYPE_CHECKING:
    from .deepseek_client import (
        DeepSeekClient,
        DeepSeekOptimizer,
        DeepSeekError,
        DeepSeekTimeoutError,
        DeepSeekRateLimitError
    )
    from .conversation_manager import ConversationManager
    from .mvp_team_manager import MVPTeamManager, TeamConfiguration

# Package metadata
__version__ = "1.0.0"
__author__ = "Conversation Service Team"

# Export all core components
__all__ = [
    # Client Components
    "DeepSeekClient",
    "DeepSeekOptimizer",
    "DeepSeekError",
    "DeepSeekTimeoutError", 
    "DeepSeekRateLimitError",
    
    # Conversation Management
    "ConversationManager",
    
    # Team Management
    "MVPTeamManager",
    "TeamConfiguration",

    # Utility functions
    "load_team_manager",
    "check_core_dependencies",
    "get_available_components",
    "get_core_config",
    "run_core_validation"
]

# Configuration par défaut
DEFAULT_CORE_CONFIG = {
    "deepseek_timeout": 60,
    "cache_enabled": True,
    "metrics_enabled": True,
    "rate_limit_enabled": True,
    "circuit_breaker_enabled": True,
    "conversation_max_turns": 20,
    "conversation_memory_backend": "memory",
    "team_workflow_timeout": 60,
    "team_health_check_interval": 300,
    "auto_recovery_enabled": True
}

def check_core_dependencies() -> dict:
    """
    Check availability of all core dependencies.

    Returns:
        Dictionary with availability status of each component
    """
    load_team_manager()

    return {
        "deepseek_client": DEEPSEEK_AVAILABLE,
        "conversation_manager": CONVERSATION_MANAGER_AVAILABLE,
        "team_manager": TEAM_MANAGER_AVAILABLE,
        "all_available": all([
            DEEPSEEK_AVAILABLE,
            CONVERSATION_MANAGER_AVAILABLE,
            TEAM_MANAGER_AVAILABLE
        ])
    }

def get_available_components() -> list:
    """
    Get list of available component names.

    Returns:
        List of available component names
    """
    load_team_manager()

    available = []
    
    if DEEPSEEK_AVAILABLE:
        available.extend([
            "DeepSeekClient", "DeepSeekOptimizer", 
            "DeepSeekError", "DeepSeekTimeoutError", "DeepSeekRateLimitError"
        ])
    
    if CONVERSATION_MANAGER_AVAILABLE:
        available.append("ConversationManager")
    
    if TEAM_MANAGER_AVAILABLE:
        available.extend(["MVPTeamManager", "TeamConfiguration"])
    
    return available

def get_core_config() -> dict:
    """
    Get core configuration with environment variable overrides.
    
    Returns:
        Core configuration dictionary
    """
    config = DEFAULT_CORE_CONFIG.copy()
    
    # Override with environment variables if present
    env_overrides = {
        "deepseek_timeout": "DEEPSEEK_TIMEOUT",
        "cache_enabled": "CACHE_ENABLED",
        "metrics_enabled": "METRICS_ENABLED",
        "rate_limit_enabled": "RATE_LIMIT_ENABLED",
        "circuit_breaker_enabled": "CIRCUIT_BREAKER_ENABLED",
        "conversation_max_turns": "MAX_CONVERSATION_HISTORY",
        "conversation_memory_backend": "CONVERSATION_MEMORY_BACKEND",
        "team_workflow_timeout": "WORKFLOW_TIMEOUT_SECONDS",
        "team_health_check_interval": "HEALTH_CHECK_INTERVAL_SECONDS",
        "auto_recovery_enabled": "AUTO_RECOVERY_ENABLED"
    }
    
    for config_key, env_var in env_overrides.items():
        env_value = getattr(settings, env_var, None)
        if env_value is not None:
            # Convert to appropriate type
            if config_key in ["deepseek_timeout", "conversation_max_turns", 
                             "team_workflow_timeout", "team_health_check_interval"]:
                try:
                    config[config_key] = int(env_value)
                except ValueError:
                    logger.warning(f"Invalid integer value for {env_var}: {env_value}")
            elif config_key in ["cache_enabled", "metrics_enabled", "rate_limit_enabled",
                               "circuit_breaker_enabled", "auto_recovery_enabled"]:
                config[config_key] = env_value.lower() in ("true", "1", "yes", "on")
            else:
                config[config_key] = env_value
    
    return config

def validate_core_setup() -> dict:
    """
    Validate the core package setup and configuration.

    Returns:
        Validation results with status and messages
    """
    load_team_manager()

    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": []
    }
    
    # Check component availability
    deps = check_core_dependencies()
    
    if not deps["deepseek_client"]:
        results["errors"].append("DeepSeekClient not available - core functionality will be limited")
        results["valid"] = False
    
    if not deps["conversation_manager"]:
        results["errors"].append("ConversationManager not available - no conversation context")
        results["valid"] = False
    
    if not deps["team_manager"]:
        results["errors"].append("MVPTeamManager not available - no team orchestration")
        results["valid"] = False
    
    # Check configuration
    config = get_core_config()
    
    if config["deepseek_timeout"] < 5:
        results["warnings"].append("DeepSeek timeout very low - may cause request failures")
    
    if config["conversation_max_turns"] < 5:
        results["warnings"].append("Max conversation turns very low - limited conversation capability")
    
    if config["team_workflow_timeout"] < 30:
        results["warnings"].append("Team workflow timeout low - complex workflows may fail")
    
    # Environment variable checks
    import os
    required_env_vars = ["DEEPSEEK_API_KEY"]
    for var in required_env_vars:
        if not getattr(settings, var, None):
            results["errors"].append(f"Required environment variable missing: {var}")
            results["valid"] = False
    
    # Info messages
    results["info"].append(f"Core package version: {__version__}")
    results["info"].append(f"Available components: {len(get_available_components())}")
    results["info"].append(f"Configuration loaded with {len(config)} parameters")

    return results

# Log core package initialization
logger.info(f"Core package initialized - version {__version__}")

# Deferred validation entrypoint
def run_core_validation() -> dict:
    """Execute core setup validation with logging.

    Returns:
        Validation results dictionary
    """
    validation = validate_core_setup()

    if not validation["valid"]:
        logger.error(f"Core setup validation failed: {validation['errors']}")
    if validation["warnings"]:
        logger.warning(f"Core setup warnings: {validation['warnings']}")

    logger.info("Core package ready for use")
    return validation
