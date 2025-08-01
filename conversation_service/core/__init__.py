"""
Core package for Conversation Service MVP.

This module provides the essential core components for the AutoGen-based
conversation service, including DeepSeek client integration, conversation
management, and team orchestration.

Exports:
    Client Components:
        - DeepSeekClient: Optimized DeepSeek API client with caching
        - DeepSeekOptimizer: Static optimization utilities for DeepSeek
    
    Conversation Management:
        - ConversationManager: Multi-turn conversation context management
    
    Team Management:
        - MVPTeamManager: 4-agent team orchestration (à venir)

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP
"""

# Import core components
from .deepseek_client import (
    DeepSeekClient,
    DeepSeekOptimizer,
    DeepSeekError,
    DeepSeekTimeoutError,
    DeepSeekRateLimitError
)

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
    "DeepSeekRateLimitError"
]

# Configuration par défaut
DEFAULT_CORE_CONFIG = {
    "deepseek_timeout": 60,
    "cache_enabled": True,
    "metrics_enabled": True,
    "rate_limit_enabled": True,
    "circuit_breaker_enabled": True
}