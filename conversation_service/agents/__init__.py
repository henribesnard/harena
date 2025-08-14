"""
AutoGen Agents Package for Conversation Service MVP.

This package contains specialized AutoGen v0.4 agents for financial conversation
processing, including intent detection, search query generation, and response
generation. All agents are optimized for DeepSeek LLM integration.

Agents:
    - BaseFinancialAgent: Base class for all financial agents
    - LLMIntentAgent: Intent detection powered by DeepSeek LLM
    - SearchQueryAgent: Search service interface + entity extraction
    - ResponseAgent: Contextual response generation
    - OrchestratorAgent: Multi-agent workflow coordination

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP
"""

from typing import TYPE_CHECKING

# Import guards for optional dependencies
try:
    from autogen import AssistantAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    AssistantAgent = None

# Conditional imports based on availability
if TYPE_CHECKING or AUTOGEN_AVAILABLE:
    from .base_financial_agent import BaseFinancialAgent
    from .hybrid_intent_agent import HybridIntentAgent

    from .llm_intent_agent import LLMIntentAgent
    from .advanced_llm_intent_agent import AdvancedLLMIntentAgent
    from .search_query_agent import SearchQueryAgent
    from .response_agent import ResponseAgent
    from .orchestrator_agent import OrchestratorAgent

__all__ = [
    "BaseFinancialAgent",
    "HybridIntentAgent",
    "LLMIntentAgent",
    "AdvancedLLMIntentAgent",
    "SearchQueryAgent",
    "ResponseAgent",
    "OrchestratorAgent"
]

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    if not AUTOGEN_AVAILABLE:
        missing_deps.append("autogen")
    
    if missing_deps:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing_deps)}. "
            f"Install with: pip install {' '.join(missing_deps)}"
        )

def get_available_agents():
    """Get list of available agent classes."""
    if not AUTOGEN_AVAILABLE:
        return []
    
    return [
        "BaseFinancialAgent",
        "HybridIntentAgent",
        "LLMIntentAgent",
        "AdvancedLLMIntentAgent",
        "SearchQueryAgent",
        "ResponseAgent",
        "OrchestratorAgent"
    ]