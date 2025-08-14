"""AutoGen Agents Package for Conversation Service MVP.

This simplified package exposes only the LLM-based intent agent used in the
conversation service. Optional dependencies such as ``autogen`` are imported
only when available so that unit tests can run without the full stack.
"""

from typing import TYPE_CHECKING

# Import guard for optional dependency
try:  # pragma: no cover - optional dependency
    from autogen import AssistantAgent  # type: ignore
    AUTOGEN_AVAILABLE = True
except Exception:  # pragma: no cover
    AUTOGEN_AVAILABLE = False
    AssistantAgent = None  # type: ignore

if TYPE_CHECKING or AUTOGEN_AVAILABLE:
    from .llm_intent_agent import LLMIntentAgent

__all__ = ["LLMIntentAgent"]


def check_dependencies() -> None:
    """Check if all required dependencies are available."""
    if not AUTOGEN_AVAILABLE:  # pragma: no cover - simple dependency check
        raise ImportError(
            "Missing required dependencies: autogen. Install with: pip install autogen"
        )


def get_available_agents() -> list[str]:
    """Return list of available agent class names."""
    if not AUTOGEN_AVAILABLE:
        return []
    return ["LLMIntentAgent"]
