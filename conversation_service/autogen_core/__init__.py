"""Core Autogen pour le service de conversation.

Ce package expose les composants principaux :
`ConversationServiceRuntime` et `AutoGenAgentFactory`.
"""

from __future__ import annotations

from .agent_runtime import ConversationServiceRuntime


class AutoGenAgentFactory:
    """Fabrique d'agents AutoGen."""

    pass


__all__ = ["ConversationServiceRuntime", "AutoGenAgentFactory"]
