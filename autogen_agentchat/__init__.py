"""Lightweight local stub for the :mod:`autogen_agentchat` package.

This module provides minimal class definitions used in tests when the
external dependency ``autogen-agentchat`` is not available. The real
package supplies advanced agent classes for chat interactions.  Only the
names required by the codebase are implemented here.
"""

from .agents import AssistantAgent, BaseChatAgent

__all__ = ["AssistantAgent", "BaseChatAgent"]
