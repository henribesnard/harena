"""Core utilities for the conversation service.

This package houses the core conversation persistence primitives and
transaction helpers used by the conversation service.  It re-exports the
primary entry points for convenience.

The top-level :mod:`core` package contains shared utilities that are used
across multiple services.  Those modules remain untouched here to keep a
clear separation between generic helpers and conversation-specific logic.
"""

from .conversation_service import ConversationService
from .transaction_manager import transaction_manager

__all__ = ["ConversationService", "transaction_manager"]
