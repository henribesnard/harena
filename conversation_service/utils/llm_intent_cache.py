"""Dedicated cache instance for LLM intent detection results."""

from __future__ import annotations

from .cache import MultiLevelCache, get_default_cache_sync

# Reuse the package-level default cache but expose a named instance for intent
# detection so agents can share cached results.
llm_intent_cache: MultiLevelCache = get_default_cache_sync()

__all__ = ["llm_intent_cache"]
