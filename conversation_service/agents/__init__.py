"""Lightweight package initializer for conversation agents.

Only small utility modules are imported eagerly so that tests can run in a
minimal environment without optional third-party dependencies. Full agent
implementations can be imported from their respective modules when required.
"""

from __future__ import annotations

# Query optimizer is used across tests; import lazily to keep dependencies light.
try:  # pragma: no cover - optional import
    from .query_generator_agent import QueryOptimizer
except Exception:  # pragma: no cover - fallback when dependency missing
    QueryOptimizer = object  # type: ignore

__all__ = [
    "QueryOptimizer",
    "intent_classifier_agent",
    "entity_extractor_agent",
    "query_generator_agent",
    "response_generator_agent",
]
