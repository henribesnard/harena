"""Lightweight namespace package for conversation agents.

Only utility modules with minimal dependencies are imported eagerly so that
unit tests can run in constrained environments.  Full agent implementations
should be imported from their respective modules when needed.
"""Lightweight package initializer for conversation agents.

Only small utility modules are imported eagerly so that tests can run in a
minimal environment without optional third-party dependencies. Full agent
implementations can be imported from their respective modules when required.
"""

from __future__ import annotations

try:  # pragma: no cover - optional utility
# Query optimizer is used across tests; import lazily to keep dependencies light.
try:  # pragma: no cover - optional import
    from .query_generator_agent import QueryOptimizer
except Exception:  # pragma: no cover - dependency missing
    QueryOptimizer = object  # type: ignore

__all__ = ["QueryOptimizer"]
try:  # pragma: no cover - optional utility
    from .query_generator_agent import QueryOptimizer
except Exception:  # pragma: no cover - fallback when dependency missing
    QueryOptimizer = object  # type: ignore[misc, assignment]

__all__ = [
    "QueryOptimizer",
    "intent_classifier_agent",
    "entity_extractor_agent",
    "query_generator_agent",
    "response_generator_agent",
]

