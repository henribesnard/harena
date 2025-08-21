"""Lightweight namespace for conversation agents."""

try:  # pragma: no cover - optional utility
    from .query_generator_agent import QueryOptimizer
except Exception:  # pragma: no cover
    QueryOptimizer = object  # type: ignore

__all__ = ["QueryOptimizer"]
"""Lightweight namespace package for conversation agents."""

__all__ = [
    "intent_classifier_agent",
    "entity_extractor_agent",
    "query_generator_agent",
    "response_generator_agent",
]
