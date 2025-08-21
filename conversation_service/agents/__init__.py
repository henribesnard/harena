"""Lightweight namespace for conversation agents."""

try:  # pragma: no cover - optional utility
    from .query_generator_agent import QueryOptimizer
except Exception:  # pragma: no cover
    QueryOptimizer = object  # type: ignore

__all__ = ["QueryOptimizer"]
