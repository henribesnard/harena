"""Lightweight namespace package for conversation agents.

Only utility modules with minimal dependencies are imported eagerly so that
unit tests can run in constrained environments.  Full agent implementations
should be imported from their respective modules when needed.
"""

from __future__ import annotations

try:  # pragma: no cover - optional utility
    from .query_generator_agent import QueryOptimizer
except Exception:  # pragma: no cover - dependency missing
    QueryOptimizer = object  # type: ignore

__all__ = ["QueryOptimizer"]
