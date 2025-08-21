"""Agent utilities for the conversation service.

Only lightweight utilities are imported eagerly to keep the package usable in
restricted test environments where optional dependencies (like ``aiohttp`` or
``autogen``) may not be installed.  The production agents can still be
imported directly from their respective modules when needed.
"""

try:  # pragma: no cover - optional imports for test friendliness
    from .query_generator_agent import QueryOptimizer
except Exception:  # pragma: no cover
    QueryOptimizer = object  # type: ignore

__all__ = ["QueryOptimizer"]

