"""Lightweight namespace package for conversation agents.

The original project exposes many agent implementations with heavy
third‑party dependencies.  For the purposes of the kata the package keeps its
initialisation minimal so that individual utility modules (like small caches or
optimisers) can be imported in isolation during testing.
"""

__all__ = [
    "BaseFinancialAgent",
    "AgentTeam",
    "ContextManager",
    "EntityExtractorAgent",
    "IntentClassifierAgent",
    "QueryGeneratorAgent",
    "ResponseGeneratorAgent",
]
