"""Lightweight package initializer for conversation agents.

The original project exposes many agent implementations which pull in optional
runtime dependencies.  For the purposes of the tests in this kata we avoid
importing those heavy modules at import time to keep the environment minimal.

Only the lightweight wrapper modules are guaranteed to be available.  They can
be imported directly, e.g. ``conversation_service.agents.intent_classifier_agent``.
"""

__all__ = [
    "intent_classifier_agent",
    "entity_extractor_agent",
    "query_generator_agent",
    "response_generator_agent",
]
