"""Intent benchmark utilities.

Provides schema validation for intent detection benchmark results.
"""
from __future__ import annotations

from typing import Any

from jsonschema import Draft7Validator

# JSON schema describing the expected structure of an intent result
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "confidence": {"type": "number"},
        "entities": {"type": "array"},
    },
    "required": ["intent", "confidence"],
}

# Reuse a single validator instance rather than recreating it for each call
INTENT_VALIDATOR = Draft7Validator(INTENT_SCHEMA)

def validate_intent_result(obj: Any) -> None:
    """Validate an intent result object against ``INTENT_SCHEMA``.

    Parameters
    ----------
    obj:
        The object to validate.
    """
    INTENT_VALIDATOR.validate(obj)
