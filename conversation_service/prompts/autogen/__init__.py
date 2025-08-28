"""Prompts AutoGen."""

from .entity_extraction_prompts import ENTITY_EXTRACTION_SYSTEM_MESSAGE
from .intent_classification_prompts import AUTOGEN_INTENT_SYSTEM_MESSAGE

__all__ = [
    "ENTITY_EXTRACTION_SYSTEM_MESSAGE",
    "AUTOGEN_INTENT_SYSTEM_MESSAGE",
]
