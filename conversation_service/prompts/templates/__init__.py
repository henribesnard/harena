"""Template definitions for conversation prompts."""

from .system_messages import (
    get_system_message,
    available_system_message_versions,
    DEFAULT_VERSION as DEFAULT_SYSTEM_VERSION,
    DEFAULT_VARIANT as DEFAULT_SYSTEM_VARIANT,
)
from .few_shot_examples import (
    get_few_shot_examples,
    available_example_versions,
    DEFAULT_VERSION as DEFAULT_EXAMPLE_VERSION,
    DEFAULT_VARIANT as DEFAULT_EXAMPLE_VARIANT,
)
from .intent_taxonomy import (
    get_intent_taxonomy,
    available_taxonomy_versions,
    DEFAULT_VERSION as DEFAULT_TAXONOMY_VERSION,
    DEFAULT_VARIANT as DEFAULT_TAXONOMY_VARIANT,
)
from .validation_prompts import (
    get_validation_prompt,
    available_validation_versions,
    DEFAULT_VERSION as DEFAULT_VALIDATION_VERSION,
    DEFAULT_VARIANT as DEFAULT_VALIDATION_VARIANT,
)

__all__ = [
    "get_system_message",
    "available_system_message_versions",
    "DEFAULT_SYSTEM_VERSION",
    "DEFAULT_SYSTEM_VARIANT",
    "get_few_shot_examples",
    "available_example_versions",
    "DEFAULT_EXAMPLE_VERSION",
    "DEFAULT_EXAMPLE_VARIANT",
    "get_intent_taxonomy",
    "available_taxonomy_versions",
    "DEFAULT_TAXONOMY_VERSION",
    "DEFAULT_TAXONOMY_VARIANT",
    "get_validation_prompt",
    "available_validation_versions",
    "DEFAULT_VALIDATION_VERSION",
    "DEFAULT_VALIDATION_VARIANT",
]
"""Prompt templates for the conversation service."""

# Placeholder for future template definitions.
