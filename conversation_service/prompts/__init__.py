"""Utilities and templates for conversation prompts.

This package centralizes prompt-related constants and exposes the
available template and utility submodules for convenience.
"""

from . import templates
from . import utils

# Default prefix used for building prompt identifiers.
DEFAULT_PROMPT_PREFIX = "CONV_PROMPT"

__all__ = ["templates", "utils", "DEFAULT_PROMPT_PREFIX"]
