"""Validation prompts used to post-process model outputs.

Templates are versioned and variant-aware to enable experimentation.
"""
from __future__ import annotations

from typing import Dict
import random

DEFAULT_VERSION = "v1"
DEFAULT_VARIANT = "A"

_VALIDATION_PROMPTS: Dict[str, Dict[str, str]] = {
    "v1": {
        "A": "Please verify the following response for correctness: {response}",
        "B": "Double-check this answer for accuracy: {response}",
    }
}


def get_validation_prompt(version: str = DEFAULT_VERSION, variant: str | None = None, **kwargs) -> str:
    """Return a validation prompt for the specified version and variant.

    Additional ``kwargs`` are formatted into the selected template.
    When ``variant`` is ``None`` a random variant is chosen.
    """

    version_prompts = _VALIDATION_PROMPTS.get(version)
    if not version_prompts:
        raise ValueError(f"Unknown validation prompt version: {version}")

    if variant is None:
        variant = random.choice(list(version_prompts))

    template = version_prompts.get(variant)
    if template is None:
        raise ValueError(f"Unknown variant '{variant}' for version '{version}'")

    return template.format(**kwargs)


def available_validation_versions() -> Dict[str, Dict[str, str]]:
    """Return mapping of available versions and variants for validation prompts."""

    return _VALIDATION_PROMPTS


__all__ = [
    "DEFAULT_VERSION",
    "DEFAULT_VARIANT",
    "get_validation_prompt",
    "available_validation_versions",
]
