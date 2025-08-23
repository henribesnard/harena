"""Intent taxonomy definitions for prompt templates.

The taxonomy is versioned and supports multiple variants for experimentation.
"""
from __future__ import annotations

from typing import Dict, List
import random

DEFAULT_VERSION = "v1"
DEFAULT_VARIANT = "A"

_INTENT_TAXONOMIES: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "v1": {
        "A": {
            "greeting": ["hello", "hi", "good morning"],
            "order_status": ["where is my order", "track order"],
        },
        "B": {
            "greeting": ["hey", "hello there"],
            "farewell": ["bye", "goodbye"],
        },
    }
}


def get_intent_taxonomy(version: str = DEFAULT_VERSION, variant: str | None = None) -> Dict[str, List[str]]:
    """Return intent taxonomy for the given version and variant.

    When ``variant`` is ``None`` a random variant is selected to facilitate
    A/B testing.
    """

    version_taxonomies = _INTENT_TAXONOMIES.get(version)
    if not version_taxonomies:
        raise ValueError(f"Unknown taxonomy version: {version}")

    if variant is None:
        variant = random.choice(list(version_taxonomies))

    taxonomy = version_taxonomies.get(variant)
    if taxonomy is None:
        raise ValueError(f"Unknown variant '{variant}' for version '{version}'")

    return taxonomy


def available_taxonomy_versions() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Return mapping of taxonomy versions and their variants."""

    return _INTENT_TAXONOMIES


__all__ = [
    "DEFAULT_VERSION",
    "DEFAULT_VARIANT",
    "get_intent_taxonomy",
    "available_taxonomy_versions",
]
