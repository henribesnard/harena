"""Few-shot example templates for prompts.

Provides versioned and variant-specific examples to aid model prompting.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import random

DEFAULT_VERSION = "v1"
DEFAULT_VARIANT = "A"

_FEW_SHOT_EXAMPLES: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "v1": {
        "A": [
            ("Hello", "Hi there! How can I assist you today?"),
            (
                "Can you help me with my order?",
                "Of course! Please provide your order ID.",
            ),
        ],
        "B": [
            ("Hello", "Greetings! What can I do for you?"),
        ],
    }
}


def get_few_shot_examples(version: str = DEFAULT_VERSION, variant: str | None = None) -> List[Tuple[str, str]]:
    """Return few-shot examples for the given version and variant.

    When ``variant`` is ``None`` a random variant is chosen for A/B testing.
    """

    version_examples = _FEW_SHOT_EXAMPLES.get(version)
    if not version_examples:
        raise ValueError(f"Unknown examples version: {version}")

    if variant is None:
        variant = random.choice(list(version_examples))

    examples = version_examples.get(variant)
    if examples is None:
        raise ValueError(f"Unknown variant '{variant}' for version '{version}'")

    return examples


def available_example_versions() -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """Return mapping of versions to their variants and examples."""

    return _FEW_SHOT_EXAMPLES


__all__ = [
    "DEFAULT_VERSION",
    "DEFAULT_VARIANT",
    "get_few_shot_examples",
    "available_example_versions",
]
