"""Common validation helpers used across models."""
from __future__ import annotations

from typing import TypeVar

T = TypeVar("T", int, float)


def non_empty_str(value: str) -> str:
    """Ensure a string is not empty or whitespace."""
    if not value or not value.strip():
        raise ValueError("value cannot be empty")
    return value.strip()


def positive_number(value: T) -> T:
    """Validate that a numeric value is positive."""
    if value <= 0:  # type: ignore[operator]
        raise ValueError("value must be positive")
    return value


def percentage(value: float, *, maximum: float = 1.0) -> float:
    """Validate that a float is between 0 and ``maximum`` (inclusive)."""
    if not (0.0 <= value <= maximum):
        raise ValueError(f"value must be between 0 and {maximum}")
    return value

__all__ = ["non_empty_str", "positive_number", "percentage"]
