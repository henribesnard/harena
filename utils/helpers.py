"""Common helper functions used across services."""

from __future__ import annotations

import os
import uuid
from typing import Generator, Iterable, List, TypeVar

T = TypeVar("T")


def chunked(iterable: Iterable[T], size: int) -> Generator[List[T], None, None]:
    """Yield successive chunks from *iterable* of length *size*.

    Args:
        iterable: Iterable to split.
        size: Desired chunk size.

    Yields:
        Lists of items from *iterable* with length up to *size*.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer")

    chunk: List[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def get_env(name: str, default: str | None = None) -> str | None:
    """Retrieve an environment variable with an optional default."""
    return os.getenv(name, default)


def generate_uuid() -> str:
    """Return a new random UUID4 string."""
    return str(uuid.uuid4())
