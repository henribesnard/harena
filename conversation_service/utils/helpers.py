"""Common helper functions used by agents."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List


def chunks(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield successive chunks from ``iterable`` of length ``size``."""

    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary using ``sep`` as separator."""

    items: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


__all__ = ["chunks", "flatten_dict"]
