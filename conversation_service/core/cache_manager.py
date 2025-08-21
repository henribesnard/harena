"""Utility caching module for Harena conversation service.

Provides a simple in-memory cache with time-to-live (TTL) support and
least-recently-used eviction.  The implementation is intentionally
lightweight and dependency free so it can be reused by agents or other
components without additional infrastructure.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["CacheManager"]


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class CacheManager:
    """Simple in-memory cache with TTL and LRU eviction."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 128) -> None:
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._store: "OrderedDict[str, _CacheEntry]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expires_at < time.time():
            del self._store[key]
            return None
        # mark as recently used
        self._store.move_to_end(key)
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self.max_size:
            # remove least recently used item
            self._store.popitem(last=False)
        expires = time.time() + (ttl if ttl is not None else self.ttl)
        self._store[key] = _CacheEntry(value=value, expires_at=expires)

    def clear(self) -> None:
        """Remove all cached items."""
        self._store.clear()
