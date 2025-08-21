"""Asynchronous cache manager backed by Redis with in-memory fallback.

This utility composes cache keys using a configurable prefix and the
``user_id`` to ensure isolation between users. It stores data in Redis
through :class:`CacheClient` when available and transparently falls back
to an in-memory LRU cache if Redis is unavailable. Each ``set`` call can
specify a TTL allowing different agents to cache values for distinct
periods.

An additional L0 in-memory layer contains precomputed responses. Keys are
prefixed with ``l0:`` and checked before hitting Redis or the fallback LRU
cache. Populate this level during application start-up via
``conversation_service.core.l0_cache.warmup`` and invalidate entries with
``conversation_service.core.l0_cache.invalidate`` when the underlying data
changes.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from ..clients.cache_client import CacheClient
from . import l0_cache

__all__ = ["CacheManager"]


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class CacheManager:
    """Manage caching with Redis and in-memory fallback."""

    def __init__(
        self,
        cache_client: Optional[CacheClient] = None,
        *,
        default_ttl: int = 300,
        max_size: int = 128,
        prefix: Optional[str] = None,
    ) -> None:
        self._client = cache_client
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._fallback: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._prefix = prefix or os.getenv("REDIS_CACHE_PREFIX", "conversation_service")

    # ------------------------------------------------------------------
    def _compose_key(self, key: str, user_id: str) -> str:
        return f"{self._prefix}:{user_id}:{key}"

    # ------------------------------------------------------------------
    async def get(self, key: str, user_id: str) -> Optional[Any]:
        """Retrieve ``key`` for ``user_id`` from cache."""

        composed = self._compose_key(key, user_id)

        # L0: in-memory table for precomputed responses
        l0_value = l0_cache.get(composed)
        if l0_value is not None:
            return l0_value

        if self._client is not None:
            try:
                value = await self._client.get(composed)
                if value is not None:
                    return value
            except Exception:
                # Redis unavailable; fall back to memory
                pass
        entry = self._fallback.get(composed)
        if not entry or entry.expires_at < time.time():
            self._fallback.pop(composed, None)
            return None
        self._fallback.move_to_end(composed)
        return entry.value

    # ------------------------------------------------------------------
    async def set(
        self,
        key: str,
        value: Any,
        user_id: str,
        *,
        ttl: Optional[int] = None,
    ) -> None:
        """Store ``value`` for ``user_id`` under ``key``."""

        ttl = ttl if ttl is not None else self._default_ttl
        composed = self._compose_key(key, user_id)
        if self._client is not None:
            try:
                await self._client.set(composed, value, ttl)
                return
            except Exception:
                # Redis unavailable; fall back to memory
                pass
        if composed in self._fallback:
            self._fallback.move_to_end(composed)
        elif len(self._fallback) >= self._max_size:
            self._fallback.popitem(last=False)
        self._fallback[composed] = _CacheEntry(value=value, expires_at=time.time() + ttl)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Clear the in-memory fallback cache."""

        self._fallback.clear()
