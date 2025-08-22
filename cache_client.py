"""Simple Redis cache client used to store service responses.

The real project relies on Redis for cross-service caching.  For the unit
tests we merely need a small abstraction over ``redis.asyncio`` that
provides ``get`` and ``set`` helpers with optional TTL handling.  Values
are serialised to JSON so arbitrary Python structures can be stored.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as redis


class CacheClient:
    """Asynchronous Redis cache helper."""

    def __init__(self, url: str, *, prefix: str = "cache:") -> None:
        self._prefix = prefix
        self._client = redis.from_url(url, decode_responses=True)

    def _format_key(self, user_id: int, key: str) -> str:
        return f"{self._prefix}{user_id}:{key}"

    async def get(self, user_id: int, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""

        raw = await self._client.get(self._format_key(user_id, key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def set(self, user_id: int, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL in seconds."""

        if not isinstance(value, str):
            value = json.dumps(value)
        await self._client.set(self._format_key(user_id, key), value, ex=ttl)

    async def delete(self, user_id: int, key: str) -> None:
        await self._client.delete(self._format_key(user_id, key))

    async def close(self) -> None:
        await self._client.close()

