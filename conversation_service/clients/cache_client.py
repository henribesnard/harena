"""Simple Redis cache client used to store service responses.

The real project relies on Redis for cross-service caching.  For the unit
tests we merely need a small abstraction over ``redis.asyncio`` that
provides ``get`` and ``set`` helpers with optional TTL handling.  Values
are serialised to JSON so arbitrary Python structures can be stored.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import redis.asyncio as redis
from redis.exceptions import RedisError


class CacheClient:
    """Asynchronous Redis cache helper with retry logic."""

    def __init__(
        self,
        url: str,
        *,
        prefix: str = "cache:",
        max_retries: int = 3,
    ) -> None:
        self._prefix = prefix
        self._client = redis.from_url(url, decode_responses=True)
        self._max_retries = max_retries

    def _format_key(self, user_id: int, key: str) -> str:
        return f"{self._prefix}{user_id}:{key}"

    async def get(self, user_id: int, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""

        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                raw = await self._client.get(self._format_key(user_id, key))
                if raw is None:
                    return None
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
            except RedisError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise last_error

    async def set(self, user_id: int, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL in seconds."""

        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                if not isinstance(value, str):
                    value = json.dumps(value)
                await self._client.set(self._format_key(user_id, key), value, ex=ttl)
                return None
            except RedisError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise last_error

    async def delete(self, user_id: int, key: str) -> None:
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                await self._client.delete(self._format_key(user_id, key))
                return None
            except RedisError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise last_error

    async def close(self) -> None:
        await self._client.close()

