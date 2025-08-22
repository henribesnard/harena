"""Asynchronous Redis cache client for the conversation service."""

from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as redis

from conversation_service.settings import settings


class CacheClient:
    """Simple wrapper around ``redis.asyncio`` with connection pooling."""

    def __init__(self) -> None:
        self._prefix = f"{settings.redis_cache_prefix}:"
        self._pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            password=settings.redis_password,
            max_connections=settings.redis_max_connections,
            health_check_interval=settings.redis_health_check_interval,
            retry_on_timeout=settings.redis_retry_on_timeout,
            decode_responses=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)

    def _format_key(self, user_id: int, key: str) -> str:
        return f"{self._prefix}{user_id}:{key}"

    async def get(self, user_id: int, key: str) -> Optional[Any]:
        """Retrieve a value from Redis."""

        raw = await self._client.get(self._format_key(user_id, key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def set(
        self, user_id: int, key: str, value: Any, ttl: Optional[int] = None
    ) -> None:
        """Store ``value`` under ``key`` with an optional TTL in seconds."""

        if not isinstance(value, str):
            value = json.dumps(value)
        await self._client.set(self._format_key(user_id, key), value, ex=ttl)

    async def delete(self, user_id: int, key: str) -> None:
        await self._client.delete(self._format_key(user_id, key))

    async def ping(self) -> bool:
        """Health check for the Redis connection."""

        return await self._client.ping()

    async def close(self) -> None:
        """Close the client and its connection pool."""

        await self._client.close()
        await self._pool.disconnect(inuse_connections=True)
