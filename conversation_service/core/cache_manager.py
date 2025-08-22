"""Two-level cache manager with in-memory and Redis backends."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, Type

from cache_client import CacheClient

from .metrics_collector import MetricsCollector
from .validators import validate_model


class CacheManager:
    """Manage an L1 in-memory cache backed by an L2 Redis cache."""

    def __init__(
        self,
        client: CacheClient,
        *,
        l1_ttl: int = 5,
        l2_ttl: int = 60,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self._client = client
        self._l1_ttl = l1_ttl
        self._l2_ttl = l2_ttl
        self._l1_cache: Dict[Tuple[int, str], Tuple[Any, float]] = {}
        self._metrics = metrics or MetricsCollector()

    # ------------------------------------------------------------------
    async def get(self, user_id: int, key: str, model: Optional[Type] = None) -> Any:
        """Retrieve ``key`` for ``user_id`` from cache."""

        now = time.time()
        l1_key = (user_id, key)
        start = time.perf_counter()
        entry = self._l1_cache.get(l1_key)
        if entry and entry[1] > now:
            self._metrics.record_hit("l1")
            self._metrics.record_latency("l1", time.perf_counter() - start)
            value = entry[0]
            return validate_model(model, value) if model else value
        if entry:
            # Expired entry
            del self._l1_cache[l1_key]
            self._metrics.record_size("l1", len(self._l1_cache))
        self._metrics.record_miss("l1")
        self._metrics.record_latency("l1", time.perf_counter() - start)

        # Fetch from L2
        start = time.perf_counter()
        data = await self._client.get(user_id, key)
        self._metrics.record_latency("l2", time.perf_counter() - start)
        if data is None:
            self._metrics.record_miss("l2")
            return None
        self._metrics.record_hit("l2")
        value = validate_model(model, data) if model else data
        # store in L1
        self._l1_cache[l1_key] = (value, now + self._l1_ttl)
        self._metrics.record_size("l1", len(self._l1_cache))
        return value

    async def set(
        self,
        user_id: int,
        key: str,
        value: Any,
        *,
        model: Optional[Type] = None,
        l1_ttl: Optional[int] = None,
        l2_ttl: Optional[int] = None,
    ) -> None:
        """Store ``value`` in both cache levels."""

        if model is not None:
            validated = validate_model(model, value)
            l1_value = validated
            try:
                l2_value = validated.model_dump()
            except AttributeError:  # pragma: no cover - support for Pydantic v1
                l2_value = validated.dict()
        else:
            l1_value = l2_value = value

        l1_ttl = l1_ttl or self._l1_ttl
        l2_ttl = l2_ttl or self._l2_ttl

        self._l1_cache[(user_id, key)] = (l1_value, time.time() + l1_ttl)
        self._metrics.record_size("l1", len(self._l1_cache))

        start = time.perf_counter()
        await self._client.set(user_id, key, l2_value, ttl=l2_ttl)
        self._metrics.record_latency("l2", time.perf_counter() - start)

    async def invalidate(self, user_id: int, key: str) -> None:
        """Remove ``key`` for ``user_id`` from both caches."""

        self._l1_cache.pop((user_id, key), None)
        self._metrics.record_size("l1", len(self._l1_cache))
        start = time.perf_counter()
        await self._client.delete(user_id, key)
        self._metrics.record_latency("l2", time.perf_counter() - start)
