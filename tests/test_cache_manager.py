import asyncio
import pytest

from cache_manager import CacheManager


class DummyRedis:
    def __init__(self):
        self.store = {}
        self.calls = {"get": 0, "set": 0, "delete": 0}

    async def get(self, key):
        self.calls["get"] += 1
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.calls["set"] += 1
        self.store[key] = value

    async def delete(self, key):
        self.calls["delete"] += 1
        self.store.pop(key, None)


class FailingRedis(DummyRedis):
    async def get(self, key):
        raise ConnectionError("Redis unavailable")

    async def set(self, key, value, ex=None):
        raise ConnectionError("Redis unavailable")

    async def delete(self, key):
        raise ConnectionError("Redis unavailable")


@pytest.mark.asyncio
async def test_l1_l2_hierarchy():
    redis = DummyRedis()
    cache = CacheManager(redis_client=redis)
    key = cache._format_key(1, "foo")
    await redis.set(key, "bar")

    first = await cache.get(1, "foo")
    assert first == "bar"
    assert redis.calls["get"] == 1

    second = await cache.get(1, "foo")
    assert second == "bar"
    assert redis.calls["get"] == 1


@pytest.mark.asyncio
async def test_ttl_expiration():
    cache = CacheManager(redis_client=DummyRedis())
    await cache.set(1, "key", "value", ttl=1)
    await asyncio.sleep(1.1)
    assert await cache.get(1, "key") is None


@pytest.mark.asyncio
async def test_targeted_invalidation():
    cache = CacheManager(redis_client=DummyRedis())
    await cache.set(1, "key", "value")
    await cache.delete(1, "key")
    assert await cache.get(1, "key") is None


@pytest.mark.asyncio
async def test_failover_to_memory():
    cache = CacheManager(redis_client=FailingRedis())
    await cache.set(1, "key", "value")
    assert await cache.get(1, "key") == "value"


@pytest.mark.asyncio
async def test_concurrent_access():
    cache = CacheManager(redis_client=DummyRedis())
    await cache.set(1, "foo", "bar")

    async def reader():
        return await cache.get(1, "foo")

    results = await asyncio.gather(*[reader() for _ in range(50)])
    assert all(r == "bar" for r in results)

import time
from typing import Optional

import pytest
from pydantic import BaseModel, ValidationError

from conversation_service.core import CacheManager, MetricsCollector


class FakeCacheClient:
    """Simple in-memory stand-in for ``CacheClient`` supporting TTL."""

    def __init__(self) -> None:
        self.store = {}

    async def get(self, user_id: int, key: str):
        item = self.store.get((user_id, key))
        if not item:
            return None
        value, expire = item
        if expire and expire < time.time():
            del self.store[(user_id, key)]
            return None
        return value

    async def set(self, user_id: int, key: str, value, ttl: Optional[int] = None):
        expire = time.time() + ttl if ttl else None
        self.store[(user_id, key)] = (value, expire)

    async def delete(self, user_id: int, key: str):
        self.store.pop((user_id, key), None)


class Item(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_two_level_cache_with_validation_and_metrics():
    client = FakeCacheClient()
    metrics = MetricsCollector()
    manager = CacheManager(client, l1_ttl=0.1, l2_ttl=0.3, metrics=metrics)

    await manager.set(1, "k", {"value": 1}, model=Item)

    # Immediate access -> L1 hit
    item = await manager.get(1, "k", model=Item)
    assert item.value == 1
    assert metrics.hits["l1"] == 1

    # After L1 TTL expires but before L2 TTL -> L1 miss, L2 hit
    await asyncio.sleep(0.15)
    item = await manager.get(1, "k", model=Item)
    assert item.value == 1
    assert metrics.misses["l1"] == 1
    assert metrics.hits["l2"] == 1

    # After L2 TTL expires -> total miss
    await asyncio.sleep(0.2)
    item = await manager.get(1, "k", model=Item)
    assert item is None
    assert metrics.misses["l2"] == 1
    assert metrics.size["l1"] == 0

    # Validation errors surface
    with pytest.raises(ValidationError):
        await manager.set(1, "bad", {"wrong": 1}, model=Item)
