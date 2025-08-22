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
