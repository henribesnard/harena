import asyncio
import time

import pytest

from conversation_service.core import CacheManager


class DummyCacheClient:
    def __init__(self):
        self.store = {}
        self.calls = {"get": 0, "set": 0, "delete": 0}

    async def get(self, user_id, key):
        self.calls["get"] += 1
        item = self.store.get((user_id, key))
        if not item:
            return None
        value, expire = item
        if expire and expire < time.time():
            del self.store[(user_id, key)]
            return None
        return value

    async def set(self, user_id, key, value, ttl=None):
        self.calls["set"] += 1
        expire = time.time() + ttl if ttl else None
        self.store[(user_id, key)] = (value, expire)

    async def delete(self, user_id, key):
        self.calls["delete"] += 1
        self.store.pop((user_id, key), None)


class FailingCacheClient(DummyCacheClient):
    async def get(self, user_id, key):
        raise ConnectionError("Redis unavailable")


@pytest.mark.asyncio
async def test_l1_l2_hierarchy():
    client = DummyCacheClient()
    cache = CacheManager(client)
    await client.set(1, "foo", "bar")

    first = await cache.get(1, "foo")
    assert first == "bar"
    assert client.calls["get"] == 1

    second = await cache.get(1, "foo")
    assert second == "bar"
    assert client.calls["get"] == 1


@pytest.mark.asyncio
async def test_ttl_expiration():
    cache = CacheManager(DummyCacheClient(), l1_ttl=1, l2_ttl=1)
    await cache.set(1, "key", "value")
    await asyncio.sleep(1.1)
    assert await cache.get(1, "key") is None


@pytest.mark.asyncio
async def test_targeted_invalidation():
    cache = CacheManager(DummyCacheClient())
    await cache.set(1, "key", "value")
    await cache.invalidate(1, "key")
    assert await cache.get(1, "key") is None


@pytest.mark.asyncio
async def test_failover_to_memory():
    cache = CacheManager(FailingCacheClient())
    await cache.set(1, "key", "value")
    assert await cache.get(1, "key") == "value"


@pytest.mark.asyncio
async def test_concurrent_access():
    cache = CacheManager(DummyCacheClient())
    await cache.set(1, "foo", "bar")

    async def reader():
        return await cache.get(1, "foo")

    results = await asyncio.gather(*[reader() for _ in range(50)])
    assert all(r == "bar" for r in results)

