import asyncio
import time
import pytest

from conversation_service.core import CacheManager


class FakeCacheClient:
    def __init__(self) -> None:
        self.store = {}
        self.calls = {"get": 0, "set": 0, "delete": 0}

    async def get(self, user_id: int, key: str):
        self.calls["get"] += 1
        item = self.store.get((user_id, key))
        if not item:
            return None
        value, expire = item
        if expire and expire < time.time():
            del self.store[(user_id, key)]
            return None
        return value

    async def set(self, user_id: int, key: str, value, ttl: int | None = None):
        self.calls["set"] += 1
        expire = time.time() + ttl if ttl else None
        self.store[(user_id, key)] = (value, expire)

    async def delete(self, user_id: int, key: str):
        self.calls["delete"] += 1
        self.store.pop((user_id, key), None)


@pytest.mark.asyncio
async def test_l1_l2_hierarchy():
    client = FakeCacheClient()
    manager = CacheManager(client, l1_ttl=60, l2_ttl=60)
    await client.set(1, "foo", "bar")

    first = await manager.get(1, "foo")
    assert first == "bar"
    assert client.calls["get"] == 1

    second = await manager.get(1, "foo")
    assert second == "bar"
    assert client.calls["get"] == 1


@pytest.mark.asyncio
async def test_ttl_expiration():
    client = FakeCacheClient()
    manager = CacheManager(client, l1_ttl=0.1, l2_ttl=0.1)
    await manager.set(1, "key", "value", l1_ttl=0.1, l2_ttl=0.1)
    await asyncio.sleep(0.15)
    assert await manager.get(1, "key") is None


@pytest.mark.asyncio
async def test_targeted_invalidation():
    client = FakeCacheClient()
    manager = CacheManager(client)
    await manager.set(1, "key", "value")
    await manager.invalidate(1, "key")
    assert await manager.get(1, "key") is None


@pytest.mark.asyncio
async def test_concurrent_access():
    client = FakeCacheClient()
    manager = CacheManager(client)
    await manager.set(1, "foo", "bar")

    async def reader():
        return await manager.get(1, "foo")

    results = await asyncio.gather(*[reader() for _ in range(50)])
    assert all(r == "bar" for r in results)
