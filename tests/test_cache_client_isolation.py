import pytest

from clients.cache_client import CacheClient


@pytest.mark.asyncio
async def test_cache_client_isolated_by_user_id():
    cache = CacheClient()
    await cache.set(1, "greeting", "hello")
    assert await cache.get(1, "greeting") == "hello"
    assert await cache.get(2, "greeting") is None
