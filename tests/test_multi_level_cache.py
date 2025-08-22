import asyncio
import pytest

from search_service.utils.cache import MultiLevelCache


@pytest.mark.asyncio
async def test_multi_level_cache_isolated_by_user_id():
    """Chaque utilisateur dispose de son propre espace de clés."""
    cache = MultiLevelCache()
    await cache.set(1, "greeting", "hello")
    assert await cache.get(1, "greeting") == "hello"
    assert await cache.get(2, "greeting") is None


@pytest.mark.asyncio
async def test_multi_level_cache_expires_entries():
    """Les entrées expirent correctement après le TTL."""
    cache = MultiLevelCache()
    await cache.set(1, "temp", "data", ttl=0.1)
    assert await cache.get(1, "temp") == "data"
    await asyncio.sleep(0.2)
    assert await cache.get(1, "temp") is None


@pytest.mark.asyncio
async def test_multi_level_cache_clear():
    """La méthode ``clear`` vide complètement le cache."""
    cache = MultiLevelCache()
    await cache.set(1, "a", "b")
    await cache.clear()
    assert await cache.get(1, "a") is None

