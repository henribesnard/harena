import pytest

from conversation_service.core.cache_manager import CacheManager
from conversation_service.core import l0_cache


@pytest.mark.asyncio
async def test_l0_cache_hit_and_invalidation():
    """L0 cache should serve values before Redis/LRU and allow invalidation."""
    l0_cache.invalidate()  # ensure clean start
    manager = CacheManager(cache_client=None, prefix="test")
    composed = manager._compose_key("foo", "user1")
    l0_cache.warmup({composed: "bar"})

    # Value should come from L0
    value = await manager.get("foo", "user1")
    assert value == "bar"

    # After invalidation the value should no longer be returned
    l0_cache.invalidate([composed])
    value = await manager.get("foo", "user1")
    assert value is None
