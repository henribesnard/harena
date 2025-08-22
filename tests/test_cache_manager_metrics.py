import asyncio
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

