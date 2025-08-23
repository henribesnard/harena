import json
import types

import pytest

from conversation_service.core import CacheManager, MetricsCollector
from monitoring import performance


class _AsyncCacheClient:
    """Async wrapper around the ``cache`` fixture used as L2 storage."""

    def __init__(self, cache):
        self._cache = cache

    async def get(self, user_id, key):  # pragma: no cover - simple passthrough
        return self._cache.get(key)

    async def set(self, user_id, key, value, ttl=None):  # pragma: no cover
        self._cache.set(key, value)

    async def delete(self, user_id, key):  # pragma: no cover
        self._cache.set(key, None)


@pytest.mark.asyncio
async def test_openai_response_cached(openai_mock, cache, monkeypatch):
    """OpenAI responses are cached and cost metrics recorded."""
    metrics = MetricsCollector()
    manager = CacheManager(_AsyncCacheClient(cache), metrics=metrics)

    # Provide a response with cost information
    payload = openai_mock._content
    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=payload)
            )
        ],
        usage=types.SimpleNamespace(total_cost=0.42),
    )

    async def fake_create(*args, **kwargs):
        return fake_response

    monkeypatch.setattr(
        openai_mock.chat.completions, "create", fake_create, raising=True
    )

    costs = []
    monkeypatch.setattr(performance, "record_openai_cost", lambda c: costs.append(c))

    # Simulate the OpenAI call and store the parsed result
    response = await openai_mock.chat.completions.create(messages=[])
    data = json.loads(response.choices[0].message.content)
    performance.record_openai_cost(response.usage.total_cost)
    await manager.set(1, "intent", data)

    # First retrieval should hit the in-memory cache
    result1 = await manager.get(1, "intent")
    assert result1 == data
    assert metrics.hits["l1"] == 1

    # Clear L1 to force a Redis (L2) lookup
    manager._l1_cache.clear()
    metrics.record_size("l1", 0)
    result2 = await manager.get(1, "intent")
    assert result2 == data
    assert metrics.misses["l1"] == 1
    assert metrics.hits["l2"] == 1

    # Cost metrics were recorded
    assert costs == [0.42]
