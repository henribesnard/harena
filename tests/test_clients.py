import asyncio
from types import SimpleNamespace

import httpx
import pytest

from clients.openai_client import OpenAIClient
from clients.search_client import SearchClient
from clients.cache_client import CacheClient


@pytest.mark.asyncio
async def test_openai_client_stream(monkeypatch):
    client = OpenAIClient()

    async def fake_create(*args, **kwargs):
        chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])
        chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="!"))])
        async def gen():
            yield chunk1
            yield chunk2
        return gen()

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)

    messages = [{"role": "user", "content": "hello"}]
    chunks = []
    async for piece in client.stream_chat(messages):
        chunks.append(piece)
    await client.close()
    assert "".join(chunks) == "hi!"


@pytest.mark.asyncio
async def test_search_client(monkeypatch):
    async def handler(request) -> httpx.Response:
        assert request.url.path == "/search"
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = SearchClient("http://test", transport=transport)
    resp = await client.search({"query": "x", "user_id": 1})
    await client.close()
    assert resp == {"ok": True}


class DummyRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        return True

    async def delete(self, key):
        self.store.pop(key, None)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_cache_client():
    cache = CacheClient(redis_client=DummyRedis())
    await cache.set("a", "1")
    assert await cache.get("a") == "1"
    await cache.delete("a")
    assert await cache.get("a") is None
    await cache.close()
