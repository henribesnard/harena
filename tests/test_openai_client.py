import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip the entire test module if openai_client is missing
openai_client = pytest.importorskip("openai_client")
OpenAIClient = openai_client.OpenAIClient


@pytest.mark.asyncio
async def test_chat_completion_success(monkeypatch):
    """The client returns the message content on a successful call."""
    client = OpenAIClient()
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))],
        usage=SimpleNamespace(total_cost=0.001),
    )
    # Patch the underlying API call
    async_create = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(
        client._client.chat.completions, "create", async_create, raising=True
    )
    result = await client.chat_completion([{"role": "user", "content": "ping"}])
    assert result == "pong"


@pytest.mark.asyncio
async def test_rate_limiting(monkeypatch):
    """Concurrent calls are serialized by the internal rate limiter."""
    client = OpenAIClient()
    # enforce a single concurrent request
    setattr(client, "_semaphore", asyncio.Semaphore(1))

    async def slow_call(*args, **kwargs):
        await asyncio.sleep(0.1)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(total_cost=0.0),
        )

    monkeypatch.setattr(
        client._client.chat.completions, "create", slow_call, raising=True
    )

    start = time.perf_counter()
    await asyncio.gather(
        client.chat_completion([{"role": "user", "content": "hi"}]),
        client.chat_completion([{"role": "user", "content": "hi"}]),
    )
    duration = time.perf_counter() - start
    assert duration >= 0.2


@pytest.mark.asyncio
async def test_cost_tracking(monkeypatch):
    """Ensure API usage cost is recorded accurately."""
    client = OpenAIClient()
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(total_cost=0.1234),
    )
    async_create = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(
        client._client.chat.completions, "create", async_create, raising=True
    )

    with patch("openai_client.record_openai_cost") as cost_mock:
        await client.chat_completion([{"role": "user", "content": "hi"}])
        cost_mock.assert_called_once_with(0.1234)


@pytest.mark.asyncio
async def test_connection_pool_reuse():
    """HTTP connection pool should be reused across calls."""
    with patch("openai_client.httpx.AsyncClient") as async_client_cls:
        client_instance = AsyncMock()
        async_client_cls.return_value = client_instance
        # Simulate API responses
        client_instance.__aenter__.return_value = client_instance
        client_instance.post.return_value = SimpleNamespace(status_code=200, json=lambda: {})

        client = OpenAIClient()
        await client.chat_completion([{"role": "user", "content": "hi"}])
        await client.chat_completion([{"role": "user", "content": "hi"}])

        # Underlying HTTP client constructed once
        assert async_client_cls.call_count == 1


@pytest.mark.asyncio
async def test_error_retry(monkeypatch):
    """Transient errors trigger a retry before succeeding."""
    client = OpenAIClient()
    attempts = 0

    async def flaky_call(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise openai_client.httpx.HTTPStatusError("boom", request=None, response=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(total_cost=0.0),
        )

    monkeypatch.setattr(
        client._client.chat.completions, "create", flaky_call, raising=True
    )
    result = await client.chat_completion([{"role": "user", "content": "hi"}])
    assert attempts == 2
    assert result == "ok"


@pytest.mark.asyncio
async def test_timeout_handling(monkeypatch):
    """Timeouts from the HTTP layer are surfaced as TimeoutError."""
    client = OpenAIClient()

    async def timeout_call(*args, **kwargs):
        raise asyncio.TimeoutError

    monkeypatch.setattr(
        client._client.chat.completions, "create", timeout_call, raising=True
    )
    with pytest.raises(asyncio.TimeoutError):
        await client.chat_completion([{"role": "user", "content": "hi"}])
