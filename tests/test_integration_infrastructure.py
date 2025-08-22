import asyncio
import json
import importlib
import os
import logging

import pytest

from monitoring.performance import track_operation, record_openai_cost


async def _generate_and_cache(client, cache):
    """Pipeline: call OpenAI and store the parsed response in cache."""
    response = await client.chat.completions.create(messages=[])
    data = json.loads(response.choices[0].message.content)
    cache.set("result", data)
    return data


def test_pipeline_openai_to_cache(openai_mock, cache):
    """The OpenAI response should be stored in the cache."""
    result = asyncio.run(_generate_and_cache(openai_mock, cache))
    assert cache.get("result") == result


def test_metrics_collection(caplog):
    """track_operation and record_openai_cost should emit logs."""
    with caplog.at_level(logging.INFO, logger="performance"):
        with track_operation("unit-test"):
            pass
        record_openai_cost(0.5)
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("unit-test took" in msg for msg in messages)
    assert any("OpenAI cost +0.5000" in msg for msg in messages)


class _FailingClient:
    class _Chat:
        class _Completions:
            async def create(self, *args, **kwargs):  # pragma: no cover - used in tests
                raise RuntimeError("boom")

        def __init__(self):
            self.completions = self._Completions()

    def __init__(self):
        self.chat = self._Chat()


def test_error_propagation(cache):
    """Errors from the OpenAI client should propagate and not cache data."""
    client = _FailingClient()

    async def pipeline():
        await _generate_and_cache(client, cache)

    with pytest.raises(RuntimeError):
        asyncio.run(pipeline())
    assert cache.get("result") is None


def test_configuration_reload(monkeypatch):
    """Reloading the configuration should pick up new environment values."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    import config_service.config as config
    original_env = os.getenv("OPENAI_CHAT_MODEL")
    new_value = "gpt-test"
    old_model = config.settings.OPENAI_CHAT_MODEL

    monkeypatch.setenv("OPENAI_CHAT_MODEL", new_value)
    importlib.reload(config)
    assert config.settings.OPENAI_CHAT_MODEL == new_value

    if original_env is None:
        monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_CHAT_MODEL", original_env)
    importlib.reload(config)
    assert config.settings.OPENAI_CHAT_MODEL == old_model
