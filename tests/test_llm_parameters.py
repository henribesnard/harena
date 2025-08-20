import types
from types import SimpleNamespace
import pytest

from conversation_service.core import deepseek_client as dsc
from conversation_service.core.deepseek_client import DeepSeekClient


class DummyAsyncOpenAI:
    def __init__(self, recorder):
        self.recorder = recorder
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=recorder)
        )


class Recorder:
    def __init__(self):
        self.kwargs = None

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            choices=[],
        )


@pytest.mark.asyncio
async def test_llm_parameters_passed(monkeypatch):
    recorder = Recorder()
    dummy_client = DummyAsyncOpenAI(recorder)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("LLM_TOP_P", "0.8")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("CIRCUIT_BREAKER_ENABLED", "false")

    monkeypatch.setattr(dsc, "AsyncOpenAI", lambda *args, **kwargs: dummy_client)
    monkeypatch.setattr(dsc, "get_default_metrics_collector", lambda: SimpleNamespace(record_deepseek_usage=lambda **k: None))

    client = DeepSeekClient(cache_enabled=False)
    await client.create_chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert recorder.kwargs["temperature"] == 0.1
    assert recorder.kwargs["top_p"] == 0.8
