import os
import pytest
from unittest.mock import AsyncMock

# Ensure required env vars for config (SECRET_KEY powers bearer token verification)
os.environ.setdefault("DEEPSEEK_API_KEY", "testkey")
os.environ.setdefault("SECRET_KEY", "x" * 33)

from conversation_service.clients.deepseek_client import DeepSeekClient


class DummyResponse:
    def __init__(self, status=200):
        self.status = status
        self.headers = {}

    async def json(self):
        return {"choices": [{"message": {"content": "{}"}}]}

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_chat_completion_includes_response_format():
    client = DeepSeekClient()
    client._initialized = True

    captured = {}

    class MockSession:
        closed = False

        def post(self, url, json):
            captured["payload"] = json
            return DummyResponse()

    client._session = MockSession()

    resp_format = {"type": "json_object"}
    await client.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        response_format=resp_format,
    )

    assert captured["payload"].get("response_format") == resp_format
