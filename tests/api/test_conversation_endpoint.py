import os
import sys
import importlib.util
from pathlib import Path

import jwt
import importlib

# Ensure we use the real httpx library (remove stub injected by tests)
for mod in list(sys.modules):
    if mod.startswith("httpx"):
        del sys.modules[mod]
httpx = importlib.import_module("httpx")

from fastapi.testclient import TestClient
from conversation_service.prompts.harena_intents import HarenaIntentType

# Ensure required environment variables for app configuration
os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
os.environ["JWT_SECRET_KEY"] = "super-secret-key-with-more-than-32-chars-123456"
os.environ["CONVERSATION_SERVICE_ENABLED"] = "true"

# Dynamically load conversation_service.main as module named 'main'
module_path = Path(__file__).resolve().parents[2] / "conversation_service" / "main.py"
spec = importlib.util.spec_from_file_location("main", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Ensure settings values are correctly set for tests
module.settings.DEEPSEEK_API_KEY = "test-deepseek-key"
module.settings.JWT_SECRET_KEY = "super-secret-key-with-more-than-32-chars-123456"
module.settings.CONVERSATION_SERVICE_ENABLED = True


class DummyDeepSeekClient:
    def __init__(self, *, invalid_json: bool = False):
        self.invalid_json = invalid_json
        self.chat_args = None
        self.chat_kwargs = None

    async def initialize(self):
        pass

    async def health_check(self) -> bool:
        return True

    async def close(self):
        pass

    async def chat_completion(self, *args, **kwargs):
        self.chat_args = args
        self.chat_kwargs = kwargs
        content = (
            "not a json"
            if self.invalid_json
            else '{"intent": "GREETING", "confidence": 0.9, "reasoning": "test"}'
        )
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 10},
        }


class DummyCacheManager:
    async def initialize(self):
        pass

    async def health_check(self) -> bool:
        return True

    async def close(self):
        pass

    async def get_semantic_cache(self, *args, **kwargs):
        return None

    async def set_semantic_cache(self, *args, **kwargs):
        return True

    async def clear_all_cache(self):
        return True


# Patch dependencies in the loaded module
module.DeepSeekClient = DummyDeepSeekClient
module.CacheManager = DummyCacheManager
sys.modules["main"] = module

from main import app, conversation_service_loader  # type: ignore  # noqa: E402

# Initialize service manually to ensure routes are registered
import asyncio
asyncio.run(conversation_service_loader.initialize_conversation_service(app))

# Remove global exception handler to surface HTTP errors in tests
app.exception_handlers.pop(Exception, None)

client = TestClient(app, raise_server_exceptions=False)


def generate_jwt(user_id: int) -> str:
    payload = {"user_id": user_id}
    secret = os.environ["JWT_SECRET_KEY"]
    return jwt.encode(payload, secret, algorithm="HS256")


def test_conversation_endpoint_success():
    token = generate_jwt(1)
    response = client.post(
        "/api/v1/conversation/1",
        json={"message": "Bonjour"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["intent"]["intent_type"] == HarenaIntentType.GREETING.value

    deepseek_client = app.state.deepseek_client
    assert deepseek_client.chat_kwargs["response_format"] == {"type": "json_object"}


def test_conversation_endpoint_invalid_json_response():
    token = generate_jwt(1)
    original_client = app.state.deepseek_client
    bad_client = DummyDeepSeekClient(invalid_json=True)
    app.state.deepseek_client = bad_client
    conversation_service_loader.deepseek_client = bad_client
    response = client.post(
        "/api/v1/conversation/1",
        json={"message": "Bonjour"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 500
    app.state.deepseek_client = original_client
    conversation_service_loader.deepseek_client = original_client


def test_conversation_endpoint_user_mismatch():
    token = generate_jwt(1)
    response = client.post(
        "/api/v1/conversation/2",
        json={"message": "Bonjour"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
