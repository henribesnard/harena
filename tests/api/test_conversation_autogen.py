import os
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request

from conversation_service.api.routes.conversation import router as conversation_router
from conversation_service.api.dependencies import (
    get_deepseek_client,
    get_cache_manager,
    get_conversation_service_status,
    validate_path_user_id,
    get_user_context,
    rate_limit_dependency,
)
from conversation_service.models.responses.conversation_responses import (
    IntentClassificationResult,
)
from conversation_service.prompts.harena_intents import HarenaIntentType

# Environment configuration for tests
os.environ.setdefault("CONVERSATION_SERVICE_ENABLED", "true")
os.environ.setdefault("ENVIRONMENT", "testing")


def create_test_app() -> FastAPI:
    """Minimal app with conversation router and dependency overrides."""
    app = FastAPI()
    app.include_router(conversation_router, prefix="/api/v1")

    async def override_validate_user(
        request: Request, path_user_id: int, token_user_id: int = 1
    ) -> int:
        request.state.user_id = path_user_id
        return path_user_id

    def override_get_user_context(request: Request, user_id: int = 1):
        return {"sub": str(getattr(request.state, "user_id", user_id))}

    async def noop_dep(request: Request = None):
        return None

    def override_service_status(request: Request):
        return {"status": "healthy"}

    app.dependency_overrides[get_deepseek_client] = noop_dep
    app.dependency_overrides[get_cache_manager] = noop_dep
    app.dependency_overrides[get_conversation_service_status] = override_service_status
    app.dependency_overrides[validate_path_user_id] = override_validate_user
    app.dependency_overrides[get_user_context] = override_get_user_context
    app.dependency_overrides[rate_limit_dependency] = noop_dep

    return app


@pytest.fixture
def client() -> TestClient:
    app = create_test_app()
    return TestClient(app, raise_server_exceptions=False)


def _patch_intent_classifier(mock_result):
    return patch(
        "conversation_service.agents.financial.intent_classifier.IntentClassifierAgent",
        **{"return_value.classify_intent": AsyncMock(return_value=mock_result)},
    )


class DummyRuntime:
    def __init__(self, result=None, should_fail=False):
        self.result = result or {"entities": {"account": "123"}, "metadata": {"engine": "autogen"}}
        self.should_fail = should_fail

    async def process_message(self, message, context):
        if self.should_fail:
            raise RuntimeError("autogen down")
        return self.result

    def health_check(self):
        return {"status": "ok"}


def _make_request(client: TestClient):
    return client.post("/api/v1/conversation/1", json={"message": "Bonjour"})


@pytest.fixture
def classification_result():
    return IntentClassificationResult(
        intent_type=HarenaIntentType.GREETING,
        confidence=0.95,
        reasoning="salut",
        original_message="Bonjour",
        category="CONVERSATIONAL",
        is_supported=True,
        alternatives=[],
        processing_time_ms=50,
    )


def test_autogen_mode_active_returns_extended_response(client, classification_result, monkeypatch):
    client.app.state.autogen_runtime = DummyRuntime()
    monkeypatch.setenv("CONVERSATION_AUTOGEN_ENABLED", "true")

    with _patch_intent_classifier(classification_result):
        response = _make_request(client)

    data = response.json()
    assert response.status_code == 200
    assert "autogen_metadata" in data
    assert "entities" in data
    assert data["autogen_metadata"].get("engine") == "autogen"


def test_autogen_runtime_failure_falls_back_to_legacy(client, classification_result, monkeypatch):
    client.app.state.autogen_runtime = DummyRuntime(should_fail=True)
    monkeypatch.setenv("CONVERSATION_AUTOGEN_ENABLED", "true")

    with _patch_intent_classifier(classification_result):
        response = _make_request(client)

    data = response.json()
    assert response.status_code == 200
    assert "autogen_metadata" not in data
    assert "entities" not in data


def test_feature_flag_disabled_uses_legacy_flow(client, classification_result, monkeypatch):
    client.app.state.autogen_runtime = DummyRuntime()
    monkeypatch.setenv("CONVERSATION_AUTOGEN_ENABLED", "false")

    with _patch_intent_classifier(classification_result):
        response = _make_request(client)

    data = response.json()
    assert response.status_code == 200
    assert "autogen_metadata" not in data
    assert "entities" not in data


def test_health_endpoint_reports_mode_availability(client, monkeypatch):
    with patch("conversation_service.api.routes.conversation.metrics_collector") as metrics:
        metrics.get_health_metrics.return_value = {
            "status": "healthy",
            "total_requests": 0,
            "error_rate_percent": 0,
            "latency_p95_ms": 0,
            "uptime_seconds": 0,
        }
        # without runtime
        resp = client.get("/api/v1/conversation/health")
        assert resp.status_code == 200
        assert resp.json()["modes"]["autogen"] is False

        # with runtime
        client.app.state.autogen_runtime = DummyRuntime()
        resp = client.get("/api/v1/conversation/health")
        assert resp.status_code == 200
        assert resp.json()["modes"]["autogen"] is True
