import time
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.testclient import TestClient
from tests.utils import assert_status

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.api.dependencies import (
    get_deepseek_client,
    get_cache_manager,
    get_conversation_service_status,
    validate_path_user_id,
    get_user_context,
    rate_limit_dependency,
    get_conversation_engine,
)
from conversation_service.api.routes.conversation import (
    _run_autogen_pipeline,
    _run_legacy_pipeline,
    conversation_health_detailed,
)
from conversation_service.prompts.harena_intents import HarenaIntentType


class DummyRuntime:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.is_initialized = True

    def get_team(self, name):  # pragma: no cover - simple stub
        runtime_self = self

        class Team:
            async def process_user_message(self, message, user_id):
                if runtime_self.should_fail:
                    raise RuntimeError("autogen failure")
                return {
                    "intent": {
                        "intent": HarenaIntentType.GREETING.value,
                        "confidence": 0.9,
                        "reasoning": "salut",
                        "category": "CONVERSATIONAL",
                    },
                    "entities": {},
                    "errors": [],
                }

        return Team

    def health_check(self):  # pragma: no cover - simple stub
        return {"status": "ok"}


class DummyLegacyAgent:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    async def classify_for_team(self, message, user_id):
        if self.should_fail:
            raise RuntimeError("legacy failure")
        return {
            "intent": HarenaIntentType.GREETING.value,
            "confidence": 0.95,
            "reasoning": "hi",
            "category": "CONVERSATIONAL",
        }


def create_test_app() -> FastAPI:
    app = FastAPI()

    @app.post("/api/v1/conversation/{user_id}")
    async def conversation_endpoint(
        user_id: int,
        req: ConversationRequest,
        engine: dict = Depends(get_conversation_engine),
    ):
        start = time.time()
        runtime = engine.get("runtime")
        legacy = engine.get("legacy_agent")
        mode = engine.get("mode", "legacy")

        if mode == "autogen" and runtime is not None:
            try:
                return await _run_autogen_pipeline(user_id, req, runtime, start)
            except Exception:
                if legacy is not None:
                    return await _run_legacy_pipeline(user_id, req, legacy, start)
                raise HTTPException(status_code=500, detail="Autogen pipeline failed")

        if legacy is not None:
            return await _run_legacy_pipeline(user_id, req, legacy, start)

        raise HTTPException(status_code=500, detail="No engine available")

    app.get("/api/v1/conversation/health")(conversation_health_detailed)

    async def override_validate(request: Request, path_user_id: int, token_user_id: int = 1) -> int:
        request.state.user_id = path_user_id
        return path_user_id

    def override_user_context(request: Request, user_id: int = 1):
        return {"sub": str(getattr(request.state, "user_id", user_id))}

    async def noop(request: Request = None):  # pragma: no cover - simple stub
        return None

    def service_status(request: Request):  # pragma: no cover - simple stub
        return {"status": "healthy"}

    app.dependency_overrides[get_deepseek_client] = noop
    app.dependency_overrides[get_cache_manager] = noop
    app.dependency_overrides[get_conversation_service_status] = service_status
    app.dependency_overrides[validate_path_user_id] = override_validate
    app.dependency_overrides[get_user_context] = override_user_context
    app.dependency_overrides[rate_limit_dependency] = noop

    return app


@pytest.fixture
def client() -> TestClient:
    app = create_test_app()
    return TestClient(app, raise_server_exceptions=False)


def _make_request(client: TestClient):
    return client.post("/api/v1/conversation/1", json={"message": "Bonjour"})


@pytest.mark.integration
@pytest.mark.api
def test_autogen_runtime_produces_enriched_response(client):
    runtime = DummyRuntime()
    legacy = DummyLegacyAgent()

    async def engine_override(request: Request):
        return {"runtime": runtime, "legacy_agent": legacy, "mode": "autogen"}

    client.app.dependency_overrides[get_conversation_engine] = engine_override
    resp = _make_request(client)
    data = resp.json()
    assert_status(resp)
    assert data["agent_metrics"]["agent_used"] == "autogen"
    assert "autogen_metadata" in data and "entities" in data


@pytest.mark.parametrize("runtime", [None, DummyRuntime(should_fail=True)])
@pytest.mark.integration
@pytest.mark.api
def test_fallback_to_legacy_when_runtime_missing_or_failing(client, runtime):
    legacy = DummyLegacyAgent()
    mode = "autogen" if runtime else "legacy"

    async def engine_override(request: Request):
        return {"runtime": runtime, "legacy_agent": legacy, "mode": mode}

    client.app.dependency_overrides[get_conversation_engine] = engine_override
    resp = _make_request(client)
    data = resp.json()
    assert_status(resp)
    assert data["agent_metrics"]["agent_used"] == "legacy_intent_classifier"
    assert "autogen_metadata" not in data


@pytest.mark.integration
@pytest.mark.api
def test_cascade_errors_return_http_500(client):
    runtime = DummyRuntime(should_fail=True)
    legacy = DummyLegacyAgent(should_fail=True)

    async def engine_override(request: Request):
        return {"runtime": runtime, "legacy_agent": legacy, "mode": "autogen"}

    client.app.dependency_overrides[get_conversation_engine] = engine_override
    resp = _make_request(client)
    assert_status(resp, 500)


@pytest.mark.integration
@pytest.mark.api
def test_health_endpoint_reports_mode_availability(client, monkeypatch):
    with patch("conversation_service.api.routes.conversation.metrics_collector") as metrics:
        metrics.get_health_metrics.return_value = {
            "status": "healthy",
            "total_requests": 0,
            "error_rate_percent": 0,
            "latency_p95_ms": 0,
            "uptime_seconds": 0,
        }

        resp = client.get("/api/v1/conversation/health")
        assert_status(resp)
        assert resp.json()["modes"]["autogen"] is False

        client.app.state.autogen_runtime = DummyRuntime()
        resp = client.get("/api/v1/conversation/health")
        assert_status(resp)
        assert resp.json()["modes"]["autogen"] is True

