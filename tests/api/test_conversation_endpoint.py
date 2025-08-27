"""Tests for conversation endpoint using AutoGen runtime."""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from jose import jwt

# ---------------------------------------------------------------------------
# Environment configuration for tests
# ---------------------------------------------------------------------------
os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key-12345"
os.environ["SECRET_KEY"] = "a" * 32 + "b" * 32
os.environ["CONVERSATION_SERVICE_ENABLED"] = "true"
os.environ["ENVIRONMENT"] = "testing"

# Ensure project modules are importable when running directly
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from conversation_service.api.routes.conversation import router as conversation_router
from conversation_service.api.middleware.auth_middleware import JWTAuthMiddleware


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def generate_test_jwt(sub: int = 1, expired: bool = False) -> str:
    """Generate a JWT compatible with user_service logic."""
    if expired:
        expire = datetime.utcnow() - timedelta(hours=1)
    else:
        expire = datetime.utcnow() + timedelta(hours=24)

    payload = {
        "exp": expire,
        "sub": str(sub),
        "permissions": ["chat:write"],
    }

    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        raise ValueError("SECRET_KEY missing in environment")

    return jwt.encode(payload, secret_key, algorithm="HS256")


def get_test_auth_headers(user_id: int = 1) -> dict:
    token = generate_test_jwt(user_id)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# App factory and fixtures
# ---------------------------------------------------------------------------
def create_test_app():
    from fastapi import FastAPI

    app = FastAPI(title="Test Conversation Service")
    app.add_middleware(JWTAuthMiddleware)
    app.include_router(conversation_router, prefix="/api/v1")
    return app


@pytest.fixture
def mock_runtime():
    """Mocked ConversationServiceRuntime with controllable behaviour."""
    runtime = AsyncMock()
    runtime.run_financial_team = AsyncMock(
        return_value={
            "final_answer": "Bonjour!",
            "context": {"foo": "bar"},
            "intermediate_steps": [
                {"agent": "analyst", "content": "Analysing: Bonjour"}
            ],
        }
    )
    return runtime


@pytest.fixture
def test_app(mock_runtime):
    """Create a FastAPI app with dependency overrides for tests."""
    app = create_test_app()

    from conversation_service.api.dependencies import (
        get_conversation_runtime,
        get_conversation_service_status,
        validate_path_user_id,
        get_user_context,
        rate_limit_dependency,
    )
    from conversation_service.api.middleware.auth_middleware import verify_user_id_match
    from fastapi import Request

    def override_get_runtime(request: Request):
        return mock_runtime

    def override_get_status(request: Request):
        return {"status": "healthy"}

    async def override_validate_user(
        request: Request, path_user_id: int, token_user_id: int = 1
    ) -> int:
        request.state.user_id = token_user_id
        await verify_user_id_match(request, path_user_id)
        return path_user_id

    def override_get_user_context(request: Request, user_id: int = 1):
        return {"sub": str(getattr(request.state, "user_id", user_id))}

    def override_rate_limit(request: Request, user_id: int = 1):
        return None

    app.dependency_overrides[get_conversation_runtime] = override_get_runtime
    app.dependency_overrides[get_conversation_service_status] = override_get_status
    app.dependency_overrides[validate_path_user_id] = override_validate_user
    app.dependency_overrides[get_user_context] = override_get_user_context
    app.dependency_overrides[rate_limit_dependency] = override_rate_limit

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# JWT compatibility tests
# ---------------------------------------------------------------------------
class TestJWTCompatibility:
    """Tests ensuring JWT generation is compatible with validators."""

    def test_jwt_token_compatibility(self):
        from conversation_service.api.middleware.auth_middleware import JWTValidator

        token = generate_test_jwt(sub=1)
        validator = JWTValidator()
        result = validator.validate_token(token)

        assert result.success
        assert result.user_id == 1
        payload = result.token_payload
        assert payload.get("sub") == "1"
        assert "chat:write" in payload.get("permissions", [])

    def test_generate_test_jwt_format(self):
        token = generate_test_jwt(sub=42)
        parts = token.split(".")
        assert len(parts) == 3
        payload = jwt.decode(token, os.environ["SECRET_KEY"], algorithms=["HS256"])
        assert payload["sub"] == "42"
        assert "exp" in payload

    def test_expired_token_generation(self):
        from conversation_service.api.middleware.auth_middleware import JWTValidator

        expired_token = generate_test_jwt(sub=1, expired=True)
        validator = JWTValidator()
        result = validator.validate_token(expired_token)
        assert not result.success
        assert result.error_code == "TOKEN_EXPIRED"


# ---------------------------------------------------------------------------
# Conversation endpoint tests
# ---------------------------------------------------------------------------
class TestConversationEndpoint:
    def test_conversation_success(self, client, mock_runtime):
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers=get_test_auth_headers(1),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert data["team_response"]["final_answer"] == "Bonjour!"
        assert len(data["team_response"]["steps"]) == 1
        mock_runtime.run_financial_team.assert_awaited_once()

    def test_conversation_runtime_error(self, client, mock_runtime):
        mock_runtime.run_financial_team.side_effect = Exception("boom")
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers=get_test_auth_headers(1),
        )
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# Authentication and authorisation tests
# ---------------------------------------------------------------------------
class TestAuthenticationAndAuthorization:
    def test_conversation_missing_authorization(self, client):
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
        )
        assert response.status_code == 401

    def test_conversation_invalid_token(self, client):
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401

    def test_conversation_expired_token(self, client):
        expired_token = generate_test_jwt(1, expired=True)
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Response structure tests
# ---------------------------------------------------------------------------
class TestResponseStructure:
    def test_conversation_response_structure(self, client):
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers=get_test_auth_headers(1),
        )
        assert response.status_code == 200
        data = response.json()

        base_fields = [
            "user_id",
            "message",
            "timestamp",
            "processing_time_ms",
            "team_response",
            "status",
            "phase",
            "request_id",
        ]
        for field in base_fields:
            assert field in data

        team_fields = ["final_answer", "steps", "context"]
        for field in team_fields:
            assert field in data["team_response"]
