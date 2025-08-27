"""Tests for conversation endpoint using AutoGen runtime."""
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import json

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
# Import dynamique avec gestion d'erreur
try:
    from fastapi.testclient import TestClient
    from conversation_service.models.requests.conversation_requests import ConversationRequest
except ImportError as e:
    # Fallback si imports directs ne fonctionnent pas
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fastapi.testclient import TestClient
    from conversation_service.models.requests.conversation_requests import ConversationRequest


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


# ============================================================================
# MOCKS POUR SERVICES EXTERNES
# ============================================================================

class MockDeepSeekClient:
    """Mock DeepSeek Client avec réponses configurables"""
    
    def __init__(self, *, invalid_json: bool = False, error_response: bool = False):
        self.invalid_json = invalid_json
        self.error_response = error_response
        self.call_count = 0
        self.last_call_args = None
        self.last_call_kwargs = None

    async def initialize(self):
        pass

    async def health_check(self) -> bool:
        return not self.error_response

    async def close(self):
        pass

    async def chat_completion(self, *args, **kwargs):
        self.call_count += 1
        self.last_call_args = args
        self.last_call_kwargs = kwargs
        
        if self.error_response:
            raise Exception("Simulated DeepSeek error")
        
        if self.invalid_json:
            content = "This is not valid JSON at all"
        else:
            content = json.dumps({
                "intent": "GREETING",
                "confidence": 0.95,
                "reasoning": "Message de salutation détecté"
            })
            
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 100}
        }

class MockCacheManager:
    """Mock Cache Manager avec fonctionnalités configurables"""
    
    def __init__(self, *, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.cache_data = {}
        self.get_calls = 0
        self.set_calls = 0

    async def initialize(self):
        pass

    async def health_check(self) -> bool:
        return self.cache_enabled

    async def close(self):
        pass

    async def get_semantic_cache(self, key: str, **kwargs):
        self.get_calls += 1
        if not self.cache_enabled:
            return None
        return self.cache_data.get(key)

    async def set_semantic_cache(self, key: str, data: dict, **kwargs):
        self.set_calls += 1
        if self.cache_enabled:
            self.cache_data[key] = data
        return self.cache_enabled

    async def get_cache_stats(self):
        return {
            "status": "active" if self.cache_enabled else "disabled",
            "keys_count": len(self.cache_data)
        }

    async def clear_all_cache(self):
        if self.cache_enabled:
            self.cache_data.clear()
            return True
        return False

class MockConversationServiceLoader:
    """Mock du service loader pour tests"""
    
    def __init__(self):
        self.service_healthy = True
        self.initialization_error = None
        self.deepseek_client = MockDeepSeekClient()
        self.cache_manager = MockCacheManager()

    async def initialize_conversation_service(self, app):
        if self.service_healthy:
            app.state.conversation_service = self
            app.state.deepseek_client = self.deepseek_client
            app.state.cache_manager = self.cache_manager
            app.state.service_config = {
                "phase": 2,
                "version": "1.0.0",
                "features": ["autogen_team"]
            }
            return True
        return False

    async def cleanup(self):
        pass

# ============================================================================
# CONFIGURATION APP DE TEST
# ============================================================================

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
def mock_service_loader():
    """Fixture providing a mock conversation service loader."""
    return MockConversationServiceLoader()


@pytest.fixture
def test_app(mock_runtime, mock_service_loader):
    """Create a FastAPI app with dependency overrides for tests."""
    app = create_test_app()

    mock_service_loader = MockConversationServiceLoader()
    app.state.deepseek_client = mock_service_loader.deepseek_client
    app.state.cache_manager = mock_service_loader.cache_manager

    from conversation_service.api.dependencies import (
        get_cache_manager,
        get_conversation_runtime,
        get_conversation_service_status,
        get_deepseek_client,
        get_user_context,
        rate_limit_dependency,
        validate_path_user_id,
        get_cache_manager,
        validate_path_user_id,
        get_user_context,
        rate_limit_dependency,
        get_deepseek_client,

    )
    from conversation_service.api.middleware.auth_middleware import verify_user_id_match
    from fastapi import Request

    def override_get_deepseek_client(request: Request):
        return mock_service_loader.deepseek_client

    def override_get_cache_manager(request: Request):
        return mock_service_loader.cache_manager

    def override_get_service_status(request: Request):
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
    app.dependency_overrides[get_conversation_service_status] = override_get_service_status
    mock_runtime = MagicMock()
    mock_runtime.run_financial_team = AsyncMock(return_value={
        "final_answer": "mock",
        "intermediate_steps": [],
        "context": {}
    })

    app.state.conversation_runtime = mock_runtime

    def override_get_conversation_runtime(request: Request):
        return mock_runtime

    # Application des overrides
    app.dependency_overrides[get_deepseek_client] = override_get_deepseek_client
    app.dependency_overrides[validate_path_user_id] = override_validate_user
    app.dependency_overrides[get_user_context] = override_get_user_context
    app.dependency_overrides[rate_limit_dependency] = override_rate_limit
    app.dependency_overrides[get_conversation_runtime] = override_get_conversation_runtime

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app, raise_server_exceptions=False)



@pytest.fixture
def runtime(test_app):
    """Accès au runtime de conversation mocké"""
    return test_app.state.conversation_runtime

# ============================================================================
# TESTS D'AUTHENTIFICATION
# ============================================================================

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
    def test_conversation_success(self, client, runtime):
        """Test conversation réussie avec réponse d'équipe AutoGen."""
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers=get_test_auth_headers(1),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert data["message"] == "Bonjour"
        assert data["team_response"]["final_answer"] == "mock"
        assert len(data["team_response"]["steps"]) == 0
        runtime.run_financial_team.assert_awaited_once_with(
            "Bonjour", {"sub": "1"}
        )

    def test_conversation_success_greeting(self, client):
        """Test conversation réussie avec salutation"""
        
        with patch("conversation_service.agents.financial.intent_classifier.IntentClassifierAgent") as MockAgent:
            # Configuration du mock agent
            mock_result = IntentClassificationResult(
                intent_type=HarenaIntentType.GREETING,
                confidence=0.95,
                reasoning="Salutation détectée",
                original_message="Bonjour",
                category="CONVERSATIONAL",
                is_supported=True,
                alternatives=[],
                processing_time_ms=150
            )
            
            mock_agent_instance = AsyncMock()
            mock_agent_instance.classify_intent = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance
            
            response = client.post(
                "/api/v1/conversation/1",
                json={"message": "Bonjour"},
                headers=get_test_auth_headers(1)
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == 1
            assert data["message"] == "Bonjour"
            assert data["intent"]["intent_type"] == "GREETING"
            assert data["intent"]["confidence"] == 0.95
            assert data["processing_time_ms"] > 0
            assert "agent_metrics" in data

    def test_conversation_success_balance_inquiry(self, client):
        """Test conversation réussie pour demande de solde"""
        
        with patch("conversation_service.agents.financial.intent_classifier.IntentClassifierAgent") as MockAgent:
            mock_result = IntentClassificationResult(
                intent_type=HarenaIntentType.BALANCE_INQUIRY,
                confidence=0.98,
                reasoning="Demande de solde claire",
                original_message="Mon solde",
                category="ACCOUNT_BALANCE",
                is_supported=True,
                alternatives=[],
                processing_time_ms=200
            )
            
            mock_agent_instance = AsyncMock()
            mock_agent_instance.classify_intent = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance
            
            response = client.post(
                "/api/v1/conversation/1",
                json={"message": "Mon solde"},
                headers=get_test_auth_headers(1)
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["intent"]["intent_type"] == "BALANCE_INQUIRY"
            assert data["intent"]["category"] == "ACCOUNT_BALANCE"
            assert data["intent"]["is_supported"] is True

    def test_conversation_runtime_injected(self, client, test_app):
        """Verify that a default runtime is injected when missing"""

        assert getattr(test_app.state, "conversation_runtime", None) is None

        with patch("conversation_service.agents.financial.intent_classifier.IntentClassifierAgent") as MockAgent:
            mock_result = IntentClassificationResult(
                intent_type=HarenaIntentType.GREETING,
                confidence=0.95,
                reasoning="Salutation détectée",
                original_message="Bonjour",
                category="CONVERSATIONAL",
                is_supported=True,
                alternatives=[],
                processing_time_ms=150,
            )

            mock_agent_instance = AsyncMock()
            mock_agent_instance.classify_intent = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance

            response = client.post(
                "/api/v1/conversation/1",
                json={"message": "Bonjour"},
                headers=get_test_auth_headers(1),
            )

            assert response.status_code == 200

        from conversation_service.core.runtime import ConversationServiceRuntime

        assert isinstance(test_app.state.conversation_runtime, ConversationServiceRuntime)

    def test_conversation_unsupported_transfer(self, client):
        """Test avec intention non supportée"""
        
        with patch("conversation_service.agents.financial.intent_classifier.IntentClassifierAgent") as MockAgent:
            mock_result = IntentClassificationResult(
                intent_type=HarenaIntentType.TRANSFER_REQUEST,
                confidence=0.96,
                reasoning="Demande de virement détectée",
                original_message="Faire un virement",
                category="UNSUPPORTED",
                is_supported=False,
                alternatives=[],
                processing_time_ms=180
            )
            
            mock_agent_instance = AsyncMock()
            mock_agent_instance.classify_intent = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance
            
            response = client.post(
                "/api/v1/conversation/1",
                json={"message": "Faire un virement"},
                headers=get_test_auth_headers(1)
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["intent"]["intent_type"] == "TRANSFER_REQUEST"
            assert data["intent"]["is_supported"] is False

        runtime.run_financial_team.return_value = {
            "final_answer": "Bonjour!",
            "intermediate_steps": [
                {"agent": "planner", "content": "analyse"},
                {"agent": "assistant", "content": "réponse"}
            ],
            "context": {"foo": "bar"}
        }

        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers=get_test_auth_headers(1)
        )

        assert response.status_code == 200
        data = response.json()

        assert data["user_id"] == 1
        assert data["message"] == "Bonjour"
        assert data["status"] == "success"
        assert data["phase"] == 2
        assert data["team_response"]["final_answer"] == "Bonjour!"
        assert len(data["team_response"]["steps"]) == 2

# ============================================================================
# TESTS D'AUTHENTIFICATION ET AUTORISATION
# ============================================================================

class TestAuthenticationAndAuthorization:
    """Tests d'authentification et autorisation"""

    def test_conversation_missing_authorization(self, client):
        """Test sans header Authorization"""
        
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
        
        assert response.status_code == 422


class TestConversationRequestModel:
    """Tests pour le modèle ConversationRequest"""

    def test_request_with_optional_fields(self):
        data = {
            "message": "Bonjour",
            "client_info": {"platform": "web", "version": "1.0.0"},
            "message_type": "text",
            "priority": "normal",
        }
        req = ConversationRequest.model_validate(data)
        assert req.message == "Bonjour"
        assert req.client_info == {"platform": "web", "version": "1.0.0"}
        assert req.message_type == "text"
        assert req.priority == "normal"

# ============================================================================
# TESTS ENDPOINTS MONITORING
# ============================================================================

# ============================================================================
# TESTS GESTION D'ERREURS
# ============================================================================

class TestErrorHandling:
    """Tests gestion des erreurs"""
    def test_conversation_agent_error(self, client, runtime):
        """Test lorsque l'équipe AutoGen renvoie une erreur"""

        runtime.run_financial_team.side_effect = Exception("Erreur technique")

        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Test"},
            headers=get_test_auth_headers(1)
        )

        assert response.status_code == 500

        runtime.run_financial_team.side_effect = None

# ============================================================================
# TESTS STRUCTURE RÉPONSE
# ============================================================================

class TestResponseStructure:
    """Tests structure de la réponse"""
    def test_conversation_response_structure(self, client, runtime):
        """Test structure complète de la réponse"""

        runtime.run_financial_team.return_value = {
            "final_answer": "Test",
            "intermediate_steps": [
                {"agent": "planner", "content": "analyse"},
                {"agent": "assistant", "content": "réponse"}
            ],
            "context": {"foo": "bar"}
        }

        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers=get_test_auth_headers(1)
        )

        assert response.status_code == 200
        data = response.json()

        # Vérification structure complète
        required_fields = [
            "user_id", "message", "timestamp", "processing_time_ms",
            "status", "phase", "team_response"
        ]

        for field in required_fields:
            assert field in data, f"Champ manquant: {field}"

        tr = data["team_response"]
        for field in ["final_answer", "steps", "context"]:
            assert field in tr, f"Champ team_response manquant: {field}"

        assert tr["final_answer"] == "Test"
        assert isinstance(tr["steps"], list) and len(tr["steps"]) == 2
        for step in tr["steps"]:
            assert "role" in step and "content" in step

    def test_conversation_performance_metrics(self, client, runtime):
        """Test métriques de performance"""

        runtime.run_financial_team.return_value = {
            "final_answer": "Solde",
            "intermediate_steps": [],
            "context": {}
        }

        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Mon solde"},
            headers=get_test_auth_headers(1)
        )

        assert response.status_code == 200
        data = response.json()

        # Vérification métriques
        assert isinstance(data["processing_time_ms"], int)
        assert data["processing_time_ms"] >= 0

# ============================================================================
# UTILITAIRES DE TEST
# ============================================================================

def assert_requires_authentication(client, method: str, url: str, **kwargs):
    """
    Assertion helper pour vérifier qu'un endpoint nécessite l'authentification
    """
    response = getattr(client, method.lower())(url, **kwargs)
    assert response.status_code == 401, \
        f"Endpoint {method.upper()} {url} devrait nécessiter l'authentification"
    
    error_detail = response.json().get("detail", "")
    assert any(keyword in error_detail.lower() for keyword in 
              ["authorization", "authentification", "auth", "token"]), \
        f"Message d'erreur d'auth inattendu: {error_detail}"

# ============================================================================
# TESTS DEBUG ET UTILITAIRES
# ============================================================================

if __name__ == "__main__":
    # Tests debug en mode standalone
    print("=== Tests JWT Compatibility Debug ===")
    
    # Test génération token
    token = generate_test_jwt(1)
    print(f"Token généré: {token[:50]}...")
    
    # Test validation
    from conversation_service.api.middleware.auth_middleware import JWTValidator
    validator = JWTValidator()
    result = validator.validate_token(token)
    print(f"Validation: {'SUCCESS' if result.success else 'FAILED'}")
    if result.success:
        print(f"User ID: {result.user_id}")
        print(f"Payload: {result.token_payload}")
    else:
        print(f"Erreur: {result.error_message}")
    
    print("=" * 40)
