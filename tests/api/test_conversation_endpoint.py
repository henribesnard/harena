"""
Tests complets pour l'endpoint principal de conversation avec mocking avancé
"""
import os
import sys
import pytest
from jose import jwt
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
import json

# Configuration environment pour tests
os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key-12345"
os.environ["SECRET_KEY"] = "super-secret-test-key-with-more-than-32-chars-123456789"
os.environ["CONVERSATION_SERVICE_ENABLED"] = "true"
os.environ["ENVIRONMENT"] = "testing"

import importlib
from pathlib import Path

# Import dynamique de l'app principale
try:
    from fastapi.testclient import TestClient
    from conversation_service.prompts.harena_intents import HarenaIntentType
    from conversation_service.models.responses.conversation_responses import (
        ConversationResponse, IntentClassificationResult, AgentMetrics
    )
    from conversation_service.models.requests.conversation_requests import ConversationRequest
except ImportError:
    # Fallback si imports directs ne fonctionnent pas
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from conversation_service.prompts.harena_intents import HarenaIntentType
    from fastapi.testclient import TestClient


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
                "phase": 1,
                "version": "1.0.0",
                "features": ["intent_classification", "json_output"]
            }
            return True
        return False

    async def cleanup(self):
        pass


# Configuration de l'app de test
def create_test_app():
    """Crée une app FastAPI pour les tests"""
    from fastapi import FastAPI
    from conversation_service.api.routes.conversation import router as conversation_router
    from conversation_service.api.middleware.auth_middleware import JWTAuthMiddleware
    
    app = FastAPI(title="Test Conversation Service")
    
    # Middleware auth (simplifié pour tests)
    # app.add_middleware(JWTAuthMiddleware)  # Désactivé pour simplifier
    
    # Routes
    app.include_router(conversation_router, prefix="/api/v1")
    
    return app


@pytest.fixture
def mock_service_loader():
    """Service loader mocké"""
    return MockConversationServiceLoader()


@pytest.fixture
def test_app(mock_service_loader):
    """App FastAPI de test configurée"""
    app = create_test_app()

    from conversation_service.api.dependencies import (
        get_deepseek_client,
        get_cache_manager,
        get_conversation_service_status,
        validate_path_user_id,
        get_user_context,
        rate_limit_dependency,
    )

    from fastapi import Request

    def override_get_deepseek_client(request: Request):
        return mock_service_loader.deepseek_client

    def override_get_cache_manager(request: Request):
        return mock_service_loader.cache_manager

    def override_get_service_status(request: Request):
        return {"status": "healthy"}

    def override_validate_user(
        request: Request, path_user_id: int, token_user_id: int = 1
    ) -> int:
        return 1

    def override_get_user_context(request: Request, user_id: int = 1):
        return {"sub": 1}

    def override_rate_limit(request: Request, user_id: int = 1):
        return None

    app.dependency_overrides[get_deepseek_client] = override_get_deepseek_client
    app.dependency_overrides[get_cache_manager] = override_get_cache_manager
    app.dependency_overrides[get_conversation_service_status] = (
        override_get_service_status
    )
    app.dependency_overrides[validate_path_user_id] = override_validate_user
    app.dependency_overrides[get_user_context] = override_get_user_context
    app.dependency_overrides[rate_limit_dependency] = override_rate_limit

    yield app


@pytest.fixture
def client(test_app):
    """Client de test FastAPI"""
    return TestClient(test_app, raise_server_exceptions=False)


def generate_test_jwt(sub: int = 1, expired: bool = False) -> str:
    """Génère un JWT pour les tests"""
    import time

    payload = {
        "sub": sub,
        "iat": int(time.time()) - (3600 if expired else 0),
        "exp": int(time.time()) + (3600 if not expired else -3600),
    }

    return jwt.encode(payload, os.environ["SECRET_KEY"], algorithm="HS256")


class TestConversationEndpoint:
    """Tests complets pour l'endpoint de conversation"""

    def test_conversation_success_greeting(self, client):
        """Test conversation réussie avec salutation"""
        
        token = generate_test_jwt(1)
        
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
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["sub"] == 1
            assert data["message"] == "Bonjour"
            assert data["intent"]["intent_type"] == "GREETING"
            assert data["intent"]["confidence"] == 0.95
            assert data["processing_time_ms"] > 0
            assert "agent_metrics" in data

    def test_conversation_success_balance_inquiry(self, client):
        """Test conversation réussie pour demande de solde"""
        
        token = generate_test_jwt(1)
        
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
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["intent"]["intent_type"] == "BALANCE_INQUIRY"
            assert data["intent"]["category"] == "ACCOUNT_BALANCE"
            assert data["intent"]["is_supported"] is True

    def test_conversation_unsupported_transfer(self, client):
        """Test avec intention non supportée"""
        
        token = generate_test_jwt(1)
        
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
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["intent"]["intent_type"] == "TRANSFER_REQUEST"
            assert data["intent"]["is_supported"] is False

    def test_conversation_missing_authorization(self, client):
        """Test sans header Authorization"""
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"}
        )
        
        assert response.status_code == 401
        assert "Authorization" in response.json()["detail"]

    def test_conversation_invalid_token(self, client):
        """Test avec token JWT invalide"""
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401

    def test_conversation_expired_token(self, client):
        """Test avec token expiré"""
        
        expired_token = generate_test_jwt(1, expired=True)
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        
        assert response.status_code == 401

    def test_conversation_sub_mismatch(self, client):
        """Test avec sub ne correspondant pas au token"""
        
        token = generate_test_jwt(1)  # Token pour user 1
        
        response = client.post(
            "/api/v1/conversation/2",  # Requête pour user 2
            json={"message": "Bonjour"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 403

    def test_conversation_empty_message(self, client):
        """Test avec message vide"""
        
        token = generate_test_jwt(1)
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": ""},
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "message"]
        assert "vide" in error["msg"]

    def test_conversation_message_too_long(self, client):
        """Test avec message trop long"""
        
        token = generate_test_jwt(1)
        long_message = "A" * 1001  # Dépasse la limite de 1000
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": long_message},
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "message"]
        assert "1000 caractères" in error["msg"]

    def test_conversation_malicious_content(self, client):
        """Test avec contenu potentiellement malveillant"""
        
        token = generate_test_jwt(1)
        malicious_message = "<script>alert('xss')</script>"
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": malicious_message},
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "message"]
        assert "malveillant" in error["msg"]

    def test_conversation_agent_error(self, client):
        """Test avec erreur de l'agent de classification"""
        
        token = generate_test_jwt(1)
        
        with patch("conversation_service.agents.financial.intent_classifier.IntentClassifierAgent") as MockAgent:
            mock_result = IntentClassificationResult(
                intent_type=HarenaIntentType.ERROR,
                confidence=0.99,
                reasoning="Erreur technique",
                original_message="Test",
                category="UNCLEAR_INTENT",
                is_supported=False,
                alternatives=[],
                processing_time_ms=100
            )
            
            mock_agent_instance = AsyncMock()
            mock_agent_instance.classify_intent = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance
            
            response = client.post(
                "/api/v1/conversation/1",
                json={"message": "Test"},
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 500

    def test_conversation_with_alternatives(self, client):
        """Test avec alternatives d'intention"""
        
        token = generate_test_jwt(1)
        
        with patch("conversation_service.agents.financial.intent_classifier.IntentClassifierAgent") as MockAgent:
            from conversation_service.models.responses.conversation_responses import IntentAlternative
            
            alternatives = [
                IntentAlternative(
                    intent_type=HarenaIntentType.SEARCH_BY_CATEGORY,
                    confidence=0.65,
                    reasoning="Alternative possible"
                )
            ]
            
            mock_result = IntentClassificationResult(
                intent_type=HarenaIntentType.SPENDING_ANALYSIS,
                confidence=0.75,
                reasoning="Analyse demandée",
                original_message="Mes dépenses restaurants",
                category="SPENDING_ANALYSIS",
                is_supported=True,
                alternatives=alternatives,
                processing_time_ms=220
            )
            
            mock_agent_instance = AsyncMock()
            mock_agent_instance.classify_intent = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance
            
            response = client.post(
                "/api/v1/conversation/1",
                json={"message": "Mes dépenses restaurants"},
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["intent"]["alternatives"]) == 1
            assert data["intent"]["alternatives"][0]["intent_type"] == "SEARCH_BY_CATEGORY"

    def test_conversation_request_validation(self, client):
        """Test validation du modèle de requête"""
        
        token = generate_test_jwt(1)
        
        # Test avec JSON invalide
        response = client.post(
            "/api/v1/conversation/1",
            data="invalid json",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        
        assert response.status_code == 422

    def test_conversation_response_structure(self, client):
        """Test structure complète de la réponse"""
        
        token = generate_test_jwt(1)
        
        with patch("conversation_service.agents.financial.intent_classifier.IntentClassifierAgent") as MockAgent:
            mock_result = IntentClassificationResult(
                intent_type=HarenaIntentType.GREETING,
                confidence=0.95,
                reasoning="Test",
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
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérification structure complète
            required_fields = [
                "sub", "message", "timestamp", "processing_time_ms",
                "intent", "agent_metrics", "phase"
            ]
            
            for field in required_fields:
                assert field in data, f"Champ manquant: {field}"
            
            # Vérification structure intent
            intent_fields = [
                "intent_type", "confidence", "reasoning", "original_message",
                "category", "is_supported", "alternatives"
            ]
            
            for field in intent_fields:
                assert field in data["intent"], f"Champ intent manquant: {field}"
            
            # Vérification structure agent_metrics
            metrics_fields = [
                "agent_used", "model_used", "tokens_consumed",
                "processing_time_ms", "confidence_threshold_met", "cache_hit"
            ]
            
            for field in metrics_fields:
                assert field in data["agent_metrics"], f"Champ metrics manquant: {field}"

    def test_conversation_performance_metrics(self, client):
        """Test métriques de performance"""
        
        token = generate_test_jwt(1)
        
        with patch("conversation_service.agents.financial.intent_classifier.IntentClassifierAgent") as MockAgent:
            mock_result = IntentClassificationResult(
                intent_type=HarenaIntentType.BALANCE_INQUIRY,
                confidence=0.9,
                reasoning="Test",
                original_message="Mon solde",
                category="ACCOUNT_BALANCE",
                is_supported=True,
                alternatives=[],
                processing_time_ms=150
            )
            
            mock_agent_instance = AsyncMock()
            mock_agent_instance.classify_intent = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance
            
            response = client.post(
                "/api/v1/conversation/1",
                json={"message": "Mon solde"},
                headers={"Authorization": f"Bearer {token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérification métriques
            assert isinstance(data["processing_time_ms"], int)
            assert data["processing_time_ms"] > 0
            assert isinstance(data["agent_metrics"]["tokens_consumed"], int)
            assert data["agent_metrics"]["tokens_consumed"] > 0


class TestConversationHealthEndpoint:
    """Tests pour l'endpoint health de conversation"""

    def test_conversation_health_success(self, client):
        """Test health check réussi"""
        
        with patch("conversation_service.api.routes.conversation.metrics_collector") as mock_metrics:
            mock_metrics.get_health_metrics.return_value = {
                "status": "healthy",
                "total_requests": 100,
                "error_rate_percent": 2.5,
                "latency_p95_ms": 250,
                "uptime_seconds": 3600
            }
            
            response = client.get("/api/v1/conversation/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["service"] == "conversation_service"
            assert data["status"] == "healthy"
            assert "health_details" in data
            assert "features" in data

    def test_conversation_health_error(self, client):
        """Test health check avec erreur"""
        
        with patch("conversation_service.api.routes.conversation.metrics_collector") as mock_metrics:
            mock_metrics.get_health_metrics.side_effect = Exception("Metrics error")
            
            response = client.get("/api/v1/conversation/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["service"] == "conversation_service"
            assert data["status"] == "error"


class TestConversationMetricsEndpoint:
    """Tests pour l'endpoint metrics"""

    def test_conversation_metrics_success(self, client):
        """Test récupération métriques réussie"""
        
        with patch("conversation_service.api.routes.conversation.metrics_collector") as mock_metrics:
            mock_metrics.get_all_metrics.return_value = {
                "timestamp": "2024-01-01T00:00:00Z",
                "counters": {"conversation.requests.total": 100},
                "histograms": {"conversation.processing_time": {"avg": 200}},
                "rates": {"conversation.requests_per_second": 1.5}
            }
            
            response = client.get("/api/v1/conversation/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "metrics" in data
            assert "service_info" in data
            assert "performance_summary" in data

    def test_conversation_metrics_error(self, client):
        """Test erreur récupération métriques"""
        
        with patch("conversation_service.api.routes.conversation.metrics_collector") as mock_metrics:
            mock_metrics.get_all_metrics.side_effect = Exception("Metrics error")

            response = client.get("/api/v1/conversation/metrics")

            assert response.status_code == 500

