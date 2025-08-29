"""
Tests complets pour l'endpoint conversation - Compatible user_service JWT
"""
import os
import sys
import pytest
from jose import jwt
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
import json
import time

# Configuration environment pour tests AVANT les imports
os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key-12345"
os.environ["SECRET_KEY"] = "a" * 32 + "b" * 32  # Même clé que conftest.py
os.environ["CONVERSATION_SERVICE_ENABLED"] = "true"
os.environ["ENVIRONMENT"] = "testing"

import importlib
from pathlib import Path

# Import dynamique avec gestion d'erreur
try:
    from fastapi.testclient import TestClient
    from conversation_service.prompts.harena_intents import HarenaIntentType
    from conversation_service.models.responses.conversation_responses import (
        ConversationResponse, IntentClassificationResult, AgentMetrics
    )
    from conversation_service.models.requests.conversation_requests import ConversationRequest
except ImportError as e:
    # Fallback si imports directs ne fonctionnent pas
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from conversation_service.prompts.harena_intents import HarenaIntentType
    from fastapi.testclient import TestClient

# ============================================================================
# GENERATION JWT COMPATIBLE user_service
# ============================================================================

def generate_test_jwt(sub: int = 1, expired: bool = False) -> str:
    """
    Génère un JWT compatible avec user_service.core.security.create_access_token
    
    Args:
        sub: Subject (user ID)
        expired: Si True, génère un token expiré
    
    Returns:
        str: Token JWT signé avec la même logique que user_service
    """
    # Logique temporelle identique à user_service.core.security.create_access_token
    if expired:
        expire = datetime.utcnow() + timedelta(hours=-1)  # Expiré depuis 1h
    else:
        expire = datetime.utcnow() + timedelta(minutes=60 * 24)  # 24h comme user_service
    
    # Payload compatible avec user_service.core.security.create_access_token
    payload = {
        "exp": expire,
        "sub": str(sub),  # user_service utilise str(subject)
        "permissions": ["chat:write"],  # Même format que user_service
    }
    
    # Utilise la même SECRET_KEY que les settings
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        raise ValueError("SECRET_KEY manquante dans l'environnement de test")
    
    return jwt.encode(payload, secret_key, algorithm="HS256")

def get_test_auth_headers(user_id: int = 1) -> dict:
    """
    Génère les headers d'authentification pour les tests
    
    Args:
        user_id: ID de l'utilisateur pour le token
        
    Returns:
        dict: Headers avec Authorization Bearer
    """
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
                "phase": 1,
                "version": "1.0.0",
                "features": ["intent_classification", "json_output"]
            }
            return True
        return False

    async def cleanup(self):
        pass

# ============================================================================
# CONFIGURATION APP DE TEST
# ============================================================================

def create_test_app():
    """Crée une app FastAPI pour les tests avec configuration complète"""
    from fastapi import FastAPI
    from conversation_service.api.routes.conversation import router as conversation_router
    from conversation_service.api.middleware.auth_middleware import JWTAuthMiddleware
    
    app = FastAPI(title="Test Conversation Service")
    
    # Middleware auth avec la configuration des tests
    app.add_middleware(JWTAuthMiddleware)
    
    # Routes
    app.include_router(conversation_router, prefix="/api/v1")
    
    return app

# ============================================================================
# FIXTURES PYTEST
# ============================================================================

@pytest.fixture
def mock_service_loader():
    """Service loader mocké"""
    return MockConversationServiceLoader()

@pytest.fixture
def test_app(mock_service_loader):
    """App FastAPI de test configurée avec dépendances mockées"""
    app = create_test_app()

    # Import des dépendances après la création de l'app
    from conversation_service.api.dependencies import (
        get_deepseek_client,
        get_cache_manager,
        get_conversation_service_status,
        validate_path_user_id,
        get_user_context,
        rate_limit_dependency,
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
        # Simule l'injection de user_id par le middleware auth
        request.state.user_id = token_user_id
        await verify_user_id_match(request, path_user_id)
        return path_user_id

    def override_get_user_context(request: Request, user_id: int = 1):
        return {"sub": str(getattr(request.state, "user_id", user_id))}

    def override_rate_limit(request: Request, user_id: int = 1):
        return None

    # Application des overrides
    app.dependency_overrides[get_deepseek_client] = override_get_deepseek_client
    app.dependency_overrides[get_cache_manager] = override_get_cache_manager
    app.dependency_overrides[get_conversation_service_status] = override_get_service_status
    app.dependency_overrides[validate_path_user_id] = override_validate_user
    app.dependency_overrides[get_user_context] = override_get_user_context
    app.dependency_overrides[rate_limit_dependency] = override_rate_limit

    yield app

@pytest.fixture
def client(test_app):
    """Client de test FastAPI"""
    return TestClient(test_app, raise_server_exceptions=False)

# ============================================================================
# TESTS D'AUTHENTIFICATION
# ============================================================================

class TestJWTCompatibility:
    """Tests de compatibilité JWT avec user_service"""

    def test_jwt_token_compatibility(self):
        """Test de compatibilité des tokens JWT entre services"""
        from conversation_service.api.middleware.auth_middleware import JWTValidator
        
        # Génère un token avec la même logique que user_service
        token = generate_test_jwt(sub=1)
        
        # Valide avec le middleware conversation_service
        validator = JWTValidator()
        result = validator.validate_token(token)
        
        assert result.success, f"Token invalide: {result.error_message}"
        assert result.user_id == 1, f"User ID incorrect: {result.user_id}"
        assert result.token_payload is not None, "Payload manquant"
        
        # Vérifications du payload
        payload = result.token_payload
        assert payload.get("sub") == "1", f"Subject incorrect: {payload.get('sub')}"
        assert "permissions" in payload, "Permissions manquantes"
        assert "chat:write" in payload["permissions"], "Permission chat:write manquante"

    def test_generate_test_jwt_format(self):
        """Test format du JWT généré"""
        token = generate_test_jwt(sub=42)
        
        # Vérification format JWT
        parts = token.split('.')
        assert len(parts) == 3, "JWT doit avoir 3 parties séparées par des points"
        
        # Décodage pour vérifier le contenu
        payload = jwt.decode(token, os.environ["SECRET_KEY"], algorithms=["HS256"])
        
        assert payload["sub"] == "42", "Subject doit être string"
        assert "exp" in payload, "Expiration manquante"
        assert "permissions" in payload, "Permissions manquantes"

    def test_expired_token_generation(self):
        """Test génération token expiré"""
        expired_token = generate_test_jwt(sub=1, expired=True)
        
        from conversation_service.api.middleware.auth_middleware import JWTValidator
        validator = JWTValidator()
        result = validator.validate_token(expired_token)
        
        assert not result.success, "Token expiré devrait être refusé"
        assert result.error_code == "TOKEN_EXPIRED", f"Code d'erreur incorrect: {result.error_code}"

# ============================================================================
# TESTS ENDPOINTS PRINCIPAUX
# ============================================================================

class TestConversationEndpoint:
    """Tests complets pour l'endpoint de conversation"""

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

    def test_conversation_with_alternatives(self, client):
        """Test avec alternatives d'intention"""
        
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
                headers=get_test_auth_headers(1)
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["intent"]["alternatives"]) == 1
            assert data["intent"]["alternatives"][0]["intent_type"] == "SEARCH_BY_CATEGORY"

# ============================================================================
# TESTS D'AUTHENTIFICATION ET AUTORISATION
# ============================================================================

class TestAuthenticationAndAuthorization:
    """Tests d'authentification et autorisation"""

    def test_conversation_missing_authorization(self, client):
        """Test sans header Authorization"""
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"}
        )
        
        assert response.status_code == 401
        detail = response.json().get("detail", "")
        assert any(keyword in detail.lower() for keyword in ["authorization", "authentification", "auth"]), \
            f"Message d'erreur inattendu: {detail}"

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
        
        # Le middleware peut échouer avant la vérification user_id mismatch
        # donc on accepte soit 401 (token validation) soit 403 (user mismatch)
        assert response.status_code in [401, 403], \
            f"Status code inattendu: {response.status_code}"
        
        # Si c'est 403, vérifier que c'est bien pour user mismatch
        if response.status_code == 403:
            error_detail = response.json().get("detail", "")
            assert "correspond" in str(error_detail).lower() or "mismatch" in str(error_detail).lower()

    def test_conversation_invalid_bearer_scheme(self, client):
        """Test avec schéma d'authentification invalide"""
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": "Bonjour"},
            headers={"Authorization": "Basic invalid-scheme"}
        )
        
        assert response.status_code == 401

# ============================================================================
# TESTS VALIDATION DONNÉES
# ============================================================================

class TestDataValidation:
    """Tests validation des données d'entrée"""

    def test_conversation_empty_message(self, client):
        """Test avec message vide"""
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": ""},
            headers=get_test_auth_headers(1)
        )

        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "message"]
        assert "vide" in error["msg"]

    def test_conversation_message_too_long(self, client):
        """Test avec message trop long"""
        
        long_message = "A" * 1001  # Dépasse la limite de 1000
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": long_message},
            headers=get_test_auth_headers(1)
        )

        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "message"]
        assert "1000 caractères" in error["msg"]

    def test_conversation_malicious_content(self, client):
        """Test avec contenu potentiellement malveillant"""
        
        malicious_message = "<script>alert('xss')</script>"
        
        response = client.post(
            "/api/v1/conversation/1",
            json={"message": malicious_message},
            headers=get_test_auth_headers(1)
        )

        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "message"]
        assert "malveillant" in error["msg"]

    def test_conversation_request_validation(self, client):
        """Test validation du modèle de requête"""
        
        # Test avec JSON invalide
        response = client.post(
            "/api/v1/conversation/1",
            data="invalid json",
            headers={**get_test_auth_headers(1), "Content-Type": "application/json"}
        )
        
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

class TestMonitoringEndpoints:
    """Tests pour les endpoints de monitoring (doivent être publics)"""

    def test_conversation_health_success(self, client):
        """Test health check réussi (sans authentification)"""
        
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
            assert data["features"] == {
                "intent_classification": True,
                "entity_extraction": True,
            }

    def test_conversation_metrics_success(self, client):
        """Test récupération métriques réussie (sans authentification)"""
        
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

    def test_conversation_status_public(self, client):
        """Test endpoint status accessible publiquement"""
        
        response = client.get("/api/v1/conversation/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "ready" in data

# ============================================================================
# TESTS GESTION D'ERREURS
# ============================================================================

class TestErrorHandling:
    """Tests gestion des erreurs"""

    def test_conversation_agent_error(self, client):
        """Test avec erreur de l'agent de classification"""
        
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
                headers=get_test_auth_headers(1)
            )
            
            assert response.status_code == 500

# ============================================================================
# TESTS STRUCTURE RÉPONSE
# ============================================================================

class TestResponseStructure:
    """Tests structure de la réponse"""

    def test_conversation_response_structure(self, client):
        """Test structure complète de la réponse"""
        
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
                headers=get_test_auth_headers(1)
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérification structure complète
            required_fields = [
                "user_id", "message", "timestamp", "processing_time_ms",
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
                headers=get_test_auth_headers(1)
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérification métriques
            assert isinstance(data["processing_time_ms"], int)
            assert data["processing_time_ms"] > 0
            assert isinstance(data["agent_metrics"]["tokens_consumed"], int)
            assert data["agent_metrics"]["tokens_consumed"] > 0

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