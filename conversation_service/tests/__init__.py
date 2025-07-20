"""
Tests module pour le Conversation Service

Ce module contient tous les tests unitaires et d'intégration :
- Tests API FastAPI
- Tests agents et clients
- Tests de performance
- Collection Postman générée
- Mocks et fixtures
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

# Test configuration
TEST_USER_ID = 34
TEST_SESSION_ID = "test_session_123"

# Test data fixtures
SAMPLE_MESSAGES = {
    "merchant": "mes achats netflix",
    "category": "mes restaurants ce mois", 
    "amount": "plus de 100 euros",
    "date": "mes transactions en janvier",
    "analysis": "combien j'ai dépensé",
    "unclear": "salut"
}

EXPECTED_INTENTS = {
    "merchant": "search_by_merchant",
    "category": "search_by_category",
    "amount": "search_by_amount", 
    "date": "search_by_date",
    "analysis": "spending_analysis",
    "unclear": "unclear_intent"
}

__version__ = "1.0.0"
__all__ = [
    "TEST_USER_ID",
    "TEST_SESSION_ID", 
    "SAMPLE_MESSAGES",
    "EXPECTED_INTENTS",
    "create_test_request",
    "create_mock_deepseek_response",
    "create_mock_intent_result",
    "get_test_cases_postman"
]

def create_test_request(message: str, user_id: int = TEST_USER_ID) -> Dict[str, Any]:
    """Crée une requête de test standardisée"""
    return {
        "user_id": user_id,
        "message": message,
        "context": {
            "session_id": TEST_SESSION_ID,
            "previous_messages": []
        }
    }

def create_mock_deepseek_response(intent: str, confidence: float, entities: Dict[str, str] = None) -> MagicMock:
    """Crée une réponse mock de DeepSeek"""
    from ..clients.deepseek_client import DeepSeekResponse
    import json
    
    response_data = {
        "intent": intent,
        "confidence": confidence,
        "entities": entities or {},
        "reasoning": f"Mock response for {intent}"
    }
    
    return DeepSeekResponse(
        content=json.dumps(response_data),
        usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        response_time=0.1
    )

def create_mock_intent_result(intent: str, confidence: float, entities: Dict[str, str] = None):
    """Crée un résultat d'intention mock"""
    from ..models.conversation_models import FinancialIntent, IntentResult, EntityHints
    
    return IntentResult(
        intent=FinancialIntent(intent),
        confidence=confidence,
        entities=EntityHints(**(entities or {})),
        reasoning=f"Mock classification for {intent}"
    )

def get_test_cases_postman() -> List[Dict[str, Any]]:
    """Retourne les cas de test pour Postman"""
    test_cases = []
    
    for message_type, message in SAMPLE_MESSAGES.items():
        expected_intent = EXPECTED_INTENTS[message_type]
        
        test_case = {
            "name": f"Test {message_type.title()} Intent",
            "request": create_test_request(message),
            "expected_intent": expected_intent,
            "expected_confidence_min": 0.8 if message_type != "unclear" else 0.0,
            "expected_confidence_max": 1.0 if message_type != "unclear" else 0.5
        }
        
        test_cases.append(test_case)
    
    return test_cases

# Fixtures pytest communes
@pytest.fixture
def test_user_id():
    """Fixture pour un ID utilisateur de test"""
    return TEST_USER_ID

@pytest.fixture  
def test_session_id():
    """Fixture pour un ID de session de test"""
    return TEST_SESSION_ID

@pytest.fixture
def sample_chat_request():
    """Fixture pour une requête de chat de test"""
    return create_test_request(SAMPLE_MESSAGES["category"])

@pytest.fixture
def mock_deepseek_client():
    """Fixture pour un client DeepSeek mocké"""
    mock_client = AsyncMock()
    mock_client.chat_completion.return_value = create_mock_deepseek_response(
        "search_by_category", 
        0.9,
        {"category": "restaurant", "period": "ce mois"}
    )
    mock_client.health_check.return_value = {
        "status": "healthy",
        "response_time": 0.1
    }
    mock_client.get_metrics.return_value = {
        "total_requests": 100,
        "successful_requests": 95,
        "cache_hit_rate": 0.3
    }
    return mock_client

@pytest.fixture
def mock_intent_classifier():
    """Fixture pour un agent de classification mocké"""
    mock_agent = AsyncMock()
    mock_agent.classify_intent.return_value = create_mock_intent_result(
        "search_by_category",
        0.9,
        {"category": "restaurant", "period": "ce mois"}
    )
    mock_agent.get_metrics.return_value = {
        "total_classifications": 50,
        "success_rate": 0.96,
        "avg_confidence": 0.87
    }
    return mock_agent

# Constantes pour les tests
TEST_API_BASE_URL = "http://testserver"
TEST_ENDPOINTS = {
    "chat": "/api/v1/conversation/chat",
    "health": "/health", 
    "metrics": "/metrics",
    "config": "/api/v1/conversation/config"
}

# Helpers pour les assertions
def assert_valid_chat_response(response_data: Dict[str, Any]):
    """Vérifie qu'une réponse de chat est valide"""
    required_fields = [
        "conversation_id", "response", "intent", "confidence", 
        "entities", "is_clear", "metadata", "timestamp"
    ]
    
    for field in required_fields:
        assert field in response_data, f"Champ manquant: {field}"
    
    assert 0.0 <= response_data["confidence"] <= 1.0, "Confidence invalide"
    assert response_data["intent"] in EXPECTED_INTENTS.values(), "Intent invalide"

def assert_valid_health_response(response_data: Dict[str, Any]):
    """Vérifie qu'une réponse de santé est valide"""
    required_fields = ["status", "timestamp", "version", "dependencies"]
    
    for field in required_fields:
        assert field in response_data, f"Champ manquant: {field}"
    
    assert response_data["status"] in ["healthy", "degraded", "unhealthy"], "Statut invalide"

def assert_valid_metrics_response(response_data: Dict[str, Any]):
    """Vérifie qu'une réponse de métriques est valide"""
    required_fields = [
        "total_classifications", "success_rate", "avg_processing_time_ms",
        "avg_confidence", "intent_distribution", "cache_hit_rate"
    ]
    
    for field in required_fields:
        assert field in response_data, f"Champ manquant: {field}"
    
    assert 0.0 <= response_data["success_rate"] <= 1.0, "Success rate invalide"
    assert 0.0 <= response_data["cache_hit_rate"] <= 1.0, "Cache hit rate invalide"