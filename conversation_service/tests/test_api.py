# conversation_service/tests/test_api.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import json

from ..main import app
from ..models.conversation import FinancialIntent, IntentResult, EntityHints
from ..clients.deepseek_client import DeepSeekResponse

# Client de test
client = TestClient(app)

class TestConversationAPI:
    """Tests pour l'API de conversation"""
    
    def test_root_endpoint(self):
        """Test de l'endpoint racine"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Conversation Service"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test de l'endpoint de santé"""
        with patch('conversation_service.clients.deepseek_client.deepseek_client.health_check') as mock_health:
            mock_health.return_value = asyncio.Future()
            mock_health.return_value.set_result({
                "status": "healthy",
                "response_time": 0.1,
                "timestamp": "2024-01-15T10:30:00Z"
            })
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "degraded"]
            assert data["version"] == "1.0.0"
            assert "dependencies" in data
    
    def test_metrics_endpoint(self):
        """Test de l'endpoint des métriques"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_classifications" in data
        assert "success_rate" in data
        assert "avg_processing_time_ms" in data
        assert "intent_distribution" in data
    
    def test_config_endpoint(self):
        """Test de l'endpoint de configuration"""
        response = client.get("/api/v1/conversation/config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service_name"] == "Conversation Service"
        assert data["version"] == "1.0.0"
        assert "configuration" in data
        assert "min_confidence_threshold" in data["configuration"]

class TestChatEndpoint:
    """Tests pour l'endpoint de chat principal"""
    
    @patch('conversation_service.agents.intent_classifier.intent_classifier.classify_intent')
    def test_chat_success_merchant_intent(self, mock_classify):
        """Test d'une classification réussie - intention marchand"""
        
        # Mock de la réponse de l'agent
        mock_classify.return_value = asyncio.Future()
        mock_classify.return_value.set_result(
            IntentResult(
                intent=FinancialIntent.SEARCH_BY_MERCHANT,
                confidence=0.95,
                entities=EntityHints(merchant="netflix"),
                reasoning="Recherche explicite par nom de marchand Netflix"
            )
        )
        
        # Requête de test
        request_data = {
            "user_id": 34,
            "message": "mes achats netflix",
            "context": {
                "session_id": "test_session",
                "previous_messages": []
            }
        }
        
        response = client.post("/api/v1/conversation/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["intent"] == "search_by_merchant"
        assert data["confidence"] == 0.95
        assert data["entities"]["merchant"] == "netflix"
        assert data["is_clear"] is True
        assert data["clarification_needed"] is None
        assert "conversation_id" in data
        assert "metadata" in data
    
    @patch('conversation_service.agents.intent_classifier.intent_classifier.classify_intent')
    def test_chat_success_category_intent(self, mock_classify):
        """Test d'une classification réussie - intention catégorie"""
        
        mock_classify.return_value = asyncio.Future()
        mock_classify.return_value.set_result(
            IntentResult(
                intent=FinancialIntent.SEARCH_BY_CATEGORY,
                confidence=0.88,
                entities=EntityHints(category="restaurant", period="ce mois"),
                reasoning="Recherche par catégorie avec période"
            )
        )
        
        request_data = {
            "user_id": 34,
            "message": "mes restaurants ce mois"
        }
        
        response = client.post("/api/v1/conversation/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["intent"] == "search_by_category"
        assert data["confidence"] == 0.88
        assert data["entities"]["category"] == "restaurant"
        assert data["entities"]["period"] == "ce mois"
        assert data["is_clear"] is True
    
    @patch('conversation_service.agents.intent_classifier.intent_classifier.classify_intent')
    def test_chat_unclear_intent(self, mock_classify):
        """Test d'une intention peu claire"""
        
        mock_classify.return_value = asyncio.Future()
        mock_classify.return_value.set_result(
            IntentResult(
                intent=FinancialIntent.UNCLEAR_INTENT,
                confidence=0.3,
                entities=EntityHints(),
                reasoning="Confiance insuffisante"
            )
        )
        
        request_data = {
            "user_id": 34,
            "message": "salut"
        }
        
        response = client.post("/api/v1/conversation/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["intent"] == "unclear_intent"
        assert data["confidence"] == 0.3
        assert data["is_clear"] is False
        assert data["clarification_needed"] is not None
    
    def test_chat_invalid_request_empty_message(self):
        """Test avec message vide"""
        
        request_data = {
            "user_id": 34,
            "message": ""
        }
        
        response = client.post("/api/v1/conversation/chat", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_invalid_request_missing_user_id(self):
        """Test avec user_id manquant"""
        
        request_data = {
            "message": "mes restaurants"
        }
        
        response = client.post("/api/v1/conversation/chat", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_invalid_request_negative_user_id(self):
        """Test avec user_id négatif"""
        
        request_data = {
            "user_id": -1,
            "message": "mes restaurants"
        }
        
        response = client.post("/api/v1/conversation/chat", json=request_data)
        
        assert response.status_code == 422  # Validation error

class TestIntentClassifierUnit:
    """Tests unitaires pour l'agent de classification"""
    
    @pytest.mark.asyncio
    @patch('conversation_service.clients.deepseek_client.deepseek_client.chat_completion')
    async def test_classify_intent_success(self, mock_chat):
        """Test de classification réussie"""
        from ..agents.intent_classifier import intent_classifier
        
        # Mock de la réponse DeepSeek
        mock_response = DeepSeekResponse(
            content='{"intent": "search_by_merchant", "confidence": 0.95, "entities": {"merchant": "netflix"}, "reasoning": "Test"}',
            usage={"total_tokens": 100},
            response_time=0.1
        )
        mock_chat.return_value = mock_response
        
        result = await intent_classifier.classify_intent("mes achats netflix")
        
        assert result.intent == FinancialIntent.SEARCH_BY_MERCHANT
        assert result.confidence == 0.95
        assert result.entities.merchant == "netflix"
    
    @pytest.mark.asyncio
    @patch('conversation_service.clients.deepseek_client.deepseek_client.chat_completion')
    async def test_classify_intent_low_confidence(self, mock_chat):
        """Test avec confiance faible"""
        from ..agents.intent_classifier import intent_classifier
        
        mock_response = DeepSeekResponse(
            content='{"intent": "search_general", "confidence": 0.4, "entities": {}, "reasoning": "Ambiguous"}',
            usage={"total_tokens": 50},
            response_time=0.1
        )
        mock_chat.return_value = mock_response
        
        result = await intent_classifier.classify_intent("quelque chose")
        
        assert result.intent == FinancialIntent.UNCLEAR_INTENT
        assert result.confidence == 0.4

class TestDeepSeekClientUnit:
    """Tests unitaires pour le client DeepSeek"""
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_chat_completion_success(self, mock_post):
        """Test d'appel réussi à DeepSeek"""
        from ..clients.deepseek_client import DeepSeekClient
        
        # Mock de la réponse HTTP
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ],
            "usage": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = DeepSeekClient()
        messages = [{"role": "user", "content": "test"}]
        
        result = await client.chat_completion(messages)
        
        assert result.content == "Test response"
        assert result.usage["total_tokens"] == 100
        assert result.response_time > 0
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self):
        """Test du fonctionnement du cache"""
        from ..clients.deepseek_client import DeepSeekClient
        
        client = DeepSeekClient()
        
        # Test génération clé cache
        messages = [{"role": "user", "content": "test"}]
        key1 = client._generate_cache_key(messages, 0.7)
        key2 = client._generate_cache_key(messages, 0.7)
        key3 = client._generate_cache_key(messages, 0.8)
        
        assert key1 == key2  # Même clé pour mêmes paramètres
        assert key1 != key3  # Clé différente pour température différente

class TestEndToEnd:
    """Tests end-to-end"""
    
    @pytest.mark.asyncio
    @patch('conversation_service.clients.deepseek_client.deepseek_client.chat_completion')
    async def test_complete_conversation_flow(self, mock_chat):
        """Test complet du flux de conversation"""
        
        # Mock DeepSeek response
        mock_response = DeepSeekResponse(
            content='{"intent": "search_by_category", "confidence": 0.90, "entities": {"category": "restaurant"}, "reasoning": "Clear category search"}',
            usage={"total_tokens": 120},
            response_time=0.15
        )
        mock_chat.return_value = mock_response
        
        # Test request
        request_data = {
            "user_id": 34,
            "message": "mes restaurants",
            "context": {
                "session_id": "e2e_test",
                "previous_messages": []
            }
        }
        
        response = client.post("/api/v1/conversation/chat", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["intent"] == "search_by_category"
        assert data["confidence"] == 0.90
        assert data["entities"]["category"] == "restaurant"
        assert data["is_clear"] is True
        assert "conversation_id" in data
        assert "metadata" in data
        assert data["metadata"]["agent_used"] == "intent_classifier"
        assert data["metadata"]["processing_time_ms"] > 0

# Collection de cas de test pour Postman
POSTMAN_TEST_CASES = [
    {
        "name": "Recherche par marchand - Netflix",
        "request": {
            "user_id": 34,
            "message": "mes achats netflix"
        },
        "expected_intent": "search_by_merchant",
        "expected_confidence_min": 0.8
    },
    {
        "name": "Recherche par catégorie - Restaurants",
        "request": {
            "user_id": 34,
            "message": "mes restaurants ce mois"
        },
        "expected_intent": "search_by_category",
        "expected_confidence_min": 0.8
    },
    {
        "name": "Recherche par montant",
        "request": {
            "user_id": 34,
            "message": "plus de 100 euros"
        },
        "expected_intent": "search_by_amount",
        "expected_confidence_min": 0.8
    },
    {
        "name": "Analyse des dépenses",
        "request": {
            "user_id": 34,
            "message": "combien j'ai dépensé en janvier"
        },
        "expected_intent": "spending_analysis",
        "expected_confidence_min": 0.8
    },
    {
        "name": "Intention peu claire",
        "request": {
            "user_id": 34,
            "message": "salut"
        },
        "expected_intent": "unclear_intent",
        "expected_confidence_max": 0.5
    }
]

def export_postman_collection():
    """Génère une collection Postman pour les tests"""
    
    collection = {
        "info": {
            "name": "Conversation Service Tests",
            "description": "Tests pour le service de classification d'intentions",
            "version": "1.0.0"
        },
        "variable": [
            {
                "key": "baseUrl",
                "value": "http://localhost:8001"
            }
        ],
        "item": []
    }
    
    for test_case in POSTMAN_TEST_CASES:
        item = {
            "name": test_case["name"],
            "request": {
                "method": "POST",
                "header": [
                    {
                        "key": "Content-Type",
                        "value": "application/json"
                    }
                ],
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(test_case["request"], indent=2)
                },
                "url": {
                    "raw": "{{baseUrl}}/api/v1/conversation/chat",
                    "host": ["{{baseUrl}}"],
                    "path": ["api", "v1", "conversation", "chat"]
                }
            },
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "pm.test('Status code is 200', function () {",
                            "    pm.response.to.have.status(200);",
                            "});",
                            "",
                            "pm.test('Response has required fields', function () {",
                            "    const jsonData = pm.response.json();",
                            "    pm.expect(jsonData).to.have.property('intent');",
                            "    pm.expect(jsonData).to.have.property('confidence');",
                            "    pm.expect(jsonData).to.have.property('entities');",
                            "    pm.expect(jsonData).to.have.property('metadata');",
                            "});",
                            "",
                            f"pm.test('Intent is {test_case['expected_intent']}', function () {{",
                            "    const jsonData = pm.response.json();",
                            f"    pm.expect(jsonData.intent).to.eql('{test_case['expected_intent']}');",
                            "});",
                        ]
                    }
                }
            ]
        }
        
        collection["item"].append(item)
    
    return collection

if __name__ == "__main__":
    # Export de la collection Postman
    collection = export_postman_collection()
    with open("conversation_service_postman.json", "w") as f:
        json.dump(collection, f, indent=2)
    print("Collection Postman exportée vers: conversation_service_postman.json")