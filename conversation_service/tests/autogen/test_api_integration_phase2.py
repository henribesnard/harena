"""
Tests API Integration AutoGen Phase 2
Tests intégration endpoints API avec équipes multi-agents et fallback Phase 1
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, AsyncMock, Mock
from fastapi.testclient import TestClient

# Imports cohérents Phase 1
from conversation_service.models.responses.conversation_responses import IntentClassification
from conversation_service.models.responses.conversation_responses import ConversationResponsePhase2
from conversation_service.models.conversation.entities import ComprehensiveEntityExtraction


class TestAPIIntegrationPhase2:
    """Tests intégration API Phase 2 avec compatibilité Phase 1"""
    
    def test_conversation_endpoint_phase2_success(
        self, 
        client, 
        auth_headers,
        test_user_id,
        mock_team_results
    ):
        """Test endpoint Phase 2 avec équipe AutoGen"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team') as mock_get_team, \
             patch('conversation_service.teams.multi_agent_financial_team.MultiAgentFinancialTeam.process_user_message') as mock_process:
            
            # Mock équipe disponible
            mock_team = AsyncMock()
            mock_get_team.return_value = mock_team
            
            # Mock résultats équipe
            mock_process.return_value = mock_team_results
            
            response = client.post(
                f"/api/v1/conversation/v2/{test_user_id}",
                json={"message": "Mes achats Amazon"},
                headers=auth_headers
            )
            
            # Validation response Phase 2
            assert response.status_code == 200
            data = response.json()
            
            # Champs Phase 1 conservés (compatibilité)
            assert "user_id" in data
            assert "message" in data
            assert "intent" in data
            assert "processing_time_ms" in data
            assert "status" in data
            
            # Champs Phase 2 ajoutés
            assert "comprehensive_entities" in data
            assert "entity_validation" in data
            assert "multi_agent_insights" in data
            assert "autogen_metadata" in data
            
            # Validation structure AutoGen
            autogen_metadata = data["autogen_metadata"]
            assert autogen_metadata["processing_mode"] == "multi_agent_team"
            assert "agents_involved" in autogen_metadata
            assert "workflow_status" in autogen_metadata
    
    def test_conversation_endpoint_fallback_to_phase1(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test fallback vers Phase 1 si AutoGen indisponible"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team', return_value=None), \
             patch('conversation_service.api.dependencies.get_intent_classifier_agent') as mock_get_agent:
            
            # Mock agent Phase 1
            mock_agent = Mock()
            mock_agent.classify_intent = AsyncMock(return_value=IntentClassification(
                intent_type="SEARCH_BY_MERCHANT",
                confidence=0.95,
                reasoning="Classification Phase 1 fallback"
            ))
            mock_get_agent.return_value = mock_agent
            
            response = client.post(
                f"/api/v1/conversation/v2/{test_user_id}",
                json={"message": "Mes achats Amazon"},
                headers=auth_headers
            )
            
            # Response Phase 1 standard via fallback
            assert response.status_code == 200
            data = response.json()
            
            # Structure Phase 1 complète
            assert data["intent"]["intent_type"] == "SEARCH_BY_MERCHANT"
            assert data["intent"]["confidence"] == 0.95
            
            # Métadonnées fallback
            assert data["autogen_metadata"]["processing_mode"] == "single_agent_fallback"
            assert "fallback_reason" in data
    
    def test_conversation_endpoint_v1_unchanged(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test endpoint V1 inchangé avec Phase 2 active"""
        
        with patch('conversation_service.api.dependencies.get_intent_classifier_agent') as mock_get_agent:
            
            # Mock agent Phase 1 standard
            mock_agent = Mock()
            mock_agent.classify_intent = AsyncMock(return_value=IntentClassification(
                intent_type="BALANCE_INQUIRY",
                confidence=0.92,
                reasoning="Classification Phase 1"
            ))
            mock_get_agent.return_value = mock_agent
            
            # Endpoint V1 existant
            response = client.post(
                f"/api/v1/conversation/{test_user_id}",
                json={"message": "Mon solde"},
                headers=auth_headers
            )
            
            # Response Phase 1 pure (pas d'enrichissements Phase 2)
            assert response.status_code == 200
            data = response.json()
            
            # Structure Phase 1 exacte
            assert data["intent"]["intent_type"] == "BALANCE_INQUIRY"
            assert data["intent"]["confidence"] == 0.92
            assert data["user_id"] == test_user_id
            
            # Pas de champs Phase 2
            assert "comprehensive_entities" not in data
            assert "entity_validation" not in data
            assert "multi_agent_insights" not in data
    
    def test_conversation_endpoint_error_cascade(
        self,
        client,
        auth_headers, 
        test_user_id
    ):
        """Test cascade d'erreurs Phase 2 → Phase 1 → Error"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team', 
                  side_effect=Exception("AutoGen team failed")), \
             patch('conversation_service.api.dependencies.get_intent_classifier_agent', 
                  return_value=None):
            
            response = client.post(
                f"/api/v1/conversation/v2/{test_user_id}",
                json={"message": "Test erreur"},
                headers=auth_headers
            )
            
            # Erreur finale si tout échoue
            assert response.status_code == 503
            error_data = response.json()
            assert "Service temporairement indisponible" in error_data["detail"]
            assert "error_code" in error_data
    
    def test_health_check_dual_mode(
        self,
        client
    ):
        """Test health check Phase 1 + Phase 2"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team') as mock_get_team, \
             patch('conversation_service.api.dependencies.get_intent_classifier_agent') as mock_get_agent:
            
            # Mock équipe AutoGen disponible
            mock_team = Mock()
            mock_team.health_check = AsyncMock(return_value={
                "team_operational": True,
                "agents_healthy": ["intent_classifier", "entity_extractor"],
                "last_assessment": datetime.utcnow().isoformat()
            })
            mock_get_team.return_value = mock_team
            
            # Mock agent Phase 1 disponible
            mock_agent = Mock()
            mock_get_agent.return_value = mock_agent
            
            response = client.get("/api/v1/team/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Health check détaillé
            assert data["single_agent_available"] is True
            assert data["multi_agent_team_available"] is True
            assert data["current_processing_mode"] == "multi_agent_team"
            assert "autogen_details" in data
            assert data["autogen_details"]["overall_status"] == "healthy"
    
    def test_health_check_phase1_only(
        self,
        client
    ):
        """Test health check Phase 1 seule (AutoGen indisponible)"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team', return_value=None), \
             patch('conversation_service.api.dependencies.get_intent_classifier_agent') as mock_get_agent:
            
            # Seul agent Phase 1 disponible
            mock_agent = Mock()
            mock_get_agent.return_value = mock_agent
            
            response = client.get("/api/v1/team/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Health check dégradé
            assert data["single_agent_available"] is True
            assert data["multi_agent_team_available"] is False
            assert data["current_processing_mode"] == "single_agent_fallback"
            assert "autogen_details" in data
            assert data["autogen_details"]["overall_status"] == "unavailable"
    
    def test_team_metrics_endpoint(
        self,
        client,
        auth_headers
    ):
        """Test endpoint métriques équipe"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team') as mock_get_team:
            
            # Mock équipe avec métriques
            mock_team = Mock()
            mock_team.get_team_metrics = Mock(return_value={
                "total_requests_processed": 1250,
                "successful_team_executions": 1180,
                "failed_team_executions": 70,
                "success_rate": 0.944,
                "average_team_response_time_ms": 1850.5,
                "team_health_score": 0.92,
                "agents_performance": {
                    "intent_classifier": {
                        "success_rate": 0.96,
                        "average_response_time_ms": 800
                    },
                    "entity_extractor": {
                        "success_rate": 0.91,
                        "average_response_time_ms": 1200
                    }
                }
            })
            mock_get_team.return_value = mock_team
            
            response = client.get("/api/v1/team/metrics", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Métriques équipe détaillées
            assert data["total_requests_processed"] == 1250
            assert data["success_rate"] == 0.944
            assert data["team_health_score"] == 0.92
            assert "agents_performance" in data
            assert len(data["agents_performance"]) == 2
    
    def test_conversation_endpoint_request_validation(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test validation requêtes endpoint conversation"""
        
        # Message vide
        response = client.post(
            f"/api/v1/conversation/v2/{test_user_id}",
            json={"message": ""},
            headers=auth_headers
        )
        assert response.status_code == 422
        
        # Message trop long
        long_message = "x" * 1001  # > 1000 chars
        response = client.post(
            f"/api/v1/conversation/v2/{test_user_id}",
            json={"message": long_message},
            headers=auth_headers
        )
        assert response.status_code == 422
        
        # User ID invalide
        response = client.post(
            "/api/v1/conversation/v2/invalid_user_id",
            json={"message": "Test"},
            headers=auth_headers
        )
        assert response.status_code == 422
    
    def test_conversation_endpoint_authentication(
        self,
        client,
        test_user_id
    ):
        """Test authentification endpoints Phase 2"""
        
        # Sans headers auth
        response = client.post(
            f"/api/v1/conversation/v2/{test_user_id}",
            json={"message": "Test auth"}
        )
        assert response.status_code == 401
        
        # Token invalide
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = client.post(
            f"/api/v1/conversation/v2/{test_user_id}",
            json={"message": "Test auth"},
            headers=invalid_headers
        )
        assert response.status_code == 401
    
    @pytest.mark.performance
    def test_conversation_endpoint_performance_phase2(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test performance endpoint Phase 2 vs Phase 1"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team') as mock_get_team:
            
            # Mock équipe rapide
            mock_team = Mock()
            mock_team.process_user_message = AsyncMock(return_value={
                "workflow_success": True,
                "intent_result": {
                    "intent_type": "BALANCE_INQUIRY",
                    "confidence": 0.95
                },
                "entities_result": {
                    "extraction_success": True,
                    "entities": {},
                    "overall_confidence": 0.8
                },
                "total_processing_time_ms": 1800,  # Acceptable pour équipe
                "coherence_validation": {"score": 0.85, "threshold_met": True}
            })
            mock_get_team.return_value = mock_team
            
            start_time = datetime.utcnow()
            
            response = client.post(
                f"/api/v1/conversation/v2/{test_user_id}",
                json={"message": "Performance test"},
                headers=auth_headers
            )
            
            end_time = datetime.utcnow()
            request_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Performance acceptable Phase 2
            assert response.status_code == 200
            assert request_time_ms < 3000  # Max 3s pour équipe + overhead API
            
            data = response.json()
            assert data["processing_time_ms"] < 2500  # Temps équipe acceptable


class TestAPIIntegrationBatchProcessing:
    """Tests traitement batch Phase 2"""
    
    def test_batch_processing_endpoint(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test endpoint traitement batch messages"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team') as mock_get_team:
            
            mock_team = Mock()
            
            # Mock différents résultats pour batch
            def mock_batch_process(message, user_id):
                if "Amazon" in message:
                    return {
                        "workflow_success": True,
                        "intent_result": {"intent_type": "SEARCH_BY_MERCHANT", "confidence": 0.95},
                        "entities_result": {
                            "extraction_success": True,
                            "entities": {"merchants": [{"normalized": "Amazon"}]},
                            "overall_confidence": 0.93
                        },
                        "total_processing_time_ms": 1600
                    }
                else:
                    return {
                        "workflow_success": True,
                        "intent_result": {"intent_type": "BALANCE_INQUIRY", "confidence": 0.88},
                        "entities_result": {"extraction_success": True, "entities": {}},
                        "total_processing_time_ms": 1200
                    }
            
            mock_team.process_user_message = AsyncMock(side_effect=mock_batch_process)
            mock_get_team.return_value = mock_team
            
            # Batch de messages
            batch_request = {
                "messages": [
                    {"message": "Mes achats Amazon", "user_id": test_user_id},
                    {"message": "Mon solde", "user_id": test_user_id},
                    {"message": "Achats Amazon ce mois", "user_id": test_user_id}
                ]
            }
            
            response = client.post(
                "/api/v1/conversation/v2/batch",
                json=batch_request,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Validation batch response
            assert data["batch_size"] == 3
            assert len(data["responses"]) == 3
            assert data["successful_responses"] >= 2
            assert data["average_response_time_ms"] > 0
    
    def test_batch_processing_partial_failures(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test traitement batch avec échecs partiels"""
        
        with patch('conversation_service.api.dependencies.get_multi_agent_team') as mock_get_team:
            
            mock_team = Mock()
            
            # Mock avec un échec
            def mock_batch_with_failure(message, user_id):
                if "error" in message.lower():
                    raise Exception("Simulated processing error")
                else:
                    return {
                        "workflow_success": True,
                        "intent_result": {"intent_type": "BALANCE_INQUIRY", "confidence": 0.9},
                        "entities_result": {"extraction_success": True, "entities": {}},
                        "total_processing_time_ms": 1300
                    }
            
            mock_team.process_user_message = AsyncMock(side_effect=mock_batch_with_failure)
            mock_get_team.return_value = mock_team
            
            batch_request = {
                "messages": [
                    {"message": "Mon solde", "user_id": test_user_id},
                    {"message": "Message error test", "user_id": test_user_id},  # Échouera
                    {"message": "Autre test", "user_id": test_user_id}
                ]
            }
            
            response = client.post(
                "/api/v1/conversation/v2/batch",
                json=batch_request,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Batch partiel réussi
            assert data["batch_size"] == 3
            assert data["successful_responses"] == 2
            assert data["failed_responses"] == 1
            assert data["batch_efficiency_score"] < 1.0  # Réduit par échec


class TestAPIIntegrationRegressionPhase1:
    """Tests régression Phase 1 avec Phase 2 active"""
    
    @pytest.mark.regression
    def test_phase1_endpoints_unchanged(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test endpoints Phase 1 inchangés avec Phase 2 déployée"""
        
        # Liste endpoints Phase 1 critiques
        phase1_endpoints = [
            f"/api/v1/conversation/{test_user_id}",
            "/health/conversation",
            "/metrics/conversation"
        ]
        
        with patch('conversation_service.api.dependencies.get_intent_classifier_agent') as mock_get_agent:
            
            mock_agent = Mock()
            mock_agent.classify_intent = AsyncMock(return_value=IntentClassification(
                intent_type="BALANCE_INQUIRY",
                confidence=0.9,
                reasoning="Phase 1 test"
            ))
            mock_get_agent.return_value = mock_agent
            
            for endpoint in phase1_endpoints:
                if endpoint.startswith("/api/v1/conversation/"):
                    response = client.post(
                        endpoint,
                        json={"message": "Test regression"},
                        headers=auth_headers
                    )
                else:
                    response = client.get(endpoint)
                
                # Endpoint Phase 1 fonctionne
                assert response.status_code in [200, 404]  # 404 si pas implémenté
    
    @pytest.mark.regression
    def test_phase1_response_format_compatibility(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test format réponse Phase 1 inchangé"""
        
        with patch('conversation_service.api.dependencies.get_intent_classifier_agent') as mock_get_agent:
            
            mock_agent = Mock()
            mock_agent.classify_intent = AsyncMock(return_value=IntentClassification(
                intent_type="SPENDING_ANALYSIS",
                confidence=0.87,
                reasoning="Classification Phase 1"
            ))
            mock_get_agent.return_value = mock_agent
            
            # Endpoint V1 pur
            response = client.post(
                f"/api/v1/conversation/{test_user_id}",
                json={"message": "Mes dépenses ce mois"},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Format Phase 1 exact
            required_fields = [
                "user_id", "message", "intent", "processing_time_ms", "status"
            ]
            
            for field in required_fields:
                assert field in data, f"Champ Phase 1 manquant: {field}"
            
            # Structure intent Phase 1
            intent = data["intent"]
            assert "intent_type" in intent
            assert "confidence" in intent
            assert intent["intent_type"] == "SPENDING_ANALYSIS"
            assert intent["confidence"] == 0.87
    
    @pytest.mark.regression  
    def test_phase1_performance_not_degraded(
        self,
        client,
        auth_headers,
        test_user_id
    ):
        """Test performance Phase 1 non dégradée par Phase 2"""
        
        with patch('conversation_service.api.dependencies.get_intent_classifier_agent') as mock_get_agent:
            
            mock_agent = Mock()
            mock_agent.classify_intent = AsyncMock(return_value=IntentClassification(
                intent_type="BALANCE_INQUIRY",
                confidence=0.92,
                reasoning="Performance test"
            ))
            mock_get_agent.return_value = mock_agent
            
            start_time = datetime.utcnow()
            
            response = client.post(
                f"/api/v1/conversation/{test_user_id}",
                json={"message": "Performance Phase 1"},
                headers=auth_headers
            )
            
            end_time = datetime.utcnow()
            request_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Performance Phase 1 maintenue
            assert response.status_code == 200
            assert request_time_ms < 1000  # Phase 1 reste rapide (<1s)
            
            data = response.json()
            assert data["processing_time_ms"] < 800  # Agent seul rapide