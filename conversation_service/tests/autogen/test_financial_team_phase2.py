"""
Tests Financial Team AutoGen Phase 2
Tests intégration équipe multi-agents avec infrastructure Phase 1
"""

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock, Mock

# Imports cohérents Phase 1
from conversation_service.models.responses.conversation_responses import IntentClassification
from conversation_service.models.conversation.entities import ComprehensiveEntityExtraction
from conversation_service.models.responses.conversation_responses_phase2 import (
    ConversationResponsePhase2,
    EntityValidationResult,
    MultiAgentProcessingInsights
)
from conversation_service.models.autogen.team_models import (
    MultiAgentTeamState,
    TeamWorkflowExecution,
    TeamCommunicationMode
)


class TestFinancialTeamPhase2:
    """Tests équipe AutoGen complète Phase 2"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_workflow_success(
        self, 
        mock_financial_team_phase2,
        mock_intent_response,
        mock_entity_response,
        sample_team_context
    ):
        """Test workflow complet Intent + Entity réussi"""
        
        team = mock_financial_team_phase2
        
        # Mock responses agents séquentiels
        intent_response_json = json.dumps(mock_intent_response)
        entity_response_json = json.dumps(mock_entity_response)
        
        with patch.object(team.intent_classifier, 'a_generate_reply', 
                         return_value=intent_response_json) as mock_intent, \
             patch.object(team.entity_extractor, 'a_generate_reply',
                         return_value=entity_response_json) as mock_entity:
            
            result = await team.process_user_message(
                "Mes achats Amazon plus de 50€", 123
            )
            
            # Validations workflow complet
            assert result["workflow_success"] is True
            assert result["intent_result"]["intent_type"] == "SEARCH_BY_MERCHANT"
            assert result["entities_result"]["extraction_success"] is True
            assert len(result["agents_sequence"]) == 2
            assert "intent_classifier" in result["agents_sequence"]
            assert "entity_extractor" in result["agents_sequence"]
            
            # Validation séquence agents
            mock_intent.assert_called_once()
            mock_entity.assert_called_once()
            
            # Validation temps traitement
            assert result["total_processing_time_ms"] > 0
            assert result["coherence_validation"]["score"] > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_coherence_validation(
        self,
        mock_financial_team_phase2,
        sample_team_context
    ):
        """Test validation cohérence intention-entités"""
        
        team = mock_financial_team_phase2
        
        # Intention et entités cohérentes
        intent_response = json.dumps({
            "intent_type": "SEARCH_BY_MERCHANT",
            "confidence": 0.95,
            "reasoning": "Recherche par marchand détectée"
        })
        
        # Entités cohérentes (marchands présents)
        coherent_entity_response = json.dumps({
            "extraction_success": True,
            "entities": {
                "merchants": [{"normalized": "Amazon", "confidence": 0.98}],
                "amounts": [{"value": 50.0, "operator": "gt", "confidence": 0.96}]
            },
            "overall_confidence": 0.97
        })
        
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         return_value=intent_response), \
             patch.object(team.entity_extractor, 'a_generate_reply',
                         return_value=coherent_entity_response):
            
            result = await team.process_user_message("Achats Amazon >50€", 123)
            
            # Score cohérence élevé pour intention/entités alignées
            coherence_score = result["coherence_validation"]["score"]
            assert coherence_score > 0.8
            assert result["coherence_validation"]["threshold_met"] is True
            
            # Détails validation
            validation_details = result["coherence_validation"]["validation_details"]
            assert validation_details["intent_entity_alignment"] > 0.8
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_incoherent_results(
        self,
        mock_financial_team_phase2
    ):
        """Test détection incohérence intention-entités"""
        
        team = mock_financial_team_phase2
        
        # Intention balance mais entités marchands (incohérent)
        intent_response = json.dumps({
            "intent_type": "BALANCE_INQUIRY",
            "confidence": 0.92,
            "reasoning": "Demande de solde"
        })
        
        incoherent_entity_response = json.dumps({
            "extraction_success": True,
            "entities": {
                "merchants": [{"normalized": "Amazon", "confidence": 0.95}]  # Incohérent avec BALANCE_INQUIRY
            },
            "overall_confidence": 0.95
        })
        
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         return_value=intent_response), \
             patch.object(team.entity_extractor, 'a_generate_reply',
                         return_value=incoherent_entity_response):
            
            result = await team.process_user_message("Mon solde Amazon", 123)
            
            # Score cohérence faible détecté
            coherence_score = result["coherence_validation"]["score"]
            assert coherence_score < 0.6
            assert result["coherence_validation"]["threshold_met"] is False
    
    @pytest.mark.asyncio 
    @pytest.mark.integration
    async def test_team_cache_integration_phase1(
        self,
        mock_financial_team_phase2,
        test_cache_manager
    ):
        """Test intégration cache équipe avec infrastructure Phase 1"""
        
        team = mock_financial_team_phase2
        
        # Mock cache hit équipe
        cached_team_result = {
            "workflow_success": True,
            "intent_result": {
                "intent_type": "BALANCE_INQUIRY",
                "confidence": 0.94
            },
            "entities_result": {
                "extraction_success": True,
                "entities": {},
                "overall_confidence": 0.8
            },
            "cache_hit": True,
            "coherence_validation": {
                "score": 0.85,
                "threshold_met": True
            }
        }
        
        with patch.object(team.cache_manager, 'get',
                         return_value=cached_team_result):
            
            result = await team.process_user_message("Mon solde", 123)
            
            # Cache équipe utilisé
            assert result["cache_hit"] is True
            assert result["intent_result"]["intent_type"] == "BALANCE_INQUIRY"
            assert result["workflow_success"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_team_metrics_update_phase1_compatible(
        self,
        mock_financial_team_phase2,
        mock_team_results,
        mock_metrics_collector
    ):
        """Test métriques équipe compatibles infrastructure Phase 1"""
        
        team = mock_financial_team_phase2
        
        # Mock workflow réussi
        intent_response = json.dumps({
            "intent_type": "SEARCH_BY_MERCHANT",
            "confidence": 0.95
        })
        
        entity_response = json.dumps({
            "extraction_success": True,
            "entities": {"merchants": [{"normalized": "Amazon"}]},
            "overall_confidence": 0.92
        })
        
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         return_value=intent_response), \
             patch.object(team.entity_extractor, 'a_generate_reply',
                         return_value=entity_response):
            
            initial_calls = mock_metrics_collector.record_metric.call_count
            
            await team.process_user_message("Test métriques", 123)
            
            # Métriques mises à jour
            assert mock_metrics_collector.record_metric.call_count > initial_calls
            
            # Vérifier métriques spécifiques équipe
            calls = mock_metrics_collector.record_metric.call_args_list
            metric_names = [call[0][0] for call in calls[-5:]]  # 5 derniers appels
            
            assert any("autogen_team" in metric for metric in metric_names)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_team_failure_handling_with_fallback(
        self,
        mock_financial_team_phase2
    ):
        """Test gestion échec équipe avec fallback Phase 1"""
        
        team = mock_financial_team_phase2
        
        # Simulation échec intent classifier
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         side_effect=Exception("Agent timeout")):
            
            result = await team.process_user_message("Test échec", 123)
            
            # Fallback appliqué (cohérent Phase 1)
            assert result["workflow_success"] is False
            assert result["fallback_applied"] is True
            assert result["intent_result"]["intent_type"] == "GENERAL_INQUIRY"  # Fallback
            assert result["error_details"]["error_type"] == "agent_failure"
            
            # Entités pas extraites si intent échoue
            assert result["entities_result"]["extraction_success"] is False
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_team_partial_failure_entity_extractor(
        self,
        mock_financial_team_phase2
    ):
        """Test échec partiel (intent réussi, entity échoué)"""
        
        team = mock_financial_team_phase2
        
        # Intent réussit
        intent_response = json.dumps({
            "intent_type": "SEARCH_BY_MERCHANT",
            "confidence": 0.95
        })
        
        # Entity extractor échoue
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         return_value=intent_response), \
             patch.object(team.entity_extractor, 'a_generate_reply',
                         side_effect=Exception("Entity extraction failed")):
            
            result = await team.process_user_message("Achats Amazon", 123)
            
            # Workflow partiellement réussi
            assert result["workflow_success"] is False  # Global échec
            assert result["intent_result"]["intent_type"] == "SEARCH_BY_MERCHANT"  # Intent OK
            assert result["entities_result"]["extraction_success"] is False  # Entity KO
            assert result["fallback_applied"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_team_workflow_performance(
        self,
        mock_financial_team_phase2,
        performance_benchmark
    ):
        """Test performance workflow équipe"""
        
        @performance_benchmark(target_time_ms=3000)  # Max 3s pour équipe
        async def team_workflow_benchmark():
            intent_response = json.dumps({
                "intent_type": "SEARCH_BY_MERCHANT",
                "confidence": 0.95
            })
            
            entity_response = json.dumps({
                "extraction_success": True,
                "entities": {"merchants": [{"normalized": "Amazon"}]},
                "overall_confidence": 0.92
            })
            
            with patch.object(mock_financial_team_phase2.intent_classifier, 'a_generate_reply',
                             return_value=intent_response), \
                 patch.object(mock_financial_team_phase2.entity_extractor, 'a_generate_reply',
                             return_value=entity_response):
                
                return await mock_financial_team_phase2.process_user_message("Benchmark test", 123)
        
        result = await team_workflow_benchmark()
        assert result["workflow_success"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_phase2_response_from_team_results(
        self,
        mock_financial_team_phase2,
        sample_comprehensive_entities,
        sample_entity_validation,
        sample_multi_agent_insights
    ):
        """Test création ConversationResponsePhase2 depuis résultats équipe"""
        
        team = mock_financial_team_phase2
        
        # Créer intent classification
        intent = IntentClassification(
            intent_type="SEARCH_BY_MERCHANT",
            confidence=0.95,
            reasoning="Recherche par marchand Amazon"
        )
        
        # Créer réponse Phase 2 complète
        phase2_response = ConversationResponsePhase2.from_multi_agent_results(
            user_id=123,
            message="Mes achats Amazon plus de 50€",
            intent=intent,
            comprehensive_entities=sample_comprehensive_entities,
            entity_validation=sample_entity_validation,
            multi_agent_insights=sample_multi_agent_insights,
            processing_time_ms=2050
        )
        
        # Validations modèle Phase 2
        assert phase2_response.user_id == 123
        assert phase2_response.intent.intent_type == "SEARCH_BY_MERCHANT"
        assert phase2_response.comprehensive_entities.entities_found is True
        assert phase2_response.entity_validation.entities_coherent is True
        assert phase2_response.multi_agent_insights.consensus_reached is True
        assert phase2_response.is_high_quality_response() is True
        
        # Vérification compatibilité Phase 1
        assert hasattr(phase2_response, 'user_id')
        assert hasattr(phase2_response, 'message')
        assert hasattr(phase2_response, 'intent')
        assert hasattr(phase2_response, 'processing_time_ms')
        assert hasattr(phase2_response, 'status')
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_team_state_management(
        self,
        mock_financial_team_phase2,
        mock_team_state
    ):
        """Test gestion état équipe"""
        
        team = mock_financial_team_phase2
        team.team_state = mock_team_state
        
        # Vérifier état initial
        assert team.team_state.is_team_operational() is True
        assert team.team_state.team_health_score > 0.5
        
        # Simuler workflow réussi
        intent_response = json.dumps({"intent_type": "BALANCE_INQUIRY", "confidence": 0.9})
        entity_response = json.dumps({
            "extraction_success": True,
            "entities": {},
            "overall_confidence": 0.8
        })
        
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         return_value=intent_response), \
             patch.object(team.entity_extractor, 'a_generate_reply',
                         return_value=entity_response):
            
            # Workflow execution
            workflow = TeamWorkflowExecution(
                user_message="Mon solde",
                communication_mode=TeamCommunicationMode.SEQUENTIAL
            )
            
            team.team_state.start_workflow(workflow)
            result = await team.process_user_message("Mon solde", 123)
            team.team_state.complete_workflow(workflow.execution_id, result["workflow_success"])
            
            # État équipe mis à jour
            assert team.team_state.total_requests_processed > 0
            if result["workflow_success"]:
                assert team.team_state.successful_team_executions > 0
    
    @pytest.mark.asyncio
    @pytest.mark.regression
    async def test_no_regression_phase1_compatibility(
        self,
        mock_financial_team_phase2
    ):
        """Test non-régression compatibilité Phase 1"""
        
        team = mock_financial_team_phase2
        
        # Inputs Phase 1 standard
        phase1_test_cases = [
            ("Mon solde", "BALANCE_INQUIRY"),
            ("Mes achats Amazon", "SEARCH_BY_MERCHANT"),
            ("Dépenses restaurants ce mois", "SPENDING_ANALYSIS")
        ]
        
        for input_msg, expected_intent in phase1_test_cases:
            intent_response = json.dumps({
                "intent_type": expected_intent,
                "confidence": 0.85
            })
            
            entity_response = json.dumps({
                "extraction_success": True,
                "entities": {"merchants": [{"normalized": "Test"}]},
                "overall_confidence": 0.8
            })
            
            with patch.object(team.intent_classifier, 'a_generate_reply',
                             return_value=intent_response), \
                 patch.object(team.entity_extractor, 'a_generate_reply',
                             return_value=entity_response):
                
                result = await team.process_user_message(input_msg, 123)
                
                # Structure réponse cohérente Phase 1
                assert result["intent_result"]["intent_type"] == expected_intent
                assert "workflow_success" in result
                assert "total_processing_time_ms" in result
                assert isinstance(result["total_processing_time_ms"], int)


class TestFinancialTeamWorkflowModes:
    """Tests modes workflow équipe AutoGen"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sequential_workflow_mode(
        self,
        mock_financial_team_phase2
    ):
        """Test mode séquentiel (intent puis entity)"""
        
        team = mock_financial_team_phase2
        
        intent_response = json.dumps({
            "intent_type": "SEARCH_BY_MERCHANT",
            "confidence": 0.95,
            "team_context": {
                "suggested_entities_focus": {
                    "priority_entities": ["merchants", "amounts"]
                }
            }
        })
        
        entity_response = json.dumps({
            "extraction_success": True,
            "entities": {
                "merchants": [{"normalized": "Amazon", "confidence": 0.98}]
            },
            "overall_confidence": 0.92,
            "context_used": {
                "intent_context": "SEARCH_BY_MERCHANT",
                "focus_entities": ["merchants", "amounts"]
            }
        })
        
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         return_value=intent_response) as mock_intent, \
             patch.object(team.entity_extractor, 'a_generate_reply',
                         return_value=entity_response) as mock_entity:
            
            result = await team.process_user_message("Achats Amazon", 123)
            
            # Validation séquence
            assert mock_intent.call_count == 1
            assert mock_entity.call_count == 1
            
            # Entity a utilisé contexte intent
            assert result["entities_result"]["context_used"]["intent_context"] == "SEARCH_BY_MERCHANT"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_with_iteration_limit(
        self,
        mock_financial_team_phase2
    ):
        """Test workflow avec limite itérations"""
        
        team = mock_financial_team_phase2
        
        # Mock conflit nécessitant plusieurs itérations
        iteration_count = 0
        
        def mock_intent_with_iterations(*args, **kwargs):
            nonlocal iteration_count
            iteration_count += 1
            
            if iteration_count == 1:
                # Première réponse ambiguë
                return json.dumps({
                    "intent_type": "GENERAL_INQUIRY",
                    "confidence": 0.6,  # Faible confiance
                    "needs_clarification": True
                })
            else:
                # Réponse clarifiée
                return json.dumps({
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "confidence": 0.92,
                    "clarified": True
                })
        
        entity_response = json.dumps({
            "extraction_success": True,
            "entities": {"merchants": [{"normalized": "Amazon"}]},
            "overall_confidence": 0.9
        })
        
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         side_effect=mock_intent_with_iterations), \
             patch.object(team.entity_extractor, 'a_generate_reply',
                         return_value=entity_response):
            
            result = await team.process_user_message("Achats Amazon", 123)
            
            # Workflow a convergé après itérations
            assert result["workflow_success"] is True
            assert iteration_count > 1  # Plusieurs appels intent
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_timeout_handling(
        self,
        mock_financial_team_phase2
    ):
        """Test gestion timeout workflow"""
        
        team = mock_financial_team_phase2
        
        async def slow_intent_response(*args, **kwargs):
            await asyncio.sleep(2)  # Simulation lenteur
            return json.dumps({
                "intent_type": "BALANCE_INQUIRY",
                "confidence": 0.9
            })
        
        with patch.object(team.intent_classifier, 'a_generate_reply',
                         side_effect=slow_intent_response):
            
            # Workflow avec timeout court
            start_time = datetime.utcnow()
            result = await team.process_user_message("Mon solde", 123, timeout_seconds=1)
            end_time = datetime.utcnow()
            
            # Timeout géré gracieusement
            execution_time = (end_time - start_time).total_seconds()
            assert execution_time < 3  # Pas d'attente excessive
            assert result["fallback_applied"] is True
            assert "timeout" in result.get("error_details", {}).get("error_type", "")