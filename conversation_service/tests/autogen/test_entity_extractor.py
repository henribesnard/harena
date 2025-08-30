"""
Tests Entity Extractor Agent AutoGen Phase 2
Tests unitaires agent extraction entités avec infrastructure Phase 1
"""

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock, Mock

# Imports cohérents Phase 1
from conversation_service.models.conversation.entities import (
    ComprehensiveEntityExtraction,
    ExtractedAmount,
    ExtractedMerchant,
    ExtractedDateRange,
    ExtractedCategory
)
from conversation_service.models.responses.conversation_responses import IntentClassification


class TestEntityExtractorAgent:
    """Tests agent extraction entités AutoGen"""
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_entities_success(
        self, 
        entity_extractor_agent, 
        sample_team_context,
        mock_entity_response,
        assert_valid_json_response
    ):
        """Test extraction réussie avec réutilisation mocks Phase 1"""
        
        # Mock réponse LLM (réutilise pattern Phase 1)
        mock_response_json = json.dumps(mock_entity_response)
        
        with patch.object(entity_extractor_agent, 'a_generate_reply', 
                         return_value=mock_response_json):
            
            result = await entity_extractor_agent.extract_entities_for_team(sample_team_context)
            
            # Validations (réutilise assertions Phase 1)
            assert_valid_json_response(result)
            assert result["extraction_success"] is True
            assert len(result["entities"]["amounts"]) == 1
            assert result["entities"]["amounts"][0]["value"] == 50.0
            assert len(result["entities"]["merchants"]) == 1
            assert result["entities"]["merchants"][0]["normalized"] == "Amazon"
            assert result["overall_confidence"] == 0.95
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_entities_with_intent_context(
        self, 
        entity_extractor_agent,
        sample_team_context
    ):
        """Test adaptation extraction selon contexte intention"""
        
        # Contexte intention SPENDING_ANALYSIS
        spending_context = {
            **sample_team_context,
            "intent_result": {
                "intent_type": "SPENDING_ANALYSIS",
                "confidence": 0.88,
                "suggested_entities_focus": {
                    "priority_entities": ["categories", "amounts", "dates"]
                }
            }
        }
        
        mock_response = json.dumps({
            "extraction_success": True,
            "entities": {
                "amounts": [{"value": 100.0, "operator": "total", "confidence": 0.91}],
                "categories": [
                    {"name": "Restaurant", "confidence": 0.87},
                    {"name": "Shopping", "confidence": 0.84}
                ],
                "dates": [{
                    "type": "relative",
                    "raw_text": "ce mois",
                    "confidence": 0.92
                }]
            },
            "overall_confidence": 0.89,
            "entities_count": 4,
            "extraction_metadata": {
                "focus_adapted": True,
                "intent_context_used": "SPENDING_ANALYSIS"
            }
        })
        
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         return_value=mock_response):
            
            result = await entity_extractor_agent.extract_entities_for_team(spending_context)
            
            # Validation adaptation selon intention
            assert result["extraction_success"] is True
            assert len(result["entities"]["categories"]) == 2
            assert "Restaurant" in [cat["name"] for cat in result["entities"]["categories"]]
            assert result["extraction_metadata"]["focus_adapted"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_entities_json_error_fallback(
        self, 
        entity_extractor_agent,
        sample_team_context
    ):
        """Test fallback JSON malformé (pattern Phase 1)"""
        
        # Réponse JSON invalide
        invalid_json = '{"extraction_success": true, "invalid": json}'
        
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         return_value=invalid_json):
            
            result = await entity_extractor_agent.extract_entities_for_team(sample_team_context)
            
            # Fallback cohérent Phase 1
            assert result["extraction_success"] is False
            assert result["fallback_used"] is True
            assert result["overall_confidence"] == 0.0
            assert "json_parse_error" in result["error_details"]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_extract_entities_cache_integration(
        self,
        entity_extractor_agent,
        sample_team_context,
        test_cache_manager
    ):
        """Test intégration cache Phase 1"""
        
        # Mock cache hit
        cached_result = {
            "extraction_success": True,
            "entities": {
                "merchants": [{"normalized": "Amazon", "confidence": 0.98}],
                "amounts": [{"value": 50.0, "confidence": 0.96}]
            },
            "overall_confidence": 0.95,
            "cache_hit": True,
            "extraction_metadata": {
                "cached_at": datetime.utcnow().isoformat()
            }
        }
        
        # Mock cache manager get
        with patch.object(entity_extractor_agent.cache_manager, 'get',
                         return_value=cached_result):
            
            result = await entity_extractor_agent.extract_entities_for_team(sample_team_context)
            
            # Validation cache utilisé
            assert result["cache_hit"] is True
            assert result["entities"]["merchants"][0]["normalized"] == "Amazon"
            assert result["overall_confidence"] == 0.95
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_entities_empty_message(
        self,
        entity_extractor_agent
    ):
        """Test gestion message vide"""
        
        empty_context = {
            "user_message": "",
            "user_id": 123,
            "intent_result": {
                "intent_type": "GENERAL_INQUIRY",
                "confidence": 0.3
            }
        }
        
        result = await entity_extractor_agent.extract_entities_for_team(empty_context)
        
        # Pas d'entités extraites pour message vide
        assert result["extraction_success"] is False
        assert result["entities"] == {}
        assert result["overall_confidence"] == 0.0
        assert "empty_message" in result.get("error_details", {})
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_entities_low_confidence_intent(
        self,
        entity_extractor_agent,
        sample_team_context
    ):
        """Test extraction avec intention faible confiance"""
        
        low_confidence_context = {
            **sample_team_context,
            "intent_result": {
                "intent_type": "GENERAL_INQUIRY",
                "confidence": 0.3,  # Très faible confiance
                "reasoning": "Intent unclear"
            }
        }
        
        # Mock réponse extraction avec confiance réduite
        mock_response = json.dumps({
            "extraction_success": True,
            "entities": {
                "merchants": [{"normalized": "Amazon", "confidence": 0.6}]  # Confiance réduite
            },
            "overall_confidence": 0.5,  # Réduite à cause intent faible
            "extraction_metadata": {
                "intent_confidence_penalty": 0.3
            }
        })
        
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         return_value=mock_response):
            
            result = await entity_extractor_agent.extract_entities_for_team(low_confidence_context)
            
            # Validation confiance ajustée
            assert result["extraction_success"] is True
            assert result["overall_confidence"] <= 0.6  # Pénalisée
            assert "intent_confidence_penalty" in result["extraction_metadata"]
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_entities_timeout_handling(
        self,
        entity_extractor_agent,
        sample_team_context
    ):
        """Test gestion timeout agent"""
        
        # Mock timeout exception
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         side_effect=asyncio.TimeoutError("Agent timeout")):
            
            result = await entity_extractor_agent.extract_entities_for_team(sample_team_context)
            
            # Fallback timeout
            assert result["extraction_success"] is False
            assert result["fallback_used"] is True
            assert result["error_details"]["error_type"] == "timeout"
            assert result["overall_confidence"] == 0.0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_extract_entities_metrics_update(
        self,
        entity_extractor_agent,
        sample_team_context,
        mock_metrics_collector
    ):
        """Test mise à jour métriques (intégration Phase 1)"""
        
        mock_response = json.dumps({
            "extraction_success": True,
            "entities": {"merchants": [{"normalized": "Amazon"}]},
            "overall_confidence": 0.95,
            "processing_time_ms": 1200
        })
        
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         return_value=mock_response):
            
            initial_calls = mock_metrics_collector.record_metric.call_count
            
            result = await entity_extractor_agent.extract_entities_for_team(sample_team_context)
            
            # Métriques mises à jour
            assert mock_metrics_collector.record_metric.call_count > initial_calls
            assert result["extraction_success"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_comprehensive_extraction(
        self,
        entity_extractor_agent,
        sample_team_context,
        mock_entity_response
    ):
        """Test création ComprehensiveEntityExtraction depuis résultats"""
        
        mock_response_json = json.dumps(mock_entity_response)
        
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         return_value=mock_response_json):
            
            # Extraction avec création modèle comprehensive
            result = await entity_extractor_agent.extract_entities_for_team(sample_team_context)
            
            # Créer modèle Pydantic depuis résultats
            comprehensive = ComprehensiveEntityExtraction(
                user_message=sample_team_context["user_message"],
                amounts=[
                    ExtractedAmount(
                        value=result["entities"]["amounts"][0]["value"],
                        original_text=result["entities"]["amounts"][0]["raw_text"],
                        confidence=result["entities"]["amounts"][0]["confidence"]
                    )
                ] if result["entities"]["amounts"] else [],
                merchants=[
                    ExtractedMerchant(
                        name=result["entities"]["merchants"][0]["normalized"],
                        original_text=result["entities"]["merchants"][0]["name"],
                        confidence=result["entities"]["merchants"][0]["confidence"]
                    )
                ] if result["entities"]["merchants"] else [],
                overall_confidence=result["overall_confidence"],
                extraction_method="multi_agent_autogen"
            )
            
            # Validation modèle Pydantic
            assert comprehensive.entities_found is True
            assert len(comprehensive.amounts) == 1
            assert comprehensive.amounts[0].value == 50.0
            assert len(comprehensive.merchants) == 1
            assert comprehensive.merchants[0].name == "Amazon"
            assert comprehensive.overall_confidence == 0.95
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_extract_entities_performance(
        self,
        entity_extractor_agent,
        sample_team_context,
        performance_benchmark
    ):
        """Test performance extraction entités"""
        
        @performance_benchmark(target_time_ms=1500)  # Max 1.5s
        async def extract_with_benchmark():
            mock_response = json.dumps({
                "extraction_success": True,
                "entities": {"merchants": [{"normalized": "Amazon"}]},
                "overall_confidence": 0.95
            })
            
            with patch.object(entity_extractor_agent, 'a_generate_reply',
                             return_value=mock_response):
                return await entity_extractor_agent.extract_entities_for_team(sample_team_context)
        
        result = await extract_with_benchmark()
        assert result["extraction_success"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.regression
    async def test_no_regression_phase1_compatibility(
        self,
        entity_extractor_agent,
        sample_team_context
    ):
        """Test non-régression compatibilité Phase 1"""
        
        # Inputs Phase 1 standard
        phase1_inputs = [
            "Mon solde",
            "Mes achats Amazon",
            "Dépenses restaurants ce mois"
        ]
        
        for input_msg in phase1_inputs:
            context = {
                **sample_team_context,
                "user_message": input_msg
            }
            
            mock_response = json.dumps({
                "extraction_success": True,
                "entities": {"merchants": [{"normalized": "Test"}]},
                "overall_confidence": 0.8
            })
            
            with patch.object(entity_extractor_agent, 'a_generate_reply',
                             return_value=mock_response):
                
                result = await entity_extractor_agent.extract_entities_for_team(context)
                
                # Structure réponse cohérente
                assert "extraction_success" in result
                assert "entities" in result
                assert "overall_confidence" in result
                assert isinstance(result["overall_confidence"], (int, float))
                assert 0.0 <= result["overall_confidence"] <= 1.0


class TestEntityExtractorAgentEdgeCases:
    """Tests cas limites Entity Extractor"""
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_multiple_currencies(
        self,
        entity_extractor_agent,
        sample_team_context
    ):
        """Test extraction montants multiples devises"""
        
        multi_currency_context = {
            **sample_team_context,
            "user_message": "J'ai dépensé 50€ et $30 hier"
        }
        
        mock_response = json.dumps({
            "extraction_success": True,
            "entities": {
                "amounts": [
                    {"value": 50.0, "currency": "EUR", "confidence": 0.96},
                    {"value": 30.0, "currency": "USD", "confidence": 0.94}
                ]
            },
            "overall_confidence": 0.95
        })
        
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         return_value=mock_response):
            
            result = await entity_extractor_agent.extract_entities_for_team(multi_currency_context)
            
            # Validation multi-devises
            assert len(result["entities"]["amounts"]) == 2
            currencies = [amount["currency"] for amount in result["entities"]["amounts"]]
            assert "EUR" in currencies
            assert "USD" in currencies
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_complex_date_expressions(
        self,
        entity_extractor_agent,
        sample_team_context
    ):
        """Test extraction expressions temporelles complexes"""
        
        complex_date_context = {
            **sample_team_context,
            "user_message": "Mes dépenses entre le 1er et le 15 janvier dernier"
        }
        
        mock_response = json.dumps({
            "extraction_success": True,
            "entities": {
                "dates": [{
                    "type": "date_range",
                    "raw_text": "entre le 1er et le 15 janvier dernier",
                    "parsed_range": {
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-15"
                    },
                    "confidence": 0.91
                }]
            },
            "overall_confidence": 0.91
        })
        
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         return_value=mock_response):
            
            result = await entity_extractor_agent.extract_entities_for_team(complex_date_context)
            
            # Validation parsing dates complexes
            date_entity = result["entities"]["dates"][0]
            assert date_entity["type"] == "date_range"
            assert "start_date" in date_entity["parsed_range"]
            assert "end_date" in date_entity["parsed_range"]
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extract_ambiguous_merchant_names(
        self,
        entity_extractor_agent,
        sample_team_context
    ):
        """Test extraction noms marchands ambigus"""
        
        ambiguous_context = {
            **sample_team_context,
            "user_message": "Mes achats chez Orange et Apple ce mois"
        }
        
        mock_response = json.dumps({
            "extraction_success": True,
            "entities": {
                "merchants": [
                    {
                        "name": "Orange",
                        "normalized": "Orange (telecom)",
                        "confidence": 0.85,
                        "disambiguation": "telecom_provider"
                    },
                    {
                        "name": "Apple",
                        "normalized": "Apple Inc",
                        "confidence": 0.92,
                        "disambiguation": "tech_company"
                    }
                ]
            },
            "overall_confidence": 0.88
        })
        
        with patch.object(entity_extractor_agent, 'a_generate_reply',
                         return_value=mock_response):
            
            result = await entity_extractor_agent.extract_entities_for_team(ambiguous_context)
            
            # Validation désambiguïsation
            merchants = result["entities"]["merchants"]
            orange_merchant = next(m for m in merchants if m["name"] == "Orange")
            apple_merchant = next(m for m in merchants if m["name"] == "Apple")
            
            assert "disambiguation" in orange_merchant
            assert "disambiguation" in apple_merchant
            assert orange_merchant["normalized"] == "Orange (telecom)"
            assert apple_merchant["normalized"] == "Apple Inc"