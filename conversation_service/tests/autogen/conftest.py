"""
Configuration fixtures AutoGen Phase 2 - Extension cohérente Phase 1
Réutilise infrastructure tests existante avec extensions AutoGen
"""

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Imports fixtures Phase 1 réutilisées
from ..conftest import (
    mock_deepseek_response, 
    test_user_id, 
    test_cache_manager,
    mock_metrics_collector,
    auth_headers
)

# Imports AutoGen et modèles
from conversation_service.models.conversation.entities import (
    ComprehensiveEntityExtraction,
    ExtractedAmount,
    ExtractedMerchant,
    ExtractedDateRange,
    ExtractedCategory
)
from conversation_service.models.responses.conversation_responses_phase2 import (
    ConversationResponsePhase2,
    EntityValidationResult,
    MultiAgentProcessingInsights
)
from conversation_service.models.autogen.team_models import (
    MultiAgentTeamState,
    TeamWorkflowExecution,
    AgentPerformanceMetrics,
    TeamCommunicationMode
)


# Configuration AutoGen Mock (standard pour tests)
OAI_CONFIG_LIST_MOCK = [
    {
        "model": "deepseek-chat",
        "api_key": "mock_deepseek_key",
        "base_url": "https://api.deepseek.com/v1"
    }
]


@pytest.fixture
def mock_llm_config():
    """Configuration LLM mockée pour AutoGen compatible DeepSeek"""
    return {
        "config_list": OAI_CONFIG_LIST_MOCK,
        "temperature": 0.1,
        "cache_seed": 42,
        "timeout": 30,
        "response_format": {"type": "json_object"},
        "model": "deepseek-chat"
    }


@pytest.fixture
def mock_deepseek_client_autogen():
    """Client DeepSeek mocké pour agents AutoGen"""
    client = Mock()
    client.model = "deepseek-chat"
    client.api_key = "mock_deepseek_key"
    client.base_url = "https://api.deepseek.com/v1"
    client.timeout = 30
    
    # Mock méthodes client
    client.create_completion = AsyncMock()
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    
    return client


@pytest.fixture
async def intent_classifier_agent(mock_llm_config, test_cache_manager, mock_metrics_collector):
    """Agent Intent Classifier pour tests AutoGen"""
    # Import dynamique pour éviter dépendances circulaires
    from conversation_service.agents.financial.intent_classifier import IntentClassifierAgent
    
    agent = IntentClassifierAgent("test_intent_classifier")
    agent.llm_config = mock_llm_config
    agent.cache_manager = test_cache_manager
    agent.metrics_collector = mock_metrics_collector
    
    return agent


@pytest.fixture
async def entity_extractor_agent(mock_llm_config, test_cache_manager, mock_metrics_collector):
    """Agent Entity Extractor pour tests AutoGen"""
    from conversation_service.agents.financial.entity_extractor import EntityExtractorAgent
    
    agent = EntityExtractorAgent("test_entity_extractor")
    agent.llm_config = mock_llm_config
    agent.cache_manager = test_cache_manager
    agent.metrics_collector = mock_metrics_collector
    
    return agent


@pytest.fixture
def sample_team_context():
    """Contexte équipe standard pour tests"""
    return {
        "user_message": "Mes achats Amazon plus de 50€ ce mois",
        "user_id": 123,
        "intent_result": {
            "intent_type": "SEARCH_BY_MERCHANT",
            "confidence": 0.95,
            "reasoning": "Message indique recherche par marchand spécifique",
            "suggested_entities_focus": {
                "priority_entities": ["merchants", "amounts"]
            }
        },
        "processing_timestamp": datetime.utcnow().isoformat(),
        "session_id": "test_session_001"
    }


@pytest.fixture
def mock_intent_response():
    """Réponse intention mockée standard"""
    return {
        "intent_type": "SEARCH_BY_MERCHANT",
        "confidence": 0.95,
        "reasoning": "Classification réussie - recherche par marchand Amazon détectée",
        "team_context": {
            "suggested_entities_focus": {
                "priority_entities": ["merchants", "amounts"],
                "context_hints": ["e-commerce", "spending_threshold"]
            },
            "processing_notes": "Confiance élevée, extraction entités recommandée"
        },
        "processing_metadata": {
            "agent_id": "intent_classifier",
            "processing_time_ms": 850,
            "model_version": "phase2"
        }
    }


@pytest.fixture
def mock_entity_response():
    """Réponse extraction entités mockée standard"""
    return {
        "extraction_success": True,
        "entities": {
            "amounts": [{
                "value": 50.0,
                "currency": "EUR",
                "operator": "gt",
                "raw_text": "plus de 50€",
                "confidence": 0.96,
                "extraction_method": "llm"
            }],
            "merchants": [{
                "name": "Amazon",
                "normalized": "Amazon",
                "confidence": 0.98,
                "category_hint": "e-commerce",
                "known_merchant": True
            }],
            "dates": [{
                "type": "relative",
                "raw_text": "ce mois",
                "parsed_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31"
                },
                "confidence": 0.92
            }]
        },
        "overall_confidence": 0.95,
        "entities_count": 3,
        "extraction_metadata": {
            "agent_id": "entity_extractor",
            "processing_time_ms": 1200,
            "extraction_method": "multi_agent_autogen"
        }
    }


@pytest.fixture
def mock_team_results():
    """Résultats équipe mockés standards"""
    return {
        "workflow_phase": "phase2_intent_entity_complete",
        "workflow_success": True,
        "intent_result": {
            "intent_type": "SEARCH_BY_MERCHANT",
            "confidence": 0.95,
            "reasoning": "Classification réussie",
            "processing_time_ms": 850
        },
        "entities_result": {
            "extraction_success": True,
            "entities": {
                "amounts": [{"value": 50.0, "operator": "gt", "confidence": 0.96}],
                "merchants": [{"normalized": "Amazon", "confidence": 0.98}],
                "dates": [{"type": "relative", "raw_text": "ce mois", "confidence": 0.92}]
            },
            "overall_confidence": 0.95,
            "entities_count": 3,
            "processing_time_ms": 1200
        },
        "coherence_validation": {
            "score": 0.89,
            "threshold_met": True,
            "validation_details": {
                "intent_entity_alignment": 0.92,
                "business_rules_compliance": 0.87
            }
        },
        "agents_sequence": ["intent_classifier", "entity_extractor"],
        "total_processing_time_ms": 2050,
        "cache_hit": False,
        "fallback_applied": False
    }


@pytest.fixture
def mock_team_state():
    """État équipe AutoGen mocké pour tests"""
    from conversation_service.models.autogen.team_models import create_financial_team_state
    
    team_state = create_financial_team_state("Test Financial Team")
    
    # Initialiser avec métriques de test
    for agent_id, performance in team_state.agent_performances.items():
        performance.update_performance(1000, True, 0.9)
        performance.update_performance(1200, True, 0.85)
        performance.update_performance(950, True, 0.92)
    
    return team_state


@pytest.fixture
def sample_comprehensive_entities():
    """Extraction entités comprehensive pour tests"""
    return ComprehensiveEntityExtraction(
        user_message="Mes achats Amazon plus de 50€ ce mois",
        amounts=[
            ExtractedAmount(
                value=50.0,
                currency="EUR",
                original_text="plus de 50€",
                confidence=0.96,
                extraction_method="llm"
            )
        ],
        merchants=[
            ExtractedMerchant(
                name="Amazon",
                original_text="Amazon",
                confidence=0.98,
                known_merchant=True,
                category_hint="e-commerce"
            )
        ],
        date_ranges=[
            ExtractedDateRange(
                original_text="ce mois",
                confidence=0.92,
                period_type="relative_period",
                relative_period="last_month"
            )
        ],
        overall_confidence=0.95,
        extraction_method="multi_agent_autogen",
        processing_time_ms=1200,
        entities_found=True,
        high_confidence_entities=3
    )


@pytest.fixture
def sample_entity_validation():
    """Résultat validation entités pour tests"""
    return EntityValidationResult(
        coherence_score=0.89,
        completeness_score=0.92,
        business_logic_score=0.87,
        entities_coherent=True,
        extraction_complete=True,
        business_rules_respected=True,
        validation_issues=[],
        validation_warnings=["Date range could be more specific"],
        recommended_actions=["Consider asking for specific date range"],
        confidence_adjustment=0.05
    )


@pytest.fixture
def sample_multi_agent_insights():
    """Insights traitement multi-agents pour tests"""
    return MultiAgentProcessingInsights(
        agents_coordination={
            "intent_classifier": {"status": "completed", "contribution_score": 0.95},
            "entity_extractor": {"status": "completed", "contribution_score": 0.92}
        },
        workflow_steps=[
            {"step": "intent_classification", "agent": "intent_classifier", "duration_ms": 850, "success": True},
            {"step": "entity_extraction", "agent": "entity_extractor", "duration_ms": 1200, "success": True}
        ],
        intent_agent_performance={
            "confidence": 0.95,
            "processing_time_ms": 850,
            "reasoning_quality": "high"
        },
        entity_agent_performance={
            "entities_found": 3,
            "confidence": 0.95,
            "processing_time_ms": 1200,
            "extraction_completeness": 0.92
        },
        orchestrator_performance={
            "workflow_efficiency": 0.88,
            "coordination_overhead_ms": 150
        },
        consensus_reached=True,
        iteration_count=1,
        convergence_achieved=True,
        intent_processing_time_ms=850,
        entity_processing_time_ms=1200,
        coordination_overhead_ms=150
    )


@pytest.fixture
async def mock_financial_team_phase2(
    mock_deepseek_client_autogen,
    test_cache_manager,
    mock_metrics_collector,
    mock_team_state
):
    """Équipe financière AutoGen Phase 2 mockée"""
    # Import dynamique
    from conversation_service.teams.multi_agent_financial_team import MultiAgentFinancialTeam
    
    team = MultiAgentFinancialTeam(
        deepseek_client=mock_deepseek_client_autogen,
        cache_manager=test_cache_manager,
        metrics_collector=mock_metrics_collector
    )
    
    # État mocké
    team.team_state = mock_team_state
    
    return team


# Fixtures utilitaires tests

@pytest.fixture
def assert_valid_json_response():
    """Helper validation réponse JSON (réutilise pattern Phase 1)"""
    def _assert_valid_json(response_data):
        """Valide structure JSON réponse"""
        assert isinstance(response_data, dict)
        assert len(response_data) > 0
        
        # Validation champs obligatoires selon contexte
        if "extraction_success" in response_data:
            assert isinstance(response_data["extraction_success"], bool)
        
        if "confidence" in response_data:
            confidence = response_data["confidence"]
            assert isinstance(confidence, (int, float))
            assert 0.0 <= confidence <= 1.0
    
    return _assert_valid_json


@pytest.fixture
def create_mock_conversation_request():
    """Factory requête conversation mockée (cohérent Phase 1)"""
    def _create_request(message: str, user_id: int = 123) -> Dict[str, Any]:
        return {
            "message": message,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": f"test_session_{user_id}",
            "request_id": f"req_{user_id}_{int(datetime.utcnow().timestamp())}"
        }
    
    return _create_request


@pytest.fixture
def performance_benchmark():
    """Helper benchmark performance (extension Phase 1)"""
    def _benchmark(target_time_ms: int = 2000):
        """Decorator benchmark temps exécution"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = datetime.utcnow()
                result = await func(*args, **kwargs)
                end_time = datetime.utcnow()
                
                execution_time_ms = (end_time - start_time).total_seconds() * 1000
                
                # Assertion performance
                assert execution_time_ms < target_time_ms, \
                    f"Performance dégradée: {execution_time_ms}ms > {target_time_ms}ms"
                
                return result
            return wrapper
        return decorator
    
    return _benchmark


# Markers pytest cohérents Phase 1

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup environnement test AutoGen (auto-applied)"""
    
    # Configuration environnement test
    import os
    os.environ["AUTOGEN_TEST_MODE"] = "true"
    os.environ["DEEPSEEK_API_KEY"] = "mock_test_key"
    os.environ["DISABLE_LLM_CALLS"] = "true"
    
    yield
    
    # Cleanup
    test_env_vars = ["AUTOGEN_TEST_MODE", "DEEPSEEK_API_KEY", "DISABLE_LLM_CALLS"]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


# Fixtures scope session pour performance

@pytest.fixture(scope="session")
def mock_autogen_runtime_session():
    """Runtime AutoGen mocké réutilisé pour session"""
    runtime = Mock()
    runtime.is_initialized = True
    runtime.teams_loaded = ["financial_analysis_phase2"]
    runtime.health_status = "healthy"
    
    runtime.get_team = AsyncMock()
    runtime.health_check = AsyncMock(return_value={
        "runtime_initialized": True,
        "teams_loaded": ["financial_analysis_phase2"],
        "overall_status": "healthy"
    })
    
    return runtime


# Configuration pytest markers

pytestmark = [
    pytest.mark.asyncio,  # Tous tests AutoGen sont async
    pytest.mark.autogen   # Marker spécifique AutoGen
]