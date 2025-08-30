"""
Tests AutoGen Phase 2 - Suite complète
Tests unitaires et intégration pour agents et équipes multi-agents
Compatible infrastructure tests Phase 1
"""

# Version tests AutoGen
__version__ = "2.0.0"

# Imports principaux pour faciliter utilisation
from .conftest import (
    mock_llm_config,
    mock_deepseek_client_autogen,
    intent_classifier_agent,
    entity_extractor_agent,
    sample_team_context,
    mock_intent_response,
    mock_entity_response,
    mock_team_results,
    mock_financial_team_phase2
)

# Re-exports fixtures utiles
__all__ = [
    "mock_llm_config",
    "mock_deepseek_client_autogen", 
    "intent_classifier_agent",
    "entity_extractor_agent",
    "sample_team_context",
    "mock_intent_response",
    "mock_entity_response",
    "mock_team_results",
    "mock_financial_team_phase2"
]

# Configuration pytest markers
pytest_markers = {
    "autogen": "AutoGen specific tests",
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for team workflows",
    "api": "API endpoint tests",
    "performance": "Performance benchmark tests", 
    "regression": "Regression tests for Phase 1 compatibility"
}

# Documentation tests
TEST_CATEGORIES = {
    "entity_extractor": {
        "description": "Tests agent extraction entités",
        "file": "test_entity_extractor.py",
        "test_count": 15,
        "coverage_target": 95
    },
    "financial_team": {
        "description": "Tests équipe multi-agents workflow",
        "file": "test_financial_team_phase2.py", 
        "test_count": 12,
        "coverage_target": 90
    },
    "api_integration": {
        "description": "Tests intégration API endpoints",
        "file": "test_api_integration_phase2.py",
        "test_count": 18,
        "coverage_target": 85
    }
}

# Commands utiles pytest
PYTEST_COMMANDS = {
    "all_autogen": "pytest tests/autogen/ -v",
    "unit_only": "pytest tests/autogen/ -m unit -v",
    "integration_only": "pytest tests/autogen/ -m integration -v", 
    "api_only": "pytest tests/autogen/ -m api -v",
    "performance": "pytest tests/autogen/ -m performance --benchmark-only",
    "regression": "pytest tests/autogen/ -m regression -v",
    "coverage": "pytest tests/autogen/ --cov=conversation_service --cov-report=html",
    "fast": "pytest tests/autogen/ -x --ff"
}