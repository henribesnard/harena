"""
🧪 Configuration Pytest Principale - Search Service
===================================================

Configuration centralisée pytest pour tous les tests Search Service.
Fixtures globales, configuration environnement, utilitaires communs.

Setup:
- Variables environnement test
- Fixtures modèles communs
- Configuration logging
- Mocks clients externes
"""

import pytest
import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH pour les imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration variables environnement pour tests
os.environ.update({
    "TESTING": "true",
    "ENVIRONMENT": "test",
    "DEBUG": "true",
    "LOG_LEVEL": "DEBUG",
    # Désactiver services externes par défaut
    "ELASTICSEARCH_HOST": "localhost",
    "ELASTICSEARCH_PORT": "9200",
    "REDIS_ENABLED": "false",
    "MOCK_ELASTICSEARCH": "true",
})

# Imports des modules après configuration environnement
try:
    from search_service.models import (
        SearchServiceQuery, SearchServiceResponse,
        QueryMetadata, SearchParameters, FilterGroup,
        TransactionResult, ResponseMetadata, PerformanceMetrics,
        SimpleLexicalSearchRequest, BaseResponse,
        FilterBuilder, AmountFilter, DateFilter,
        ElasticsearchQuery, ESQueryClause, ESQueryType,
        IntentType, FilterOperator, ResponseStatus
    )
    
    # Configuration disponible
    try:
        from config_service.search_config import SearchServiceConfig
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False
        
except ImportError as e:
    pytest.skip(f"Cannot import search_service modules: {e}", allow_module_level=True)

from datetime import datetime, date
from typing import Dict, Any, List
from unittest.mock import MagicMock


# =============================================================================
# 🧪 CONFIGURATION PYTEST
# =============================================================================

def pytest_configure(config):
    """Configuration pytest au démarrage."""
    # Markers personnalisés
    config.addinivalue_line("markers", "unit: Tests unitaires rapides")
    config.addinivalue_line("markers", "integration: Tests intégration")
    config.addinivalue_line("markers", "contracts: Tests contrats interface")
    config.addinivalue_line("markers", "models: Tests modèles Pydantic")
    config.addinivalue_line("markers", "performance: Tests performance")

def pytest_collection_modifyitems(config, items):
    """Modification automatique items selon nom fichier."""
    for item in items:
        # Auto-marker selon nom fichier
        if "test_models" in item.nodeid:
            item.add_marker(pytest.mark.models)
        if "test_contracts" in item.nodeid:
            item.add_marker(pytest.mark.contracts)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)


# =============================================================================
# 🔧 FIXTURES CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def test_environment():
    """Configuration environnement test."""
    return {
        "TESTING": True,
        "ENVIRONMENT": "test",
        "DEBUG": True,
        "MOCK_ELASTICSEARCH": True,
        "REDIS_ENABLED": False,
    }

@pytest.fixture
def search_config():
    """Configuration Search Service pour tests."""
    if not CONFIG_AVAILABLE:
        pytest.skip("SearchServiceConfig not available")
    
    return SearchServiceConfig(
        ENVIRONMENT="test",
        DEBUG=True,
        TESTING_MODE=True,
        MOCK_ELASTICSEARCH=True,
        REDIS_ENABLED=False,
        MAX_SEARCH_RESULTS=100,
        DEFAULT_SEARCH_LIMIT=10,
        DEFAULT_SEARCH_TIMEOUT=1000,
    )


# =============================================================================
# 📋 FIXTURES DONNÉES TEST
# =============================================================================

@pytest.fixture
def sample_user_id():
    """ID utilisateur test standard."""
    return 34

@pytest.fixture
def sample_transaction():
    """Transaction exemple pour tests."""
    return {
        "transaction_id": "user_34_tx_12345",
        "user_id": 34,
        "account_id": 101,
        "amount": -45.67,
        "amount_abs": 45.67,
        "transaction_type": "debit",
        "currency_code": "EUR",
        "date": "2024-01-15",
        "primary_description": "RESTAURANT LE BISTROT",
        "merchant_name": "Le Bistrot",
        "category_name": "restaurant",
        "operation_type": "card_payment",
        "month_year": "2024-01",
        "weekday": "Monday",
        "searchable_text": "restaurant le bistrot cuisine française"
    }

@pytest.fixture
def sample_transaction_result(sample_transaction):
    """TransactionResult valide pour tests."""
    return TransactionResult(
        transaction_id=sample_transaction["transaction_id"],
        user_id=sample_transaction["user_id"],
        amount=sample_transaction["amount"],
        amount_abs=sample_transaction["amount_abs"],
        transaction_type=sample_transaction["transaction_type"],
        currency_code=sample_transaction["currency_code"],
        date=sample_transaction["date"],
        primary_description=sample_transaction["primary_description"],
        merchant_name=sample_transaction["merchant_name"],
        category_name=sample_transaction["category_name"],
        score=0.95
    )

@pytest.fixture
def sample_transactions_list():
    """Liste transactions pour tests."""
    return [
        {
            "transaction_id": "user_34_tx_001",
            "user_id": 34,
            "amount": -25.50,
            "amount_abs": 25.50,
            "category_name": "restaurant",
            "merchant_name": "PIZZA ROMA",
            "primary_description": "PIZZA ROMA",
            "date": "2024-01-10",
            "transaction_type": "debit",
            "currency_code": "EUR",
            "score": 0.92
        },
        {
            "transaction_id": "user_34_tx_002", 
            "user_id": 34,
            "amount": -67.80,
            "amount_abs": 67.80,
            "category_name": "transport",
            "merchant_name": "SNCF CONNECT",
            "primary_description": "SNCF CONNECT TRAIN",
            "date": "2024-01-12",
            "transaction_type": "debit",
            "currency_code": "EUR",
            "score": 0.88
        }
    ]


# =============================================================================
# 🤝 FIXTURES CONTRATS
# =============================================================================

@pytest.fixture
def sample_query_metadata(sample_user_id):
    """QueryMetadata valide pour tests."""
    return QueryMetadata(
        user_id=sample_user_id,
        intent_type=IntentType.SEARCH_BY_CATEGORY,
        confidence=0.95,
        agent_name="test_agent",
        team_name="test_team"
    )

@pytest.fixture
def sample_search_parameters():
    """SearchParameters valides pour tests."""
    return SearchParameters(
        query_type="filtered_search",
        fields=["searchable_text", "primary_description", "merchant_name"],
        limit=20,
        timeout_ms=5000
    )

@pytest.fixture
def sample_filter_group(sample_user_id):
    """FilterGroup valide pour tests."""
    return FilterGroup(
        required=[
            {"field": "user_id", "operator": FilterOperator.EQ, "value": sample_user_id},
            {"field": "category_name", "operator": FilterOperator.EQ, "value": "restaurant"}
        ]
    )

@pytest.fixture
def sample_search_service_query(sample_query_metadata, sample_search_parameters, sample_filter_group):
    """SearchServiceQuery complet pour tests."""
    return SearchServiceQuery(
        query_metadata=sample_query_metadata,
        search_parameters=sample_search_parameters,
        filters=sample_filter_group
    )

@pytest.fixture
def sample_response_metadata():
    """ResponseMetadata valide pour tests."""
    return ResponseMetadata(
        query_id="test-query-123",
        execution_time_ms=45,
        total_hits=10,
        returned_hits=2,
        has_more=False,
        cache_hit=False,
        elasticsearch_took=30
    )

@pytest.fixture
def sample_performance_metrics():
    """PerformanceMetrics valides pour tests."""
    return PerformanceMetrics(
        query_complexity="simple",
        optimization_applied=["user_filter", "category_filter"],
        index_used="harena_transactions",
        shards_queried=1
    )

@pytest.fixture
def sample_search_service_response(sample_response_metadata, sample_performance_metrics, sample_transactions_list):
    """SearchServiceResponse complet pour tests."""
    # Convertir transactions en TransactionResult
    results = []
    for tx_data in sample_transactions_list:
        results.append(TransactionResult(**tx_data))
    
    return SearchServiceResponse(
        response_metadata=sample_response_metadata,
        results=results,
        performance=sample_performance_metrics,
        context_enrichment={
            "search_intent_matched": True,
            "result_quality_score": 0.95,
            "suggested_followup_questions": []
        }
    )


# =============================================================================
# 📥📤 FIXTURES REQUÊTES/RÉPONSES
# =============================================================================

@pytest.fixture
def sample_simple_search_request(sample_user_id):
    """SimpleLexicalSearchRequest pour tests."""
    return SimpleLexicalSearchRequest(
        query="restaurant italien",
        user_id=sample_user_id,
        fields=["searchable_text", "primary_description"],
        limit=20
    )

@pytest.fixture
def sample_base_response():
    """BaseResponse pour tests."""
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message="Test operation completed",
        execution_time_ms=150
    )


# =============================================================================
# 🔧 FIXTURES FILTRES
# =============================================================================

@pytest.fixture
def sample_amount_filter():
    """AmountFilter pour tests."""
    from decimal import Decimal
    return AmountFilter(
        amount_type="absolute",
        min_amount=Decimal("10.0"),
        max_amount=Decimal("100.0")
    )

@pytest.fixture
def sample_date_filter():
    """DateFilter pour tests."""
    return DateFilter(
        filter_type="date_range",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31)
    )

@pytest.fixture
def sample_composite_filter(sample_user_id):
    """CompositeFilter complet pour tests."""
    return (FilterBuilder(sample_user_id)
            .with_categories(["restaurant"])
            .with_amount_range(10.0, 100.0)
            .build())


# =============================================================================
# 🔍 FIXTURES ELASTICSEARCH
# =============================================================================

@pytest.fixture
def sample_es_query_clause():
    """ESQueryClause pour tests."""
    return ESQueryClause(
        query_type=ESQueryType.TERM,
        field="user_id",
        value=34
    )

@pytest.fixture
def sample_elasticsearch_query():
    """ElasticsearchQuery pour tests."""
    return ElasticsearchQuery(
        query=ESQueryClause(query_type=ESQueryType.MATCH_ALL),
        size=20,
        from_=0
    )


# =============================================================================
# 🎭 FIXTURES MOCKS
# =============================================================================

@pytest.fixture
def mock_elasticsearch_response():
    """Mock réponse Elasticsearch."""
    return {
        "took": 15,
        "timed_out": False,
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "max_score": 1.0,
            "hits": [
                {
                    "_index": "harena_transactions",
                    "_id": "user_34_tx_001",
                    "_score": 0.92,
                    "_source": {
                        "transaction_id": "user_34_tx_001",
                        "user_id": 34,
                        "amount": -25.50,
                        "category_name": "restaurant"
                    }
                }
            ]
        }
    }

@pytest.fixture
def mock_elasticsearch_client():
    """Mock client Elasticsearch."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.search.return_value = {
        "took": 5,
        "hits": {"total": {"value": 0}, "hits": []}
    }
    return mock_client


# =============================================================================
# 🛠️ FIXTURES UTILITAIRES
# =============================================================================

@pytest.fixture
def test_data_factory():
    """Factory pour créer données test."""
    class TestDataFactory:
        @staticmethod
        def create_transaction(user_id: int = 34, **overrides) -> Dict[str, Any]:
            base_data = {
                "transaction_id": f"user_{user_id}_tx_test",
                "user_id": user_id,
                "amount": -50.0,
                "amount_abs": 50.0,
                "transaction_type": "debit",
                "currency_code": "EUR",
                "date": "2024-01-15",
                "primary_description": "TEST TRANSACTION",
                "score": 0.9
            }
            base_data.update(overrides)
            return base_data
        
        @staticmethod
        def create_search_query(user_id: int = 34, **overrides) -> SearchServiceQuery:
            base_query = {
                "query_metadata": {
                    "user_id": user_id,
                    "intent_type": IntentType.TEXT_SEARCH,
                    "confidence": 0.9,
                    "agent_name": "test_agent"
                },
                "search_parameters": {
                    "query_type": "simple_search",
                    "fields": ["searchable_text"],
                    "limit": 10
                },
                "filters": {
                    "required": [
                        {"field": "user_id", "operator": FilterOperator.EQ, "value": user_id}
                    ]
                }
            }
            # Merger overrides de manière récursive si nécessaire
            return SearchServiceQuery.parse_obj(base_query)
    
    return TestDataFactory()

@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Nettoyage automatique après chaque test."""
    yield
    # Nettoyage variables d'environnement ajoutées par tests
    test_vars = [k for k in os.environ.keys() if k.startswith("TEST_")]
    for var in test_vars:
        os.environ.pop(var, None)


# =============================================================================
# 🧹 FINALISATION
# =============================================================================

def pytest_runtest_setup(item):
    """Setup avant chaque test."""
    # Marquer début test pour debugging
    print(f"\n🧪 Running test: {item.name}")

def pytest_runtest_teardown(item, nextitem):
    """Nettoyage après chaque test."""
    # Cleanup spécifique si nécessaire
    pass