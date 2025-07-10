"""
Configuration pytest pour les tests Search Service.

Ce fichier contient :
- Configuration globale pytest
- Fixtures partagées
- Markers personnalisés
- Setup et teardown globaux
- Mocks communs
"""

import pytest
import logging
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Generator, AsyncGenerator

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration du logging pour les tests
logging.getLogger("search_service").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ==================== CONFIGURATION PYTEST ====================

def pytest_configure(config):
    """Configuration globale pytest."""
    # Enregistrer les markers personnalisés
    config.addinivalue_line(
        "markers", "unit: marque les tests unitaires"
    )
    config.addinivalue_line(
        "markers", "integration: marque les tests d'intégration"
    )
    config.addinivalue_line(
        "markers", "security: marque les tests de sécurité"
    )
    config.addinivalue_line(
        "markers", "performance: marque les tests de performance"
    )
    config.addinivalue_line(
        "markers", "slow: marque les tests lents"
    )
    
    # Variables d'environnement pour les tests
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'WARNING'


def pytest_collection_modifyitems(config, items):
    """Modifie la collection de tests."""
    # Ajouter le marker 'unit' par défaut si aucun marker spécifique
    for item in items:
        if not any(mark.name in ['integration', 'security', 'performance', 'slow'] 
                  for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# ==================== FIXTURES GLOBALES ====================

@pytest.fixture(scope="session")
def event_loop():
    """Fixture pour la boucle d'événements asyncio."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Configuration de test globale."""
    return {
        "TESTING": True,
        "LOG_LEVEL": "WARNING",
        "ELASTICSEARCH_HOST": "localhost",
        "ELASTICSEARCH_PORT": 9200,
        "ELASTICSEARCH_TIMEOUT": 5,
        "SEARCH_CACHE_SIZE": 100,
        "SEARCH_CACHE_TTL": 30,
        "PROJECT_NAME": "Test Search Service",
        "API_V1_STR": "/api/v1",
        "CORS_ORIGINS": "*"
    }


@pytest.fixture
def mock_settings(test_config):
    """Mock des settings pour les tests."""
    mock_settings = Mock()
    for key, value in test_config.items():
        setattr(mock_settings, key, value)
    return mock_settings


# ==================== FIXTURES ELASTICSEARCH ====================

@pytest.fixture
def mock_elasticsearch_client():
    """Mock du client Elasticsearch."""
    client = AsyncMock()
    
    # Configuration des méthodes principales
    client.connect.return_value = None
    client.close.return_value = None
    client.health_check.return_value = {
        "status": "healthy",
        "cluster_name": "test-cluster"
    }
    
    # Mock des réponses de recherche
    client.search.return_value = {
        "took": 5,
        "hits": {
            "total": {"value": 0, "relation": "eq"},
            "hits": []
        }
    }
    
    # Mock des informations cluster
    client.info.return_value = {
        "version": {"number": "7.10.0"},
        "cluster_name": "test-cluster"
    }
    
    return client


@pytest.fixture
def elasticsearch_response():
    """Réponse Elasticsearch type pour les tests."""
    return {
        "took": 10,
        "timed_out": False,
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "max_score": 1.0,
            "hits": [
                {
                    "_id": "user_12345_tx_1",
                    "_score": 1.0,
                    "_source": {
                        "transaction_id": "user_12345_tx_1",
                        "user_id": 12345,
                        "amount": -45.67,
                        "merchant_name": "Restaurant Le Gourmet",
                        "clean_description": "Payment at restaurant",
                        "category_name": "Restaurant",
                        "transaction_date": "2024-01-15T12:30:00Z"
                    }
                },
                {
                    "_id": "user_12345_tx_2", 
                    "_score": 0.8,
                    "_source": {
                        "transaction_id": "user_12345_tx_2",
                        "user_id": 12345,
                        "amount": -23.45,
                        "merchant_name": "Cafe Central",
                        "clean_description": "Coffee purchase",
                        "category_name": "Restaurant",
                        "transaction_date": "2024-01-16T08:15:00Z"
                    }
                }
            ]
        }
    }


# ==================== FIXTURES CORE ====================

@pytest.fixture
def mock_lexical_engine():
    """Mock du moteur de recherche lexicale."""
    engine = AsyncMock()
    
    engine.search.return_value = {
        "results": [],
        "total": 0,
        "took": 10,
        "query_id": "test-query-123"
    }
    
    engine.close.return_value = None
    
    return engine


@pytest.fixture
def mock_query_executor():
    """Mock de l'exécuteur de requêtes."""
    executor = AsyncMock()
    
    executor.execute.return_value = {
        "hits": [],
        "total": 0,
        "took": 5
    }
    
    return executor


@pytest.fixture
def mock_result_processor():
    """Mock du processeur de résultats."""
    processor = Mock()
    
    processor.process.return_value = {
        "results": [],
        "total": 0,
        "metadata": {}
    }
    
    return processor


# ==================== FIXTURES CACHE ====================

@pytest.fixture
def mock_cache():
    """Mock du cache LRU."""
    cache = AsyncMock()
    
    cache.get.return_value = None  # Cache miss par défaut
    cache.set.return_value = None
    cache.clear.return_value = None
    cache.stats.return_value = {
        "hits": 0,
        "misses": 0,
        "size": 0
    }
    
    return cache


# ==================== FIXTURES MÉTRIQUES ====================

@pytest.fixture
def mock_metrics_collector():
    """Mock du collecteur de métriques."""
    collector = Mock()
    
    collector.record_query.return_value = None
    collector.record_cache_hit.return_value = None
    collector.record_cache_miss.return_value = None
    collector.get_stats.return_value = {
        "queries_total": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "avg_response_time": 0
    }
    
    return collector


# ==================== FIXTURES REQUÊTES ====================

@pytest.fixture
def sample_search_request():
    """Requête de recherche type pour les tests."""
    return {
        "query": "restaurant",
        "user_id": 12345,
        "filters": {
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-01-31"
            },
            "amount_range": {
                "min": -100.0,
                "max": -10.0
            },
            "categories": ["Restaurant", "Food"]
        },
        "options": {
            "size": 20,
            "sort": "date_desc",
            "highlight": True
        }
    }


@pytest.fixture
def sample_user_data():
    """Données utilisateur type pour les tests."""
    return {
        "user_id": 12345,
        "username": "test_user",
        "email": "test@example.com",
        "permissions": ["search", "read"],
        "preferences": {
            "default_limit": 20,
            "sort_preference": "date_desc"
        }
    }


# ==================== FIXTURES FASTAPI ====================

@pytest.fixture
def test_app():
    """Application FastAPI de test."""
    # Import conditionnel pour éviter les erreurs si modules manquants
    try:
        from search_service.main import create_app
        return create_app()
    except ImportError:
        # Créer une app minimale pour les tests
        from fastapi import FastAPI
        app = FastAPI(title="Test App")
        
        @app.get("/health")
        def health():
            return {"status": "ok"}
        
        return app


@pytest.fixture
def test_client(test_app):
    """Client de test FastAPI."""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


# ==================== FIXTURES UTILITAIRES ====================

@pytest.fixture
def temp_directory(tmp_path):
    """Répertoire temporaire pour les tests."""
    return tmp_path


@pytest.fixture
def sample_transactions():
    """Transactions d'exemple pour les tests."""
    return [
        {
            "transaction_id": "user_12345_tx_1",
            "user_id": 12345,
            "amount": -45.67,
            "merchant_name": "Restaurant Le Gourmet",
            "clean_description": "Payment at restaurant",
            "category_name": "Restaurant",
            "transaction_date": "2024-01-15T12:30:00Z",
            "searchable_text": "restaurant le gourmet payment"
        },
        {
            "transaction_id": "user_12345_tx_2",
            "user_id": 12345,
            "amount": -23.45,
            "merchant_name": "Super Market",
            "clean_description": "Grocery shopping",
            "category_name": "Groceries",
            "transaction_date": "2024-01-16T14:20:00Z",
            "searchable_text": "super market grocery shopping"
        },
        {
            "transaction_id": "user_12345_tx_3",
            "user_id": 12345,
            "amount": -89.12,
            "merchant_name": "Gas Station",
            "clean_description": "Fuel purchase",
            "category_name": "Transport",
            "transaction_date": "2024-01-17T09:45:00Z",
            "searchable_text": "gas station fuel purchase"
        }
    ]


# ==================== FIXTURES CONTEXTUELLES ====================

@pytest.fixture
def with_mocked_dependencies():
    """Context manager pour mocker toutes les dépendances."""
    with patch('search_service.CONFIG_AVAILABLE', True), \
         patch('search_service.CORE_AVAILABLE', True), \
         patch('search_service.API_AVAILABLE', True), \
         patch('search_service.MODELS_AVAILABLE', True), \
         patch('search_service.CLIENTS_AVAILABLE', True):
        yield


# ==================== HELPERS DE TEST ====================

class TestHelpers:
    """Helpers pour faciliter l'écriture de tests."""
    
    @staticmethod
    def create_mock_response(hits=None, total=0, took=10):
        """Crée une réponse Elasticsearch mockée."""
        if hits is None:
            hits = []
        
        return {
            "took": took,
            "timed_out": False,
            "hits": {
                "total": {"value": total, "relation": "eq"},
                "hits": hits
            }
        }
    
    @staticmethod
    def create_mock_transaction(transaction_id="test_tx_1", user_id=12345, **kwargs):
        """Crée une transaction mockée."""
        defaults = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "amount": -25.50,
            "merchant_name": "Test Merchant",
            "clean_description": "Test transaction",
            "category_name": "Test",
            "transaction_date": "2024-01-15T12:00:00Z"
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def assert_valid_search_response(response):
        """Valide qu'une réponse de recherche a la bonne structure."""
        assert isinstance(response, dict)
        assert "results" in response
        assert "total" in response
        assert isinstance(response["results"], list)
        assert isinstance(response["total"], int)


@pytest.fixture
def helpers():
    """Fixture pour les helpers de test."""
    return TestHelpers()


# ==================== CLEANUP ====================

@pytest.fixture(autouse=True)
def cleanup_environment():
    """Nettoyage automatique après chaque test."""
    # Setup
    original_env = os.environ.copy()
    
    yield
    
    # Cleanup
    # Restaurer les variables d'environnement
    os.environ.clear()
    os.environ.update(original_env)
    
    # Nettoyer les imports
    modules_to_remove = [
        module for module in sys.modules.keys() 
        if module.startswith('search_service') and 'test' not in module
    ]
    
    # Ne pas supprimer les modules pendant les tests pour éviter les erreurs
    # for module in modules_to_remove:
    #     if module in sys.modules:
    #         del sys.modules[module]


# ==================== MARKERS ET FILTRES ====================

def pytest_runtest_setup(item):
    """Setup avant chaque test."""
    # Skip les tests slow sauf si explicitement demandés
    if "slow" in item.keywords and not item.config.getoption("--runslow", False):
        pytest.skip("test nécessite --runslow pour s'exécuter")


def pytest_addoption(parser):
    """Ajouter des options de ligne de commande."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="exécuter les tests lents"
    )
    parser.addoption(
        "--integration", 
        action="store_true", 
        default=False, 
        help="exécuter les tests d'intégration"
    )


# ==================== LOGGING POUR LES TESTS ====================

@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure le logging pour les tests."""
    # Réduire le niveau de log pendant les tests
    loggers = [
        'search_service',
        'elasticsearch',
        'urllib3',
        'asyncio'
    ]
    
    original_levels = {}
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)
    
    yield
    
    # Restaurer les niveaux originaux
    for logger_name, level in original_levels.items():
        logging.getLogger(logger_name).setLevel(level)