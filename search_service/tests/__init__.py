"""
Tests unitaires pour Search Service

Suite de tests complète pour valider les modules config/, models/ et templates/
avant le développement des autres composants.

Architecture de tests:
- Test isolés par module avec mocks
- Validation des contrats d'interface 
- Tests de performance et sécurité
- Fixtures réutilisables et mocks
- Tests d'intégration entre modules

Modules testés:
- config/: Configuration et validation
- models/: Modèles Pydantic et contrats
- templates/: Templates Elasticsearch et agrégations
"""

import pytest
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import os

# Configuration des logs pour les tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ajout du path parent pour imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Constants pour les tests
TEST_DATA_DIR = current_dir / "test_data"
FIXTURES_DIR = current_dir / "fixtures"
MOCKS_DIR = current_dir / "mocks"

# Configuration de test globale
TEST_CONFIG = {
    "elasticsearch": {
        "mock_mode": True,
        "test_index": "test_harena_transactions",
        "test_host": "localhost:9200"
    },
    "cache": {
        "enabled": False,
        "test_backend": "memory"
    },
    "performance": {
        "max_response_time_ms": 100,
        "max_query_size": 1000,
        "max_aggregation_buckets": 1000
    },
    "security": {
        "test_user_id": 12345,
        "validate_user_isolation": True
    }
}

# Fixtures communes pour tous les tests
@pytest.fixture
def test_config():
    """Configuration de test globale"""
    return TEST_CONFIG

@pytest.fixture
def sample_user_id():
    """ID utilisateur pour les tests"""
    return TEST_CONFIG["security"]["test_user_id"]

@pytest.fixture
def sample_transaction_data():
    """Données de transaction pour les tests"""
    return {
        "transaction_id": "user_12345_tx_67890",
        "user_id": 12345,
        "account_id": 101,
        "amount": -45.67,
        "amount_abs": 45.67,
        "transaction_type": "debit",
        "currency_code": "EUR",
        "date": "2024-01-15",
        "primary_description": "RESTAURANT LE BISTROT",
        "merchant_name": "Le Bistrot",
        "category_name": "Restaurant",
        "operation_type": "card_payment",
        "month_year": "2024-01",
        "weekday": "Monday",
        "searchable_text": "restaurant le bistrot italian cuisine"
    }

@pytest.fixture
def sample_elasticsearch_response():
    """Réponse Elasticsearch simulée"""
    return {
        "took": 23,
        "timed_out": False,
        "hits": {
            "total": {"value": 156, "relation": "eq"},
            "max_score": 1.0,
            "hits": [
                {
                    "_index": "harena_transactions",
                    "_id": "user_12345_tx_67890",
                    "_score": 1.0,
                    "_source": {
                        "transaction_id": "user_12345_tx_67890",
                        "user_id": 12345,
                        "amount": -45.67,
                        "amount_abs": 45.67,
                        "category_name": "Restaurant",
                        "merchant_name": "Le Bistrot",
                        "date": "2024-01-15"
                    }
                }
            ]
        },
        "aggregations": {
            "total_amount": {"value": 1247.89},
            "transaction_count": {"value": 156}
        }
    }

# Utilitaires de test
class TestHelpers:
    """Utilitaires pour les tests"""
    
    @staticmethod
    def create_test_query_metadata(**overrides):
        """Crée des métadonnées de requête pour tests"""
        base = {
            "query_id": "test-query-123",
            "user_id": TEST_CONFIG["security"]["test_user_id"],
            "intent_type": "SEARCH_BY_CATEGORY",
            "confidence": 0.95,
            "agent_name": "test_agent",
            "team_name": "test_team",
            "timestamp": "2024-01-15T10:30:00Z",
            "execution_context": {
                "conversation_id": "test_conv_123",
                "turn_number": 1,
                "agent_chain": ["intent_classifier", "query_generator"]
            }
        }
        base.update(overrides)
        return base
    
    @staticmethod
    def create_test_search_parameters(**overrides):
        """Crée des paramètres de recherche pour tests"""
        base = {
            "query_type": "filtered_search",
            "fields": ["user_id", "category_name", "merchant_name"],
            "limit": 20,
            "offset": 0,
            "timeout_ms": 5000
        }
        base.update(overrides)
        return base
    
    @staticmethod
    def create_test_filters(**overrides):
        """Crée des filtres pour tests"""
        base = {
            "required": [
                {"field": "user_id", "operator": "eq", "value": TEST_CONFIG["security"]["test_user_id"]}
            ],
            "optional": [],
            "ranges": [],
            "text_search": None
        }
        base.update(overrides)
        return base

# Marqueurs pytest personnalisés
def pytest_configure(config):
    """Configuration pytest personnalisée"""
    config.addinivalue_line(
        "markers", "unit: Tests unitaires rapides"
    )
    config.addinivalue_line(
        "markers", "integration: Tests d'intégration"
    )
    config.addinivalue_line(
        "markers", "performance: Tests de performance"
    )
    config.addinivalue_line(
        "markers", "security: Tests de sécurité"
    )
    config.addinivalue_line(
        "markers", "slow: Tests lents"
    )

# Hooks pytest pour logging
def pytest_runtest_setup(item):
    """Setup avant chaque test"""
    logging.info(f"🧪 Running test: {item.nodeid}")

def pytest_runtest_teardown(item):
    """Cleanup après chaque test"""
    logging.info(f"✅ Completed test: {item.nodeid}")

# Classes de base pour les tests
class BaseTestCase:
    """Classe de base pour tous les tests"""
    
    def setup_method(self):
        """Setup avant chaque méthode de test"""
        self.test_config = TEST_CONFIG
        self.helpers = TestHelpers()
    
    def teardown_method(self):
        """Cleanup après chaque méthode de test"""
        pass

class ConfigTestCase(BaseTestCase):
    """Classe de base pour les tests de configuration"""
    
    def setup_method(self):
        super().setup_method()
        # Import sécurisé des modules config
        try:
            from search_service.config import settings
            self.settings = settings
        except ImportError:
            self.settings = None

class ModelsTestCase(BaseTestCase):
    """Classe de base pour les tests de modèles"""
    
    def setup_method(self):
        super().setup_method()
        # Import sécurisé des modules models
        try:
            from search_service.models import service_contracts
            self.contracts = service_contracts
        except ImportError:
            self.contracts = None

class TemplatesTestCase(BaseTestCase):
    """Classe de base pour les tests de templates"""
    
    def setup_method(self):
        super().setup_method()
        # Import sécurisé des modules templates
        try:
            from search_service.templates import query_templates, aggregation_templates
            self.query_templates = query_templates
            self.aggregation_templates = aggregation_templates
        except ImportError:
            self.query_templates = None
            self.aggregation_templates = None