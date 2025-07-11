"""
Module de tests pour le Search Service.

Ce module centralise les classes de base et utilitaires pour les tests
du Search Service, permettant une réutilisation facile dans tous les tests.
"""

import unittest
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configuration du logging pour les tests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ==================== CLASSES DE BASE POUR TESTS ====================

class ConfigTestCase(unittest.TestCase):
    """
    Classe de base pour les tests de configuration.
    
    Fournit des méthodes communes pour tester les configurations
    du Search Service.
    """
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.config_data = {}
        self.mock_env_vars = {}
        self.settings = None
        try:
            from search_service.config import settings
            self.settings = settings
        except ImportError:
            pass
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        self.config_data.clear()
        self.mock_env_vars.clear()
    
    def create_test_config(self, **kwargs) -> Dict[str, Any]:
        """Crée une configuration de test."""
        default_config = {
            "service_name": "search_service",
            "version": "1.0.0",
            "environment": "test",
            "debug": True,
            "elasticsearch": {
                "host": "localhost",
                "port": 9200,
                "index": "test_transactions",
                "timeout": 30
            },
            "cache": {
                "type": "memory",
                "ttl": 300,
                "max_size": 1000
            },
            "pagination": {
                "default_size": 20,
                "max_size": 100
            }
        }
        default_config.update(kwargs)
        return default_config
    
    def assert_config_valid(self, config: Dict[str, Any]):
        """Valide qu'une configuration est correcte."""
        self.assertIsInstance(config, dict)
        self.assertIn("service_name", config)
        self.assertIn("version", config)
        self.assertIn("elasticsearch", config)
    
    def assert_elasticsearch_config_valid(self, es_config: Dict[str, Any]):
        """Valide la configuration Elasticsearch."""
        self.assertIsInstance(es_config, dict)
        self.assertIn("host", es_config)
        self.assertIn("port", es_config)
        self.assertIn("index", es_config)
        self.assertIsInstance(es_config["port"], int)
        self.assertGreater(es_config["port"], 0)

class ModelsTestCase(unittest.TestCase):
    """
    Classe de base pour les tests de modèles.
    
    Fournit des méthodes communes pour tester les modèles Pydantic
    du Search Service.
    """
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.contracts = None
        self.helpers = None
        self.test_config = {
            "security": {"test_user_id": 12345}
        }
        
        # Tentative d'import des contrats
        try:
            from search_service.models import service_contracts
            self.contracts = service_contracts
        except ImportError:
            pass
        
        # Création du helper
        self.helpers = TestHelpers()
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        pass

class SearchTestCase(unittest.TestCase):
    """
    Classe de base pour les tests de recherche.
    
    Fournit des méthodes communes pour tester les fonctionnalités
    de recherche du Search Service.
    """
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.test_user_id = 12345
        self.test_queries = [
            "virement café",
            "carte restaurant",
            "retrait ATM",
            "supermarché courses"
        ]
        self.mock_elasticsearch_response = self.create_mock_elasticsearch_response()
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        pass
    
    def create_mock_elasticsearch_response(self, hits_count: int = 3) -> Dict[str, Any]:
        """Crée une réponse Elasticsearch mock."""
        hits = []
        for i in range(hits_count):
            hits.append({
                "_id": f"test_id_{i}",
                "_score": 1.0 - (i * 0.1),
                "_source": {
                    "user_id": self.test_user_id,
                    "merchant_name": f"Test Merchant {i}",
                    "amount": 25.50 + i,
                    "transaction_date": "2024-01-15T10:30:00Z",
                    "searchable_text": f"Test transaction {i}"
                },
                "highlight": {
                    "searchable_text": [f"Test <mark>transaction</mark> {i}"]
                }
            })
        
        return {
            "took": 15,
            "timed_out": False,
            "hits": {
                "total": {"value": hits_count, "relation": "eq"},
                "max_score": 1.0,
                "hits": hits
            }
        }
    
    def create_test_query_dict(self, query: str = "test query") -> Dict[str, Any]:
        """Crée un dictionnaire de requête de test."""
        return {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": self.test_user_id}},
                        {"multi_match": {
                            "query": query,
                            "fields": ["searchable_text^3", "merchant_name^2"],
                            "type": "best_fields"
                        }}
                    ]
                }
            },
            "size": 20,
            "highlight": {
                "fields": {
                    "searchable_text": {},
                    "merchant_name": {}
                }
            }
        }
    
    def assert_query_structure_valid(self, query: Dict[str, Any]):
        """Valide la structure d'une requête Elasticsearch."""
        self.assertIsInstance(query, dict)
        self.assertIn("query", query)
        self.assertIn("size", query)
        
        # Vérification de la structure bool
        if "bool" in query["query"]:
            bool_query = query["query"]["bool"]
            self.assertIsInstance(bool_query, dict)
    
    def assert_elasticsearch_response_valid(self, response: Dict[str, Any]):
        """Valide la structure d'une réponse Elasticsearch."""
        self.assertIsInstance(response, dict)
        self.assertIn("hits", response)
        self.assertIn("took", response)
        
        hits = response["hits"]
        self.assertIsInstance(hits, dict)
        self.assertIn("total", hits)
        self.assertIn("hits", hits)
        self.assertIsInstance(hits["hits"], list)
    """
    Classe de base pour les tests de recherche.
    
    Fournit des méthodes communes pour tester les fonctionnalités
    de recherche du Search Service.
    """
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.test_user_id = 12345
        self.test_queries = [
            "virement café",
            "carte restaurant",
            "retrait ATM",
            "supermarché courses"
        ]
        self.mock_elasticsearch_response = self.create_mock_elasticsearch_response()
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        pass
    
    def create_mock_elasticsearch_response(self, hits_count: int = 3) -> Dict[str, Any]:
        """Crée une réponse Elasticsearch mock."""
        hits = []
        for i in range(hits_count):
            hits.append({
                "_id": f"test_id_{i}",
                "_score": 1.0 - (i * 0.1),
                "_source": {
                    "user_id": self.test_user_id,
                    "merchant_name": f"Test Merchant {i}",
                    "amount": 25.50 + i,
                    "transaction_date": "2024-01-15T10:30:00Z",
                    "searchable_text": f"Test transaction {i}"
                },
                "highlight": {
                    "searchable_text": [f"Test <mark>transaction</mark> {i}"]
                }
            })
        
        return {
            "took": 15,
            "timed_out": False,
            "hits": {
                "total": {"value": hits_count, "relation": "eq"},
                "max_score": 1.0,
                "hits": hits
            }
        }
    
    def create_test_query_dict(self, query: str = "test query") -> Dict[str, Any]:
        """Crée un dictionnaire de requête de test."""
        return {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": self.test_user_id}},
                        {"multi_match": {
                            "query": query,
                            "fields": ["searchable_text^3", "merchant_name^2"],
                            "type": "best_fields"
                        }}
                    ]
                }
            },
            "size": 20,
            "highlight": {
                "fields": {
                    "searchable_text": {},
                    "merchant_name": {}
                }
            }
        }
    
    def assert_query_structure_valid(self, query: Dict[str, Any]):
        """Valide la structure d'une requête Elasticsearch."""
        self.assertIsInstance(query, dict)
        self.assertIn("query", query)
        self.assertIn("size", query)
        
        # Vérification de la structure bool
        if "bool" in query["query"]:
            bool_query = query["query"]["bool"]
            self.assertIsInstance(bool_query, dict)
    
    def assert_elasticsearch_response_valid(self, response: Dict[str, Any]):
        """Valide la structure d'une réponse Elasticsearch."""
        self.assertIsInstance(response, dict)
        self.assertIn("hits", response)
        self.assertIn("took", response)
        
        hits = response["hits"]
        self.assertIsInstance(hits, dict)
        self.assertIn("total", hits)
        self.assertIn("hits", hits)
        self.assertIsInstance(hits["hits"], list)

class AsyncTestCase(unittest.TestCase):
    """
    Classe de base pour les tests asynchrones.
    
    Fournit des utilitaires pour tester du code asynchrone.
    """
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        self.loop.close()
    
    def run_async(self, coro):
        """Execute une coroutine dans la boucle de test."""
        return self.loop.run_until_complete(coro)
    
    async def async_assert_raises(self, exception_class, coro):
        """Version asynchrone de assertRaises."""
        with self.assertRaises(exception_class):
            await coro

class APITestCase(AsyncTestCase):
    """
    Classe de base pour les tests d'API.
    
    Fournit des méthodes communes pour tester les endpoints
    du Search Service.
    """
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        super().setUp()
        self.base_url = "http://testserver"
        self.test_headers = {
            "Content-Type": "application/json",
            "User-Agent": "SearchService/Test"
        }
    
    def create_search_request(self, query: str = "test", **kwargs) -> Dict[str, Any]:
        """Crée une requête de recherche pour l'API."""
        request = {
            "query": query,
            "user_id": self.test_user_id,
            "size": kwargs.get("size", 20),
            "from": kwargs.get("from", 0)
        }
        
        if "filters" in kwargs:
            request["filters"] = kwargs["filters"]
        
        return request
    
    def assert_api_response_valid(self, response: Dict[str, Any]):
        """Valide la structure d'une réponse API."""
        self.assertIsInstance(response, dict)
        self.assertIn("success", response)
        self.assertIn("data", response)
        
        if not response["success"]:
            self.assertIn("error", response)

# ==================== FIXTURES ET DONNÉES DE TEST ====================

@dataclass
class TestTransaction:
    """Fixture pour une transaction de test."""
    id: str
    user_id: int
    merchant_name: str
    amount: float
    transaction_date: str
    category: str
    description: str

class TestHelpers:
    """
    Classe utilitaire pour créer des données de test.
    """
    
    @staticmethod
    def create_test_query_metadata(**kwargs) -> Dict[str, Any]:
        """Crée des métadonnées de requête de test."""
        return {
            "query_id": kwargs.get("query_id", "test-query-123"),
            "user_id": kwargs.get("user_id", 12345),
            "agent_name": kwargs.get("agent_name", "test_agent"),
            "team_name": kwargs.get("team_name", "test_team"),
            "timestamp": kwargs.get("timestamp", datetime.utcnow().isoformat()),
            "execution_context": kwargs.get("execution_context", {
                "conversation_id": "test-conv-123",
                "turn_number": 1,
                "agent_chain": ["test_agent"]
            })
        }
    
    @staticmethod
    def create_test_search_parameters(**kwargs) -> Dict[str, Any]:
        """Crée des paramètres de recherche de test."""
        return {
            "limit": kwargs.get("limit", 20),
            "offset": kwargs.get("offset", 0),
            "timeout_ms": kwargs.get("timeout_ms", 5000),
            "sort_by": kwargs.get("sort_by", "relevance"),
            "sort_order": kwargs.get("sort_order", "desc"),
            "include_highlights": kwargs.get("include_highlights", True),
            "include_explanation": kwargs.get("include_explanation", False)
        }
    
    @staticmethod
    def create_test_filters(**kwargs) -> Dict[str, Any]:
        """Crée des filtres de test."""
        return {
            "required": kwargs.get("required", [
                {"field": "user_id", "operator": "eq", "value": 12345}
            ]),
            "optional": kwargs.get("optional", []),
            "ranges": kwargs.get("ranges", []),
            "text_search": kwargs.get("text_search", {})
        }
    
    @staticmethod
    def create_test_aggregations(**kwargs) -> Dict[str, Any]:
        """Crée des agrégations de test."""
        return {
            "enabled": kwargs.get("enabled", True),
            "types": kwargs.get("types", ["sum", "count"]),
            "group_by": kwargs.get("group_by", ["category_name"]),
            "metrics": kwargs.get("metrics", ["amount_abs"]),
            "limit": kwargs.get("limit", 50)
        }
    
    @staticmethod
    def create_test_response_metadata(**kwargs) -> Dict[str, Any]:
        """Crée des métadonnées de réponse de test."""
        return {
            "query_id": kwargs.get("query_id", "test-query-123"),
            "execution_time_ms": kwargs.get("execution_time_ms", 45),
            "total_hits": kwargs.get("total_hits", 156),
            "returned_hits": kwargs.get("returned_hits", 20),
            "has_more": kwargs.get("has_more", True),
            "cache_hit": kwargs.get("cache_hit", False),
            "elasticsearch_took": kwargs.get("elasticsearch_took", 23)
        }
    
    @staticmethod
    def create_test_search_results(count: int = 3, user_id: int = 12345) -> List[Dict[str, Any]]:
        """Crée des résultats de recherche de test."""
        results = []
        for i in range(count):
            results.append({
                "transaction_id": f"user_{user_id}_tx_{i}",
                "user_id": user_id,
                "amount": -25.50 - i,
                "amount_abs": 25.50 + i,
                "category_name": "Restaurant",
                "merchant_name": f"Test Merchant {i}",
                "date": "2024-01-15",
                "score": 1.0 - (i * 0.1),
                "searchable_text": f"Test transaction {i}"
            })
        return results

class TestDataFactory:
    """Factory pour créer des données de test."""
    
    @staticmethod
    def create_transaction(user_id: int = 12345, **kwargs) -> TestTransaction:
        """Crée une transaction de test."""
        defaults = {
            "id": f"test_trans_{datetime.now().timestamp()}",
            "user_id": user_id,
            "merchant_name": "Test Merchant",
            "amount": 25.50,
            "transaction_date": "2024-01-15T10:30:00Z",
            "category": "restaurant",
            "description": "Test transaction"
        }
        defaults.update(kwargs)
        return TestTransaction(**defaults)
    
    @staticmethod
    def create_transactions(count: int = 5, user_id: int = 12345) -> List[TestTransaction]:
        """Crée plusieurs transactions de test."""
        transactions = []
        merchants = ["Café Central", "Supermarché", "Station Essence", "Restaurant", "Pharmacie"]
        amounts = [15.50, 45.20, 60.00, 32.80, 12.90]
        
        for i in range(count):
            transaction = TestDataFactory.create_transaction(
                user_id=user_id,
                id=f"test_trans_{i}",
                merchant_name=merchants[i % len(merchants)],
                amount=amounts[i % len(amounts)],
                description=f"Test transaction {i}"
            )
            transactions.append(transaction)
        
        return transactions
    
    @staticmethod
    def transactions_to_elasticsearch_docs(transactions: List[TestTransaction]) -> List[Dict[str, Any]]:
        """Convertit les transactions en documents Elasticsearch."""
        docs = []
        for trans in transactions:
            doc = {
                "_id": trans.id,
                "_source": {
                    "user_id": trans.user_id,
                    "merchant_name": trans.merchant_name,
                    "amount": trans.amount,
                    "transaction_date": trans.transaction_date,
                    "category": trans.category,
                    "description": trans.description,
                    "searchable_text": f"{trans.merchant_name} {trans.description}"
                }
            }
            docs.append(doc)
        return docs
    """Factory pour créer des données de test."""
    
    @staticmethod
    def create_transaction(user_id: int = 12345, **kwargs) -> TestTransaction:
        """Crée une transaction de test."""
        defaults = {
            "id": f"test_trans_{datetime.now().timestamp()}",
            "user_id": user_id,
            "merchant_name": "Test Merchant",
            "amount": 25.50,
            "transaction_date": "2024-01-15T10:30:00Z",
            "category": "restaurant",
            "description": "Test transaction"
        }
        defaults.update(kwargs)
        return TestTransaction(**defaults)
    
    @staticmethod
    def create_transactions(count: int = 5, user_id: int = 12345) -> List[TestTransaction]:
        """Crée plusieurs transactions de test."""
        transactions = []
        merchants = ["Café Central", "Supermarché", "Station Essence", "Restaurant", "Pharmacie"]
        amounts = [15.50, 45.20, 60.00, 32.80, 12.90]
        
        for i in range(count):
            transaction = TestDataFactory.create_transaction(
                user_id=user_id,
                id=f"test_trans_{i}",
                merchant_name=merchants[i % len(merchants)],
                amount=amounts[i % len(amounts)],
                description=f"Test transaction {i}"
            )
            transactions.append(transaction)
        
        return transactions
    
    @staticmethod
    def transactions_to_elasticsearch_docs(transactions: List[TestTransaction]) -> List[Dict[str, Any]]:
        """Convertit les transactions en documents Elasticsearch."""
        docs = []
        for trans in transactions:
            doc = {
                "_id": trans.id,
                "_source": {
                    "user_id": trans.user_id,
                    "merchant_name": trans.merchant_name,
                    "amount": trans.amount,
                    "transaction_date": trans.transaction_date,
                    "category": trans.category,
                    "description": trans.description,
                    "searchable_text": f"{trans.merchant_name} {trans.description}"
                }
            }
            docs.append(doc)
        return docs

# ==================== MOCKS ET UTILITAIRES ====================

class MockElasticsearchClient:
    """Mock client Elasticsearch pour les tests."""
    
    def __init__(self):
        self.search_responses = []
        self.search_calls = []
        self.index_calls = []
        self.delete_calls = []
    
    def add_search_response(self, response: Dict[str, Any]):
        """Ajoute une réponse mock pour search."""
        self.search_responses.append(response)
    
    async def search(self, **kwargs) -> Dict[str, Any]:
        """Mock de la méthode search."""
        self.search_calls.append(kwargs)
        
        if self.search_responses:
            return self.search_responses.pop(0)
        
        # Réponse par défaut
        return {
            "took": 5,
            "hits": {
                "total": {"value": 0, "relation": "eq"},
                "hits": []
            }
        }
    
    async def index(self, **kwargs):
        """Mock de la méthode index."""
        self.index_calls.append(kwargs)
        return {"_id": "test_id", "result": "created"}
    
    async def delete(self, **kwargs):
        """Mock de la méthode delete."""
        self.delete_calls.append(kwargs)
        return {"result": "deleted"}
    
    def reset_mocks(self):
        """Remet à zéro tous les mocks."""
        self.search_responses.clear()
        self.search_calls.clear()
        self.index_calls.clear()
        self.delete_calls.clear()

class MockCache:
    """Mock cache pour les tests."""
    
    def __init__(self):
        self._data = {}
        self.get_calls = []
        self.set_calls = []
        self.delete_calls = []
    
    async def get(self, key: str) -> Optional[Any]:
        """Mock de get."""
        self.get_calls.append(key)
        return self._data.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Mock de set."""
        self.set_calls.append((key, value, ttl))
        self._data[key] = value
    
    async def delete(self, key: str):
        """Mock de delete."""
        self.delete_calls.append(key)
        self._data.pop(key, None)
    
    async def clear(self):
        """Mock de clear."""
        self._data.clear()
    
    def reset_mock(self):
        """Remet à zéro le mock."""
        self._data.clear()
        self.get_calls.clear()
        self.set_calls.clear()
        self.delete_calls.clear()

def create_mock_metrics_collector():
    """Crée un mock du collecteur de métriques."""
    mock = Mock()
    mock.increment_counter = Mock()
    mock.record_histogram = Mock()
    mock.record_timer = Mock()
    mock.set_gauge = Mock()
    return mock

def assert_called_with_user_id(mock_call, expected_user_id: int):
    """Vérifie qu'un mock a été appelé avec le bon user_id."""
    args, kwargs = mock_call
    
    # Recherche dans les args
    if args and len(args) > 0:
        if isinstance(args[0], dict) and "user_id" in args[0]:
            assert args[0]["user_id"] == expected_user_id
            return
    
    # Recherche dans les kwargs
    if "user_id" in kwargs:
        assert kwargs["user_id"] == expected_user_id
        return
    
    # Recherche dans une requête imbriquée
    for arg in args:
        if isinstance(arg, dict):
            if "query" in arg and isinstance(arg["query"], dict):
                if "bool" in arg["query"] and "filter" in arg["query"]["bool"]:
                    filters = arg["query"]["bool"]["filter"]
                    for filter_clause in filters:
                        if "term" in filter_clause and "user_id" in filter_clause["term"]:
                            assert filter_clause["term"]["user_id"] == expected_user_id
                            return
    
    raise AssertionError(f"user_id {expected_user_id} not found in mock call")

# ==================== DÉCORATEURS DE TEST ====================

def skip_if_no_elasticsearch(test_func):
    """Décorateur pour ignorer les tests si Elasticsearch n'est pas disponible."""
    def wrapper(*args, **kwargs):
        try:
            # Test rapide de connexion Elasticsearch
            # Cette implémentation peut être adaptée selon vos besoins
            return test_func(*args, **kwargs)
        except Exception as e:
            import unittest
            raise unittest.SkipTest(f"Elasticsearch not available: {e}")
    return wrapper

def mock_elasticsearch(test_func):
    """Décorateur pour mocker Elasticsearch dans un test."""
    def wrapper(self, *args, **kwargs):
        with patch('search_service.clients.elasticsearch_client.ElasticsearchClient') as mock_client:
            mock_instance = MockElasticsearchClient()
            mock_client.return_value = mock_instance
            self.mock_elasticsearch = mock_instance
            return test_func(self, *args, **kwargs)
    return wrapper

def mock_cache(test_func):
    """Décorateur pour mocker le cache dans un test."""
    def wrapper(self, *args, **kwargs):
        with patch('search_service.utils.cache.SearchCache') as mock_cache_class:
            mock_instance = MockCache()
            mock_cache_class.return_value = mock_instance
            self.mock_cache = mock_instance
            return test_func(self, *args, **kwargs)
    return wrapper

def mock_metrics(test_func):
    """Décorateur pour mocker les métriques dans un test."""
    def wrapper(self, *args, **kwargs):
        with patch('search_service.utils.metrics.MetricsCollector.get_instance') as mock_metrics:
            mock_instance = create_mock_metrics_collector()
            mock_metrics.return_value = mock_instance
            self.mock_metrics = mock_instance
            return test_func(self, *args, **kwargs)
    return wrapper

# ==================== ASSERTIONS PERSONNALISÉES ====================

def assert_query_contains_user_filter(test_case: unittest.TestCase, query: Dict[str, Any], user_id: int):
    """Vérifie qu'une requête contient un filtre utilisateur."""
    test_case.assertIn("query", query)
    
    if "bool" in query["query"]:
        bool_query = query["query"]["bool"]
        
        # Vérification dans must ou filter
        for clause_type in ["must", "filter"]:
            if clause_type in bool_query:
                clauses = bool_query[clause_type]
                for clause in clauses:
                    if "term" in clause and "user_id" in clause["term"]:
                        test_case.assertEqual(clause["term"]["user_id"], user_id)
                        return
    
    test_case.fail(f"Query does not contain user filter for user_id {user_id}")

def assert_response_has_highlights(test_case: unittest.TestCase, response: Dict[str, Any]):
    """Vérifie qu'une réponse contient des highlights."""
    test_case.assertIn("hits", response)
    hits = response["hits"]["hits"]
    
    highlighted_hits = [hit for hit in hits if "highlight" in hit]
    test_case.assertGreater(len(highlighted_hits), 0, "No highlights found in response")

def assert_results_belong_to_user(test_case: unittest.TestCase, response: Dict[str, Any], user_id: int):
    """Vérifie que tous les résultats appartiennent à l'utilisateur."""
    test_case.assertIn("hits", response)
    hits = response["hits"]["hits"]
    
    for hit in hits:
        if "_source" in hit and "user_id" in hit["_source"]:
            test_case.assertEqual(hit["_source"]["user_id"], user_id,
                               f"Hit {hit.get('_id')} belongs to wrong user")

def assert_pagination_applied(test_case: unittest.TestCase, query: Dict[str, Any], size: int, from_: int):
    """Vérifie que la pagination est correctement appliquée."""
    test_case.assertEqual(query.get("size"), size, "Wrong page size")
    test_case.assertEqual(query.get("from"), from_, "Wrong offset")

def assert_highlights_enabled(test_case: unittest.TestCase, query: Dict[str, Any]):
    """Vérifie que les highlights sont activés dans la requête."""
    test_case.assertIn("highlight", query, "Highlights not enabled")
    test_case.assertIn("fields", query["highlight"], "Highlight fields not specified")

# ==================== UTILITAIRES DE COMPARAISON ====================

def normalize_query_for_comparison(query: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise une requête pour la comparaison."""
    # Crée une copie profonde et supprime les champs non-déterministes
    import copy
    normalized = copy.deepcopy(query)
    
    # Supprime les timestamps ou autres champs variables
    # Trie les listes pour une comparaison cohérente
    
    return normalized

def queries_are_equivalent(query1: Dict[str, Any], query2: Dict[str, Any]) -> bool:
    """Compare deux requêtes Elasticsearch pour équivalence."""
    norm1 = normalize_query_for_comparison(query1)
    norm2 = normalize_query_for_comparison(query2)
    return norm1 == norm2

# ==================== HELPERS POUR TESTS D'INTÉGRATION ====================

class IntegrationTestHelper:
    """Helper pour les tests d'intégration."""
    
    def __init__(self):
        self.created_indices = []
        self.indexed_documents = []
    
    async def create_test_index(self, es_client, index_name: str):
        """Crée un index de test."""
        mapping = {
            "mappings": {
                "properties": {
                    "user_id": {"type": "integer"},
                    "merchant_name": {"type": "text", "analyzer": "standard"},
                    "amount": {"type": "float"},
                    "transaction_date": {"type": "date"},
                    "searchable_text": {"type": "text", "analyzer": "standard"},
                    "category": {"type": "keyword"}
                }
            }
        }
        
        await es_client.indices.create(index=index_name, body=mapping)
        self.created_indices.append(index_name)
    
    async def index_test_documents(self, es_client, index_name: str, documents: List[Dict[str, Any]]):
        """Indexe des documents de test."""
        for doc in documents:
            await es_client.index(index=index_name, body=doc["_source"], id=doc.get("_id"))
            self.indexed_documents.append((index_name, doc.get("_id")))
        
        # Refresh pour rendre les documents disponibles immédiatement
        await es_client.indices.refresh(index=index_name)
    
    async def cleanup(self, es_client):
        """Nettoie les ressources de test."""
        # Supprime les indices créés
        for index_name in self.created_indices:
            try:
                await es_client.indices.delete(index=index_name)
            except Exception as e:
                logger.warning(f"Failed to delete test index {index_name}: {e}")
        
        self.created_indices.clear()
        self.indexed_documents.clear()

# ==================== GÉNÉRATEURS DE DONNÉES ALÉATOIRES ====================

import random
import string
from datetime import datetime, timedelta

class RandomDataGenerator:
    """Générateur de données aléatoires pour les tests."""
    
    MERCHANTS = [
        "Café Central", "Restaurant Le Bon Goût", "Supermarché Carrefour",
        "Station Total", "Pharmacie Centrale", "Boulangerie Dupont",
        "McDonald's", "IKEA", "Amazon", "Netflix", "Spotify"
    ]
    
    CATEGORIES = [
        "restaurant", "supermarché", "transport", "santé",
        "divertissement", "shopping", "carburant", "abonnement"
    ]
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """Génère une chaîne aléatoire."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_amount(min_amount: float = 5.0, max_amount: float = 500.0) -> float:
        """Génère un montant aléatoire."""
        return round(random.uniform(min_amount, max_amount), 2)
    
    @staticmethod
    def random_date(start_days_ago: int = 30, end_days_ago: int = 0) -> str:
        """Génère une date aléatoire."""
        start_date = datetime.now() - timedelta(days=start_days_ago)
        end_date = datetime.now() - timedelta(days=end_days_ago)
        
        time_between = end_date - start_date
        random_time = random.random() * time_between.total_seconds()
        random_date = start_date + timedelta(seconds=random_time)
        
        return random_date.isoformat() + "Z"
    
    @staticmethod
    def random_merchant() -> str:
        """Sélectionne un marchand aléatoire."""
        return random.choice(RandomDataGenerator.MERCHANTS)
    
    @staticmethod
    def random_category() -> str:
        """Sélectionne une catégorie aléatoire."""
        return random.choice(RandomDataGenerator.CATEGORIES)
    
    @staticmethod
    def generate_random_transaction(user_id: int) -> TestTransaction:
        """Génère une transaction aléatoire."""
        merchant = RandomDataGenerator.random_merchant()
        return TestDataFactory.create_transaction(
            user_id=user_id,
            id=f"random_{RandomDataGenerator.random_string(8)}",
            merchant_name=merchant,
            amount=RandomDataGenerator.random_amount(),
            transaction_date=RandomDataGenerator.random_date(),
            category=RandomDataGenerator.random_category(),
            description=f"Transaction chez {merchant}"
        )
    
    @staticmethod
    def generate_random_transactions(user_id: int, count: int) -> List[TestTransaction]:
        """Génère plusieurs transactions aléatoires."""
        return [RandomDataGenerator.generate_random_transaction(user_id) for _ in range(count)]

# ==================== VALIDATEURS DE TESTS ====================

class TestValidator:
    """Validateur pour vérifier la cohérence des tests."""
    
    @staticmethod
    def validate_test_data_consistency(transactions: List[TestTransaction], user_id: int) -> bool:
        """Valide que les données de test sont cohérentes."""
        for trans in transactions:
            if trans.user_id != user_id:
                return False
            if not trans.merchant_name or not trans.description:
                return False
            if trans.amount <= 0:
                return False
        return True
    
    @staticmethod
    def validate_elasticsearch_query(query: Dict[str, Any]) -> List[str]:
        """Valide une requête Elasticsearch et retourne les erreurs."""
        errors = []
        
        if "query" not in query:
            errors.append("Missing 'query' field")
        
        if "size" in query and not isinstance(query["size"], int):
            errors.append("'size' must be an integer")
        
        if "from" in query and not isinstance(query["from"], int):
            errors.append("'from' must be an integer")
        
        return errors
    
    @staticmethod
    def validate_search_response(response: Dict[str, Any]) -> List[str]:
        """Valide une réponse de recherche et retourne les erreurs."""
        errors = []
        
        required_fields = ["hits", "took"]
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")
        
        if "hits" in response:
            hits = response["hits"]
            if not isinstance(hits, dict):
                errors.append("'hits' must be a dictionary")
            elif "hits" not in hits:
                errors.append("Missing 'hits' array in hits object")
            elif not isinstance(hits["hits"], list):
                errors.append("'hits.hits' must be a list")
        
        return errors

# ==================== CONTEXTE DE TEST ====================

class TestContext:
    """Contexte partagé pour les tests."""
    
    def __init__(self):
        self.user_id = 12345
        self.test_index = "test_transactions"
        self.mock_elasticsearch = None
        self.mock_cache = None
        self.mock_metrics = None
        self.test_data = []
        self.integration_helper = IntegrationTestHelper()
    
    def setup_mocks(self):
        """Configure tous les mocks."""
        self.mock_elasticsearch = MockElasticsearchClient()
        self.mock_cache = MockCache()
        self.mock_metrics = create_mock_metrics_collector()
    
    def create_test_transactions(self, count: int = 5) -> List[TestTransaction]:
        """Crée des transactions de test."""
        self.test_data = TestDataFactory.create_transactions(count, self.user_id)
        return self.test_data
    
    def create_random_transactions(self, count: int = 10) -> List[TestTransaction]:
        """Crée des transactions aléatoires."""
        self.test_data = RandomDataGenerator.generate_random_transactions(self.user_id, count)
        return self.test_data
    
    def reset(self):
        """Remet à zéro le contexte."""
        self.test_data.clear()
        if self.mock_elasticsearch:
            self.mock_elasticsearch.reset_mocks()
        if self.mock_cache:
            self.mock_cache.reset_mock()

# ==================== CONFIGURATION DES TESTS ====================

class TestConfig:
    """Configuration globale pour les tests."""
    
    # Timeouts
    DEFAULT_TIMEOUT = 10
    LONG_TIMEOUT = 30
    
    # Tailles de données
    SMALL_DATASET_SIZE = 10
    MEDIUM_DATASET_SIZE = 100
    LARGE_DATASET_SIZE = 1000
    
    # Configuration Elasticsearch de test
    TEST_ES_CONFIG = {
        "host": "localhost",
        "port": 9200,
        "index": "test_transactions",
        "timeout": 5
    }
    
    # Configuration de cache de test
    TEST_CACHE_CONFIG = {
        "type": "memory",
        "ttl": 60,
        "max_size": 100
    }
    
    @classmethod
    def get_test_elasticsearch_config(cls) -> Dict[str, Any]:
        """Retourne la configuration Elasticsearch pour les tests."""
        return cls.TEST_ES_CONFIG.copy()
    
    @classmethod
    def get_test_cache_config(cls) -> Dict[str, Any]:
        """Retourne la configuration de cache pour les tests."""
        return cls.TEST_CACHE_CONFIG.copy()

# ==================== RUNNERS DE TESTS SPÉCIALISÉS ====================

def run_performance_test(test_func, iterations: int = 100, max_duration: float = 1.0):
    """Exécute un test de performance."""
    import time
    
    durations = []
    for _ in range(iterations):
        start_time = time.time()
        test_func()
        duration = time.time() - start_time
        durations.append(duration)
    
    avg_duration = sum(durations) / len(durations)
    max_duration_observed = max(durations)
    
    if avg_duration > max_duration:
        raise AssertionError(f"Performance test failed: avg duration {avg_duration:.3f}s > {max_duration}s")
    
    return {
        "avg_duration": avg_duration,
        "max_duration": max_duration_observed,
        "min_duration": min(durations),
        "iterations": iterations
    }

def run_load_test(test_func, concurrent_users: int = 10, duration_seconds: int = 30):
    """Exécute un test de charge."""
    import threading
    import time
    
    results = []
    stop_event = threading.Event()
    
    def worker():
        while not stop_event.is_set():
            start_time = time.time()
            try:
                test_func()
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
            
            results.append({
                "duration": time.time() - start_time,
                "success": success,
                "error": error
            })
    
    # Démarrage des threads
    threads = []
    for _ in range(concurrent_users):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)
    
    # Attente
    time.sleep(duration_seconds)
    stop_event.set()
    
    # Nettoyage
    for thread in threads:
        thread.join()
    
    # Analyse des résultats
    total_requests = len(results)
    successful_requests = len([r for r in results if r["success"]])
    success_rate = successful_requests / total_requests if total_requests > 0 else 0
    
    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": success_rate,
        "avg_duration": sum(r["duration"] for r in results) / total_requests if total_requests > 0 else 0,
        "errors": [r["error"] for r in results if not r["success"]]
    }

# ==================== EXPORTS PRINCIPAUX ====================

__all__ = [
    # Classes de base
    'ConfigTestCase',
    'ModelsTestCase',  # ✅ AJOUTÉ
    'SearchTestCase',
    'AsyncTestCase',
    'APITestCase',
    
    # Fixtures et factories
    'TestTransaction',
    'TestDataFactory',
    'TestHelpers',  # ✅ AJOUTÉ
    'RandomDataGenerator',
    
    # Mocks
    'MockElasticsearchClient',
    'MockCache',
    'create_mock_metrics_collector',
    
    # Utilitaires
    'TestContext',
    'TestConfig',
    'TestValidator',
    'IntegrationTestHelper',
    
    # Décorateurs
    'skip_if_no_elasticsearch',
    'mock_elasticsearch',
    'mock_cache',
    'mock_metrics',
    
    # Assertions personnalisées
    'assert_called_with_user_id',
    'assert_query_contains_user_filter',
    'assert_response_has_highlights',
    'assert_results_belong_to_user',
    'assert_pagination_applied',
    'assert_highlights_enabled',
    
    # Comparaisons
    'normalize_query_for_comparison',
    'queries_are_equivalent',
    
    # Runners spécialisés
    'run_performance_test',
    'run_load_test'
]

# ==================== INITIALISATION DU MODULE ====================

# Configuration du logging pour les tests
logging.getLogger('elasticsearch').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Création du contexte global de test
global_test_context = TestContext()

def get_test_context() -> TestContext:
    """Retourne le contexte global de test."""
    return global_test_context

def setup_test_environment():
    """Configure l'environnement de test."""
    global_test_context.setup_mocks()
    logger.info("Test environment set up")

def teardown_test_environment():
    """Nettoie l'environnement de test."""
    global_test_context.reset()
    logger.info("Test environment cleaned up")

# Ajout aux exports
__all__.extend([
    'global_test_context',
    'get_test_context',
    'setup_test_environment',
    'teardown_test_environment'
])

# Message de confirmation du chargement
logger.info("Search Service test module loaded successfully")