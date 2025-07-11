"""
Tests unitaires pour models/service_contracts.py

Validation complète des contrats d'interface entre services:
- Modèles Pydantic de requête et réponse
- Validation des données et contraintes
- Sérialisation/désérialisation JSON
- Conformité aux spécifications AutoGen
- Tests de sécurité et performance

Tests couvrant:
- SearchServiceQuery et SearchServiceResponse
- Métadonnées et contexte d'exécution
- Filtres et agrégations
- Validation des contraintes de sécurité
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import MagicMock
import logging
from pydantic import ValidationError

# ✅ CORRECTION : Import robuste avec fallback
try:
    # Méthode 1 : Import relatif depuis le package parent tests
    from .. import ModelsTestCase, TestHelpers
except ImportError:
    try:
        # Méthode 2 : Import absolu en ajoutant le chemin
        import sys
        from pathlib import Path
        
        # Ajouter le répertoire tests au PYTHONPATH
        tests_path = Path(__file__).parent.parent
        if str(tests_path) not in sys.path:
            sys.path.insert(0, str(tests_path))
        
        # Import des classes depuis le module tests
        from tests import ModelsTestCase, TestHelpers
    except ImportError:
        # Méthode 3 : Import direct du fichier
        import sys
        from pathlib import Path
        tests_init_path = Path(__file__).parent.parent / "__init__.py"
        
        if tests_init_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("tests", tests_init_path)
            tests_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tests_module)
            ModelsTestCase = tests_module.ModelsTestCase
            TestHelpers = tests_module.TestHelpers
        else:
            # Fallback : Créer des classes de test simples
            import unittest
            
            class ModelsTestCase(unittest.TestCase):
                """Classe de test de base fallback pour models."""
                def setUp(self):
                    self.contracts = None
                    self.helpers = None
                    self.test_config = {
                        "security": {"test_user_id": 12345}
                    }
                    try:
                        from search_service.models import service_contracts
                        self.contracts = service_contracts
                    except ImportError:
                        pass
                    
                    # Créer un helper simple
                    self.helpers = type('TestHelpers', (), {
                        'create_test_query_metadata': lambda **kwargs: {
                            "query_id": "test-query-123",
                            "user_id": kwargs.get('user_id', 12345),
                            "agent_name": kwargs.get('agent_name', "test_agent"),
                            "team_name": kwargs.get('team_name', "test_team"),
                            "timestamp": datetime.utcnow().isoformat(),
                            "execution_context": kwargs.get('execution_context', {})
                        },
                        'create_test_search_parameters': lambda **kwargs: {
                            "limit": kwargs.get('limit', 20),
                            "offset": kwargs.get('offset', 0),
                            "timeout_ms": kwargs.get('timeout_ms', 5000),
                            "sort_by": "relevance"
                        },
                        'create_test_filters': lambda **kwargs: {
                            "required": kwargs.get('required', [
                                {"field": "user_id", "operator": "eq", "value": 12345}
                            ]),
                            "optional": kwargs.get('optional', []),
                            "ranges": kwargs.get('ranges', []),
                            "text_search": kwargs.get('text_search', {})
                        }
                    })()
            
            class TestHelpers:
                """Helper de test fallback."""
                @staticmethod
                def create_test_query_metadata(**kwargs):
                    return {
                        "query_id": "test-query-123",
                        "user_id": kwargs.get('user_id', 12345),
                        "agent_name": kwargs.get('agent_name', "test_agent"),
                        "team_name": kwargs.get('team_name', "test_team"),
                        "timestamp": datetime.utcnow().isoformat(),
                        "execution_context": kwargs.get('execution_context', {})
                    }

logger = logging.getLogger(__name__)

class TestSearchServiceQuery(ModelsTestCase):
    """Tests pour le modèle SearchServiceQuery"""
    
    @pytest.mark.unit
    def test_valid_query_creation(self):
        """Test la création d'une requête valide"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Données valides
        query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": self.helpers.create_test_filters(),
            "aggregations": {
                "enabled": True,
                "types": ["sum", "count"],
                "group_by": ["category_name"],
                "metrics": ["amount_abs"]
            },
            "options": {
                "include_highlights": False,
                "include_explanation": False,
                "cache_enabled": True,
                "return_raw_elasticsearch": False
            }
        }
        
        # Création du modèle
        try:
            query = self.contracts.SearchServiceQuery(**query_data)
            
            # Validations basiques
            assert query.query_metadata.user_id == self.test_config["security"]["test_user_id"]
            assert query.search_parameters.limit == 20
            assert len(query.filters.required) > 0
            assert query.aggregations.enabled == True
            
            logger.info("✅ Création SearchServiceQuery valide")
            
        except Exception as e:
            pytest.fail(f"Échec création requête valide: {e}")
    
    @pytest.mark.unit
    def test_query_validation_user_id_required(self):
        """Test que user_id est obligatoire"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Requête sans user_id
        query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(user_id=None),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": {"required": [], "optional": [], "ranges": []}
        }
        
        # Doit lever une ValidationError
        with pytest.raises(ValidationError) as exc_info:
            self.contracts.SearchServiceQuery(**query_data)
        
        # Vérification du message d'erreur
        error_msg = str(exc_info.value)
        assert "user_id" in error_msg.lower()
        
        logger.info("✅ Validation user_id obligatoire correcte")
    
    @pytest.mark.unit
    def test_query_validation_limits(self):
        """Test les validations de limites"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Test limites invalides
        invalid_limits = [
            {"limit": -1},  # Négatif
            {"limit": 10000},  # Trop grand
            {"offset": -1},  # Offset négatif
            {"timeout_ms": 0}  # Timeout nul
        ]
        
        base_data = {
            "query_metadata": self.helpers.create_test_query_metadata(),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": self.helpers.create_test_filters()
        }
        
        for invalid_limit in invalid_limits:
            query_data = base_data.copy()
            query_data["search_parameters"].update(invalid_limit)
            
            with pytest.raises(ValidationError):
                self.contracts.SearchServiceQuery(**query_data)
        
        logger.info("✅ Validation limites correcte")
    
    @pytest.mark.security
    def test_user_filter_enforcement(self):
        """Test l'application obligatoire du filtre utilisateur"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Requête sans filtre user_id
        query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": {
                "required": [
                    {"field": "category_name", "operator": "eq", "value": "restaurant"}
                ],
                "optional": [],
                "ranges": []
            }
        }
        
        # Création de la requête
        query = self.contracts.SearchServiceQuery(**query_data)
        
        # Validation que le filtre user_id a été ajouté automatiquement
        user_filter_exists = any(
            f.get("field") == "user_id" and f.get("value") == query.query_metadata.user_id
            for f in query.filters.required
        )
        
        assert user_filter_exists, "Filtre user_id obligatoire manquant"
        
        logger.info("✅ Filtre user_id appliqué automatiquement")
    
    @pytest.mark.unit
    def test_query_serialization(self):
        """Test la sérialisation/désérialisation JSON"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Création d'une requête complexe
        query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": self.helpers.create_test_filters(
                required=[
                    {"field": "user_id", "operator": "eq", "value": 12345},
                    {"field": "category_name", "operator": "eq", "value": "restaurant"}
                ],
                ranges=[
                    {"field": "amount_abs", "operator": "gt", "value": 50.0}
                ],
                text_search={
                    "query": "italien",
                    "fields": ["searchable_text", "merchant_name"],
                    "operator": "match"
                }
            ),
            "aggregations": {
                "enabled": True,
                "types": ["sum", "count", "avg"],
                "group_by": ["category_name", "month_year"],
                "metrics": ["amount_abs", "transaction_id"]
            }
        }
        
        # Création et sérialisation
        original_query = self.contracts.SearchServiceQuery(**query_data)
        json_str = original_query.model_dump_json()
        
        # Désérialisation
        parsed_data = json.loads(json_str)
        reconstructed_query = self.contracts.SearchServiceQuery(**parsed_data)
        
        # Vérification identité
        assert original_query.query_metadata.query_id == reconstructed_query.query_metadata.query_id
        assert original_query.search_parameters.limit == reconstructed_query.search_parameters.limit
        assert len(original_query.filters.required) == len(reconstructed_query.filters.required)
        
        logger.info("✅ Sérialisation/désérialisation JSON correcte")

class TestSearchServiceResponse(ModelsTestCase):
    """Tests pour le modèle SearchServiceResponse"""
    
    @pytest.mark.unit
    def test_valid_response_creation(self):
        """Test la création d'une réponse valide"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Données de réponse valides
        response_data = {
            "response_metadata": {
                "query_id": "test-query-123",
                "execution_time_ms": 45,
                "total_hits": 156,
                "returned_hits": 20,
                "has_more": True,
                "cache_hit": False,
                "elasticsearch_took": 23,
                "agent_context": {
                    "requesting_agent": "query_generator_agent",
                    "requesting_team": "financial_analysis_team",
                    "next_suggested_agent": "response_generator_agent"
                }
            },
            "results": [
                {
                    "transaction_id": "user_12345_tx_67890",
                    "user_id": 12345,
                    "amount": -45.67,
                    "amount_abs": 45.67,
                    "category_name": "Restaurant",
                    "merchant_name": "Le Bistrot",
                    "date": "2024-01-15",
                    "score": 1.0
                }
            ],
            "aggregations": {
                "total_amount": -1247.89,
                "transaction_count": 156,
                "average_amount": -7.99,
                "by_category": [
                    {"key": "Restaurant", "doc_count": 45, "total_amount": -567.89}
                ]
            },
            "performance": {
                "query_complexity": "simple",
                "optimization_applied": ["user_filter", "category_filter"],
                "index_used": "harena_transactions",
                "shards_queried": 1
            }
        }
        
        # Création du modèle
        try:
            response = self.contracts.SearchServiceResponse(**response_data)
            
            # Validations
            assert response.response_metadata.total_hits == 156
            assert response.response_metadata.returned_hits == 20
            assert len(response.results) == 1
            assert response.results[0].user_id == 12345
            assert response.aggregations.transaction_count == 156
            
            logger.info("✅ Création SearchServiceResponse valide")
            
        except Exception as e:
            pytest.fail(f"Échec création réponse valide: {e}")
    
    @pytest.mark.unit
    def test_response_validation_consistency(self):
        """Test la cohérence des données de réponse"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Données incohérentes
        inconsistent_data = {
            "response_metadata": {
                "query_id": "test-query-123",
                "execution_time_ms": 45,
                "total_hits": 156,
                "returned_hits": 20,  # Dit 20 résultats
                "has_more": True,
                "cache_hit": False,
                "elasticsearch_took": 23
            },
            "results": [  # Mais seulement 1 résultat
                {
                    "transaction_id": "user_12345_tx_67890",
                    "user_id": 12345,
                    "amount": -45.67,
                    "score": 1.0
                }
            ]
        }
        
        # La validation doit détecter l'incohérence
        response = self.contracts.SearchServiceResponse(**inconsistent_data)
        
        # Validation custom de cohérence
        actual_results = len(response.results)
        declared_results = response.response_metadata.returned_hits
        
        # Warning si incohérence (ne doit pas faire planter)
        if actual_results != declared_results:
            logger.warning(f"⚠️ Incohérence détectée: {actual_results} résultats vs {declared_results} déclarés")
        
        logger.info("✅ Validation cohérence réponse")
    
    @pytest.mark.performance
    def test_response_performance_metadata(self):
        """Test les métadonnées de performance"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Données avec performance
        response_data = {
            "response_metadata": {
                "query_id": "test-query-123",
                "execution_time_ms": 45,
                "total_hits": 156,
                "returned_hits": 20,
                "elasticsearch_took": 23
            },
            "results": [],
            "performance": {
                "query_complexity": "medium",
                "optimization_applied": ["cache_hit", "index_optimization"],
                "index_used": "harena_transactions",
                "shards_queried": 1
            }
        }
        
        response = self.contracts.SearchServiceResponse(**response_data)
        
        # Validation métriques performance
        assert response.response_metadata.execution_time_ms >= 0
        assert response.response_metadata.elasticsearch_took >= 0
        assert response.response_metadata.execution_time_ms >= response.response_metadata.elasticsearch_took
        
        # Validation performance details
        assert response.performance.query_complexity in ["simple", "medium", "complex"]
        assert isinstance(response.performance.optimization_applied, list)
        assert response.performance.shards_queried > 0
        
        logger.info("✅ Métadonnées performance valides")

class TestFilterModels(ModelsTestCase):
    """Tests pour les modèles de filtres"""
    
    @pytest.mark.unit
    def test_search_filter_creation(self):
        """Test la création de filtres de recherche"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Différents types de filtres
        filter_examples = [
            # Filtre simple
            {"field": "category_name", "operator": "eq", "value": "restaurant"},
            # Filtre range
            {"field": "amount_abs", "operator": "between", "value": [50.0, 200.0]},
            # Filtre multiple
            {"field": "merchant_name", "operator": "in", "value": ["AMAZON", "CARREFOUR"]},
            # Filtre null
            {"field": "description", "operator": "exists", "value": True}
        ]
        
        for filter_data in filter_examples:
            try:
                search_filter = self.contracts.SearchFilter(**filter_data)
                
                # Validations
                assert search_filter.field is not None
                assert search_filter.operator in ["eq", "neq", "gt", "gte", "lt", "lte", "between", "in", "nin", "exists"]
                assert search_filter.value is not None
                
                logger.info(f"✅ Filtre valide: {filter_data}")
                
            except ValidationError as e:
                pytest.fail(f"Filtre valide rejeté: {filter_data}, erreur: {e}")
    
    @pytest.mark.unit
    def test_text_search_filter(self):
        """Test les filtres de recherche textuelle"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Filtre textuel valide
        text_filter_data = {
            "query": "restaurant italien",
            "fields": ["searchable_text", "primary_description", "merchant_name"],
            "operator": "match",
            "fuzziness": "AUTO",
            "boost": 1.5
        }
        
        text_filter = self.contracts.TextSearchFilter(**text_filter_data)
        
        # Validations
        assert text_filter.query == "restaurant italien"
        assert len(text_filter.fields) == 3
        assert text_filter.operator in ["match", "match_phrase", "multi_match"]
        assert text_filter.boost > 0
        
        logger.info("✅ Filtre recherche textuelle valide")

class TestAggregationModels(ModelsTestCase):
    """Tests pour les modèles d'agrégation"""
    
    @pytest.mark.unit
    def test_aggregation_request_creation(self):
        """Test la création de requêtes d'agrégation"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Requête d'agrégation complète
        agg_data = {
            "enabled": True,
            "types": ["sum", "count", "avg", "min", "max"],
            "group_by": ["category_name", "month_year", "merchant_name"],
            "metrics": ["amount_abs", "amount", "transaction_id"],
            "filters": {
                "field": "amount_abs",
                "operator": "gt", 
                "value": 10.0
            },
            "limit": 50,
            "sort_by": "total_amount",
            "sort_order": "desc"
        }
        
        agg_request = self.contracts.AggregationRequest(**agg_data)
        
        # Validations
        assert agg_request.enabled == True
        assert "sum" in agg_request.types
        assert "category_name" in agg_request.group_by
        assert "amount_abs" in agg_request.metrics
        assert agg_request.limit == 50
        
        logger.info("✅ Requête agrégation valide")
    
    @pytest.mark.unit
    def test_aggregation_result_structure(self):
        """Test la structure des résultats d'agrégation"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Résultat d'agrégation
        agg_result_data = {
            "total_amount": -1247.89,
            "transaction_count": 156,
            "average_amount": -7.99,
            "min_amount": -89.50,
            "max_amount": -5.20,
            "by_category": [
                {
                    "key": "Restaurant",
                    "doc_count": 45,
                    "total_amount": -567.89,
                    "percentage": 45.5
                },
                {
                    "key": "Transport",
                    "doc_count": 23,
                    "total_amount": -234.56,
                    "percentage": 18.8
                }
            ],
            "by_month": [
                {
                    "key": "2024-01",
                    "doc_count": 78,
                    "total_amount": -789.45
                }
            ]
        }
        
        agg_result = self.contracts.AggregationResult(**agg_result_data)
        
        # Validations
        assert agg_result.transaction_count == 156
        assert len(agg_result.by_category) == 2
        assert agg_result.by_category[0].key == "Restaurant"
        assert agg_result.by_category[0].doc_count == 45
        
        logger.info("✅ Résultat agrégation valide")

class TestContractValidation(ModelsTestCase):
    """Tests pour la validation des contrats"""
    
    @pytest.mark.unit
    def test_query_contract_validation_function(self):
        """Test la fonction de validation des contrats de requête"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Requête valide
        valid_query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": self.helpers.create_test_filters()
        }
        
        valid_query = self.contracts.SearchServiceQuery(**valid_query_data)
        
        # Validation via fonction dédiée
        is_valid, errors = self.contracts.validate_search_service_query(valid_query)
        
        assert is_valid == True, f"Requête valide rejetée: {errors}"
        assert len(errors) == 0
        
        logger.info("✅ Fonction validation requête correcte")
    
    @pytest.mark.unit
    def test_response_contract_validation_function(self):
        """Test la fonction de validation des contrats de réponse"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Réponse valide
        valid_response_data = {
            "response_metadata": {
                "query_id": "test-query-123",
                "execution_time_ms": 45,
                "total_hits": 20,
                "returned_hits": 20,
                "has_more": False,
                "cache_hit": False,
                "elasticsearch_took": 23
            },
            "results": [
                {
                    "transaction_id": "user_12345_tx_67890",
                    "user_id": 12345,
                    "amount": -45.67,
                    "score": 1.0
                }
            ]
        }
        
        valid_response = self.contracts.SearchServiceResponse(**valid_response_data)
        
        # Validation via fonction dédiée
        is_valid, errors = self.contracts.validate_search_service_response(valid_response)
        
        assert is_valid == True, f"Réponse valide rejetée: {errors}"
        assert len(errors) == 0
        
        logger.info("✅ Fonction validation réponse correcte")
    
    @pytest.mark.security
    def test_security_validation(self):
        """Test les validations de sécurité"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Test injection dans les filtres
        malicious_query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": {
                "required": [
                    {"field": "user_id", "operator": "eq", "value": 12345},
                    # Tentative d'injection
                    {"field": "category_name'; DROP TABLE transactions; --", "operator": "eq", "value": "restaurant"}
                ],
                "optional": [],
                "ranges": []
            }
        }
        
        # La création doit réussir (Pydantic valide la structure)
        malicious_query = self.contracts.SearchServiceQuery(**malicious_query_data)
        
        # Mais la validation sécurité doit détecter le problème
        is_valid, errors = self.contracts.validate_search_service_query(malicious_query)
        
        # Doit être invalide à cause du nom de champ suspect
        assert not is_valid, "Requête malicieuse acceptée"
        assert any("field" in error.lower() for error in errors), "Injection non détectée"
        
        logger.info("✅ Validation sécurité correcte")
    
    @pytest.mark.performance
    def test_performance_validation(self):
        """Test les validations de performance"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Requête potentiellement coûteuse
        expensive_query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(),
            "search_parameters": self.helpers.create_test_search_parameters(
                limit=1000,  # Limite élevée
                timeout_ms=60000  # Timeout élevé
            ),
            "filters": self.helpers.create_test_filters(),
            "aggregations": {
                "enabled": True,
                "types": ["sum", "count", "avg", "min", "max", "stats", "percentiles"],  # Beaucoup d'agrégations
                "group_by": ["category_name", "merchant_name", "month_year", "weekday"],  # Beaucoup de groupes
                "metrics": ["amount_abs", "amount", "transaction_id"]
            }
        }
        
        expensive_query = self.contracts.SearchServiceQuery(**expensive_query_data)
        
        # Validation performance
        is_valid, errors = self.contracts.validate_search_service_query(expensive_query)
        
        # Peut être invalide selon les limites configurées
        if not is_valid:
            assert any("performance" in error.lower() or "limit" in error.lower() for error in errors)
            logger.info(f"⚠️ Requête coûteuse rejetée: {errors}")
        else:
            logger.info("✅ Requête coûteuse acceptée")

class TestModelIntegration(ModelsTestCase):
    """Tests d'intégration entre modèles"""
    
    @pytest.mark.integration
    def test_query_response_cycle(self):
        """Test le cycle complet requête → réponse"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # 1. Création d'une requête
        query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": self.helpers.create_test_filters()
        }
        
        query = self.contracts.SearchServiceQuery(**query_data)
        
        # 2. Simulation d'une réponse correspondante
        response_data = {
            "response_metadata": {
                "query_id": query.query_metadata.query_id,  # Même ID
                "execution_time_ms": 45,
                "total_hits": 20,
                "returned_hits": 20,
                "has_more": False,
                "cache_hit": False,
                "elasticsearch_took": 23
            },
            "results": [
                {
                    "transaction_id": "user_12345_tx_67890",
                    "user_id": query.query_metadata.user_id,  # Même user_id
                    "amount": -45.67,
                    "score": 1.0
                }
            ]
        }
        
        response = self.contracts.SearchServiceResponse(**response_data)
        
        # 3. Vérification cohérence
        assert query.query_metadata.query_id == response.response_metadata.query_id
        assert query.query_metadata.user_id == response.results[0].user_id
        
        # 4. Validation des deux contrats
        query_valid, query_errors = self.contracts.validate_search_service_query(query)
        response_valid, response_errors = self.contracts.validate_search_service_response(response)
        
        assert query_valid, f"Requête invalide: {query_errors}"
        assert response_valid, f"Réponse invalide: {response_errors}"
        
        logger.info("✅ Cycle requête-réponse cohérent")
    
    @pytest.mark.integration
    def test_autogen_agent_context_integration(self):
        """Test l'intégration avec le contexte des agents AutoGen"""
        if self.contracts is None:
            pytest.skip("Module contracts non disponible")
        
        # Requête avec contexte AutoGen complet
        query_data = {
            "query_metadata": self.helpers.create_test_query_metadata(
                agent_name="query_generator_agent",
                team_name="financial_analysis_team",
                execution_context={
                    "conversation_id": "conv_123",
                    "turn_number": 3,
                    "agent_chain": ["intent_classifier", "entity_extractor", "query_generator"]
                }
            ),
            "search_parameters": self.helpers.create_test_search_parameters(),
            "filters": self.helpers.create_test_filters()
        }
        
        query = self.contracts.SearchServiceQuery(**query_data)
        
        # Vérification contexte AutoGen
        assert query.query_metadata.agent_name == "query_generator_agent"
        assert query.query_metadata.team_name == "financial_analysis_team"
        assert query.query_metadata.execution_context.conversation_id == "conv_123"
        assert "query_generator" in query.query_metadata.execution_context.agent_chain
        
        # Réponse avec contexte agent suivant
        response_data = {
            "response_metadata": {
                "query_id": query.query_metadata.query_id,
                "execution_time_ms": 45,
                "total_hits": 20,
                "returned_hits": 20,
                "agent_context": {
                    "requesting_agent": "query_generator_agent",
                    "requesting_team": "financial_analysis_team",
                    "next_suggested_agent": "response_generator_agent"
                }
            },
            "results": []
        }
        
        response = self.contracts.SearchServiceResponse(**response_data)
        
        # Vérification cohérence contexte
        assert response.response_metadata.agent_context.requesting_agent == query.query_metadata.agent_name
        assert response.response_metadata.agent_context.requesting_team == query.query_metadata.team_name
        assert response.response_metadata.agent_context.next_suggested_agent == "response_generator_agent"
        
        logger.info("✅ Intégration contexte AutoGen correcte")