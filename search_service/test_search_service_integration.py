"""
🧪 Test d'Intégration Complet Search Service
===========================================

Test exhaustif pour vérifier que tous les appels de fonctions entre
les fichiers du search_service fonctionnent correctement.

Objectifs:
- Vérifier qu'aucune méthode n'est appelée sans être implémentée
- Vérifier que toutes les variables/attributs existent
- Tester les imports et dépendances entre modules
- Valider la cohérence de l'architecture

Usage:
    python -m pytest test_search_service_integration.py -v
"""

import asyncio
import sys
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
import importlib
import inspect

import pytest


# === CONFIGURATION TEST ===

# Ajouter le projet au path pour les imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration environnement test
import os
os.environ.update({
    "TESTING": "true",
    "ENVIRONMENT": "test",
    "LOG_LEVEL": "DEBUG",
    "ELASTICSEARCH_HOST": "localhost",
    "ELASTICSEARCH_PORT": "9200",
    "REDIS_ENABLED": "false",
    "CACHE_ENABLED": "false",
    "METRICS_ENABLED": "true",
    "MOCK_ELASTICSEARCH": "true"
})


# === FIXTURES ET UTILITAIRES ===

class ImportTracker:
    """Traque les imports et détecte les problèmes"""
    
    def __init__(self):
        self.import_errors = []
        self.missing_attributes = []
        self.circular_imports = []
        self.successful_imports = []
    
    def track_import(self, module_name: str, from_module: str = None):
        """Tente d'importer un module et traque les erreurs"""
        try:
            if from_module:
                module = importlib.import_module(f"{from_module}.{module_name}")
            else:
                module = importlib.import_module(module_name)
            
            self.successful_imports.append(module_name)
            return module
            
        except ImportError as e:
            error_info = {
                "module": module_name,
                "from_module": from_module,
                "error": str(e),
                "type": "ImportError"
            }
            self.import_errors.append(error_info)
            return None
            
        except Exception as e:
            error_info = {
                "module": module_name,
                "from_module": from_module,
                "error": str(e),
                "type": type(e).__name__
            }
            self.import_errors.append(error_info)
            return None
    
    def check_attribute(self, obj, attr_name: str, context: str = ""):
        """Vérifie qu'un attribut existe sur un objet"""
        try:
            value = getattr(obj, attr_name)
            return value
        except AttributeError as e:
            self.missing_attributes.append({
                "object": str(obj),
                "attribute": attr_name,
                "context": context,
                "error": str(e)
            })
            return None


@pytest.fixture
def import_tracker():
    """Fixture pour tracker les imports"""
    return ImportTracker()


# === TESTS D'IMPORTS MODULAIRES ===

class TestModuleImports:
    """Tests des imports entre modules"""
    
    def test_config_imports(self, import_tracker):
        """Test imports du module config"""
        print("\n🔧 Test imports config...")
        
        # Import du module config principal
        config = import_tracker.track_import("search_service.config")
        assert config is not None, "Impossible d'importer search_service.config"
        
        # Vérification de l'existence de settings
        settings = import_tracker.check_attribute(config, "settings", "config module")
        assert settings is not None, "settings manquant dans config"
        
        # Test des attributs essentiels de settings
        essential_settings = [
            "elasticsearch_host", "elasticsearch_port", "environment",
            "log_level", "cache_enabled", "metrics_enabled"
        ]
        
        for setting in essential_settings:
            value = import_tracker.check_attribute(settings, setting, f"settings.{setting}")
            print(f"  ✓ settings.{setting} = {value}")
        
        print("✅ Config imports OK")
    
    def test_models_imports(self, import_tracker):
        """Test imports du module models"""
        print("\n📋 Test imports models...")
        
        # Import du module models principal
        models = import_tracker.track_import("search_service.models")
        assert models is not None, "Impossible d'importer search_service.models"
        
        # Test des contrats principaux
        main_contracts = [
            "SearchServiceQuery", "SearchServiceResponse",
            "ContractValidator", "QueryMetadata", "SearchParameters"
        ]
        
        for contract in main_contracts:
            obj = import_tracker.check_attribute(models, contract, f"models.{contract}")
            assert obj is not None, f"Contrat {contract} manquant dans models"
            print(f"  ✓ {contract} disponible")
        
        # Test des enums
        enums = ["QueryType", "FilterOperator", "AggregationType"]
        for enum_name in enums:
            enum_obj = import_tracker.check_attribute(models, enum_name, f"models.{enum_name}")
            assert enum_obj is not None, f"Enum {enum_name} manquant"
            print(f"  ✓ {enum_name} enum disponible")
        
        print("✅ Models imports OK")
    
    def test_clients_imports(self, import_tracker):
        """Test imports du module clients"""
        print("\n🔌 Test imports clients...")
        
        # Import du module clients
        clients = import_tracker.track_import("search_service.clients")
        assert clients is not None, "Impossible d'importer search_service.clients"
        
        # Test des clients principaux
        client_classes = ["ElasticsearchClient", "BaseClient"]
        for client_class in client_classes:
            client = import_tracker.check_attribute(clients, client_class, f"clients.{client_class}")
            assert client is not None, f"Client {client_class} manquant"
            print(f"  ✓ {client_class} disponible")
        
        print("✅ Clients imports OK")
    
    def test_core_imports(self, import_tracker):
        """Test imports du module core"""
        print("\n🔧 Test imports core...")
        
        # Import du module core
        core = import_tracker.track_import("search_service.core")
        assert core is not None, "Impossible d'importer search_service.core"
        
        # Test des gestionnaires principaux
        core_components = ["CoreManager", "core_manager"]
        for component in core_components:
            obj = import_tracker.check_attribute(core, component, f"core.{component}")
            assert obj is not None, f"Composant {component} manquant dans core"
            print(f"  ✓ {component} disponible")
        
        print("✅ Core imports OK")
    
    def test_api_imports(self, import_tracker):
        """Test imports du module api"""
        print("\n🌐 Test imports api...")
        
        # Import du module api
        api = import_tracker.track_import("search_service.api")
        assert api is not None, "Impossible d'importer search_service.api"
        
        # Test des composants API
        api_components = ["APIManager", "api_manager", "create_search_service_app"]
        for component in api_components:
            obj = import_tracker.check_attribute(api, component, f"api.{component}")
            assert obj is not None, f"Composant {component} manquant dans api"
            print(f"  ✓ {component} disponible")
        
        print("✅ API imports OK")
    
    def test_utils_imports(self, import_tracker):
        """Test imports du module utils"""
        print("\n🛠️ Test imports utils...")
        
        # Import du module utils
        utils = import_tracker.track_import("search_service.utils")
        assert utils is not None, "Impossible d'importer search_service.utils"
        
        # Test des utilitaires principaux
        util_components = [
            "CacheManager", "MetricsCollector", "ElasticsearchHelper"
        ]
        
        for component in util_components:
            obj = import_tracker.check_attribute(utils, component, f"utils.{component}")
            assert obj is not None, f"Utilitaire {component} manquant"
            print(f"  ✓ {component} disponible")
        
        print("✅ Utils imports OK")


# === TESTS DE CRÉATION D'OBJETS ===

class TestObjectCreation:
    """Tests de création d'objets avec les contrats"""
    
    def test_create_search_query(self):
        """Test création SearchServiceQuery"""
        print("\n🏗️ Test création SearchServiceQuery...")
        
        from search_service.models import (
            SearchServiceQuery, QueryMetadata, SearchParameters,
            SearchFilters, SearchFilter, FilterOperator
        )
        
        # Création des métadonnées
        metadata = QueryMetadata(
            user_id=123,
            intent_type="text_search",
            agent_name="test_agent",
            confidence=0.95
        )
        print("  ✓ QueryMetadata créé")
        
        # Création des paramètres
        parameters = SearchParameters(
            limit=20,
            offset=0,
            timeout_ms=5000
        )
        print("  ✓ SearchParameters créé")
        
        # Création des filtres
        filters = SearchFilters(
            required=[
                SearchFilter(
                    field="user_id",
                    value=123,
                    operator=FilterOperator.EQUALS
                )
            ]
        )
        print("  ✓ SearchFilters créé")
        
        # Création de la requête complète
        query = SearchServiceQuery(
            query_metadata=metadata,
            text_search="test recherche",
            search_parameters=parameters,
            filters=filters
        )
        print("  ✓ SearchServiceQuery créé")
        
        # Vérifications
        assert query.query_metadata.user_id == 123
        assert query.text_search == "test recherche"
        assert len(query.filters.required) == 1
        
        print("✅ Création SearchServiceQuery OK")
    
    def test_create_managers(self):
        """Test création des gestionnaires"""
        print("\n👔 Test création gestionnaires...")
        
        from search_service.core import CoreManager
        from search_service.api import APIManager
        
        # Test CoreManager
        core_manager = CoreManager()
        assert core_manager is not None
        assert hasattr(core_manager, '_initialized')
        print("  ✓ CoreManager créé")
        
        # Test APIManager  
        api_manager = APIManager()
        assert api_manager is not None
        assert hasattr(api_manager, '_initialized')
        print("  ✓ APIManager créé")
        
        print("✅ Création gestionnaires OK")


# === TESTS DE VALIDATION ===

class TestContractValidation:
    """Tests de validation des contrats"""
    
    def test_contract_validator(self):
        """Test du validateur de contrats"""
        print("\n✅ Test validation contrats...")
        
        from search_service.models import (
            SearchServiceQuery, ContractValidator,
            QueryMetadata, SearchParameters, SearchFilters, SearchFilter,
            FilterOperator
        )
        
        # Création d'une requête valide
        valid_query = SearchServiceQuery(
            query_metadata=QueryMetadata(
                user_id=123,
                intent_type="text_search",
                agent_name="test_agent"
            ),
            text_search="test",
            search_parameters=SearchParameters(limit=10),
            filters=SearchFilters(
                required=[
                    SearchFilter(
                        field="user_id",
                        value=123,
                        operator=FilterOperator.EQUALS
                    )
                ]
            )
        )
        
        # Test validation
        try:
            ContractValidator.validate_search_query(valid_query)
            print("  ✓ Validation requête valide OK")
        except Exception as e:
            pytest.fail(f"Validation échouée pour requête valide: {e}")
        
        print("✅ Validation contrats OK")


# === TESTS D'INTÉGRATION ASYNC ===

class TestAsyncIntegration:
    """Tests d'intégration asynchrone"""
    
    @pytest.mark.asyncio
    async def test_search_service_initialization(self):
        """Test initialisation du service principal"""
        print("\n🚀 Test initialisation SearchService...")
        
        from search_service import SearchService
        
        # Création du service
        service = SearchService(
            cache_enabled=False,  # Désactiver pour les tests
            metrics_enabled=True
        )
        
        assert service is not None
        assert not service.is_initialized
        print("  ✓ SearchService créé")
        
        # Test des propriétés avant initialisation
        assert service.is_initialized == False
        print("  ✓ État initial correct")
        
        # Note: On ne teste pas l'initialisation complète car elle nécessite Elasticsearch
        # mais on vérifie que les méthodes existent
        assert hasattr(service, 'initialize')
        assert hasattr(service, 'shutdown')
        assert hasattr(service, 'search')
        assert hasattr(service, 'health_check')
        print("  ✓ Méthodes principales présentes")
        
        print("✅ Initialisation SearchService OK")
    
    @pytest.mark.asyncio 
    async def test_quick_search_function(self):
        """Test de la fonction quick_search"""
        print("\n⚡ Test fonction quick_search...")
        
        from search_service import quick_search
        
        # Vérifier que la fonction existe et est appelable
        assert callable(quick_search)
        assert inspect.iscoroutinefunction(quick_search)
        print("  ✓ quick_search est une fonction async")
        
        # Note: On ne peut pas tester l'exécution réelle sans Elasticsearch
        # mais on vérifie la signature
        sig = inspect.signature(quick_search)
        expected_params = {'text', 'user_id', 'filters', 'limit'}
        actual_params = set(sig.parameters.keys())
        
        assert expected_params.issubset(actual_params), f"Paramètres manquants: {expected_params - actual_params}"
        print("  ✓ Signature quick_search correcte")
        
        print("✅ Fonction quick_search OK")


# === TESTS DE COHÉRENCE ARCHITECTURALE ===

class TestArchitecturalConsistency:
    """Tests de cohérence architecturale"""
    
    def test_main_module_exports(self):
        """Test des exports du module principal"""
        print("\n📦 Test exports module principal...")
        
        import search_service
        
        # Vérifier les exports principaux
        expected_exports = [
            'SearchService', 'create_app', 'SearchServiceQuery',
            'SearchServiceResponse', 'get_service', 'quick_search'
        ]
        
        for export in expected_exports:
            assert hasattr(search_service, export), f"Export {export} manquant"
            print(f"  ✓ {export} exporté")
        
        print("✅ Exports module principal OK")
    
    def test_version_and_metadata(self):
        """Test version et métadonnées"""
        print("\n🏷️ Test version et métadonnées...")
        
        import search_service
        
        # Vérifier les métadonnées
        metadata_attrs = ['__version__', '__author__', '__description__']
        for attr in metadata_attrs:
            value = getattr(search_service, attr, None)
            assert value is not None, f"Métadonnée {attr} manquante"
            print(f"  ✓ {attr} = {value}")
        
        print("✅ Version et métadonnées OK")
    
    def test_settings_consistency(self):
        """Test cohérence des settings"""
        print("\n⚙️ Test cohérence settings...")
        
        from search_service.config import settings
        
        # Vérifier les settings essentiels
        essential_settings = [
            ('elasticsearch_host', str),
            ('elasticsearch_port', int),
            ('environment', str),
            ('cache_enabled', bool),
            ('metrics_enabled', bool)
        ]
        
        for setting_name, expected_type in essential_settings:
            value = getattr(settings, setting_name, None)
            assert value is not None, f"Setting {setting_name} manquant"
            assert isinstance(value, expected_type), f"Setting {setting_name} type incorrect: {type(value)} vs {expected_type}"
            print(f"  ✓ {setting_name} = {value} ({expected_type.__name__})")
        
        print("✅ Cohérence settings OK")


# === TEST PRINCIPAL D'INTÉGRATION ===

class TestCompleteIntegration:
    """Test d'intégration complet"""
    
    def test_complete_import_chain(self, import_tracker):
        """Test de la chaîne d'imports complète"""
        print("\n🔗 Test chaîne d'imports complète...")
        
        # Test import du module principal
        search_service = import_tracker.track_import("search_service")
        assert search_service is not None, "Import search_service échoué"
        print("  ✓ search_service importé")
        
        # Test création d'un service
        SearchService = import_tracker.check_attribute(search_service, "SearchService")
        assert SearchService is not None, "SearchService non disponible"
        
        service_instance = SearchService()
        assert service_instance is not None, "Impossible de créer SearchService"
        print("  ✓ SearchService instancié")
        
        # Test fonction utilitaire
        quick_search = import_tracker.check_attribute(search_service, "quick_search")
        assert quick_search is not None, "quick_search non disponible"
        assert callable(quick_search), "quick_search non callable"
        print("  ✓ quick_search accessible")
        
        # Vérifier qu'il n'y a pas d'erreurs d'import
        if import_tracker.import_errors:
            print("\n❌ Erreurs d'import détectées:")
            for error in import_tracker.import_errors:
                print(f"  - {error['module']}: {error['error']}")
            pytest.fail(f"{len(import_tracker.import_errors)} erreurs d'import détectées")
        
        # Vérifier qu'il n'y a pas d'attributs manquants
        if import_tracker.missing_attributes:
            print("\n❌ Attributs manquants détectés:")
            for missing in import_tracker.missing_attributes:
                print(f"  - {missing['object']}.{missing['attribute']}: {missing['error']}")
            pytest.fail(f"{len(import_tracker.missing_attributes)} attributs manquants détectés")
        
        print("✅ Chaîne d'imports complète OK")
        print(f"✅ {len(import_tracker.successful_imports)} imports réussis")


# === FONCTION PRINCIPALE POUR EXÉCUTION DIRECTE ===

def run_integration_tests():
    """Exécute tous les tests d'intégration"""
    print("🧪 DÉMARRAGE TESTS D'INTÉGRATION SEARCH SERVICE")
    print("=" * 60)
    
    # Configuration pytest
    pytest_args = [
        __file__,
        "-v",  # verbose
        "--tb=short",  # traceback court
        "--no-header",  # pas d'header pytest
        "-x",  # arrêt au premier échec
    ]
    
    # Exécution
    try:
        exit_code = pytest.main(pytest_args)
        
        if exit_code == 0:
            print("\n🎉 TOUS LES TESTS D'INTÉGRATION PASSENT!")
            print("✅ Aucune méthode manquante détectée")
            print("✅ Toutes les variables existent")
            print("✅ Architecture cohérente")
        else:
            print("\n❌ ÉCHEC DES TESTS D'INTÉGRATION")
            print("Des problèmes ont été détectés dans l'architecture")
            
        return exit_code == 0
        
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)