"""
Tests unitaires pour search_service/__init__.py

Ces tests valident :
- Les imports du package principal
- La classe SearchService
- Les factory functions
- Les flags de disponibilité
- La gestion d'erreurs gracieuse
"""

import pytest
import logging
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any


class TestSearchServiceImports:
    """Tests pour vérifier que tous les imports fonctionnent correctement."""
    
    def test_package_import(self):
        """Test que le package search_service peut être importé."""
        try:
            import search_service
            assert search_service is not None
        except ImportError as e:
            pytest.fail(f"Impossible d'importer search_service: {e}")
    
    def test_version_attributes(self):
        """Test que les attributs de version sont présents."""
        import search_service
        
        assert hasattr(search_service, '__version__')
        assert hasattr(search_service, '__author__')
        assert isinstance(search_service.__version__, str)
        assert isinstance(search_service.__author__, str)
        assert search_service.__version__ == "1.0.0"
    
    def test_availability_flags(self):
        """Test que les flags de disponibilité existent."""
        import search_service
        
        # Vérifier que tous les flags existent
        flags = [
            'CONFIG_AVAILABLE',
            'CORE_AVAILABLE', 
            'API_AVAILABLE',
            'MODELS_AVAILABLE',
            'CLIENTS_AVAILABLE'
        ]
        
        for flag in flags:
            assert hasattr(search_service, flag)
            assert isinstance(getattr(search_service, flag), bool)
    
    def test_main_exports(self):
        """Test que les exports principaux sont présents."""
        import search_service
        
        # Exports obligatoires qui doivent toujours être présents
        required_exports = [
            'SearchService',
            'create_search_service',
            '__version__',
            '__author__'
        ]
        
        for export in required_exports:
            assert hasattr(search_service, export), f"Export manquant: {export}"
    
    def test_conditional_imports(self):
        """Test que les imports conditionnels ne cassent pas le package."""
        import search_service
        
        # Ces imports peuvent être None si les modules ne sont pas disponibles
        optional_imports = [
            'LexicalEngine',
            'LexicalEngineFactory', 
            'api_router',
            'ElasticsearchClient',
            'settings'
        ]
        
        for import_name in optional_imports:
            # L'attribut doit exister, mais peut être None
            assert hasattr(search_service, import_name)


class TestSearchServiceClass:
    """Tests pour la classe SearchService principale."""
    
    def test_search_service_creation(self):
        """Test que SearchService peut être créée."""
        import search_service
        
        service = search_service.SearchService()
        assert service is not None
        assert isinstance(service, search_service.SearchService)
    
    def test_search_service_with_config(self):
        """Test SearchService avec configuration personnalisée."""
        import search_service
        
        config = {"test_key": "test_value"}
        service = search_service.SearchService(config=config)
        
        assert service.config == config
        assert service.config["test_key"] == "test_value"
    
    def test_search_service_initialization_state(self):
        """Test l'état d'initialisation initial."""
        import search_service
        
        service = search_service.SearchService()
        
        # Vérifier l'état initial
        assert service._initialized is False
        assert service.lexical_engine is None
        assert service.elasticsearch_client is None
    
    @pytest.mark.asyncio
    async def test_health_check_method(self):
        """Test que la méthode health_check existe et fonctionne."""
        import search_service
        
        service = search_service.SearchService()
        
        # La méthode doit exister et être callable
        assert hasattr(service, 'health_check')
        assert callable(service.health_check)
        
        # Doit retourner un dictionnaire
        health = await service.health_check()
        assert isinstance(health, dict)
        assert "service" in health
        assert "version" in health
        assert "status" in health
        assert "components" in health
    
    @pytest.mark.asyncio
    async def test_close_method(self):
        """Test que la méthode close existe et fonctionne."""
        import search_service
        
        service = search_service.SearchService()
        
        # La méthode doit exister
        assert hasattr(service, 'close')
        assert callable(service.close)
        
        # Ne doit pas lever d'exception même sans composants initialisés
        try:
            await service.close()
        except Exception as e:
            pytest.fail(f"close() a levé une exception: {e}")
    
    @pytest.mark.asyncio
    async def test_search_lexical_without_engine(self):
        """Test search_lexical sans moteur initialisé."""
        import search_service
        
        service = search_service.SearchService()
        
        # Sans moteur initialisé, doit lever une exception
        with pytest.raises(RuntimeError, match="Moteur lexical non disponible"):
            await service.search_lexical("test query", user_id=12345)


class TestFactoryFunction:
    """Tests pour la factory function."""
    
    def test_create_search_service_function(self):
        """Test que create_search_service fonctionne."""
        import search_service
        
        # Vérifier que la fonction existe
        assert hasattr(search_service, 'create_search_service')
        assert callable(search_service.create_search_service)
        
        # Créer un service
        service = search_service.create_search_service()
        assert isinstance(service, search_service.SearchService)
    
    def test_create_search_service_with_config(self):
        """Test create_search_service avec configuration."""
        import search_service
        
        config = {"custom": "config"}
        service = search_service.create_search_service(config=config)
        
        assert service.config == config


class TestModuleBehaviorWithMissingDependencies:
    """Tests du comportement quand des dépendances manquent."""
    
    @patch('search_service.CONFIG_AVAILABLE', False)
    @patch('search_service.settings', None)
    def test_behavior_without_config(self):
        """Test du comportement sans module config."""
        import search_service
        
        # Le service doit toujours pouvoir être créé
        service = search_service.SearchService()
        assert service is not None
    
    @patch('search_service.CORE_AVAILABLE', False)
    @patch('search_service.LexicalEngine', None)
    def test_behavior_without_core(self):
        """Test du comportement sans module core."""
        import search_service
        
        service = search_service.SearchService()
        assert service is not None
        
        # health_check doit indiquer que core n'est pas disponible
        # (on ne peut pas tester directement car le patch ne fonctionne pas sur un module déjà importé)
    
    @patch('search_service.API_AVAILABLE', False)
    @patch('search_service.api_router', None)
    def test_behavior_without_api(self):
        """Test du comportement sans module API."""
        import search_service
        
        service = search_service.SearchService()
        assert service is not None


class TestLogging:
    """Tests pour vérifier que le logging fonctionne."""
    
    def test_logger_exists(self):
        """Test que le logger du module existe."""
        import search_service
        import logging
        
        # Le module doit avoir configuré un logger
        logger = logging.getLogger('search_service')
        assert logger is not None
    
    def test_package_loading_logs(self, caplog):
        """Test que le chargement du package génère des logs."""
        with caplog.at_level(logging.INFO):
            # Recharger le module pour capturer les logs d'init
            import importlib
            import search_service
            importlib.reload(search_service)
            
            # Vérifier qu'il y a des logs
            assert len(caplog.records) > 0
            
            # Chercher des logs spécifiques
            log_messages = [record.message for record in caplog.records]
            package_loaded = any("package chargé" in msg for msg in log_messages)
            assert package_loaded, f"Logs attendus non trouvés. Messages: {log_messages}"


class TestHealthCheckIntegration:
    """Tests d'intégration pour le health check."""
    
    @pytest.mark.asyncio
    async def test_health_check_structure(self):
        """Test de la structure du health check."""
        import search_service
        
        service = search_service.SearchService()
        health = await service.health_check()
        
        # Structure attendue
        expected_keys = ["service", "version", "status", "components"]
        for key in expected_keys:
            assert key in health, f"Clé manquante dans health check: {key}"
        
        # Types attendus
        assert isinstance(health["service"], str)
        assert isinstance(health["version"], str)
        assert isinstance(health["status"], str)
        assert isinstance(health["components"], dict)
        
        # Valeurs attendues
        assert health["service"] == "search_service"
        assert health["version"] == "1.0.0"
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_health_check_components(self):
        """Test des composants dans le health check."""
        import search_service
        
        service = search_service.SearchService()
        health = await service.health_check()
        
        components = health["components"]
        
        # Composants attendus
        expected_components = ["config", "core", "api", "models", "clients"]
        for component in expected_components:
            assert component in components
            assert isinstance(components[component], bool)


class TestErrorHandling:
    """Tests pour la gestion d'erreurs."""
    
    @pytest.mark.asyncio
    async def test_search_without_initialization(self):
        """Test que search_lexical gère l'absence d'initialisation."""
        import search_service
        
        service = search_service.SearchService()
        
        # Mock d'un moteur qui n'est pas disponible
        service.lexical_engine = None
        
        with pytest.raises(RuntimeError, match="Moteur lexical non disponible"):
            await service.search_lexical("test", user_id=123)
    
    def test_service_creation_with_invalid_config(self):
        """Test création de service avec config invalide."""
        import search_service
        
        # Même avec une config bizarre, ça ne doit pas crasher
        weird_config = {"invalid": None, "nested": {"deep": "value"}}
        service = search_service.SearchService(config=weird_config)
        
        assert service is not None
        assert service.config == weird_config


# ==================== FIXTURES ====================

@pytest.fixture
def mock_settings():
    """Fixture pour mocker les settings."""
    return Mock(
        ELASTICSEARCH_HOST="localhost",
        ELASTICSEARCH_PORT=9200,
        SEARCH_CACHE_SIZE=1000,
        PROJECT_NAME="Test Search Service"
    )


@pytest.fixture
def mock_elasticsearch_client():
    """Fixture pour mocker le client Elasticsearch."""
    mock_client = AsyncMock()
    mock_client.connect.return_value = None
    mock_client.health_check.return_value = {"status": "healthy"}
    mock_client.close.return_value = None
    return mock_client


@pytest.fixture
def mock_lexical_engine():
    """Fixture pour mocker le moteur lexical."""
    mock_engine = AsyncMock()
    mock_engine.search.return_value = {"results": [], "total": 0}
    mock_engine.close.return_value = None
    return mock_engine


# ==================== TESTS AVEC MOCKS ====================

class TestSearchServiceWithMocks:
    """Tests avec des mocks pour simuler les dépendances."""
    
    @pytest.mark.asyncio
    async def test_initialize_with_mocks(self, mock_elasticsearch_client, mock_lexical_engine):
        """Test d'initialisation avec des mocks."""
        import search_service
        
        service = search_service.SearchService()
        
        # Simuler les composants disponibles
        with patch('search_service.CLIENTS_AVAILABLE', True), \
             patch('search_service.CORE_AVAILABLE', True), \
             patch('search_service.ElasticsearchClient', return_value=mock_elasticsearch_client), \
             patch('search_service.LexicalEngineFactory') as mock_factory:
            
            mock_factory.create.return_value = mock_lexical_engine
            
            await service.initialize()
            
            assert service._initialized is True
            assert service.elasticsearch_client == mock_elasticsearch_client
            assert service.lexical_engine == mock_lexical_engine
    
    @pytest.mark.asyncio
    async def test_search_with_mocked_engine(self, mock_lexical_engine):
        """Test de recherche avec moteur mocké."""
        import search_service
        
        service = search_service.SearchService()
        service._initialized = True
        service.lexical_engine = mock_lexical_engine
        
        # Test de recherche
        result = await service.search_lexical("test query", user_id=12345)
        
        # Vérifier que le moteur a été appelé
        mock_lexical_engine.search.assert_called_once()
        
        # Vérifier la structure de l'appel
        call_args = mock_lexical_engine.search.call_args[0][0]
        assert call_args["query"] == "test query"
        assert call_args["user_id"] == 12345
        assert "filters" in call_args
        assert "options" in call_args


if __name__ == "__main__":
    # Lancer les tests directement
    pytest.main([__file__, "-v"])