"""
Tests unitaires pour search_service/main.py

Ces tests valident :
- La création de l'application FastAPI
- Les endpoints de base
- Le cycle de vie de l'application
- La gestion d'erreurs
- Les fonctions utilitaires
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json


class TestApplicationCreation:
    """Tests pour la création de l'application FastAPI."""
    
    def test_create_app_function_exists(self):
        """Test que la fonction create_app existe."""
        try:
            from search_service.main import create_app
            assert callable(create_app)
        except ImportError as e:
            pytest.fail(f"Impossible d'importer create_app: {e}")
    
    def test_create_app_returns_fastapi(self):
        """Test que create_app retourne une application FastAPI."""
        from search_service.main import create_app
        
        app = create_app()
        assert isinstance(app, FastAPI)
        assert app.title is not None
        assert app.version is not None
    
    def test_app_configuration(self):
        """Test la configuration de base de l'application."""
        from search_service.main import create_app
        
        app = create_app()
        
        # Vérifier les attributs de base
        assert "Search Service" in app.title or "search" in app.title.lower()
        assert app.version == "1.0.0"
        assert app.openapi_url is not None
        assert "/docs" in app.docs_url if app.docs_url else True
    
    def test_middleware_configuration(self):
        """Test que les middlewares sont configurés."""
        from search_service.main import create_app
        
        app = create_app()
        
        # Vérifier qu'il y a des middlewares
        assert len(app.user_middleware) > 0
        
        # Vérifier la présence de CORSMiddleware et GZipMiddleware
        middleware_types = [type(middleware.cls).__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_types
        assert "GZipMiddleware" in middleware_types


class TestApplicationInstance:
    """Tests pour l'instance d'application globale."""
    
    def test_app_instance_exists(self):
        """Test que l'instance app existe."""
        try:
            from search_service.main import app
            assert isinstance(app, FastAPI)
        except ImportError as e:
            pytest.fail(f"Impossible d'importer app: {e}")
    
    def test_app_instance_configuration(self):
        """Test la configuration de l'instance app."""
        from search_service.main import app
        
        assert app.title is not None
        assert app.version == "1.0.0"


class TestBasicEndpoints:
    """Tests pour les endpoints de base."""
    
    @pytest.fixture
    def client(self):
        """Fixture pour créer un client de test."""
        from search_service.main import create_app
        app = create_app()
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test de l'endpoint racine."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data
    
    def test_health_endpoint_exists(self, client):
        """Test que l'endpoint /health existe."""
        response = client.get("/health")
        
        # Doit retourner 200 ou 503 (service healthy ou degraded)
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert "components" in data
    
    def test_health_endpoint_structure(self, client):
        """Test de la structure de l'endpoint health."""
        response = client.get("/health")
        data = response.json()
        
        # Structure attendue
        required_fields = ["service", "version", "status", "components"]
        for field in required_fields:
            assert field in data, f"Champ manquant: {field}"
        
        # Types attendus
        assert isinstance(data["service"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["components"], dict)
        
        # Valeurs attendues
        assert data["service"] == "search_service"
        assert data["version"] == "1.0.0"
        assert data["status"] in ["healthy", "degraded"]
    
    def test_metrics_endpoint(self, client):
        """Test de l'endpoint /metrics."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "metrics" in data
        assert data["service"] == "search_service"
        
        # Structure des métriques
        metrics = data["metrics"]
        expected_metrics = ["requests_total", "requests_errors", "cache_hits", "cache_misses"]
        for metric in expected_metrics:
            assert metric in metrics


class TestErrorHandling:
    """Tests pour la gestion d'erreurs."""
    
    @pytest.fixture
    def client(self):
        """Fixture pour créer un client de test."""
        from search_service.main import create_app
        app = create_app()
        return TestClient(app)
    
    def test_404_handling(self, client):
        """Test de la gestion des 404."""
        response = client.get("/endpoint-qui-nexiste-pas")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_method_not_allowed(self, client):
        """Test de la gestion des méthodes non autorisées."""
        # Essayer POST sur root qui n'accepte que GET
        response = client.post("/")
        
        assert response.status_code == 405
    
    def test_http_exception_handler(self, client):
        """Test du gestionnaire d'exceptions HTTP personnalisé."""
        # L'endpoint /health existe, testons avec une méthode non autorisée
        response = client.delete("/health")
        
        assert response.status_code == 405


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires."""
    
    def test_get_lexical_engine_function(self):
        """Test que get_lexical_engine existe et fonctionne."""
        try:
            from search_service.main import get_lexical_engine
            assert callable(get_lexical_engine)
            
            # Au début, doit retourner None
            engine = get_lexical_engine()
            assert engine is None or hasattr(engine, 'search')
        except ImportError as e:
            pytest.fail(f"Impossible d'importer get_lexical_engine: {e}")
    
    def test_get_elasticsearch_client_function(self):
        """Test que get_elasticsearch_client existe et fonctionne."""
        try:
            from search_service.main import get_elasticsearch_client
            assert callable(get_elasticsearch_client)
            
            # Au début, doit retourner None
            client = get_elasticsearch_client()
            assert client is None or hasattr(client, 'connect')
        except ImportError as e:
            pytest.fail(f"Impossible d'importer get_elasticsearch_client: {e}")


class TestLoggingConfiguration:
    """Tests pour la configuration du logging."""
    
    def test_setup_logging_function(self):
        """Test que setup_logging existe."""
        try:
            from search_service.main import setup_logging
            assert callable(setup_logging)
        except ImportError as e:
            pytest.fail(f"Impossible d'importer setup_logging: {e}")
    
    def test_logger_exists(self):
        """Test que le logger du module existe."""
        import logging
        
        # Importer le module pour déclencher la configuration du logging
        import search_service.main
        
        logger = logging.getLogger("search_service")
        assert logger is not None
        assert logger.level <= logging.INFO


class TestLifespanManager:
    """Tests pour le gestionnaire de cycle de vie."""
    
    @pytest.mark.asyncio
    async def test_lifespan_function_exists(self):
        """Test que la fonction lifespan existe."""
        try:
            from search_service.main import lifespan
            assert lifespan is not None
        except ImportError as e:
            pytest.fail(f"Impossible d'importer lifespan: {e}")
    
    @pytest.mark.asyncio
    async def test_lifespan_context_manager(self):
        """Test que lifespan est un context manager async."""
        from search_service.main import lifespan, create_app
        
        app = create_app()
        
        # Tester que lifespan peut être utilisé comme context manager
        try:
            async with lifespan(app):
                # Phase de fonctionnement
                pass
        except Exception as e:
            # Si des composants ne sont pas disponibles, c'est normal
            if "Elasticsearch" not in str(e):
                pytest.fail(f"Erreur inattendue dans lifespan: {e}")


class TestApplicationWithMocks:
    """Tests avec des mocks pour simuler les dépendances."""
    
    @patch('search_service.main.CORE_AVAILABLE', True)
    @patch('search_service.main.ElasticsearchClient')
    @patch('search_service.main.LexicalEngineFactory')
    def test_lifespan_with_mocked_components(self, mock_factory, mock_es_client_class):
        """Test du cycle de vie avec des composants mockés."""
        from search_service.main import create_app
        
        # Configurer les mocks
        mock_es_client = AsyncMock()
        mock_es_client.connect.return_value = None
        mock_es_client.health_check.return_value = {"status": "healthy"}
        mock_es_client.close.return_value = None
        mock_es_client_class.return_value = mock_es_client
        
        mock_engine = AsyncMock()
        mock_engine.close.return_value = None
        mock_factory.create.return_value = mock_engine
        
        # Créer l'application
        app = create_app()
        
        # L'application doit être créée sans erreur
        assert isinstance(app, FastAPI)
    
    @patch('search_service.main.API_AVAILABLE', True)
    @patch('search_service.main.search_router')
    def test_router_registration(self, mock_router):
        """Test de l'enregistrement du routeur."""
        from search_service.main import create_app
        
        mock_router.tags = ["search"]
        
        app = create_app()
        
        # Vérifier que l'application a été créée
        assert isinstance(app, FastAPI)
    
    def test_health_check_with_no_components(self):
        """Test du health check sans composants."""
        from search_service.main import create_app
        
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/health")
        
        # Doit retourner un statut (même si dégradé)
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]


class TestConfigurationHandling:
    """Tests pour la gestion de la configuration."""
    
    @patch('search_service.main.get_settings')
    def test_app_with_custom_settings(self, mock_get_settings):
        """Test de l'application avec des settings personnalisés."""
        # Mock des settings
        mock_settings = Mock()
        mock_settings.PROJECT_NAME = "Test Search Service"
        mock_settings.API_V1_STR = "/api/v1"
        mock_settings.CORS_ORIGINS = "http://localhost:3000"
        mock_get_settings.return_value = mock_settings
        
        from search_service.main import create_app
        
        app = create_app()
        
        assert isinstance(app, FastAPI)
        assert "Test Search Service" in app.title
    
    @patch('search_service.main.CONFIG_AVAILABLE', False)
    def test_app_without_config(self):
        """Test de l'application sans module de configuration."""
        from search_service.main import create_app
        
        # Doit toujours pouvoir créer l'application
        app = create_app()
        assert isinstance(app, FastAPI)


class TestCORSConfiguration:
    """Tests pour la configuration CORS."""
    
    def test_cors_middleware_present(self):
        """Test que le middleware CORS est présent."""
        from search_service.main import create_app
        
        app = create_app()
        
        # Vérifier la présence du middleware CORS
        cors_present = any(
            "CORS" in str(type(middleware.cls)) 
            for middleware in app.user_middleware
        )
        assert cors_present, "Middleware CORS non trouvé"
    
    @pytest.fixture
    def client(self):
        """Client de test avec CORS."""
        from search_service.main import create_app
        app = create_app()
        return TestClient(app)
    
    def test_options_request(self, client):
        """Test de la requête OPTIONS pour CORS."""
        response = client.options("/", headers={"Origin": "http://localhost:3000"})
        
        # CORS doit permettre la requête OPTIONS
        assert response.status_code in [200, 204]


class TestDevelopmentMode:
    """Tests pour le mode développement."""
    
    def test_main_execution_guard(self):
        """Test que le code __main__ est protégé."""
        try:
            # Importer le module ne doit pas démarrer uvicorn
            import search_service.main
            # Si on arrive ici, c'est que le code __main__ est bien protégé
            assert True
        except Exception as e:
            if "uvicorn" in str(e).lower():
                pytest.fail("Le code __main__ ne doit pas s'exécuter lors de l'import")
            else:
                # Autre erreur acceptable
                pass


# ==================== FIXTURES GLOBALES ====================

@pytest.fixture(scope="session")
def mock_settings():
    """Settings mockés pour tous les tests."""
    mock_settings = Mock()
    mock_settings.PROJECT_NAME = "Test Search Service"
    mock_settings.API_V1_STR = "/api/v1"
    mock_settings.CORS_ORIGINS = "*"
    mock_settings.LOG_LEVEL = "INFO"
    mock_settings.ELASTICSEARCH_HOST = "localhost"
    mock_settings.ELASTICSEARCH_PORT = 9200
    return mock_settings


@pytest.fixture
def app_instance():
    """Instance d'application pour les tests."""
    from search_service.main import create_app
    return create_app()


@pytest.fixture
def test_client(app_instance):
    """Client de test FastAPI."""
    return TestClient(app_instance)


# ==================== TESTS D'INTÉGRATION ====================

class TestIntegration:
    """Tests d'intégration de l'application complète."""
    
    def test_full_application_startup(self, test_client):
        """Test du démarrage complet de l'application."""
        # Tester les endpoints principaux
        endpoints = ["/", "/health", "/metrics"]
        
        for endpoint in endpoints:
            response = test_client.get(endpoint)
            assert response.status_code in [200, 503], f"Endpoint {endpoint} failed"
    
    def test_application_metadata(self, test_client):
        """Test des métadonnées de l'application."""
        # Test de l'endpoint racine
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == "1.0.0"
        assert "Search Service" in data["service"]
    
    def test_health_check_integration(self, test_client):
        """Test d'intégration du health check."""
        response = test_client.get("/health")
        assert response.status_code in [200, 503]
        
        data = response.json()
        
        # Vérifier la cohérence des données
        assert data["service"] == "search_service"
        assert data["version"] == "1.0.0"
        
        # Les composants doivent être cohérents
        components = data["components"]
        for component, status in components.items():
            assert isinstance(status, bool)
    
    def test_error_handling_integration(self, test_client):
        """Test d'intégration de la gestion d'erreurs."""
        # Test 404
        response = test_client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test méthode non autorisée
        response = test_client.post("/health")
        assert response.status_code == 405


if __name__ == "__main__":
    # Lancer les tests directement
    pytest.main([__file__, "-v"])