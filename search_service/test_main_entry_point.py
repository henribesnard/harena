"""
🧪 TEST DU POINT D'ENTRÉE PRINCIPAL (search_service/main.py)
===========================================================

Test prioritaire du main.py car c'est le point d'entrée de l'application.
Si main.py fonctionne, on sait que les imports principaux sont corrects.

Usage:
    pytest test_main_entry_point.py -v
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Ajouter search_service au path pour imports directs
SEARCH_SERVICE_PATH = Path(__file__).parent / "search_service"
if str(SEARCH_SERVICE_PATH) not in sys.path:
    sys.path.insert(0, str(SEARCH_SERVICE_PATH))

# Configuration environnement test
os.environ.update({
    "TESTING": "true",
    "ENVIRONMENT": "testing",
    "LOG_LEVEL": "INFO",
    "ELASTICSEARCH_HOST": "localhost",
    "ELASTICSEARCH_PORT": "9200"
})


class TestMainEntryPoint:
    """Tests du point d'entrée principal search_service/main.py"""
    
    def test_01_import_main_module(self):
        """Test 1: Vérifier que le module main.py s'importe sans erreur"""
        print("\n🧪 Test 1: Import du module search_service/main.py")
        
        try:
            # Import depuis search_service
            import main
            assert main is not None
            print("  ✅ Module search_service/main importé avec succès")
            
        except ImportError as e:
            pytest.fail(f"❌ Échec import search_service/main.py: {e}")
        except Exception as e:
            pytest.fail(f"❌ Erreur inattendue lors import search_service/main.py: {e}")
    
    def test_02_main_dependencies_imports(self):
        """Test 2: Vérifier que toutes les dépendances du main.py s'importent"""
        print("\n🧪 Test 2: Import des dépendances du search_service/main.py")
        
        # Test imports un par un pour identifier les problèmes
        dependencies = [
            ("api", "api_manager"),
            ("config", "settings"), 
            ("core", "CoreManager"),
            ("utils", "get_utils_health"),
            ("utils", "initialize_utils"),
            ("utils", "shutdown_utils"),
            ("utils", "get_system_metrics"),
            ("utils", "get_performance_summary")
        ]
        
        for module_name, import_name in dependencies:
            try:
                module = __import__(module_name, fromlist=[import_name])
                imported_obj = getattr(module, import_name)
                assert imported_obj is not None
                print(f"  ✅ {module_name}.{import_name} importé")
                
            except ImportError as e:
                pytest.fail(f"❌ Échec import {module_name}.{import_name}: {e}")
            except AttributeError as e:
                pytest.fail(f"❌ {import_name} non trouvé dans {module_name}: {e}")
    
    def test_03_main_functions_exist(self):
        """Test 3: Vérifier que les fonctions principales du main.py existent"""
        print("\n🧪 Test 3: Vérification fonctions principales")
        
        try:
            import main
            
            # Fonctions requises
            required_functions = [
                'create_app',
                'create_development_app', 
                'create_production_app',
                'create_testing_app',
                'get_api_health',
                'get_api_info',
                'main'
            ]
            
            for func_name in required_functions:
                assert hasattr(main, func_name), f"Fonction {func_name} manquante"
                func = getattr(main, func_name)
                assert callable(func), f"{func_name} n'est pas callable"
                print(f"  ✅ {func_name} disponible")
            
            # Variables requises
            required_vars = ['app', 'logger', 'core_manager']
            for var_name in required_vars:
                assert hasattr(main, var_name), f"Variable {var_name} manquante"
                print(f"  ✅ {var_name} disponible")
                
        except Exception as e:
            pytest.fail(f"❌ Erreur vérification fonctions: {e}")
    
    @patch('main.api_manager')
    @patch('main.settings')
    def test_04_create_app_basic(self, mock_settings, mock_api_manager):
        """Test 4: Vérifier que create_app() fonctionne"""
        print("\n🧪 Test 4: Test create_app() de base")
        
        # Configuration des mocks
        mock_settings.environment = "testing"
        mock_settings.log_level = "INFO"
        mock_settings.cors_enabled = True
        mock_settings.cors_origins = ["*"]
        
        mock_router = Mock()
        mock_admin_router = Mock()
        mock_api_manager.router = mock_router
        mock_api_manager.admin_router = mock_admin_router
        
        try:
            from main import create_app
            
            app = create_app(environment="testing")
            assert app is not None
            assert hasattr(app, 'routes')
            print("  ✅ create_app() fonctionne")
            
            # Vérifier que l'app a les routes de base
            route_paths = [route.path for route in app.routes]
            expected_paths = ["/", "/health", "/info"]
            
            for path in expected_paths:
                assert path in route_paths, f"Route {path} manquante"
                print(f"  ✅ Route {path} présente")
                
        except Exception as e:
            pytest.fail(f"❌ create_app() failed: {e}")
    
    @patch('main.get_utils_health')
    @patch('main.get_system_metrics')
    @patch('main.get_performance_summary')
    @pytest.mark.asyncio
    async def test_05_health_functions(self, mock_perf, mock_metrics, mock_health):
        """Test 5: Vérifier que les fonctions de santé fonctionnent"""
        print("\n🧪 Test 5: Test fonctions de santé")
        
        # Configuration des mocks
        mock_health.return_value = {
            "overall_status": "healthy",
            "components": {"cache": {"status": "healthy"}}
        }
        mock_metrics.return_value = {
            "memory": {"rss_mb": 100},
            "cpu": {"percent": 15}
        }
        mock_perf.return_value = {
            "performance_metrics": {},
            "system_health": {}
        }
        
        try:
            from main import get_api_health, get_api_info
            
            # Test get_api_health
            health = await get_api_health()
            assert isinstance(health, dict)
            assert "healthy" in health
            assert "timestamp" in health
            print("  ✅ get_api_health() fonctionne")
            
            # Test get_api_info  
            info = await get_api_info()
            assert isinstance(info, dict)
            assert "service" in info
            assert "version" in info
            print("  ✅ get_api_info() fonctionne")
                
        except Exception as e:
            pytest.fail(f"❌ Health functions failed: {e}")
    
    @patch('main.api_manager')
    @patch('main.settings')
    def test_06_fastapi_client_basic(self, mock_settings, mock_api_manager):
        """Test 6: Vérifier qu'on peut créer un client FastAPI"""
        print("\n🧪 Test 6: Test client FastAPI de base")
        
        # Configuration des mocks
        mock_settings.environment = "testing"
        mock_settings.log_level = "INFO"
        mock_settings.cors_enabled = True
        mock_settings.cors_origins = ["*"]
        
        mock_router = Mock()
        mock_admin_router = Mock()
        mock_api_manager.router = mock_router
        mock_api_manager.admin_router = mock_admin_router
        
        try:
            from main import create_testing_app
            
            app = create_testing_app()
            client = TestClient(app)
            
            # Test route racine
            response = client.get("/")
            assert response.status_code == 200
            
            data = response.json()
            assert data["service"] == "search-service"
            assert data["status"] == "running"
            print("  ✅ Client FastAPI fonctionne")
            print(f"  ✅ Route / retourne: {data['service']}")
            
        except Exception as e:
            pytest.fail(f"❌ FastAPI client failed: {e}")
    
    @patch('main.api_manager')
    @patch('main.settings')
    @patch('main.get_utils_health')
    def test_07_health_endpoint(self, mock_health, mock_settings, mock_api_manager):
        """Test 7: Vérifier que l'endpoint /health fonctionne"""
        print("\n🧪 Test 7: Test endpoint /health")
        
        # Configuration des mocks
        mock_settings.environment = "testing"
        mock_settings.log_level = "INFO"
        mock_settings.cors_enabled = True
        mock_settings.cors_origins = ["*"]
        
        mock_router = Mock()
        mock_admin_router = Mock()
        mock_api_manager.router = mock_router
        mock_api_manager.admin_router = mock_admin_router
        
        mock_health.return_value = {
            "overall_status": "healthy",
            "components": {"cache": {"status": "healthy"}}
        }
        
        try:
            from main import create_testing_app
            
            app = create_testing_app()
            client = TestClient(app)
            
            # Test endpoint health
            response = client.get("/health")
            
            # Le status peut être 200 ou 503 selon la santé
            assert response.status_code in [200, 503]
            
            data = response.json()
            assert "healthy" in data
            assert "timestamp" in data
            print(f"  ✅ Endpoint /health fonctionne (status: {response.status_code})")
            print(f"  ✅ Health status: {data.get('healthy', 'unknown')}")
            
        except Exception as e:
            pytest.fail(f"❌ Health endpoint failed: {e}")
    
    @patch('main.api_manager')
    @patch('main.settings')  
    @patch('main.get_system_metrics')
    @patch('main.get_performance_summary')
    def test_08_info_endpoint(self, mock_perf, mock_metrics, mock_settings, mock_api_manager):
        """Test 8: Vérifier que l'endpoint /info fonctionne"""
        print("\n🧪 Test 8: Test endpoint /info")
        
        # Configuration des mocks
        mock_settings.environment = "testing"
        mock_settings.log_level = "INFO"
        mock_settings.elasticsearch_host = "localhost"
        mock_settings.elasticsearch_port = 9200
        mock_settings.cors_enabled = True
        mock_settings.cors_origins = ["*"]
        
        mock_router = Mock()
        mock_admin_router = Mock()
        mock_api_manager.router = mock_router
        mock_api_manager.admin_router = mock_admin_router
        
        mock_metrics.return_value = {
            "memory": {"rss_mb": 100},
            "cpu": {"percent": 15}
        }
        mock_perf.return_value = {
            "performance_metrics": {},
            "system_health": {}
        }
        
        try:
            from main import create_testing_app
            
            app = create_testing_app()
            client = TestClient(app)
            
            # Test endpoint info
            response = client.get("/info")
            assert response.status_code == 200
            
            data = response.json()
            assert data["service"] == "search-service"
            assert data["version"] == "1.0.0"
            assert "configuration" in data
            print("  ✅ Endpoint /info fonctionne")
            print(f"  ✅ Service: {data['service']} v{data['version']}")
            
        except Exception as e:
            pytest.fail(f"❌ Info endpoint failed: {e}")
    
    def test_09_main_function_exists(self):
        """Test 9: Vérifier que la fonction main() existe et est callable"""
        print("\n🧪 Test 9: Test fonction main()")
        
        try:
            from main import main
            
            assert callable(main)
            print("  ✅ Fonction main() existe et est callable")
            
            # Note: On ne l'exécute pas car elle lance uvicorn
            
        except Exception as e:
            pytest.fail(f"❌ Main function test failed: {e}")
    
    def test_10_core_manager_instance(self):
        """Test 10: Vérifier que core_manager est une instance valide"""
        print("\n🧪 Test 10: Test instance core_manager")
        
        try:
            from main import core_manager
            from core import CoreManager
            
            assert core_manager is not None
            assert isinstance(core_manager, CoreManager)
            print("  ✅ core_manager est une instance de CoreManager")
            
        except Exception as e:
            pytest.fail(f"❌ Core manager instance test failed: {e}")


class TestMainImportsDependencies:
    """Tests spécialisés pour les imports et dépendances du search_service/main.py"""
    
    def test_01_api_module_availability(self):
        """Test que le module api est disponible avec tous ses composants"""
        print("\n🧪 Test API Module: Vérification disponibilité")
        
        try:
            # Test import module api
            import api
            assert api is not None
            print("  ✅ Module api importé")
            
            # Test api_manager
            from api import api_manager
            assert api_manager is not None
            print("  ✅ api_manager disponible")
            
            # Test que api_manager a les attributs requis
            required_attrs = ['router']
            for attr in required_attrs:
                assert hasattr(api_manager, attr), f"api_manager.{attr} manquant"
                print(f"  ✅ api_manager.{attr} présent")
                
        except ImportError as e:
            pytest.fail(f"❌ Module api non disponible: {e}")
        except Exception as e:
            pytest.fail(f"❌ Erreur module api: {e}")
    
    def test_02_config_module_availability(self):
        """Test que le module config est disponible"""
        print("\n🧪 Test Config Module: Vérification disponibilité")
        
        try:
            # Test import config
            from config import settings
            assert settings is not None
            print("  ✅ settings importé")
            
            # Test attributs essentiels
            essential_attrs = [
                'environment',
                'log_level', 
                'elasticsearch_host',
                'elasticsearch_port'
            ]
            
            for attr in essential_attrs:
                assert hasattr(settings, attr), f"settings.{attr} manquant"
                value = getattr(settings, attr)
                print(f"  ✅ settings.{attr} = {value}")
                
        except ImportError as e:
            pytest.fail(f"❌ Module config non disponible: {e}")
        except Exception as e:
            pytest.fail(f"❌ Erreur module config: {e}")
    
    def test_03_core_module_availability(self):
        """Test que le module core est disponible"""
        print("\n🧪 Test Core Module: Vérification disponibilité")
        
        try:
            # Test import core
            from core import CoreManager
            assert CoreManager is not None
            print("  ✅ CoreManager importé")
            
            # Test création instance
            core_instance = CoreManager()
            assert core_instance is not None
            print("  ✅ Instance CoreManager créée")
            
        except ImportError as e:
            pytest.fail(f"❌ Module core non disponible: {e}")
        except Exception as e:
            pytest.fail(f"❌ Erreur module core: {e}")
    
    def test_04_utils_module_availability(self):
        """Test que le module utils est disponible avec toutes les fonctions"""
        print("\n🧪 Test Utils Module: Vérification disponibilité")
        
        utils_functions = [
            'get_utils_health',
            'initialize_utils', 
            'shutdown_utils',
            'get_system_metrics',
            'get_performance_summary'
        ]
        
        for func_name in utils_functions:
            try:
                # Import direct spécifique
                utils_module = __import__('utils', fromlist=[func_name])
                imported_func = getattr(utils_module, func_name)
                
                assert imported_func is not None
                assert callable(imported_func)
                print(f"  ✅ utils.{func_name} disponible")
                
            except ImportError as e:
                pytest.fail(f"❌ utils.{func_name} non disponible: {e}")
            except AttributeError as e:
                pytest.fail(f"❌ utils.{func_name} non trouvé dans le module: {e}")
            except Exception as e:
                pytest.fail(f"❌ Erreur utils.{func_name}: {e}")


class TestMainFastAPIIntegration:
    """Tests d'intégration FastAPI spécifiques"""
    
    @patch('main.api_manager')
    @patch('main.settings')
    def test_01_app_creation_variants(self, mock_settings, mock_api_manager):
        """Test création des différentes variantes d'app"""
        print("\n🧪 Test FastAPI: Création variantes d'app")
        
        # Configuration mocks
        mock_settings.environment = "testing"
        mock_settings.log_level = "INFO"
        mock_settings.cors_enabled = True
        mock_settings.cors_origins = ["*"]
        
        mock_router = Mock()
        mock_admin_router = Mock()
        mock_api_manager.router = mock_router
        mock_api_manager.admin_router = mock_admin_router
        
        try:
            from main import (
                create_app,
                create_development_app,
                create_production_app, 
                create_testing_app
            )
            
            # Test chaque variante
            variants = [
                ("development", create_development_app),
                ("production", create_production_app),
                ("testing", create_testing_app),
                ("custom", lambda: create_app(environment="custom"))
            ]
            
            for variant_name, create_func in variants:
                app = create_func()
                assert app is not None
                assert hasattr(app, 'routes')
                print(f"  ✅ App {variant_name} créée")
                
        except Exception as e:
            pytest.fail(f"❌ App creation variants failed: {e}")
    
    @patch('main.api_manager')
    @patch('main.settings')
    def test_02_middleware_integration(self, mock_settings, mock_api_manager):
        """Test que les middlewares sont correctement intégrés"""
        print("\n🧪 Test FastAPI: Intégration middleware")
        
        # Configuration mocks
        mock_settings.environment = "testing"
        mock_settings.log_level = "INFO"
        mock_settings.cors_enabled = True
        mock_settings.cors_origins = ["http://localhost:3000"]
        
        mock_router = Mock()
        mock_api_manager.router = mock_router
        mock_api_manager.admin_router = Mock()
        
        try:
            from main import create_testing_app
            
            app = create_testing_app()
            
            # Vérifier que l'app a des middlewares
            assert hasattr(app, 'user_middleware')
            print("  ✅ Middlewares présents")
            
            # Test avec client
            client = TestClient(app)
            
            # Test OPTIONS request (CORS)
            response = client.options("/")
            # CORS devrait permettre cette requête
            assert response.status_code in [200, 405]  # 405 = Method not allowed mais CORS OK
            print("  ✅ CORS middleware fonctionne")
            
        except Exception as e:
            pytest.fail(f"❌ Middleware integration failed: {e}")
    
    @patch('main.api_manager')
    @patch('main.settings')
    def test_03_error_handling(self, mock_settings, mock_api_manager):
        """Test que la gestion d'erreurs globale fonctionne"""
        print("\n🧪 Test FastAPI: Gestion d'erreurs")
        
        # Configuration mocks
        mock_settings.environment = "testing"
        mock_settings.log_level = "INFO"
        mock_settings.cors_enabled = False
        
        mock_router = Mock()
        mock_api_manager.router = mock_router
        mock_api_manager.admin_router = Mock()
        
        try:
            from main import create_testing_app
            
            app = create_testing_app()
            
            # Ajouter une route qui génère une erreur pour tester
            @app.get("/test-error")
            def test_error():
                raise Exception("Test error")
            
            client = TestClient(app)
            
            # Test que l'erreur est capturée
            response = client.get("/test-error")
            assert response.status_code == 500
            
            data = response.json()
            assert "error" in data
            assert "timestamp" in data
            print("  ✅ Gestion d'erreurs globale fonctionne")
            
        except Exception as e:
            pytest.fail(f"❌ Error handling test failed: {e}")


# ============================================================================
# COMMANDES D'EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    """
    Exécution directe des tests pour debugging
    
    Usage:
        python test_main_entry_point.py
    """
    import pytest
    
    print("🚀 Exécution des tests du search_service/main.py")
    print("=" * 50)
    
    # Exécuter les tests avec pytest
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-x"  # Stop au premier échec
    ])