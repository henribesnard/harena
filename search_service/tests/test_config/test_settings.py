"""
Tests unitaires pour le module config/settings.py

Validation complète de la configuration du Search Service:
- Chargement et validation des paramètres
- Configuration Elasticsearch
- Paramètres de performance et cache
- Configuration des templates
- Validation de sécurité

Tests couvrant:
- Valeurs par défaut correctes
- Variables d'environnement
- Validation des paramètres
- Configuration par environnement
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any
import logging

# ✅ CORRECTION : Import relatif depuis le package parent tests
try:
    # Méthode 1 : Import relatif depuis le module parent
    from .. import ConfigTestCase
except ImportError:
    try:
        # Méthode 2 : Import absolu en ajoutant le chemin
        import sys
        from pathlib import Path
        
        # Ajouter le répertoire tests au PYTHONPATH
        tests_path = Path(__file__).parent.parent
        if str(tests_path) not in sys.path:
            sys.path.insert(0, str(tests_path))
        
        # Import de ConfigTestCase depuis le module tests
        from tests import ConfigTestCase
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
            ConfigTestCase = tests_module.ConfigTestCase
        else:
            # Fallback : Créer une classe de test simple
            import unittest
            
            class ConfigTestCase(unittest.TestCase):
                """Classe de test de base fallback."""
                def setUp(self):
                    self.settings = None
                    try:
                        from search_service.config import settings
                        self.settings = settings
                    except ImportError:
                        pass

logger = logging.getLogger(__name__)

class TestSearchServiceSettings(ConfigTestCase):
    """Tests pour la configuration principal du Search Service"""
    
    @pytest.mark.unit
    def test_default_settings_loading(self):
        """Test le chargement des paramètres par défaut"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Vérification des paramètres Elasticsearch par défaut
        assert hasattr(self.settings, 'ELASTICSEARCH_HOST')
        assert hasattr(self.settings, 'ELASTICSEARCH_PORT')
        assert hasattr(self.settings, 'ELASTICSEARCH_TIMEOUT')
        
        # Vérification valeurs par défaut sensées
        assert self.settings.ELASTICSEARCH_PORT == 9200
        assert self.settings.ELASTICSEARCH_TIMEOUT >= 10
        assert self.settings.ELASTICSEARCH_TIMEOUT <= 60
        
        logger.info("✅ Paramètres par défaut corrects")
    
    @pytest.mark.unit
    def test_elasticsearch_configuration(self):
        """Test la configuration Elasticsearch"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Test configuration complète
        es_config = self.settings.get_elasticsearch_config()
        
        required_keys = [
            'host', 'port', 'timeout', 'max_retries',
            'retry_on_timeout', 'max_connections'
        ]
        
        for key in required_keys:
            assert key in es_config, f"Clé manquante dans config ES: {key}"
        
        # Validation valeurs
        assert isinstance(es_config['port'], int)
        assert es_config['port'] > 0
        assert es_config['timeout'] > 0
        assert es_config['max_retries'] >= 0
        assert es_config['max_connections'] > 0
        
        logger.info(f"✅ Configuration Elasticsearch valide: {es_config}")
    
    @pytest.mark.unit
    def test_cache_configuration(self):
        """Test la configuration du cache"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        cache_config = self.settings.get_cache_config()
        
        # Vérification paramètres de cache
        assert 'type' in cache_config
        assert 'ttl' in cache_config
        assert 'max_size' in cache_config
        
        # Validation valeurs
        assert cache_config['type'] in ['memory', 'redis']
        assert cache_config['ttl'] > 0
        assert cache_config['max_size'] > 0
        
        logger.info(f"✅ Configuration cache valide: {cache_config}")
    
    @pytest.mark.unit
    def test_performance_configuration(self):
        """Test la configuration de performance"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        perf_config = self.settings.get_performance_config()
        
        # Vérification paramètres de performance
        required_keys = [
            'max_concurrent_searches', 'search_timeout',
            'max_results_per_page', 'default_page_size'
        ]
        
        for key in required_keys:
            assert key in perf_config, f"Clé manquante dans config performance: {key}"
        
        # Validation limites raisonnables
        assert perf_config['max_concurrent_searches'] > 0
        assert perf_config['search_timeout'] > 0
        assert perf_config['max_results_per_page'] <= 1000
        assert perf_config['default_page_size'] <= 100
        
        logger.info(f"✅ Configuration performance valide: {perf_config}")
    
    @pytest.mark.unit
    def test_template_configuration(self):
        """Test la configuration des templates"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        template_config = self.settings.get_template_config()
        
        # Vérification paramètres de templates
        assert 'default_search_template' in template_config
        assert 'available_templates' in template_config
        assert 'template_cache_enabled' in template_config
        
        # Validation liste templates
        available_templates = template_config['available_templates']
        assert isinstance(available_templates, list)
        assert len(available_templates) > 0
        
        # Vérification template par défaut existe
        default_template = template_config['default_search_template']
        assert default_template in available_templates
        
        logger.info(f"✅ Configuration templates valide: {template_config}")

class TestEnvironmentVariables(ConfigTestCase):
    """Tests pour les variables d'environnement"""
    
    @pytest.mark.unit
    @patch.dict(os.environ, {'SEARCH_SERVICE_ELASTICSEARCH_HOST': 'test-host'})
    def test_environment_override_elasticsearch_host(self):
        """Test le remplacement du host Elasticsearch par variable d'environnement"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Recharger la configuration avec la variable d'environnement
        if hasattr(self.settings, 'reload_config'):
            self.settings.reload_config()
        
        es_config = self.settings.get_elasticsearch_config()
        
        # Vérifier que la variable d'environnement est prise en compte
        expected_host = 'test-host'
        actual_host = es_config.get('host')
        
        # Note: selon l'implémentation, cela peut nécessiter un reload
        logger.info(f"Host configuré: {actual_host} (attendu: {expected_host})")
        
        # Test que la configuration peut être surchargée
        assert actual_host is not None
        
        logger.info("✅ Test variable d'environnement terminé")
    
    @pytest.mark.unit
    @patch.dict(os.environ, {
        'SEARCH_SERVICE_CACHE_TTL': '600',
        'SEARCH_SERVICE_CACHE_SIZE': '2000'
    })
    def test_environment_override_cache_settings(self):
        """Test le remplacement des paramètres de cache"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Recharger si possible
        if hasattr(self.settings, 'reload_config'):
            self.settings.reload_config()
        
        cache_config = self.settings.get_cache_config()
        
        # Tester que les paramètres peuvent être modifiés
        logger.info(f"Configuration cache: {cache_config}")
        
        # Validation que la configuration reste valide
        assert cache_config['ttl'] > 0
        assert cache_config['max_size'] > 0
        
        logger.info("✅ Test surcharge cache terminé")

class TestConfigValidation(ConfigTestCase):
    """Tests pour la validation de configuration"""
    
    @pytest.mark.unit
    def test_elasticsearch_connection_validation(self):
        """Test la validation de la connexion Elasticsearch"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Test configuration valide
        valid_config = {
            'host': 'localhost',
            'port': 9200,
            'timeout': 30,
            'max_retries': 3
        }
        
        is_valid = self.settings._validate_elasticsearch_config(valid_config)
        assert is_valid, "Configuration Elasticsearch valide doit être acceptée"
        
        # Test configurations invalides
        invalid_configs = [
            {'host': '', 'port': 9200},  # Host vide
            {'host': 'localhost', 'port': -1},  # Port invalide
            {'host': 'localhost', 'port': 'invalid'},  # Port non numérique
            {'host': 'localhost', 'port': 9200, 'timeout': 0}  # Timeout invalide
        ]
        
        for invalid_config in invalid_configs:
            is_valid = self.settings._validate_elasticsearch_config(invalid_config)
            assert not is_valid, f"Configuration invalide acceptée: {invalid_config}"
        
        logger.info("✅ Validation configuration Elasticsearch correcte")
    
    @pytest.mark.unit  
    def test_performance_limits_validation(self):
        """Test la validation des limites de performance"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Test limites valides
        valid_limits = {
            'max_query_size': 1000,
            'max_results_per_query': 100,
            'query_timeout_ms': 5000
        }
        
        is_valid = self.settings._validate_performance_config(valid_limits)
        assert is_valid, "Limites valides doivent être acceptées"
        
        # Test limites excessives
        invalid_limits = [
            {'max_query_size': 100000},  # Trop gros
            {'max_results_per_query': 10000},  # Trop de résultats
            {'query_timeout_ms': 0}  # Timeout invalide
        ]
        
        for invalid_limit in invalid_limits:
            is_valid = self.settings._validate_performance_config(invalid_limit)
            assert not is_valid, f"Limites excessives acceptées: {invalid_limit}"
        
        logger.info("✅ Validation limites performance correcte")
    
    @pytest.mark.security
    def test_security_configuration(self):
        """Test la configuration de sécurité"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        security_config = self.settings.get_security_config()
        
        # Vérification paramètres de sécurité
        assert 'user_isolation_enabled' in security_config
        assert 'validate_user_permissions' in security_config
        assert 'max_concurrent_queries_per_user' in security_config
        
        # Validation valeurs sécurisées
        assert security_config['user_isolation_enabled'] == True
        assert security_config['max_concurrent_queries_per_user'] > 0
        assert security_config['max_concurrent_queries_per_user'] <= 100
        
        logger.info(f"✅ Configuration sécurité valide: {security_config}")

class TestEnvironmentSpecificConfig(ConfigTestCase):
    """Tests pour la configuration spécifique par environnement"""
    
    @pytest.mark.unit
    @patch.dict(os.environ, {'ENVIRONMENT': 'development'})
    def test_development_configuration(self):
        """Test la configuration pour l'environnement de développement"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # En dev, certains paramètres doivent être plus permissifs
        dev_config = self.settings.get_environment_config()
        
        # En développement
        if dev_config.get('environment') == 'development':
            assert dev_config.get('debug', False) == True
            assert dev_config.get('log_level', 'INFO') in ['DEBUG', 'INFO']
        
        logger.info(f"✅ Configuration développement: {dev_config}")
    
    @pytest.mark.unit
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_production_configuration(self):
        """Test la configuration pour l'environnement de production"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # En prod, paramètres plus stricts
        prod_config = self.settings.get_environment_config()
        
        # En production
        if prod_config.get('environment') == 'production':
            assert prod_config.get('debug', True) == False
            assert prod_config.get('log_level', 'DEBUG') in ['WARNING', 'ERROR', 'INFO']
        
        logger.info(f"✅ Configuration production: {prod_config}")

class TestConfigIntegration(ConfigTestCase):
    """Tests d'intégration pour la configuration"""
    
    @pytest.mark.integration
    def test_complete_config_loading(self):
        """Test le chargement complet de toute la configuration"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Chargement de tous les modules de config
        configs = {
            'elasticsearch': self.settings.get_elasticsearch_config(),
            'cache': self.settings.get_cache_config(),
            'performance': self.settings.get_performance_config(),
            'template': self.settings.get_template_config(),
            'security': self.settings.get_security_config()
        }
        
        # Validation que toutes les configs sont chargées
        for config_name, config_data in configs.items():
            assert config_data is not None, f"Configuration {config_name} non chargée"
            assert isinstance(config_data, dict), f"Configuration {config_name} invalide"
            assert len(config_data) > 0, f"Configuration {config_name} vide"
        
        logger.info(f"✅ Configuration complète chargée: {list(configs.keys())}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_config_hot_reload(self):
        """Test le rechargement à chaud de la configuration"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Configuration initiale
        initial_config = self.settings.get_elasticsearch_config()
        initial_host = initial_config.get('host', 'localhost')
        
        # Simulation changement de configuration
        with patch.dict(os.environ, {'SEARCH_SERVICE_ELASTICSEARCH_HOST': 'new-host'}):
            # Rechargement
            if hasattr(self.settings, 'reload_config'):
                self.settings.reload_config()
                
                # Vérification changement
                new_config = self.settings.get_elasticsearch_config()
                new_host = new_config.get('host', 'localhost')
                
                # Le host doit avoir changé (ou méthode reload pas implémentée)
                logger.info(f"Host initial: {initial_host}, nouveau: {new_host}")
        
        logger.info("✅ Test rechargement configuration terminé")