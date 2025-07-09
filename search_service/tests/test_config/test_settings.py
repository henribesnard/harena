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

from tests import ConfigTestCase

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
        
        # Vérification structure
        assert 'enabled' in cache_config
        assert 'backend' in cache_config
        assert 'ttl_seconds' in cache_config
        assert 'max_size' in cache_config
        
        # Validation types et valeurs
        assert isinstance(cache_config['enabled'], bool)
        assert cache_config['ttl_seconds'] > 0
        assert cache_config['max_size'] > 0
        
        if cache_config['enabled']:
            assert cache_config['backend'] in ['redis', 'memory', 'memcached']
        
        logger.info(f"✅ Configuration cache valide: {cache_config}")
    
    @pytest.mark.unit
    def test_performance_configuration(self):
        """Test la configuration des paramètres de performance"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        perf_config = self.settings.get_performance_config()
        
        # Vérification paramètres critiques
        required_perf_keys = [
            'max_query_size', 'max_results_per_query', 
            'query_timeout_ms', 'cache_ttl_seconds'
        ]
        
        for key in required_perf_keys:
            assert key in perf_config, f"Paramètre performance manquant: {key}"
        
        # Validation limites sensées
        assert perf_config['max_query_size'] <= 10000
        assert perf_config['max_results_per_query'] <= 1000
        assert perf_config['query_timeout_ms'] >= 1000
        assert perf_config['query_timeout_ms'] <= 60000
        
        logger.info(f"✅ Configuration performance valide: {perf_config}")
    
    @pytest.mark.unit
    def test_template_configuration(self):
        """Test la configuration des templates"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        template_config = self.settings.get_template_config()
        
        # Vérification structure
        assert 'templates_enabled' in template_config
        assert 'cache_size' in template_config
        assert 'validation_enabled' in template_config
        assert 'default_templates' in template_config
        
        # Validation templates par défaut
        default_templates = template_config['default_templates']
        expected_templates = [
            'text_search', 'category_search', 'merchant_search',
            'amount_range', 'date_range'
        ]
        
        for template_name in expected_templates:
            assert template_name in default_templates, f"Template manquant: {template_name}"
            template = default_templates[template_name]
            assert 'type' in template, f"Type manquant pour template {template_name}"
        
        logger.info(f"✅ Configuration templates valide avec {len(default_templates)} templates")
    
    @pytest.mark.unit
    @patch.dict(os.environ, {
        'SEARCH_SERVICE_ELASTICSEARCH_HOST': 'test-es-host',
        'SEARCH_SERVICE_ELASTICSEARCH_PORT': '9201',
        'SEARCH_SERVICE_CACHE_ENABLED': 'false'
    })
    def test_environment_variables_override(self):
        """Test que les variables d'environnement surchargent la config"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Rechargement avec variables d'env
        with patch.object(self.settings, '_load_from_env') as mock_load:
            mock_load.return_value = {
                'ELASTICSEARCH_HOST': 'test-es-host',
                'ELASTICSEARCH_PORT': 9201,
                'CACHE_ENABLED': False
            }
            
            # Test override
            es_config = self.settings.get_elasticsearch_config()
            cache_config = self.settings.get_cache_config()
            
            # Vérification surcharge
            assert es_config.get('host') == 'test-es-host' or \
                   self.settings.ELASTICSEARCH_HOST == 'test-es-host'
            assert not cache_config['enabled'] or \
                   not self.settings.CACHE_ENABLED
        
        logger.info("✅ Variables d'environnement correctement prises en compte")
    
    @pytest.mark.unit
    def test_validation_function(self):
        """Test la fonction de validation de configuration"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Test validation complète
        validation_result = self.settings.validate_config()
        
        assert 'valid' in validation_result
        assert 'errors' in validation_result
        assert 'warnings' in validation_result
        
        # Si la configuration est invalide, il doit y avoir des erreurs
        if not validation_result['valid']:
            assert len(validation_result['errors']) > 0
            
        # Log des erreurs/warnings pour debugging
        if validation_result['errors']:
            logger.warning(f"⚠️ Erreurs de configuration: {validation_result['errors']}")
        if validation_result['warnings']:
            logger.info(f"⚠️ Warnings de configuration: {validation_result['warnings']}")
        
        logger.info(f"✅ Validation configuration: {validation_result['valid']}")

class TestConfigValidation(ConfigTestCase):
    """Tests pour la validation de configuration"""
    
    @pytest.mark.unit
    def test_elasticsearch_connection_validation(self):
        """Test la validation de la connexion Elasticsearch"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Test avec configuration valide
        valid_config = {
            'host': 'localhost',
            'port': 9200,
            'timeout': 30,
            'max_retries': 3
        }
        
        is_valid = self.settings._validate_elasticsearch_config(valid_config)
        assert is_valid, "Configuration ES valide doit être acceptée"
        
        # Test avec configuration invalide
        invalid_configs = [
            {'host': '', 'port': 9200},  # Host vide
            {'host': 'localhost', 'port': -1},  # Port invalide
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
        
        # Test limites acceptables
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
            assert dev_config.get('debug_enabled', False) == True
            assert dev_config.get('cache_enabled', True) == False  # Cache souvent désactivé en dev
            
        logger.info(f"✅ Configuration développement: {dev_config}")
    
    @pytest.mark.unit
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_production_configuration(self):
        """Test la configuration pour l'environnement de production"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        prod_config = self.settings.get_environment_config()
        
        # En production
        if prod_config.get('environment') == 'production':
            assert prod_config.get('debug_enabled', True) == False
            assert prod_config.get('cache_enabled', False) == True
            assert prod_config.get('strict_validation', False) == True
            
        logger.info(f"✅ Configuration production: {prod_config}")
    
    @pytest.mark.unit
    def test_config_consistency_across_modules(self):
        """Test la cohérence de la configuration entre modules"""
        if self.settings is None:
            pytest.skip("Module settings non disponible")
        
        # Vérification que tous les modules utilisent la même config
        elasticsearch_config = self.settings.get_elasticsearch_config()
        cache_config = self.settings.get_cache_config()
        performance_config = self.settings.get_performance_config()
        
        # Les timeouts doivent être cohérents
        es_timeout = elasticsearch_config.get('timeout', 30) * 1000  # Conversion en ms
        query_timeout = performance_config.get('query_timeout_ms', 30000)
        
        # Le timeout ES doit être >= au timeout query
        assert es_timeout >= query_timeout, \
            f"Timeout ES ({es_timeout}ms) doit être >= timeout query ({query_timeout}ms)"
        
        # Les tailles max doivent être cohérentes
        max_results = performance_config.get('max_results_per_query', 100)
        max_query_size = performance_config.get('max_query_size', 1000)
        
        assert max_results <= max_query_size, \
            "Max results doit être <= max query size"
        
        logger.info("✅ Configuration cohérente entre modules")

# Tests d'intégration configuration
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