"""
Configuration centralisée pour le Search Service.

Ce module expose la configuration optimisée pour le service de recherche lexicale
spécialisé dans les transactions financières, avec intégration au config_service
centralisé et paramètres spécifiques Elasticsearch.

ARCHITECTURE CONFIGURATION:
- Configuration centralisée via config_service
- Paramètres spécialisés Elasticsearch
- Validation stricte des paramètres
- Support des environnements (dev/test/prod)
- Configuration cache et performance

RESPONSABILITÉS:
✅ Paramètres Elasticsearch optimisés
✅ Configuration cache Redis
✅ Limites et quotas recherche
✅ Templates requêtes par défaut
✅ Validation et sécurité
✅ Métriques et monitoring

USAGE:
    from search_service.config import settings, get_elasticsearch_config
    
    # Configuration principale
    es_config = get_elasticsearch_config()
    
    # Paramètres spécifiques
    timeout = settings.ELASTICSEARCH_TIMEOUT
    max_results = settings.SEARCH_MAX_LIMIT
"""

import os
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache

# Configuration principale
from .settings import (
    # Classe de configuration principale
    SearchServiceSettings,
    LogLevel,
    
    # Instance globale
    settings,
    
    # Fonctions de configuration
    get_settings,
    load_config_from_service,
    
    # Configurations par environnement
    get_development_config,
    get_test_config,
    get_production_config
)

# Logger pour ce module
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ELASTICSEARCH ====================

@lru_cache(maxsize=1)
def get_elasticsearch_config() -> Dict[str, Any]:
    """
    Configuration Elasticsearch optimisée pour le Search Service.
    
    Returns:
        Dict contenant tous les paramètres Elasticsearch
    """
    return {
        "hosts": [settings.elasticsearch_url],
        "timeout": settings.ELASTICSEARCH_TIMEOUT,
        "max_retries": settings.ELASTICSEARCH_MAX_RETRIES,
        "retry_on_timeout": settings.ELASTICSEARCH_RETRY_ON_TIMEOUT,
        "http_auth": (
            settings.ELASTICSEARCH_USERNAME,
            settings.ELASTICSEARCH_PASSWORD
        ) if settings.ELASTICSEARCH_USERNAME and settings.ELASTICSEARCH_PASSWORD else None,
        "verify_certs": settings.ELASTICSEARCH_VERIFY_CERTS,
        "use_ssl": settings.ELASTICSEARCH_USE_SSL,
        "ssl_show_warn": settings.ELASTICSEARCH_SSL_SHOW_WARN,
        "sniff_on_start": settings.ELASTICSEARCH_SNIFF_ON_START,
        "sniff_on_connection_fail": settings.ELASTICSEARCH_SNIFF_ON_CONNECTION_FAIL,
        "sniffer_timeout": settings.ELASTICSEARCH_SNIFFER_TIMEOUT,
        "connection_class": "default",
        "headers": {
            "Content-Type": "application/json",
            "User-Agent": f"search_service/{settings.SERVICE_VERSION}"
        }
    }

@lru_cache(maxsize=1)
def get_search_config() -> Dict[str, Any]:
    """
    Configuration recherche optimisée pour les transactions financières.
    
    Returns:
        Dict contenant les paramètres de recherche
    """
    return {
        "default_limit": settings.SEARCH_DEFAULT_LIMIT,
        "max_limit": settings.SEARCH_MAX_LIMIT,
        "default_timeout_ms": settings.SEARCH_DEFAULT_TIMEOUT_MS,
        "max_timeout_ms": settings.SEARCH_MAX_TIMEOUT_MS,
        "min_score": settings.SEARCH_MIN_SCORE,
        "boost_config": {
            "searchable_text": settings.BOOST_SEARCHABLE_TEXT,
            "primary_description": settings.BOOST_PRIMARY_DESCRIPTION,
            "merchant_name": settings.BOOST_MERCHANT_NAME,
            "category_name": settings.BOOST_CATEGORY_NAME,
            "clean_description": settings.BOOST_CLEAN_DESCRIPTION,
            "exact_phrase": settings.BOOST_EXACT_PHRASE
        },
        "features": {
            "fuzzy_enabled": settings.ENABLE_FUZZY,
            "wildcards_enabled": settings.ENABLE_WILDCARDS,
            "synonyms_enabled": settings.ENABLE_SYNONYMS,
            "highlighting_enabled": settings.HIGHLIGHT_ENABLED,
            "aggregations_enabled": settings.ENABLE_AGGREGATIONS
        },
        "highlighting": {
            "fragment_size": settings.HIGHLIGHT_FRAGMENT_SIZE,
            "max_fragments": settings.HIGHLIGHT_MAX_FRAGMENTS,
            "pre_tags": settings.HIGHLIGHT_PRE_TAGS,
            "post_tags": settings.HIGHLIGHT_POST_TAGS
        },
        "aggregations": {
            "max_buckets": settings.MAX_AGGREGATION_BUCKETS,
            "timeout_ms": settings.AGGREGATION_TIMEOUT_MS
        }
    }

@lru_cache(maxsize=1)
def get_cache_config() -> Dict[str, Any]:
    """
    Configuration cache pour les résultats de recherche.
    
    Returns:
        Dict contenant les paramètres de cache
    """
    return {
        "enabled": settings.CACHE_ENABLED,
        "type": settings.CACHE_TYPE,
        "redis_url": settings.REDIS_URL,
        "ttl_seconds": settings.CACHE_TTL_SECONDS,
        "max_size": settings.CACHE_MAX_SIZE,
        "key_prefix": settings.CACHE_KEY_PREFIX,
        "compression": settings.CACHE_COMPRESSION,
        "search_cache": {
            "enabled": settings.SEARCH_CACHE_ENABLED,
            "ttl": settings.SEARCH_CACHE_TTL,
            "max_size": settings.SEARCH_CACHE_MAX_SIZE
        },
        "template_cache": {
            "enabled": settings.TEMPLATE_CACHE_ENABLED,
            "size": settings.TEMPLATE_CACHE_SIZE
        }
    }

@lru_cache(maxsize=1)
def get_security_config() -> Dict[str, Any]:
    """
    Configuration sécurité pour les requêtes.
    
    Returns:
        Dict contenant les paramètres de sécurité
    """
    return {
        "require_user_id": settings.REQUIRE_USER_ID,
        "max_query_length": settings.MAX_QUERY_LENGTH,
        "allowed_fields": settings.ALLOWED_SEARCH_FIELDS,
        "forbidden_fields": settings.FORBIDDEN_SEARCH_FIELDS,
        "rate_limiting": {
            "enabled": settings.ENABLE_RATE_LIMITING,
            "requests_per_minute": settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "burst_size": settings.RATE_LIMIT_BURST_SIZE
        },
        "validation": {
            "strict_mode": settings.STRICT_VALIDATION,
            "validate_elasticsearch_syntax": settings.VALIDATE_ELASTICSEARCH_SYNTAX,
            "sanitize_inputs": settings.SANITIZE_INPUTS
        }
    }

@lru_cache(maxsize=1)
def get_monitoring_config() -> Dict[str, Any]:
    """
    Configuration monitoring et métriques.
    
    Returns:
        Dict contenant les paramètres de monitoring
    """
    return {
        "metrics_enabled": settings.ENABLE_METRICS,
        "detailed_logging": settings.DETAILED_LOGGING,
        "log_level": settings.LOG_LEVEL.value,
        "performance_monitoring": {
            "enabled": settings.PERFORMANCE_MONITORING_ENABLED,
            "slow_query_threshold_ms": settings.SLOW_QUERY_THRESHOLD_MS,
            "track_query_performance": settings.TRACK_QUERY_PERFORMANCE
        },
        "health_checks": {
            "elasticsearch_timeout": settings.HEALTH_CHECK_ELASTICSEARCH_TIMEOUT,
            "cache_timeout": settings.HEALTH_CHECK_CACHE_TIMEOUT,
            "interval_seconds": settings.HEALTH_CHECK_INTERVAL_SECONDS
        },
        "metrics_export": {
            "enabled": settings.METRICS_EXPORT_ENABLED,
            "endpoint": settings.METRICS_EXPORT_ENDPOINT,
            "interval_seconds": settings.METRICS_EXPORT_INTERVAL_SECONDS
        }
    }

# ==================== CONFIGURATION TEMPLATES ====================

@lru_cache(maxsize=1)
def get_query_templates_config() -> Dict[str, Any]:
    """
    Configuration templates de requêtes par intention.
    
    Returns:
        Dict contenant les paramètres des templates
    """
    return {
        "templates_enabled": settings.ENABLE_QUERY_TEMPLATES,
        "cache_size": settings.TEMPLATE_CACHE_SIZE,
        "validation_enabled": settings.ENABLE_TEMPLATE_VALIDATION,
        "default_templates": {
            "text_search": {
                "type": "multi_match",
                "fields": [
                    "searchable_text^2.0",
                    "primary_description^1.5",
                    "merchant_name^1.8",
                    "category_name^1.2"
                ],
                "fuzziness": "AUTO",
                "tie_breaker": 0.3
            },
            "category_search": {
                "type": "bool",
                "must": [
                    {"term": {"category_name.keyword": "{{category}}"}}
                ],
                "filter": [
                    {"term": {"user_id": "{{user_id}}"}}
                ]
            },
            "merchant_search": {
                "type": "bool",
                "must": [
                    {"term": {"merchant_name.keyword": "{{merchant}}"}}
                ],
                "filter": [
                    {"term": {"user_id": "{{user_id}}"}}
                ]
            },
            "amount_range": {
                "type": "bool",
                "must": [
                    {"range": {"amount_abs": {"gte": "{{min_amount}}", "lte": "{{max_amount}}"}}}
                ],
                "filter": [
                    {"term": {"user_id": "{{user_id}}"}}
                ]
            },
            "date_range": {
                "type": "bool",
                "must": [
                    {"range": {"date": {"gte": "{{start_date}}", "lte": "{{end_date}}"}}}
                ],
                "filter": [
                    {"term": {"user_id": "{{user_id}}"}}
                ]
            }
        }
    }

# ==================== CONFIGURATION VALIDATION ====================

def validate_config() -> Dict[str, Any]:
    """
    Valide la configuration complète du Search Service.
    
    Returns:
        Dict contenant les résultats de validation
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {}
    }
    
    try:
        # Validation Elasticsearch
        if not settings.ELASTICSEARCH_HOST:
            validation_results["errors"].append("ELASTICSEARCH_HOST is required")
            validation_results["valid"] = False
        
        if not (1 <= settings.ELASTICSEARCH_PORT <= 65535):
            validation_results["errors"].append("ELASTICSEARCH_PORT must be between 1 and 65535")
            validation_results["valid"] = False
        
        # Validation limites de recherche
        if settings.SEARCH_DEFAULT_LIMIT <= 0:
            validation_results["errors"].append("SEARCH_DEFAULT_LIMIT must be positive")
            validation_results["valid"] = False
        
        if settings.SEARCH_MAX_LIMIT < settings.SEARCH_DEFAULT_LIMIT:
            validation_results["errors"].append("SEARCH_MAX_LIMIT must be >= SEARCH_DEFAULT_LIMIT")
            validation_results["valid"] = False
        
        # Validation cache
        if settings.CACHE_ENABLED and not settings.REDIS_URL:
            validation_results["warnings"].append("Cache enabled but REDIS_URL not configured")
        
        # Validation timeouts
        if settings.SEARCH_DEFAULT_TIMEOUT_MS <= 0:
            validation_results["errors"].append("SEARCH_DEFAULT_TIMEOUT_MS must be positive")
            validation_results["valid"] = False
        
        if settings.SEARCH_MAX_TIMEOUT_MS < settings.SEARCH_DEFAULT_TIMEOUT_MS:
            validation_results["errors"].append("SEARCH_MAX_TIMEOUT_MS must be >= SEARCH_DEFAULT_TIMEOUT_MS")
            validation_results["valid"] = False
        
        # Checks réussis
        validation_results["checks"] = {
            "elasticsearch_config": len([e for e in validation_results["errors"] if "ELASTICSEARCH" in e]) == 0,
            "search_limits": settings.SEARCH_DEFAULT_LIMIT > 0 and settings.SEARCH_MAX_LIMIT >= settings.SEARCH_DEFAULT_LIMIT,
            "timeouts": settings.SEARCH_DEFAULT_TIMEOUT_MS > 0 and settings.SEARCH_MAX_TIMEOUT_MS >= settings.SEARCH_DEFAULT_TIMEOUT_MS,
            "cache_config": not settings.CACHE_ENABLED or bool(settings.REDIS_URL),
            "security_config": settings.REQUIRE_USER_ID,
            "monitoring_config": settings.ENABLE_METRICS
        }
        
        logger.info(f"Configuration validation: {'✅ VALID' if validation_results['valid'] else '❌ INVALID'}")
        
    except Exception as e:
        validation_results["errors"].append(f"Configuration validation failed: {str(e)}")
        validation_results["valid"] = False
        logger.error(f"Configuration validation error: {e}")
    
    return validation_results

# ==================== CONFIGURATION ENVIRONMENT ====================

def get_config_for_environment(env: str) -> SearchServiceSettings:
    """
    Retourne la configuration pour un environnement spécifique.
    
    Args:
        env: Environnement (dev, test, prod)
        
    Returns:
        Instance de configuration adaptée
    """
    env_configs = {
        "development": get_development_config,
        "dev": get_development_config,
        "test": get_test_config,
        "testing": get_test_config,
        "production": get_production_config,
        "prod": get_production_config
    }
    
    config_func = env_configs.get(env.lower())
    if not config_func:
        logger.warning(f"Unknown environment '{env}', using default configuration")
        return settings
    
    return config_func()

# ==================== CONFIGURATION SUMMARY ====================

def get_config_summary() -> Dict[str, Any]:
    """
    Retourne un résumé de la configuration pour debugging.
    
    Returns:
        Dict contenant les informations essentielles
    """
    return {
        "service": {
            "name": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "debug": settings.DEBUG,
            "log_level": settings.LOG_LEVEL.value
        },
        "elasticsearch": {
            "host": settings.ELASTICSEARCH_HOST,
            "port": settings.ELASTICSEARCH_PORT,
            "index": settings.ELASTICSEARCH_INDEX,
            "timeout": settings.ELASTICSEARCH_TIMEOUT,
            "auth_configured": bool(settings.ELASTICSEARCH_USERNAME)
        },
        "search": {
            "default_limit": settings.SEARCH_DEFAULT_LIMIT,
            "max_limit": settings.SEARCH_MAX_LIMIT,
            "default_timeout_ms": settings.SEARCH_DEFAULT_TIMEOUT_MS,
            "fuzzy_enabled": settings.ENABLE_FUZZY,
            "aggregations_enabled": settings.ENABLE_AGGREGATIONS
        },
        "cache": {
            "enabled": settings.CACHE_ENABLED,
            "type": settings.CACHE_TYPE,
            "ttl_seconds": settings.CACHE_TTL_SECONDS,
            "redis_configured": bool(settings.REDIS_URL)
        },
        "security": {
            "require_user_id": settings.REQUIRE_USER_ID,
            "rate_limiting_enabled": settings.ENABLE_RATE_LIMITING,
            "strict_validation": settings.STRICT_VALIDATION
        },
        "monitoring": {
            "metrics_enabled": settings.ENABLE_METRICS,
            "detailed_logging": settings.DETAILED_LOGGING,
            "performance_monitoring": settings.PERFORMANCE_MONITORING_ENABLED
        }
    }

# ==================== EXPORTS ====================

__all__ = [
    # Configuration principale
    "SearchServiceSettings",
    "LogLevel",
    "settings",
    "get_settings",
    
    # Configuration par environnement
    "get_development_config",
    "get_test_config", 
    "get_production_config",
    "get_config_for_environment",
    
    # Configuration spécialisée
    "get_elasticsearch_config",
    "get_search_config",
    "get_cache_config",
    "get_security_config",
    "get_monitoring_config",
    "get_query_templates_config",
    
    # Validation et utilities
    "validate_config",
    "get_config_summary",
    "load_config_from_service"
]

# ==================== INITIALISATION ====================

def _initialize_config():
    """Initialise la configuration au démarrage."""
    try:
        # Validation de la configuration
        validation = validate_config()
        
        if not validation["valid"]:
            logger.error("Configuration validation failed!")
            for error in validation["errors"]:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")
        
        # Log des warnings
        for warning in validation["warnings"]:
            logger.warning(f"Configuration warning: {warning}")
        
        # Log du résumé
        logger.info("Search Service configuration loaded successfully")
        logger.debug(f"Configuration summary: {get_config_summary()}")
        
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        raise

# Initialisation automatique
_initialize_config()