"""
Configuration centralisée pour le Search Service.

Ce module expose la configuration optimisée pour le service de recherche lexicale
avec gestion d'erreurs robuste et fallbacks.
"""

import logging

logger = logging.getLogger(__name__)

# ==================== IMPORTS AVEC GESTION D'ERREURS ====================

try:
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
        get_production_config,
        get_elasticsearch_config,
        validate_settings,
        
        # Flags
        PYDANTIC_AVAILABLE
    )
    
    SETTINGS_AVAILABLE = True
    logger.info("✅ Module settings chargé avec succès")
    
except ImportError as e:
    logger.error(f"❌ Erreur import settings: {e}")
    
    # Configuration fallback minimale
    class FallbackSettings:
        PROJECT_NAME = "Search Service"
        VERSION = "1.0.0"
        API_V1_STR = "/api/v1"
        LOG_LEVEL = "INFO"
        CORS_ORIGINS = "*"
        
        ELASTICSEARCH_HOST = "localhost"
        ELASTICSEARCH_PORT = 9200
        ELASTICSEARCH_TIMEOUT = 30
        ELASTICSEARCH_INDEX = "transactions"
        
        SEARCH_CACHE_SIZE = 1000
        SEARCH_CACHE_TTL = 300
        SEARCH_MAX_LIMIT = 1000
        SEARCH_DEFAULT_SIZE = 20
        MAX_SEARCH_RESULTS = 1000
        
        RATE_LIMIT_ENABLED = False
        RATE_LIMIT_REQUESTS = 60
        RATE_LIMIT_PERIOD = 60
        
        LOG_TO_FILE = False
        LOG_FILE = "search_service.log"
        TESTING = False
    
    # Remplacements fallback
    SearchServiceSettings = FallbackSettings
    
    class LogLevel:
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
    
    settings = FallbackSettings()
    
    def get_settings():
        return settings
    
    def load_config_from_service():
        return None
    
    def get_development_config():
        return {"LOG_LEVEL": "DEBUG"}
    
    def get_test_config():
        return {"TESTING": True, "LOG_LEVEL": "WARNING"}
    
    def get_production_config():
        return {"LOG_LEVEL": "INFO"}
    
    def get_elasticsearch_config():
        return {
            "hosts": ["localhost:9200"],
            "timeout": 30,
            "max_retries": 3
        }
    
    def validate_settings(settings_obj):
        return True, []
    
    PYDANTIC_AVAILABLE = False
    SETTINGS_AVAILABLE = False
    
    logger.warning("⚠️ Utilisation de la configuration fallback")

# ==================== EXPORTS PRINCIPAUX ====================

__all__ = [
    # Classes
    'SearchServiceSettings',
    'LogLevel',
    
    # Instance globale
    'settings',
    
    # Fonctions
    'get_settings',
    'load_config_from_service',
    'get_development_config',
    'get_test_config',
    'get_production_config', 
    'get_elasticsearch_config',
    'validate_settings',
    
    # Flags
    'PYDANTIC_AVAILABLE',
    'SETTINGS_AVAILABLE'
]

# ==================== UTILITAIRES ====================

def get_config_summary() -> dict:
    """
    Retourne un résumé de la configuration actuelle.
    
    Returns:
        Dictionnaire avec résumé configuration
    """
    current_settings = get_settings()
    
    summary = {
        "module_status": "available" if SETTINGS_AVAILABLE else "fallback",
        "pydantic_available": PYDANTIC_AVAILABLE,
        "project_name": getattr(current_settings, 'PROJECT_NAME', 'Unknown'),
        "version": getattr(current_settings, 'VERSION', '1.0.0'),
        "log_level": getattr(current_settings, 'LOG_LEVEL', 'INFO'),
        "testing_mode": getattr(current_settings, 'TESTING', False),
        "elasticsearch": {
            "host": getattr(current_settings, 'ELASTICSEARCH_HOST', 'localhost'),
            "port": getattr(current_settings, 'ELASTICSEARCH_PORT', 9200),
            "timeout": getattr(current_settings, 'ELASTICSEARCH_TIMEOUT', 30)
        },
        "cache": {
            "size": getattr(current_settings, 'SEARCH_CACHE_SIZE', 1000),
            "ttl": getattr(current_settings, 'SEARCH_CACHE_TTL', 300)
        }
    }
    
    return summary

def check_config_health() -> tuple[bool, list]:
    """
    Vérifie la santé de la configuration.
    
    Returns:
        Tuple (is_healthy, issues)
    """
    issues = []
    
    if not SETTINGS_AVAILABLE:
        issues.append("Configuration utilise le mode fallback")
    
    current_settings = get_settings()
    
    # Vérifications de base
    if not hasattr(current_settings, 'ELASTICSEARCH_HOST'):
        issues.append("ELASTICSEARCH_HOST manquant")
    
    if not hasattr(current_settings, 'MAX_SEARCH_RESULTS'):
        issues.append("MAX_SEARCH_RESULTS manquant")
    
    # Validation avec la fonction dédiée si disponible
    try:
        is_valid, validation_errors = validate_settings(current_settings)
        if not is_valid:
            issues.extend(validation_errors)
    except Exception as e:
        issues.append(f"Erreur validation: {e}")
    
    is_healthy = len(issues) == 0
    
    return is_healthy, issues

# ==================== INITIALISATION ====================

# Vérification de santé au chargement
config_healthy, config_issues = check_config_health()

if config_healthy:
    logger.info("✅ Configuration Search Service en bonne santé")
else:
    logger.warning(f"⚠️ Problèmes de configuration détectés: {config_issues}")

# Log du résumé de configuration
config_summary = get_config_summary()
logger.info(f"Configuration chargée: {config_summary['project_name']} v{config_summary['version']} "
           f"(Status: {config_summary['module_status']})")