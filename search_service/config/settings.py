"""
Configuration du Search Service.

Ce module gère la configuration centralisée avec fallbacks
pour éviter les erreurs de dépendances manquantes.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# ==================== ENUMS ====================

class LogLevel(str, Enum):
    """Niveaux de log disponibles."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# ==================== CONFIGURATION DE BASE ====================

class SearchServiceSettings:
    """Configuration du Search Service avec fallbacks."""
    
    def __init__(self):
        # Configuration générale
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", "Harena Search Service")
        self.VERSION = "1.0.0"
        self.API_V1_STR = os.getenv("API_V1_STR", "/api/v1")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Configuration CORS
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
        
        # Configuration Elasticsearch
        self.ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
        self.ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
        self.ELASTICSEARCH_TIMEOUT = int(os.getenv("ELASTICSEARCH_TIMEOUT", "30"))
        self.ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "transactions")
        
        # Configuration cache
        self.SEARCH_CACHE_SIZE = int(os.getenv("SEARCH_CACHE_SIZE", "1000"))
        self.SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "300"))
        
        # Configuration recherche
        self.SEARCH_MAX_LIMIT = int(os.getenv("SEARCH_MAX_LIMIT", "1000"))
        self.SEARCH_DEFAULT_SIZE = int(os.getenv("SEARCH_DEFAULT_SIZE", "20"))
        self.MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "1000"))
        
        # Configuration sécurité
        self.RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
        self.RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))
        
        # Configuration logging
        self.LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
        self.LOG_FILE = os.getenv("LOG_FILE", "search_service.log")
        
        # Mode test
        self.TESTING = os.getenv("TESTING", "false").lower() == "true"

# ==================== CONFIGURATION AVEC PYDANTIC (OPTIONNEL) ====================

try:
    # Essayer d'importer pydantic-settings si disponible
    from pydantic_settings import BaseSettings
    from pydantic import Field
    
    class PydanticSearchServiceSettings(BaseSettings):
        """Configuration avec validation Pydantic si disponible."""
        
        # Configuration générale
        PROJECT_NAME: str = Field(default="Harena Search Service")
        VERSION: str = Field(default="1.0.0")
        API_V1_STR: str = Field(default="/api/v1")
        LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)
        
        # Configuration CORS
        CORS_ORIGINS: str = Field(default="*")
        
        # Configuration Elasticsearch
        ELASTICSEARCH_HOST: str = Field(default="localhost")
        ELASTICSEARCH_PORT: int = Field(default=9200)
        ELASTICSEARCH_TIMEOUT: int = Field(default=30)
        ELASTICSEARCH_INDEX: str = Field(default="transactions")
        
        # Configuration cache
        SEARCH_CACHE_SIZE: int = Field(default=1000)
        SEARCH_CACHE_TTL: int = Field(default=300)
        
        # Configuration recherche
        SEARCH_MAX_LIMIT: int = Field(default=1000)
        SEARCH_DEFAULT_SIZE: int = Field(default=20)
        MAX_SEARCH_RESULTS: int = Field(default=1000)
        
        # Configuration sécurité
        RATE_LIMIT_ENABLED: bool = Field(default=False)
        RATE_LIMIT_REQUESTS: int = Field(default=60)
        RATE_LIMIT_PERIOD: int = Field(default=60)
        
        # Configuration logging
        LOG_TO_FILE: bool = Field(default=False)
        LOG_FILE: str = Field(default="search_service.log")
        
        # Mode test
        TESTING: bool = Field(default=False)
        
        class Config:
            env_file = ".env"
            case_sensitive = True
    
    # Utiliser la version Pydantic si disponible
    SearchServiceSettingsClass = PydanticSearchServiceSettings
    PYDANTIC_AVAILABLE = True
    logger.info("✅ Configuration avec Pydantic disponible")
    
except ImportError as e:
    logger.warning(f"Pydantic non disponible - utilisation configuration simple: {e}")
    SearchServiceSettingsClass = SearchServiceSettings
    PYDANTIC_AVAILABLE = False

# ==================== CONFIGURATION GLOBALE ====================

# Instance globale de configuration
_settings_instance = None

def get_settings() -> SearchServiceSettings:
    """
    Récupère l'instance de configuration globale.
    
    Returns:
        Instance de configuration
    """
    global _settings_instance
    
    if _settings_instance is None:
        try:
            _settings_instance = SearchServiceSettingsClass()
            logger.info("✅ Configuration initialisée")
        except Exception as e:
            logger.error(f"Erreur initialisation configuration: {e}")
            # Fallback vers configuration simple
            _settings_instance = SearchServiceSettings()
            logger.info("✅ Configuration fallback utilisée")
    
    return _settings_instance

# Instance par défaut
settings = get_settings()

# ==================== FONCTIONS UTILITAIRES ====================

def load_config_from_service():
    """
    Tente de charger la configuration depuis config_service.
    
    Returns:
        Configuration chargée ou None si échec
    """
    try:
        from config_service.config import settings as global_settings
        
        # Copier les paramètres pertinents
        config_dict = {}
        
        # Mapping des attributs
        mapping = {
            'PROJECT_NAME': 'PROJECT_NAME',
            'API_V1_STR': 'API_V1_STR', 
            'LOG_LEVEL': 'LOG_LEVEL',
            'CORS_ORIGINS': 'CORS_ORIGINS'
        }
        
        for search_attr, global_attr in mapping.items():
            if hasattr(global_settings, global_attr):
                config_dict[search_attr] = getattr(global_settings, global_attr)
        
        logger.info("✅ Configuration chargée depuis config_service")
        return config_dict
        
    except ImportError as e:
        logger.warning(f"config_service non disponible: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur chargement config_service: {e}")
        return None

def get_development_config() -> Dict[str, Any]:
    """Configuration pour l'environnement de développement."""
    return {
        "LOG_LEVEL": "DEBUG",
        "ELASTICSEARCH_HOST": "localhost",
        "ELASTICSEARCH_PORT": 9200,
        "SEARCH_CACHE_SIZE": 100,
        "SEARCH_CACHE_TTL": 60,
        "CORS_ORIGINS": "*",
        "RATE_LIMIT_ENABLED": False
    }

def get_test_config() -> Dict[str, Any]:
    """Configuration pour les tests."""
    return {
        "TESTING": True,
        "LOG_LEVEL": "WARNING",
        "ELASTICSEARCH_HOST": "localhost",
        "ELASTICSEARCH_PORT": 9200,
        "SEARCH_CACHE_SIZE": 10,
        "SEARCH_CACHE_TTL": 30,
        "CORS_ORIGINS": "*",
        "RATE_LIMIT_ENABLED": False,
        "MAX_SEARCH_RESULTS": 100
    }

def get_production_config() -> Dict[str, Any]:
    """Configuration pour la production."""
    return {
        "LOG_LEVEL": "INFO",
        "LOG_TO_FILE": True,
        "SEARCH_CACHE_SIZE": 5000,
        "SEARCH_CACHE_TTL": 600,
        "RATE_LIMIT_ENABLED": True,
        "RATE_LIMIT_REQUESTS": 100,
        "RATE_LIMIT_PERIOD": 60
    }

def get_elasticsearch_config() -> Dict[str, Any]:
    """
    Configuration Elasticsearch optimisée.
    
    Returns:
        Configuration pour le client Elasticsearch
    """
    config = {
        "hosts": [f"{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"],
        "timeout": settings.ELASTICSEARCH_TIMEOUT,
        "max_retries": 3,
        "retry_on_timeout": True,
        "retry_on_status": [502, 503, 504],
        "request_timeout": settings.ELASTICSEARCH_TIMEOUT
    }
    
    return config

# ==================== VALIDATION ====================

def validate_settings(settings_obj) -> tuple[bool, List[str]]:
    """
    Valide la configuration.
    
    Args:
        settings_obj: Instance de configuration à valider
        
    Returns:
        Tuple (is_valid, errors)
    """
    errors = []
    
    # Validation des valeurs obligatoires
    if not hasattr(settings_obj, 'ELASTICSEARCH_HOST') or not settings_obj.ELASTICSEARCH_HOST:
        errors.append("ELASTICSEARCH_HOST manquant")
    
    if not hasattr(settings_obj, 'ELASTICSEARCH_PORT') or settings_obj.ELASTICSEARCH_PORT <= 0:
        errors.append("ELASTICSEARCH_PORT invalide")
    
    if hasattr(settings_obj, 'SEARCH_CACHE_SIZE') and settings_obj.SEARCH_CACHE_SIZE <= 0:
        errors.append("SEARCH_CACHE_SIZE doit être positif")
    
    if hasattr(settings_obj, 'SEARCH_CACHE_TTL') and settings_obj.SEARCH_CACHE_TTL <= 0:
        errors.append("SEARCH_CACHE_TTL doit être positif")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info("✅ Configuration validée")
    else:
        logger.error(f"❌ Configuration invalide: {errors}")
    
    return is_valid, errors

# ==================== EXPORTS ====================

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
    'PYDANTIC_AVAILABLE'
]

# Validation au chargement
is_valid, validation_errors = validate_settings(settings)
if not is_valid:
    logger.warning(f"Configuration avec erreurs: {validation_errors}")

logger.info(f"Configuration Search Service chargée (Pydantic: {PYDANTIC_AVAILABLE})")