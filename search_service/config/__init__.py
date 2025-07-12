"""
Module de configuration du Search Service
Expose les settings et constantes principales
"""

from .settings import (
    # Configuration principale
    settings,
    Settings,
    LogLevel,
    CacheBackend,
    
    # Constantes métier
    SUPPORTED_INTENT_TYPES,
    SUPPORTED_FILTER_OPERATORS,
    SUPPORTED_AGGREGATION_TYPES,
    INDEXED_FIELDS,
)

# Version du module de configuration
__version__ = "1.0.0"

# Exports principaux
__all__ = [
    # Configuration
    "settings",
    "Settings", 
    "LogLevel",
    "CacheBackend",
    
    # Constantes
    "SUPPORTED_INTENT_TYPES",
    "SUPPORTED_FILTER_OPERATORS", 
    "SUPPORTED_AGGREGATION_TYPES",
    "INDEXED_FIELDS",
    
    # Helpers
    "get_elasticsearch_config",
    "get_cache_config",
    "is_production",
    "is_development",
]

# Helpers pour accès rapide aux configurations
def get_elasticsearch_config():
    """Retourne la configuration Elasticsearch"""
    return settings.get_elasticsearch_config()

def get_cache_config():
    """Retourne la configuration cache"""
    return settings.get_cache_config()

def is_production():
    """Vérifie si on est en production"""
    return settings.is_production()

def is_development():
    """Vérifie si on est en développement"""
    return settings.is_development()

# Validation de la configuration au démarrage
def validate_config():
    """Valide la configuration au démarrage du service"""
    errors = []
    
    # Vérification Elasticsearch
    if not settings.elasticsearch_url:
        errors.append("elasticsearch_url est requis")
    
    if not settings.elasticsearch_index_name:
        errors.append("elasticsearch_index_name est requis")
    
    # Vérification limites cohérentes
    if settings.default_results_limit > settings.max_results_per_query:
        errors.append("default_results_limit ne peut pas être supérieur à max_results_per_query")
    
    if settings.default_query_timeout_ms > settings.max_query_timeout_ms:
        errors.append("default_query_timeout_ms ne peut pas être supérieur à max_query_timeout_ms")
    
    # Vérification cache Redis si activé
    if settings.cache_backend == CacheBackend.REDIS:
        if not settings.redis_url:
            errors.append("redis_url est requis quand cache_backend=redis")
    
    # Vérification champs obligatoires
    required_search_fields = ["searchable_text", "primary_description"]
    missing_fields = [f for f in required_search_fields if f not in settings.default_search_fields]
    if missing_fields:
        errors.append(f"Champs de recherche manquants: {missing_fields}")
    
    if errors:
        raise ValueError(f"Configuration invalide: {'; '.join(errors)}")
    
    return True

# Auto-validation au import si pas en mode test
import os
if not os.getenv("PYTEST_CURRENT_TEST"):
    try:
        validate_config()
    except Exception as e:
        print(f"⚠️  Avertissement configuration: {e}")