"""
Configuration centralisée pour le Search Service.

Toutes les variables de configuration sont définies ici
et chargées depuis les variables d'environnement via config_service.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum


class LogLevel(str, Enum):
    """Niveaux de log supportés."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SearchServiceSettings(BaseSettings):
    """
    Configuration principale du Search Service.
    
    Toutes les variables sont chargées depuis l'environnement
    avec des valeurs par défaut sensées pour le développement.
    """
    
    # === SERVICE CONFIGURATION ===
    SERVICE_NAME: str = Field(default="search_service", description="Nom du service")
    SERVICE_VERSION: str = Field(default="1.0.0", description="Version du service")
    DEBUG: bool = Field(default=False, description="Mode debug")
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, description="Niveau de log")
    
    # === API CONFIGURATION ===
    API_HOST: str = Field(default="0.0.0.0", description="Host API")
    API_PORT: int = Field(default=8001, description="Port API")
    API_PREFIX: str = Field(default="/api/v1", description="Préfixe API")
    API_DOCS_URL: Optional[str] = Field(default="/docs", description="URL documentation API")
    API_REDOC_URL: Optional[str] = Field(default="/redoc", description="URL ReDoc")
    
    # === ELASTICSEARCH CONFIGURATION ===
    ELASTICSEARCH_HOST: str = Field(default="localhost", description="Host Elasticsearch")
    ELASTICSEARCH_PORT: int = Field(default=9200, description="Port Elasticsearch")
    ELASTICSEARCH_SCHEME: str = Field(default="http", description="Schéma Elasticsearch")
    ELASTICSEARCH_USERNAME: Optional[str] = Field(default=None, description="Nom utilisateur ES")
    ELASTICSEARCH_PASSWORD: Optional[str] = Field(default=None, description="Mot de passe ES")
    ELASTICSEARCH_INDEX: str = Field(default="harena_transactions", description="Index principal")
    ELASTICSEARCH_TIMEOUT: int = Field(default=30, description="Timeout requêtes ES (s)")
    ELASTICSEARCH_MAX_RETRIES: int = Field(default=3, description="Nombre max de retry")
    ELASTICSEARCH_RETRY_ON_TIMEOUT: bool = Field(default=True, description="Retry sur timeout")
    
    # === SEARCH CONFIGURATION ===
    SEARCH_DEFAULT_LIMIT: int = Field(default=20, description="Limite par défaut résultats")
    SEARCH_MAX_LIMIT: int = Field(default=100, description="Limite max résultats")
    SEARCH_DEFAULT_TIMEOUT_MS: int = Field(default=5000, description="Timeout recherche (ms)")
    SEARCH_MAX_TIMEOUT_MS: int = Field(default=10000, description="Timeout max recherche (ms)")
    SEARCH_ENABLE_HIGHLIGHTING: bool = Field(default=True, description="Activer highlighting")
    SEARCH_HIGHLIGHT_FRAGMENT_SIZE: int = Field(default=150, description="Taille fragments highlight")
    SEARCH_HIGHLIGHT_MAX_FRAGMENTS: int = Field(default=3, description="Max fragments highlight")
    
    # === CACHE CONFIGURATION ===
    CACHE_ENABLED: bool = Field(default=True, description="Activer cache")
    CACHE_TYPE: str = Field(default="memory", description="Type de cache (memory/redis)")
    CACHE_TTL_SECONDS: int = Field(default=300, description="TTL cache (s)")
    CACHE_MAX_SIZE: int = Field(default=1000, description="Taille max cache")
    REDIS_URL: Optional[str] = Field(default=None, description="URL Redis pour cache")
    
    # === PERFORMANCE CONFIGURATION ===
    ENABLE_QUERY_OPTIMIZATION: bool = Field(default=True, description="Optimisation requêtes")
    ENABLE_RESULT_CACHING: bool = Field(default=True, description="Cache résultats")
    ENABLE_METRICS: bool = Field(default=True, description="Collecte métriques")
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, description="Max requêtes concurrentes")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=30, description="Timeout requêtes HTTP")
    
    # === SECURITY CONFIGURATION ===
    ENABLE_RATE_LIMITING: bool = Field(default=True, description="Limitation débit")
    RATE_LIMIT_PER_MINUTE: int = Field(default=1000, description="Limite requêtes/minute")
    REQUIRE_USER_ID: bool = Field(default=True, description="user_id obligatoire")
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="Origins CORS autorisées")
    
    # === MONITORING CONFIGURATION ===
    ENABLE_HEALTH_CHECKS: bool = Field(default=True, description="Health checks")
    HEALTH_CHECK_INTERVAL_SECONDS: int = Field(default=30, description="Intervalle health checks")
    METRICS_EXPORT_INTERVAL_SECONDS: int = Field(default=60, description="Export métriques")
    
    # === TEMPLATE CONFIGURATION ===
    ENABLE_QUERY_TEMPLATES: bool = Field(default=True, description="Templates requêtes")
    TEMPLATE_CACHE_SIZE: int = Field(default=100, description="Cache templates")
    ENABLE_TEMPLATE_VALIDATION: bool = Field(default=True, description="Validation templates")
    
    # === AGGREGATION CONFIGURATION ===
    ENABLE_AGGREGATIONS: bool = Field(default=True, description="Agrégations")
    MAX_AGGREGATION_BUCKETS: int = Field(default=1000, description="Max buckets agrégation")
    AGGREGATION_TIMEOUT_MS: int = Field(default=5000, description="Timeout agrégations")
    
    @validator('ELASTICSEARCH_HOST')
    def validate_elasticsearch_host(cls, v):
        if not v or not v.strip():
            raise ValueError("ELASTICSEARCH_HOST ne peut pas être vide")
        return v.strip()
    
    @validator('ELASTICSEARCH_PORT')
    def validate_elasticsearch_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError("ELASTICSEARCH_PORT doit être entre 1 et 65535")
        return v
    
    @validator('SEARCH_DEFAULT_LIMIT')
    def validate_search_default_limit(cls, v):
        if v <= 0:
            raise ValueError("SEARCH_DEFAULT_LIMIT doit être positif")
        return v
    
    @validator('SEARCH_MAX_LIMIT')
    def validate_search_max_limit(cls, v, values):
        default_limit = values.get('SEARCH_DEFAULT_LIMIT', 20)
        if v < default_limit:
            raise ValueError("SEARCH_MAX_LIMIT doit être >= SEARCH_DEFAULT_LIMIT")
        return v
    
    @validator('CACHE_TTL_SECONDS')
    def validate_cache_ttl(cls, v):
        if v <= 0:
            raise ValueError("CACHE_TTL_SECONDS doit être positif")
        return v
    
    @property
    def elasticsearch_url(self) -> str:
        """URL complète Elasticsearch."""
        auth_part = ""
        if self.ELASTICSEARCH_USERNAME and self.ELASTICSEARCH_PASSWORD:
            auth_part = f"{self.ELASTICSEARCH_USERNAME}:{self.ELASTICSEARCH_PASSWORD}@"
        
        return (f"{self.ELASTICSEARCH_SCHEME}://{auth_part}"
                f"{self.ELASTICSEARCH_HOST}:{self.ELASTICSEARCH_PORT}")
    
    @property
    def is_production(self) -> bool:
        """Indique si on est en production."""
        return not self.DEBUG and self.LOG_LEVEL != LogLevel.DEBUG
    
    @property
    def cache_config(self) -> Dict[str, Any]:
        """Configuration cache formatée."""
        return {
            "enabled": self.CACHE_ENABLED,
            "type": self.CACHE_TYPE,
            "ttl_seconds": self.CACHE_TTL_SECONDS,
            "max_size": self.CACHE_MAX_SIZE,
            "redis_url": self.REDIS_URL
        }
    
    @property
    def elasticsearch_config(self) -> Dict[str, Any]:
        """Configuration Elasticsearch formatée."""
        config = {
            "hosts": [self.elasticsearch_url],
            "timeout": self.ELASTICSEARCH_TIMEOUT,
            "max_retries": self.ELASTICSEARCH_MAX_RETRIES,
            "retry_on_timeout": self.ELASTICSEARCH_RETRY_ON_TIMEOUT
        }
        
        if self.ELASTICSEARCH_USERNAME and self.ELASTICSEARCH_PASSWORD:
            config["http_auth"] = (
                self.ELASTICSEARCH_USERNAME, 
                self.ELASTICSEARCH_PASSWORD
            )
        
        return config
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_prefix = "SEARCH_SERVICE_"


# Instance globale des settings
settings = SearchServiceSettings()


def get_settings() -> SearchServiceSettings:
    """
    Récupère l'instance des settings.
    
    Utilisé pour l'injection de dépendances FastAPI.
    """
    return settings


def load_config_from_service(config_service_url: Optional[str] = None) -> SearchServiceSettings:
    """
    Charge la configuration depuis le config_service.
    
    Args:
        config_service_url: URL du service de configuration
        
    Returns:
        Instance de configuration mise à jour
        
    Note:
        Cette fonction sera appelée au démarrage pour récupérer
        la configuration centralisée depuis le config_service.
    """
    # TODO: Implémenter l'appel au config_service
    # Pour l'instant, on utilise les variables d'environnement
    return settings


# === CONFIGURATION PAR ENVIRONNEMENT ===

def get_development_config() -> SearchServiceSettings:
    """Configuration pour l'environnement de développement."""
    return SearchServiceSettings(
        DEBUG=True,
        LOG_LEVEL=LogLevel.DEBUG,
        ELASTICSEARCH_HOST="localhost",
        CACHE_ENABLED=True,
        ENABLE_METRICS=True,
        REQUIRE_USER_ID=False  # Plus flexible en dev
    )


def get_test_config() -> SearchServiceSettings:
    """Configuration pour les tests."""
    return SearchServiceSettings(
        DEBUG=True,
        LOG_LEVEL=LogLevel.DEBUG,
        ELASTICSEARCH_HOST="localhost",
        ELASTICSEARCH_INDEX="test_harena_transactions",
        CACHE_ENABLED=False,  # Pas de cache en test
        ENABLE_RATE_LIMITING=False,  # Pas de rate limiting en test
        REQUIRE_USER_ID=False
    )


def get_production_config() -> SearchServiceSettings:
    """Configuration pour la production."""
    return SearchServiceSettings(
        DEBUG=False,
        LOG_LEVEL=LogLevel.INFO,
        CACHE_ENABLED=True,
        ENABLE_RATE_LIMITING=True,
        REQUIRE_USER_ID=True,
        ENABLE_METRICS=True
    )


# === HELPER FUNCTIONS ===

def get_config_by_environment(env: str = None) -> SearchServiceSettings:
    """
    Récupère la configuration selon l'environnement.
    
    Args:
        env: Environnement (development/test/production)
        
    Returns:
        Configuration adaptée à l'environnement
    """
    env = env or os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "test":
        return get_test_config()
    elif env == "production":
        return get_production_config()
    else:
        return get_development_config()