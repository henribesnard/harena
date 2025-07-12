"""
Configuration centralisée pour le Search Service
Spécialisé pour la recherche lexicale Elasticsearch haute performance
"""

import os
from typing import Dict, List, Optional
from enum import Enum

# === IMPORT PYDANTIC COMPATIBLE V1/V2 ===
try:
    # Pydantic v2 - nouveau package
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
except ImportError:
    # Pydantic v1 - ancien import
    from pydantic import BaseSettings, Field, validator


class LogLevel(str, Enum):
    """Niveaux de logging disponibles"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class CacheBackend(str, Enum):
    """Types de cache supportés"""
    MEMORY = "memory"
    REDIS = "redis"


class Settings(BaseSettings):
    """Configuration principale du Search Service"""
    
    # === CONFIGURATION GÉNÉRALE ===
    app_name: str = Field(default="Search Service", description="Nom de l'application")
    app_version: str = Field(default="1.0.0", description="Version du service")
    environment: str = Field(default="development", description="Environnement (dev/staging/prod)")
    debug: bool = Field(default=False, description="Mode debug")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Niveau de logging")
    
    # === CONFIGURATION SERVEUR ===
    host: str = Field(default="0.0.0.0", description="Host du serveur")
    port: int = Field(default=8001, description="Port du serveur")
    workers: int = Field(default=1, description="Nombre de workers uvicorn")
    reload: bool = Field(default=False, description="Auto-reload en développement")
    
    # === CONFIGURATION ELASTICSEARCH ===
    # Connexion principale
    elasticsearch_host: str = Field(default="localhost", description="Host Elasticsearch")
    elasticsearch_port: int = Field(default=9200, description="Port Elasticsearch")
    elasticsearch_url: str = Field(
        default="http://localhost:9200",
        description="URL Elasticsearch"
    )
    elasticsearch_username: Optional[str] = Field(
        default=None,
        description="Username Elasticsearch (optionnel)"
    )
    elasticsearch_password: Optional[str] = Field(
        default=None,
        description="Password Elasticsearch (optionnel)"
    )
    elasticsearch_use_ssl: bool = Field(
        default=False,
        description="Utiliser SSL pour Elasticsearch"
    )
    elasticsearch_verify_certs: bool = Field(
        default=True,
        description="Vérifier certificats SSL"
    )
    
    # Performance Elasticsearch
    elasticsearch_timeout: int = Field(
        default=30,
        description="Timeout connexion Elasticsearch (secondes)"
    )
    elasticsearch_max_retries: int = Field(
        default=3,
        description="Nombre maximum de tentatives"
    )
    elasticsearch_retry_on_timeout: bool = Field(
        default=True,
        description="Retry automatique sur timeout"
    )
    elasticsearch_max_connections: int = Field(
        default=20,
        description="Pool de connexions maximum"
    )
    
    # Index et configuration
    elasticsearch_index_name: str = Field(
        default="harena_transactions",
        description="Nom de l'index principal"
    )
    elasticsearch_doc_type: str = Field(
        default="_doc",
        description="Type de document Elasticsearch"
    )
    
    # === CONFIGURATION CACHE ===
    cache_enabled: bool = Field(default=True, description="Activer le cache")
    cache_backend: CacheBackend = Field(
        default=CacheBackend.MEMORY,
        description="Backend de cache"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="TTL par défaut du cache (5 min)"
    )
    cache_max_size: int = Field(
        default=1000,
        description="Taille maximum cache en mémoire"
    )
    
    # Redis (si utilisé)
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="URL Redis pour cache"
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Password Redis (optionnel)"
    )
    redis_timeout: int = Field(
        default=5,
        description="Timeout Redis (secondes)"
    )
    
    # === LIMITES ET QUOTAS ===
    # Limites par requête
    max_results_per_query: int = Field(
        default=100,
        description="Nombre maximum de résultats par requête"
    )
    default_results_limit: int = Field(
        default=20,
        description="Limite par défaut si non spécifiée"
    )
    max_query_timeout_ms: int = Field(
        default=10000,
        description="Timeout maximum par requête (10s)"
    )
    default_query_timeout_ms: int = Field(
        default=5000,
        description="Timeout par défaut (5s)"
    )
    
    # Rate limiting
    rate_limit_requests_per_minute: int = Field(
        default=100,
        description="Requêtes max par minute par client"
    )
    rate_limit_burst: int = Field(
        default=20,
        description="Burst maximum autorisé"
    )
    
    # === CONFIGURATION RECHERCHE ===
    # Scoring et pertinence
    default_min_score: float = Field(
        default=0.1,
        description="Score minimum pour considérer un résultat pertinent"
    )
    text_search_boost_multiplier: float = Field(
        default=2.0,
        description="Multiplicateur boost pour recherche textuelle"
    )
    
    # Pagination
    max_pagination_offset: int = Field(
        default=10000,
        description="Offset maximum pour pagination (limite Elasticsearch)"
    )
    
    # Agrégations
    max_aggregation_buckets: int = Field(
        default=1000,
        description="Nombre maximum de buckets par agrégation"
    )
    enable_aggregations_by_default: bool = Field(
        default=True,
        description="Activer agrégations par défaut"
    )
    
    # === TEMPLATES DE REQUÊTES ===
    template_cache_enabled: bool = Field(
        default=True,
        description="Activer cache des templates"
    )
    template_validation_strict: bool = Field(
        default=True,
        description="Validation stricte des templates"
    )
    
    # === MONITORING ET MÉTRIQUES ===
    metrics_enabled: bool = Field(
        default=True,
        description="Activer collecte de métriques"
    )
    metrics_prefix: str = Field(
        default="search_service",
        description="Préfixe pour les métriques"
    )
    health_check_timeout: int = Field(
        default=5,
        description="Timeout health check (secondes)"
    )
    
    # Logging avancé
    log_elasticsearch_queries: bool = Field(
        default=False,
        description="Logger les requêtes Elasticsearch (debug only)"
    )
    log_performance_slow_queries_ms: int = Field(
        default=1000,
        description="Seuil pour logger les requêtes lentes"
    )
    
    # === SÉCURITÉ ===
    allow_all_origins: bool = Field(
        default=False,
        description="Autoriser toutes les origines CORS (dev only)"
    )
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"],
        description="Origines autorisées pour CORS"
    )
    
    # Validation requêtes
    validate_user_id_required: bool = Field(
        default=True,
        description="user_id obligatoire dans toutes les requêtes"
    )
    max_text_search_length: int = Field(
        default=500,
        description="Longueur maximum recherche textuelle"
    )
    
    # === CONFIGURATION AVANCÉE ===
    # Optimisations Elasticsearch
    elasticsearch_preference: str = Field(
        default="_local",
        description="Préférence de routing Elasticsearch"
    )
    elasticsearch_request_cache: bool = Field(
        default=True,
        description="Utiliser cache de requêtes Elasticsearch"
    )
    elasticsearch_search_type: str = Field(
        default="dfs_query_then_fetch",
        description="Type de recherche Elasticsearch"
    )
    
    # Champs par défaut
    default_search_fields: List[str] = Field(
        default_factory=lambda: [
            "searchable_text",
            "primary_description", 
            "merchant_name",
            "category_name"
        ],
        description="Champs de recherche par défaut"
    )
    
    default_return_fields: List[str] = Field(
        default_factory=lambda: [
            "transaction_id",
            "user_id",
            "account_id",
            "amount",
            "amount_abs",
            "transaction_type",
            "currency_code",
            "date",
            "primary_description",
            "merchant_name",
            "category_name",
            "operation_type",
            "month_year"
        ],
        description="Champs retournés par défaut"
    )

    # === CONFIGURATION PYDANTIC ===
    class Config:
        """Configuration Pydantic"""
        env_file = ".env"
        env_prefix = "SEARCH_SERVICE_"
        case_sensitive = False
        validate_assignment = True

    @validator("elasticsearch_timeout")
    def validate_elasticsearch_timeout(cls, v):
        """Valide le timeout Elasticsearch"""
        if v < 1 or v > 300:
            raise ValueError("Elasticsearch timeout doit être entre 1 et 300 secondes")
        return v

    @validator("max_results_per_query")
    def validate_max_results(cls, v):
        """Valide la limite de résultats"""
        if v < 1 or v > 10000:
            raise ValueError("max_results_per_query doit être entre 1 et 10000")
        return v

    @validator("cache_ttl_seconds")
    def validate_cache_ttl(cls, v):
        """Valide le TTL du cache"""
        if v < 60 or v > 3600:
            raise ValueError("cache_ttl_seconds doit être entre 60 et 3600 secondes")
        return v

    @validator("rate_limit_requests_per_minute")
    def validate_rate_limit(cls, v):
        """Valide le rate limiting"""
        if v < 1 or v > 10000:
            raise ValueError("rate_limit_requests_per_minute doit être entre 1 et 10000")
        return v

    @validator("default_search_fields")
    def validate_search_fields(cls, v):
        """Valide les champs de recherche"""
        required_fields = ["searchable_text", "primary_description"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Champ obligatoire manquant: {field}")
        return v

    def get_elasticsearch_config(self) -> Dict:
        """Retourne la configuration Elasticsearch optimisée"""
        config = {
            "hosts": [self.elasticsearch_url],
            "timeout": self.elasticsearch_timeout,
            "max_retries": self.elasticsearch_max_retries,
            "retry_on_timeout": self.elasticsearch_retry_on_timeout,
            "maxsize": self.elasticsearch_max_connections,
        }
        
        # Authentification si configurée
        if self.elasticsearch_username and self.elasticsearch_password:
            config["http_auth"] = (self.elasticsearch_username, self.elasticsearch_password)
        
        # SSL si activé
        if self.elasticsearch_use_ssl:
            config["use_ssl"] = True
            config["verify_certs"] = self.elasticsearch_verify_certs
        
        return config

    def get_cache_config(self) -> Dict:
        """Retourne la configuration cache"""
        if self.cache_backend == CacheBackend.REDIS:
            return {
                "backend": "redis",
                "url": self.redis_url,
                "password": self.redis_password,
                "timeout": self.redis_timeout,
                "ttl": self.cache_ttl_seconds
            }
        else:
            return {
                "backend": "memory",
                "max_size": self.cache_max_size,
                "ttl": self.cache_ttl_seconds
            }

    def is_production(self) -> bool:
        """Vérifie si on est en production"""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Vérifie si on est en développement"""
        return self.environment.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        """True si environnement de test"""
        return self.environment.lower() == "testing"
    
    @property
    def cors_origins(self) -> List[str]:
        """Retourne les origines CORS"""
        if self.allow_all_origins:
            return ["*"]
        return self.allowed_origins
    
    @property  
    def access_log_enabled(self) -> bool:
        """Logs d'accès activés"""
        return not self.is_production() or self.debug


# Instance globale des settings
settings = Settings()


# === CONSTANTES MÉTIER ===

# Types d'intentions supportés (synchronisés avec conversation_service)
SUPPORTED_INTENT_TYPES = [
    "SEARCH_BY_CATEGORY",
    "SEARCH_BY_MERCHANT", 
    "SEARCH_BY_AMOUNT",
    "SEARCH_BY_DATE",
    "TEXT_SEARCH",
    "COUNT_OPERATIONS",
    "TEMPORAL_ANALYSIS",
    "AGGREGATE_BY_CATEGORY",
    "AGGREGATE_BY_MERCHANT",
    "BALANCE_CALCULATION",
    "SPENDING_PATTERNS",
    "TOP_MERCHANTS"
]

# Opérateurs de filtre supportés
SUPPORTED_FILTER_OPERATORS = [
    "eq",      # égal
    "ne",      # différent
    "gt",      # supérieur
    "gte",     # supérieur ou égal
    "lt",      # inférieur
    "lte",     # inférieur ou égal
    "between", # entre deux valeurs
    "in",      # dans une liste
    "not_in",  # pas dans une liste
    "exists",  # champ existe
    "match",   # recherche textuelle
    "prefix"   # préfixe
]

# Types d'agrégation disponibles
SUPPORTED_AGGREGATION_TYPES = [
    "sum",
    "avg", 
    "min",
    "max",
    "count",
    "cardinality",
    "percentiles",
    "stats",
    "terms",
    "date_histogram",
    "histogram"
]

# Champs indexés disponibles pour recherche/filtrage
INDEXED_FIELDS = {
    # Champs de recherche textuelle
    "searchable_text": {"type": "text", "boost": 2.0},
    "primary_description": {"type": "text", "boost": 1.5}, 
    "merchant_name": {"type": "text", "boost": 1.8},
    "category_name": {"type": "text", "boost": 1.2},
    
    # Champs de filtrage exact
    "user_id": {"type": "integer", "required": True},
    "transaction_id": {"type": "keyword"},
    "account_id": {"type": "integer"},
    "transaction_type": {"type": "keyword"},
    "currency_code": {"type": "keyword"},
    "operation_type": {"type": "keyword"},
    "month_year": {"type": "keyword"},
    "weekday": {"type": "keyword"},
    
    # Champs numériques/temporels
    "amount": {"type": "float"},
    "amount_abs": {"type": "float"},
    "date": {"type": "date"},
    
    # Champs keyword pour agrégations
    "merchant_name.keyword": {"type": "keyword"},
    "category_name.keyword": {"type": "keyword"},
    "operation_type.keyword": {"type": "keyword"}
}