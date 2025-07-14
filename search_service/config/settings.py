"""
Configuration centralisée pour le Search Service
Spécialisé pour la recherche lexicale Elasticsearch haute performance
Version corrigée - Pydantic V2 compatible avec tous les fallbacks nécessaires
"""

import os
from typing import Dict, List, Optional, Any
from enum import Enum

# === IMPORT PYDANTIC V2 COMPATIBLE ===
try:
    # Pydantic v2
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator, ConfigDict
    PYDANTIC_V2 = True
except ImportError:
    try:
        # Pydantic v1 fallback
        from pydantic import BaseSettings, Field, validator
        PYDANTIC_V2 = False
    except ImportError:
        raise ImportError("Pydantic requis (v1 ou v2)")


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
    debug_mode: bool = Field(default=False, description="Mode debug détaillé")
    development_mode: bool = Field(default=True, description="Mode développement")
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
    elasticsearch_index: str = Field(
        default="harena_transactions",
        description="Alias pour elasticsearch_index_name"
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
    
    # Variables manquantes identifiées avec fallbacks
    lexical_cache_size: int = Field(
        default=1000,
        description="Taille cache lexical"
    )
    max_concurrent_queries: int = Field(
        default=10,
        description="Nombre max de requêtes concurrentes"
    )
    
    # ✅ VARIABLES MANQUANTES CRITIQUES AJOUTÉES
    query_cache_size: int = Field(
        default=1000,
        description="Taille du cache des requêtes"
    )
    search_cache_size: int = Field(
        default=1000, 
        description="Taille du cache de recherche"
    )
    result_cache_size: int = Field(
        default=500,
        description="Taille du cache des résultats"
    )
    aggregation_cache_size: int = Field(
        default=200,
        description="Taille du cache des agrégations"
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
    metrics_retention_hours: int = Field(
        default=24,
        description="Rétention des métriques en heures"
    )
    export_metrics_on_shutdown: bool = Field(
        default=True,
        description="Exporter métriques à l'arrêt"
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
    cors_enabled: bool = Field(
        default=True,
        description="Activer CORS"
    )
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

    # === Configuration avancée ajoutée ===
    field_configurations: Dict[str, Any] = Field(
        default_factory=lambda: {
            "searchable_text": {"type": "text", "boost": 2.0},
            "primary_description": {"type": "text", "boost": 1.5},
            "merchant_name": {"type": "text", "boost": 1.8},
            "category_name": {"type": "text", "boost": 1.2}
        },
        description="Configuration des champs indexés"
    )

    # === CONFIGURATION PYDANTIC ===
    if PYDANTIC_V2:
        model_config = ConfigDict(
            env_file=".env",
            env_prefix="",  # Pas de préfixe pour compatibilité
            case_sensitive=False,
            validate_assignment=True,
            extra="ignore"
        )
    else:
        class Config:
            """Configuration Pydantic V1"""
            env_file = ".env"
            env_prefix = ""  # Pas de préfixe pour compatibilité
            case_sensitive = False
            validate_assignment = True

    # === VALIDATORS V2/V1 COMPATIBLE ===
    if PYDANTIC_V2:
        @field_validator("elasticsearch_timeout")
        @classmethod
        def validate_elasticsearch_timeout(cls, v: int) -> int:
            """Valide le timeout Elasticsearch"""
            if v < 1 or v > 300:
                raise ValueError("Elasticsearch timeout doit être entre 1 et 300 secondes")
            return v

        @field_validator("max_results_per_query")
        @classmethod
        def validate_max_results(cls, v: int) -> int:
            """Valide la limite de résultats"""
            if v < 1 or v > 10000:
                raise ValueError("max_results_per_query doit être entre 1 et 10000")
            return v

        @field_validator("cache_ttl_seconds")
        @classmethod
        def validate_cache_ttl(cls, v: int) -> int:
            """Valide le TTL du cache"""
            if v < 60 or v > 3600:
                raise ValueError("cache_ttl_seconds doit être entre 60 et 3600 secondes")
            return v

        @field_validator("rate_limit_requests_per_minute")
        @classmethod
        def validate_rate_limit(cls, v: int) -> int:
            """Valide le rate limiting"""
            if v < 1 or v > 10000:
                raise ValueError("rate_limit_requests_per_minute doit être entre 1 et 10000")
            return v

        @field_validator("default_search_fields")
        @classmethod
        def validate_search_fields(cls, v: List[str]) -> List[str]:
            """Valide les champs de recherche"""
            required_fields = ["searchable_text", "primary_description"]
            for field in required_fields:
                if field not in v:
                    raise ValueError(f"Champ obligatoire manquant: {field}")
            return v
    else:
        # Validators Pydantic V1
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

    def get_elasticsearch_config(self) -> Dict[str, Any]:
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

    def get_cache_config(self) -> Dict[str, Any]:
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
    
    # ✅ PROPRIÉTÉS DE COMPATIBILITÉ COMPLÈTES AVEC FALLBACKS
    @property
    def ELASTICSEARCH_INDEX(self) -> str:
        """Alias pour elasticsearch_index_name"""
        return self.elasticsearch_index_name
    
    @property
    def FIELD_CONFIGURATIONS(self) -> Dict[str, Any]:
        """Alias pour field_configurations"""
        return self.field_configurations
    
    @property
    def LEXICAL_CACHE_SIZE(self) -> int:
        """Alias pour lexical_cache_size"""
        return self.lexical_cache_size
    
    @property
    def CACHE_TTL_SECONDS(self) -> int:
        """Alias pour cache_ttl_seconds"""
        return self.cache_ttl_seconds
    
    @property
    def MAX_CONCURRENT_QUERIES(self) -> int:
        """Alias pour max_concurrent_queries"""
        return self.max_concurrent_queries
    
    @property
    def DEFAULT_QUERY_TIMEOUT_MS(self) -> int:
        """Alias pour default_query_timeout_ms"""
        return self.default_query_timeout_ms
    
    # ✅ NOUVELLES PROPRIÉTÉS CRITIQUES AJOUTÉES
    @property
    def QUERY_CACHE_SIZE(self) -> int:
        """Taille du cache des requêtes"""
        return self.query_cache_size
    
    @property
    def SEARCH_CACHE_SIZE(self) -> int:
        """Taille du cache de recherche"""
        return self.search_cache_size
    
    @property
    def RESULT_CACHE_SIZE(self) -> int:
        """Taille du cache des résultats"""
        return self.result_cache_size
    
    @property
    def AGGREGATION_CACHE_SIZE(self) -> int:
        """Taille du cache des agrégations"""
        return self.aggregation_cache_size
    
    # === FALLBACKS POUR VARIABLES COURAMMENT UTILISÉES ===
    @property
    def DEFAULT_CACHE_SIZE(self) -> int:
        """Cache size par défaut (fallback)"""
        return getattr(self, 'default_cache_size', self.cache_max_size)
    
    @property
    def DEFAULT_CACHE_TTL(self) -> int:
        """TTL par défaut (fallback)"""
        return getattr(self, 'default_cache_ttl', self.cache_ttl_seconds)
    
    @property
    def DEFAULT_BATCH_SIZE(self) -> int:
        """Batch size par défaut (fallback)"""
        return getattr(self, 'default_batch_size', 100)
    
    @property
    def DEFAULT_PAGE_SIZE(self) -> int:
        """Page size par défaut (fallback)"""
        return getattr(self, 'default_page_size', self.default_results_limit)
    
    @property
    def DEFAULT_TIMEOUT(self) -> int:
        """Timeout par défaut (fallback)"""
        return getattr(self, 'default_timeout', self.elasticsearch_timeout)
    
    @property
    def MAX_RETRIES(self) -> int:
        """Max retries (fallback)"""
        return getattr(self, 'max_retries', self.elasticsearch_max_retries)
    
    @property
    def RETRY_DELAY(self) -> float:
        """Delay retry (fallback)"""
        return getattr(self, 'retry_delay', 1.0)
    
    @property
    def SEARCH_TIMEOUT(self) -> float:
        """Timeout de recherche (fallback)"""
        return getattr(self, 'search_timeout', float(self.default_query_timeout_ms / 1000))
    
    @property
    def METRICS_ENABLED(self) -> bool:
        """Métriques activées (fallback)"""
        return self.metrics_enabled
    
    @property
    def DETAILED_LOGGING(self) -> bool:
        """Logging détaillé (fallback)"""
        return getattr(self, 'detailed_logging', self.debug_mode)
    
    @property
    def PERFORMANCE_MONITORING(self) -> bool:
        """Monitoring performance (fallback)"""
        return getattr(self, 'performance_monitoring', self.metrics_enabled)
    
    @property
    def ENABLE_DETAILED_METRICS(self) -> bool:
        """Métriques détaillées (fallback)"""
        return getattr(self, 'enable_detailed_metrics', self.metrics_enabled and self.debug_mode)
    
    # === PROPRIÉTÉS POUR VARIABLES DE LIMITE ===
    @property
    def MAX_SEARCH_RESULTS(self) -> int:
        """Max résultats recherche (fallback)"""
        return getattr(self, 'max_search_results', self.max_results_per_query)
    
    @property
    def MAX_SEARCH_TIMEOUT(self) -> float:
        """Max timeout recherche (fallback)"""
        return getattr(self, 'max_search_timeout', float(self.max_query_timeout_ms / 1000))
    
    @property
    def MAX_SEARCH_LIMIT(self) -> int:
        """Max limite recherche (fallback)"""
        return getattr(self, 'max_search_limit', self.max_results_per_query)
    
    @property
    def MAX_SEARCH_OFFSET(self) -> int:
        """Max offset recherche (fallback)"""
        return getattr(self, 'max_search_offset', self.max_pagination_offset)
    
    @property
    def DEFAULT_SEARCH_TIMEOUT(self) -> float:
        """Timeout recherche par défaut (fallback)"""
        return getattr(self, 'default_search_timeout', float(self.default_query_timeout_ms / 1000))
    
    @property
    def MAX_QUERY_LENGTH(self) -> int:
        """Longueur max requête (fallback)"""
        return getattr(self, 'max_query_length', self.max_text_search_length)
    
    @property
    def MAX_PREVIOUS_QUERIES(self) -> int:
        """Max requêtes précédentes (fallback)"""
        return getattr(self, 'max_previous_queries', 10)


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

# === FONCTIONS UTILITAIRES ===

def get_setting_with_fallback(attr_name: str, default_value: Any) -> Any:
    """
    Récupère un attribut des settings avec fallback robuste
    
    Args:
        attr_name: Nom de l'attribut
        default_value: Valeur par défaut
        
    Returns:
        Valeur de l'attribut ou fallback
    """
    try:
        return getattr(settings, attr_name, default_value)
    except Exception:
        return default_value

def get_cache_size(cache_type: str = "default") -> int:
    """
    Retourne la taille du cache selon le type
    
    Args:
        cache_type: Type de cache (query, search, result, aggregation)
        
    Returns:
        Taille du cache
    """
    cache_sizes = {
        "query": settings.QUERY_CACHE_SIZE,
        "search": settings.SEARCH_CACHE_SIZE,
        "result": settings.RESULT_CACHE_SIZE,
        "aggregation": settings.AGGREGATION_CACHE_SIZE,
        "lexical": settings.LEXICAL_CACHE_SIZE,
        "default": settings.DEFAULT_CACHE_SIZE
    }
    
    return cache_sizes.get(cache_type, settings.DEFAULT_CACHE_SIZE)

# Validation au démarrage
def validate_settings():
    """Valide la configuration au démarrage"""
    errors = []
    
    if not settings.elasticsearch_url:
        errors.append("elasticsearch_url est requis")
    
    if settings.default_results_limit > settings.max_results_per_query:
        errors.append("default_results_limit ne peut pas être > max_results_per_query")
    
    if errors:
        raise ValueError(f"Configuration invalide: {'; '.join(errors)}")

# Auto-validation
if not os.getenv("PYTEST_CURRENT_TEST"):
    try:
        validate_settings()
    except Exception as e:
        # En cas d'erreur de validation, continuer avec des warnings
        import logging
        logging.getLogger(__name__).warning(f"Validation settings: {e}")

# === EXPORTS ===
__all__ = [
    "Settings",
    "settings", 
    "LogLevel",
    "CacheBackend",
    "SUPPORTED_INTENT_TYPES",
    "SUPPORTED_FILTER_OPERATORS", 
    "SUPPORTED_AGGREGATION_TYPES",
    "INDEXED_FIELDS",
    "get_setting_with_fallback",
    "get_cache_size",
    "validate_settings"
]