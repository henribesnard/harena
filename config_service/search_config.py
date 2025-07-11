"""
🔍 Configuration Centralisée Search Service
============================================

Configuration spécialisée pour le Search Service focalisé sur la recherche lexicale pure
via Elasticsearch. Cette configuration suit l'architecture hybride séparant clairement
les responsabilités entre Search Service (lexical) et Conversation Service (IA).

Architecture:
- Search Service: Recherche lexicale haute performance Elasticsearch
- Contrats standardisés: Interface stable entre services  
- Performance optimisée: <50ms temps réponse, cache intelligent
- Sécurité: Validation stricte, isolation utilisateur obligatoire
"""

import os
import logging
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseSettings, Field, validator
from pathlib import Path

# Configuration du logging pour le module de configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchServiceConfig(BaseSettings):
    """
    Configuration spécialisée Search Service pour recherche lexicale pure Elasticsearch.
    
    Responsabilités:
    - Configuration Elasticsearch optimisée performance
    - Paramètres cache Redis pour résultats
    - Limites et quotas recherche sécurisés
    - Templates requêtes par défaut
    - Validation stricte paramètres
    """
    
    # =============================================================================
    # 🌍 CONFIGURATION ENVIRONNEMENT
    # =============================================================================
    
    ENVIRONMENT: str = Field(default="development", description="Environnement déploiement")
    DEBUG: bool = Field(default=True, description="Mode debug détaillé")
    LOG_LEVEL: str = Field(default="INFO", description="Niveau logging")
    
    # =============================================================================
    # 🔍 CONFIGURATION ELASTICSEARCH CORE
    # =============================================================================
    
    # Connexion Elasticsearch principale
    ELASTICSEARCH_HOST: str = Field(default="localhost", description="Host Elasticsearch")
    ELASTICSEARCH_PORT: int = Field(default=9200, description="Port Elasticsearch")
    ELASTICSEARCH_SCHEME: str = Field(default="http", description="Schéma connexion (http/https)")
    ELASTICSEARCH_USERNAME: Optional[str] = Field(default=None, description="Utilisateur Elasticsearch")
    ELASTICSEARCH_PASSWORD: Optional[str] = Field(default=None, description="Mot de passe Elasticsearch")
    
    # Index et configuration
    ELASTICSEARCH_INDEX: str = Field(default="harena_transactions", description="Index principal transactions")
    ELASTICSEARCH_TIMEOUT: int = Field(default=30, description="Timeout connexion Elasticsearch (secondes)")
    ELASTICSEARCH_MAX_RETRIES: int = Field(default=3, description="Nombre max tentatives reconnexion")
    ELASTICSEARCH_RETRY_ON_TIMEOUT: bool = Field(default=True, description="Retry sur timeout")
    
    # Pool de connexions optimisé
    ELASTICSEARCH_MAX_CONNECTIONS: int = Field(default=20, description="Max connexions simultanées")
    ELASTICSEARCH_POOL_CONNECTIONS: int = Field(default=10, description="Taille pool connexions")
    ELASTICSEARCH_POOL_MAXSIZE: int = Field(default=20, description="Taille max pool")
    
    # =============================================================================
    # ⚡ CONFIGURATION PERFORMANCE RECHERCHE
    # =============================================================================
    
    # Limites temps et taille
    MAX_SEARCH_TIMEOUT: int = Field(default=10000, description="Timeout max recherche (ms)")
    DEFAULT_SEARCH_TIMEOUT: int = Field(default=5000, description="Timeout par défaut recherche (ms)")
    MAX_SEARCH_RESULTS: int = Field(default=1000, description="Résultats max par recherche")
    DEFAULT_SEARCH_LIMIT: int = Field(default=20, description="Limite par défaut résultats")
    MAX_SEARCH_OFFSET: int = Field(default=10000, description="Offset max pagination")
    
    # Optimisations Elasticsearch
    SEARCH_REQUEST_CACHE: bool = Field(default=True, description="Cache requêtes Elasticsearch")
    SEARCH_PREFERENCE: str = Field(default="_local", description="Préférence routage recherche")
    SEARCH_BATCHED_REDUCE_SIZE: int = Field(default=512, description="Taille batch réduction")
    SEARCH_MAX_CONCURRENT_SHARD_REQUESTS: int = Field(default=5, description="Requêtes shard concurrentes max")
    
    # =============================================================================
    # 💾 CONFIGURATION CACHE REDIS
    # =============================================================================
    
    # Cache Redis pour résultats
    REDIS_ENABLED: bool = Field(default=True, description="Activation cache Redis")
    REDIS_HOST: str = Field(default="localhost", description="Host Redis")
    REDIS_PORT: int = Field(default=6379, description="Port Redis")
    REDIS_DB: int = Field(default=0, description="Base Redis (0-15)")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Mot de passe Redis")
    REDIS_SOCKET_TIMEOUT: int = Field(default=5, description="Timeout socket Redis")
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(default=5, description="Timeout connexion Redis")
    
    # Configuration cache recherche
    SEARCH_CACHE_TTL: int = Field(default=300, description="TTL cache résultats (secondes)")
    SEARCH_CACHE_MAX_SIZE: int = Field(default=1000, description="Taille max cache LRU")
    SEARCH_CACHE_KEY_PREFIX: str = Field(default="search:", description="Préfixe clés cache")
    QUERY_CACHE_TTL: int = Field(default=600, description="TTL cache requêtes (secondes)")
    AGGREGATION_CACHE_TTL: int = Field(default=1800, description="TTL cache agrégations (secondes)")
    
    # =============================================================================
    # 🔒 CONFIGURATION SÉCURITÉ ET VALIDATION
    # =============================================================================
    
    # Limites validation critiques sécurité
    MAX_QUERY_LENGTH: int = Field(default=1000, description="Longueur max requête")
    MAX_FILTER_VALUES: int = Field(default=100, description="Valeurs max par filtre")
    MAX_FILTERS_PER_GROUP: int = Field(default=20, description="Filtres max par groupe")
    MAX_AGGREGATIONS: int = Field(default=10, description="Agrégations max par requête")
    MAX_AGGREGATION_BUCKETS: int = Field(default=1000, description="Buckets max par agrégation")
    MAX_BOOL_CLAUSES: int = Field(default=1024, description="Clauses bool max Elasticsearch")
    MAX_SEARCH_FIELDS: int = Field(default=50, description="Champs recherche max")
    
    # Champs autorisés pour sécurité
    ALLOWED_SEARCH_FIELDS: Set[str] = Field(
        default={
            "searchable_text", "primary_description", "merchant_name",
            "category_name", "operation_type"
        },
        description="Champs autorisés recherche textuelle"
    )
    
    ALLOWED_FILTER_FIELDS: Set[str] = Field(
        default={
            "user_id", "category_name", "merchant_name", "transaction_type",
            "currency_code", "operation_type", "month_year", "weekday",
            "amount", "amount_abs", "date"
        },
        description="Champs autorisés filtrage"
    )
    
    SENSITIVE_FIELDS: Set[str] = Field(
        default={"account_id", "card_number", "iban"},
        description="Champs sensibles interdits"
    )
    
    # Filtres obligatoires sécurité
    REQUIRED_FILTERS: Set[str] = Field(
        default={"user_id"},
        description="Filtres obligatoires pour isolation utilisateur"
    )
    
    # =============================================================================
    # 📊 CONFIGURATION MÉTRIQUES ET MONITORING
    # =============================================================================
    
    # Métriques performance
    METRICS_ENABLED: bool = Field(default=True, description="Activation métriques")
    METRICS_DETAILED: bool = Field(default=False, description="Métriques détaillées debug")
    PERFORMANCE_TRACKING: bool = Field(default=True, description="Suivi performance requêtes")
    SLOW_QUERY_THRESHOLD: int = Field(default=1000, description="Seuil requête lente (ms)")
    
    # Health checks
    HEALTH_CHECK_ENABLED: bool = Field(default=True, description="Activation health checks")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, description="Intervalle health check (secondes)")
    ELASTICSEARCH_HEALTH_TIMEOUT: int = Field(default=5, description="Timeout health check ES")
    
    # =============================================================================
    # 🎯 CONFIGURATION TEMPLATES REQUÊTES
    # =============================================================================
    
    # Templates par défaut
    DEFAULT_QUERY_TEMPLATE: str = Field(default="simple_match", description="Template requête par défaut")
    ENABLE_QUERY_TEMPLATES: bool = Field(default=True, description="Activation système templates")
    TEMPLATE_CACHE_ENABLED: bool = Field(default=True, description="Cache templates")
    TEMPLATE_VALIDATION: bool = Field(default=True, description="Validation templates avant usage")
    
    # Scoring et pertinence
    DEFAULT_BM25_K1: float = Field(default=1.2, description="Paramètre BM25 k1")
    DEFAULT_BM25_B: float = Field(default=0.75, description="Paramètre BM25 b")
    MIN_SCORE_THRESHOLD: float = Field(default=0.0, description="Score minimum résultats")
    
    # =============================================================================
    # 🚀 CONFIGURATION OPTIMISATIONS AVANCÉES
    # =============================================================================
    
    # Optimisations batch et parallélisme
    BATCH_SIZE: int = Field(default=100, description="Taille batch traitement")
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, description="Requêtes concurrentes max")
    CIRCUIT_BREAKER_ENABLED: bool = Field(default=True, description="Circuit breaker protection")
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, description="Seuil échecs circuit breaker")
    
    # Pre-warming et optimisations
    ENABLE_QUERY_WARMUP: bool = Field(default=True, description="Préchauffage requêtes fréquentes")
    WARMUP_QUERIES: List[str] = Field(
        default=[
            "category:restaurant", "category:transport", "category:alimentation"
        ],
        description="Requêtes préchauffage"
    )
    
    # =============================================================================
    # 🤝 CONFIGURATION CONTRATS SERVICES
    # =============================================================================
    
    # Interface standardisée avec Conversation Service
    ENABLE_CONTRACT_VALIDATION: bool = Field(default=True, description="Validation contrats API")
    STRICT_CONTRACT_MODE: bool = Field(default=True, description="Mode strict contrats")
    CONTRACT_VERSION: str = Field(default="v1.0", description="Version contrats API")
    
    # Métadonnées requêtes
    REQUIRE_QUERY_METADATA: bool = Field(default=True, description="Métadonnées requête obligatoires")
    REQUIRE_USER_ID: bool = Field(default=True, description="user_id obligatoire")
    ENABLE_QUERY_TRACING: bool = Field(default=True, description="Traçage requêtes bout-en-bout")
    
    # =============================================================================
    # 🧪 CONFIGURATION TESTS ET MOCKS
    # =============================================================================
    
    # Mode test
    TESTING_MODE: bool = Field(default=False, description="Mode test activation")
    MOCK_ELASTICSEARCH: bool = Field(default=False, description="Mock Elasticsearch pour tests")
    TEST_DATA_PATH: str = Field(default="tests/fixtures", description="Chemin données test")
    
    # =============================================================================
    # ⚙️ MÉTHODES VALIDATION ET UTILITAIRES
    # =============================================================================
    
    @validator('ELASTICSEARCH_PORT')
    def validate_elasticsearch_port(cls, v):
        """Validation port Elasticsearch valide."""
        if not 1 <= v <= 65535:
            raise ValueError('Port Elasticsearch doit être entre 1 et 65535')
        return v
    
    @validator('MAX_SEARCH_TIMEOUT')
    def validate_search_timeout(cls, v):
        """Validation timeout recherche raisonnable."""
        if v < 100 or v > 30000:
            raise ValueError('Timeout recherche doit être entre 100ms et 30s')
        return v
    
    @validator('MAX_SEARCH_RESULTS')
    def validate_max_results(cls, v):
        """Validation limite résultats raisonnable."""
        if v < 1 or v > 10000:
            raise ValueError('Limite résultats doit être entre 1 et 10000')
        return v
    
    @validator('SEARCH_CACHE_TTL')
    def validate_cache_ttl(cls, v):
        """Validation TTL cache raisonnable."""
        if v < 60 or v > 3600:
            raise ValueError('TTL cache doit être entre 60s et 1h')
        return v
    
    @property
    def elasticsearch_url(self) -> str:
        """URL complète Elasticsearch."""
        auth = ""
        if self.ELASTICSEARCH_USERNAME and self.ELASTICSEARCH_PASSWORD:
            auth = f"{self.ELASTICSEARCH_USERNAME}:{self.ELASTICSEARCH_PASSWORD}@"
        
        return f"{self.ELASTICSEARCH_SCHEME}://{auth}{self.ELASTICSEARCH_HOST}:{self.ELASTICSEARCH_PORT}"
    
    @property
    def redis_url(self) -> str:
        """URL complète Redis."""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """Configuration complète client Elasticsearch."""
        return {
            "hosts": [self.elasticsearch_url],
            "timeout": self.ELASTICSEARCH_TIMEOUT,
            "max_retries": self.ELASTICSEARCH_MAX_RETRIES,
            "retry_on_timeout": self.ELASTICSEARCH_RETRY_ON_TIMEOUT,
            "maxsize": self.ELASTICSEARCH_MAX_CONNECTIONS,
            "request_timeout": self.DEFAULT_SEARCH_TIMEOUT / 1000,  # Conversion ms -> s
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Configuration complète client Redis."""
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DB,
            "password": self.REDIS_PASSWORD,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": self.REDIS_SOCKET_CONNECT_TIMEOUT,
            "decode_responses": True,
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Configuration cache recherche optimisée."""
        return {
            "enabled": self.REDIS_ENABLED,
            "ttl": self.SEARCH_CACHE_TTL,
            "max_size": self.SEARCH_CACHE_MAX_SIZE,
            "key_prefix": self.SEARCH_CACHE_KEY_PREFIX,
            "query_ttl": self.QUERY_CACHE_TTL,
            "aggregation_ttl": self.AGGREGATION_CACHE_TTL,
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Configuration sécurité et validation."""
        return {
            "max_query_length": self.MAX_QUERY_LENGTH,
            "max_filter_values": self.MAX_FILTER_VALUES,
            "max_filters_per_group": self.MAX_FILTERS_PER_GROUP,
            "max_aggregations": self.MAX_AGGREGATIONS,
            "allowed_search_fields": self.ALLOWED_SEARCH_FIELDS,
            "allowed_filter_fields": self.ALLOWED_FILTER_FIELDS,
            "sensitive_fields": self.SENSITIVE_FIELDS,
            "required_filters": self.REQUIRED_FILTERS,
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Configuration optimisations performance."""
        return {
            "max_timeout": self.MAX_SEARCH_TIMEOUT,
            "default_timeout": self.DEFAULT_SEARCH_TIMEOUT,
            "max_results": self.MAX_SEARCH_RESULTS,
            "default_limit": self.DEFAULT_SEARCH_LIMIT,
            "request_cache": self.SEARCH_REQUEST_CACHE,
            "preference": self.SEARCH_PREFERENCE,
            "circuit_breaker": self.CIRCUIT_BREAKER_ENABLED,
            "max_concurrent": self.MAX_CONCURRENT_REQUESTS,
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validation complète configuration avec diagnostics."""
        errors = []
        warnings = []
        
        # Validation cohérence timeouts
        if self.DEFAULT_SEARCH_TIMEOUT > self.MAX_SEARCH_TIMEOUT:
            errors.append("DEFAULT_SEARCH_TIMEOUT ne peut pas être > MAX_SEARCH_TIMEOUT")
        
        if self.DEFAULT_SEARCH_TIMEOUT > self.ELASTICSEARCH_TIMEOUT * 1000:
            warnings.append("DEFAULT_SEARCH_TIMEOUT > ELASTICSEARCH_TIMEOUT peut causer des timeouts")
        
        # Validation cohérence limites
        if self.DEFAULT_SEARCH_LIMIT > self.MAX_SEARCH_RESULTS:
            errors.append("DEFAULT_SEARCH_LIMIT ne peut pas être > MAX_SEARCH_RESULTS")
        
        # Validation champs obligatoires
        if not self.REQUIRED_FILTERS:
            errors.append("REQUIRED_FILTERS ne peut pas être vide (sécurité)")
        
        if "user_id" not in self.REQUIRED_FILTERS:
            errors.append("user_id doit être dans REQUIRED_FILTERS (isolation utilisateur)")
        
        # Validation cache
        if self.REDIS_ENABLED and self.SEARCH_CACHE_TTL <= 0:
            errors.append("SEARCH_CACHE_TTL doit être > 0 si cache activé")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "config_summary": {
                "elasticsearch": f"{self.ELASTICSEARCH_HOST}:{self.ELASTICSEARCH_PORT}",
                "index": self.ELASTICSEARCH_INDEX,
                "cache_enabled": self.REDIS_ENABLED,
                "max_results": self.MAX_SEARCH_RESULTS,
                "default_timeout": self.DEFAULT_SEARCH_TIMEOUT,
            }
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Informations debug configuration complète."""
        return {
            "environment": self.ENVIRONMENT,
            "elasticsearch_url": self.elasticsearch_url,
            "redis_url": self.redis_url if self.REDIS_ENABLED else "disabled",
            "cache_enabled": self.REDIS_ENABLED,
            "metrics_enabled": self.METRICS_ENABLED,
            "contract_validation": self.ENABLE_CONTRACT_VALIDATION,
            "security_config": self.get_security_config(),
            "performance_config": self.get_performance_config(),
        }
    
    class Config:
        """Configuration Pydantic."""
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore les variables supplémentaires


# =============================================================================
# 🚀 INITIALISATION CONFIGURATION GLOBALE
# =============================================================================

# Instance singleton configuration
search_config = SearchServiceConfig()

# Validation automatique au démarrage
config_validation = search_config.validate_config()

# Logging initialisation
logger.info(f"🔍 Search Service Configuration chargée - Environnement: {search_config.ENVIRONMENT}")
logger.info(f"📊 Elasticsearch: {search_config.elasticsearch_url}")
logger.info(f"💾 Cache Redis: {'activé' if search_config.REDIS_ENABLED else 'désactivé'}")
logger.info(f"🔒 Validation contrats: {'activée' if search_config.ENABLE_CONTRACT_VALIDATION else 'désactivée'}")

# Validation configuration
if not config_validation["valid"]:
    logger.error(f"❌ Configuration invalide: {config_validation['errors']}")
    raise ValueError(f"Configuration Search Service invalide: {config_validation['errors']}")

if config_validation["warnings"]:
    for warning in config_validation["warnings"]:
        logger.warning(f"⚠️ {warning}")

# Log configuration performance si debug
if search_config.DEBUG:
    debug_info = search_config.get_debug_info()
    logger.debug(f"🔧 Configuration debug: {debug_info}")

logger.info("✅ Search Service Configuration initialisée avec succès")


# =============================================================================
# 🎯 EXPORTS PUBLICS
# =============================================================================

__all__ = [
    "SearchServiceConfig",
    "search_config",
]