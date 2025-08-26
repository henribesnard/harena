from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import secrets
import os
from pydantic import field_validator
from urllib.parse import quote_plus
import logging

# Configurer un logger pour le module
logger = logging.getLogger(__name__)

class GlobalSettings(BaseSettings):
    """
    Configuration globale pour tous les services Harena.
    Cette classe centralise toutes les variables d'environnement utilisées par les différents services.
    """
    # ==========================================
    # CONFIGURATION GÉNÉRALE DE L'APPLICATION
    # ==========================================
    PROJECT_NAME: str = "Harena Finance Platform"
    API_V1_STR: str = "/api/v1"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 8))  # 8 jours par défaut
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "production")
    
    # ==========================================
    # CONFIGURATION BASE DE DONNÉES
    # ==========================================
    POSTGRES_SERVER: str = os.environ.get("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "")
    POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "harena")
    POSTGRES_PORT: str = os.environ.get("POSTGRES_PORT", "5432")
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL", None)
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
    # ==========================================
    # CONFIGURATION BRIDGE API
    # ==========================================
    BRIDGE_API_URL: str = os.environ.get("BRIDGE_API_URL", "https://api.bridgeapi.io/v3")
    BRIDGE_API_VERSION: str = os.environ.get("BRIDGE_API_VERSION", "2025-01-15")
    BRIDGE_CLIENT_ID: str = os.environ.get("BRIDGE_CLIENT_ID", "")
    BRIDGE_CLIENT_SECRET: str = os.environ.get("BRIDGE_CLIENT_SECRET", "")
    
    # ==========================================
    # CONFIGURATION DES WEBHOOKS
    # ==========================================
    BRIDGE_WEBHOOK_SECRET: str = os.environ.get("BRIDGE_WEBHOOK_SECRET", "")
    WEBHOOK_BASE_URL: str = os.environ.get("WEBHOOK_BASE_URL", "https://api.harena.app")
    
    # ==========================================
    # CONFIGURATION D'ALERTES ET EMAIL
    # ==========================================
    ALERT_EMAIL_ENABLED: bool = os.environ.get("ALERT_EMAIL_ENABLED", "False").lower() == "true"
    ALERT_EMAIL_FROM: str = os.environ.get("ALERT_EMAIL_FROM", "alerts@harena.app")
    ALERT_EMAIL_TO: str = os.environ.get("ALERT_EMAIL_TO", "admin@harena.app")
    SMTP_SERVER: str = os.environ.get("SMTP_SERVER", "smtp.sendgrid.net")
    SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.environ.get("SMTP_USERNAME", "apikey")
    SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
    
    # ==========================================
    # CONFIGURATION LOGGING
    # ==========================================
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.environ.get("LOG_FILE", "harena.log")
    LOG_TO_FILE: bool = os.environ.get("LOG_TO_FILE", "False").lower() == "true"
    
    # ==========================================
    # CONFIGURATION ELASTICSEARCH / SEARCHBOX / BONSAI
    # ==========================================
    SEARCHBOX_URL: str = os.environ.get("SEARCHBOX_URL", "")
    SEARCHBOX_API_KEY: str = os.environ.get("SEARCHBOX_API_KEY", "")
    BONSAI_URL: str = os.environ.get("BONSAI_URL", "")
    BONSAI_ACCESS_KEY: str = os.environ.get("BONSAI_ACCESS_KEY", "")
    BONSAI_SECRET_KEY: str = os.environ.get("BONSAI_SECRET_KEY", "")
    DISABLE_INDEX_TEMPLATE: bool = os.environ.get("DISABLE_INDEX_TEMPLATE", "False").lower() == "true"
    
    # Configuration Elasticsearch générale
    ELASTICSEARCH_HOST: str = os.environ.get("ELASTICSEARCH_HOST", "localhost")
    ELASTICSEARCH_PORT: int = int(os.environ.get("ELASTICSEARCH_PORT", "9200"))
    ELASTICSEARCH_INDEX: str = os.environ.get("ELASTICSEARCH_INDEX", "harena_transactions")
    ELASTICSEARCH_URL: str = os.environ.get("ELASTICSEARCH_URL", "")
    
    # ==========================================
    # CONFIGURATION QDRANT POUR LE STOCKAGE VECTORIEL
    # ==========================================
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION_NAME", "financial_transactions")
    QDRANT_VECTOR_SIZE: int = int(os.environ.get("QDRANT_VECTOR_SIZE", "1536"))
    QDRANT_DISTANCE_METRIC: str = os.environ.get("QDRANT_DISTANCE_METRIC", "cosine")
    
    # ==========================================
    # CONFIGURATION OPENAI
    # ==========================================
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_CHAT_MODEL: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_REASONER_MODEL: str = os.environ.get("OPENAI_REASONER_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = int(os.environ.get("OPENAI_MAX_TOKENS", "2048"))
    OPENAI_TEMPERATURE: float = float(os.environ.get("OPENAI_TEMPERATURE", "1.0"))
    OPENAI_TOP_P: float = float(os.environ.get("OPENAI_TOP_P", "0.95"))
    OPENAI_TIMEOUT: int = int(os.environ.get("OPENAI_TIMEOUT", "30"))
    OPENAI_MAX_PROMPT_CHARS: int = int(os.environ.get("OPENAI_MAX_PROMPT_CHARS", "2000"))

    # Configuration OpenAI par tâche - Conversation Service
    OPENAI_INTENT_MAX_TOKENS: int = int(os.environ.get("OPENAI_INTENT_MAX_TOKENS", "80"))
    OPENAI_INTENT_TEMPERATURE: float = float(os.environ.get("OPENAI_INTENT_TEMPERATURE", "0.1"))
    OPENAI_INTENT_TIMEOUT: int = int(os.environ.get("OPENAI_INTENT_TIMEOUT", "6"))
    OPENAI_INTENT_TOP_P: float = float(os.environ.get("OPENAI_INTENT_TOP_P", "0.9"))

    OPENAI_ENTITY_MAX_TOKENS: int = int(os.environ.get("OPENAI_ENTITY_MAX_TOKENS", "60"))
    OPENAI_ENTITY_TEMPERATURE: float = float(os.environ.get("OPENAI_ENTITY_TEMPERATURE", "0.05"))
    OPENAI_ENTITY_TIMEOUT: int = int(os.environ.get("OPENAI_ENTITY_TIMEOUT", "5"))
    OPENAI_ENTITY_TOP_P: float = float(os.environ.get("OPENAI_ENTITY_TOP_P", "0.8"))

    OPENAI_QUERY_MAX_TOKENS: int = int(os.environ.get("OPENAI_QUERY_MAX_TOKENS", "200"))
    OPENAI_QUERY_TEMPERATURE: float = float(os.environ.get("OPENAI_QUERY_TEMPERATURE", "0.2"))
    OPENAI_QUERY_TIMEOUT: int = int(os.environ.get("OPENAI_QUERY_TIMEOUT", "8"))
    OPENAI_QUERY_TOP_P: float = float(os.environ.get("OPENAI_QUERY_TOP_P", "0.9"))

    OPENAI_RESPONSE_MAX_TOKENS: int = int(os.environ.get("OPENAI_RESPONSE_MAX_TOKENS", "300"))
    OPENAI_RESPONSE_TEMPERATURE: float = float(os.environ.get("OPENAI_RESPONSE_TEMPERATURE", "0.7"))
    OPENAI_RESPONSE_TIMEOUT: int = int(os.environ.get("OPENAI_RESPONSE_TIMEOUT", "12"))
    OPENAI_RESPONSE_TOP_P: float = float(os.environ.get("OPENAI_RESPONSE_TOP_P", "0.95"))

    OPENAI_EXPECTED_LATENCY_MS: int = int(os.environ.get("OPENAI_EXPECTED_LATENCY_MS", "1500"))

    # ==========================================
    # CONFIGURATION LLM-ONLY
    # ==========================================
    LLM_TIMEOUT: int = int(os.environ.get("LLM_TIMEOUT", "10"))
    MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
    LLM_CACHE_ENABLED: bool = os.environ.get("LLM_CACHE_ENABLED", "True").lower() == "true"
    LLM_CACHE_TTL: int = int(os.environ.get("LLM_CACHE_TTL", "300"))
    LLM_CACHE_MAX_SIZE: int = int(os.environ.get("LLM_CACHE_MAX_SIZE", "1000"))
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
    LLM_TOP_P: float = float(os.environ.get("LLM_TOP_P", os.environ.get("TOP_P", "0.95")))
    INTENT_TIMEOUT_SECONDS: int = int(os.environ.get("INTENT_TIMEOUT_SECONDS", "10"))
    INTENT_MAX_RETRIES: int = int(os.environ.get("INTENT_MAX_RETRIES", "3"))
    INTENT_BACKOFF_BASE: float = float(os.environ.get("INTENT_BACKOFF_BASE", "1"))
    
    # ==========================================
    # CONFIGURATION OPENAI POUR LES EMBEDDINGS
    # ==========================================
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # ==========================================
    # CONFIGURATION COHERE POUR LE RERANKING
    # ==========================================
    COHERE_KEY: str = os.environ.get("COHERE_KEY", "")
    
    # ==========================================
    # CONFIGURATION REDIS
    # ==========================================
    REDIS_URL: str = os.environ.get("REDIS_URL", "")  # Required
    REDISCLOUD_URL: str = os.environ.get("REDISCLOUD_URL", "")  # Heroku Redis
    REDIS_PASSWORD: Optional[str] = os.environ.get("REDIS_PASSWORD", None)
    REDIS_DB: int = int(os.environ.get("REDIS_DB", "0"))
    REDIS_CACHE_PREFIX: str = "harena_conv"
    REDIS_MAX_CONNECTIONS: int = int(os.environ.get("REDIS_MAX_CONNECTIONS", "20"))
    REDIS_RETRY_ON_TIMEOUT: bool = os.environ.get("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    REDIS_HEALTH_CHECK_INTERVAL: int = int(os.environ.get("REDIS_HEALTH_CHECK_INTERVAL", "30"))

    @field_validator("REDIS_URL")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if not v:
            raise ValueError("REDIS_URL environment variable is required")
        return v
    
    # ==========================================
    # CONFIGURATION DE PERFORMANCE ET CACHE
    # ==========================================
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))
    CACHE_TTL: int = int(os.environ.get("CACHE_TTL", "3600"))
    MEMORY_CACHE_TTL: int = int(os.environ.get("MEMORY_CACHE_TTL", "3600"))
    MEMORY_CACHE_MAX_SIZE: int = int(os.environ.get("MEMORY_CACHE_MAX_SIZE", "10000"))
    MEMORY_CACHE_SIZE: int = int(os.environ.get("MEMORY_CACHE_SIZE", "2000"))
    
    # ==========================================
    # CONFIGURATION DE TAUX DE LIMITE
    # ==========================================
    RATE_LIMIT_ENABLED: bool = os.environ.get("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_PERIOD: int = int(os.environ.get("RATE_LIMIT_PERIOD", "60"))
    RATE_LIMIT_REQUESTS: int = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
    RATE_LIMIT_BURST_SIZE: int = int(os.environ.get("RATE_LIMIT_BURST_SIZE", "10"))
    SEARCH_RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(
        os.environ.get("SEARCH_RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
    )
    SEARCH_RATE_LIMIT_WINDOW_SECONDS: int = int(
        os.environ.get("SEARCH_RATE_LIMIT_WINDOW_SECONDS", "60")
    )
    
    # ==========================================
    # PARAMÈTRES DE RECHERCHE PAR DÉFAUT
    # ==========================================
    DEFAULT_TOP_K_INITIAL: int = int(os.environ.get("DEFAULT_TOP_K_INITIAL", "50"))
    DEFAULT_TOP_K_FINAL: int = int(os.environ.get("DEFAULT_TOP_K_FINAL", "10"))
    
    # ==========================================
    # CONFIGURATION DES COÛTS (POUR LE COMPTAGE DE TOKENS)
    # ==========================================
    ENABLE_TOKEN_COUNTING: bool = os.environ.get("ENABLE_TOKEN_COUNTING", "True").lower() == "true"
    COST_PER_1K_INPUT_TOKENS: float = float(os.environ.get("COST_PER_1K_INPUT_TOKENS", "0.0005"))
    COST_PER_1K_OUTPUT_TOKENS: float = float(os.environ.get("COST_PER_1K_OUTPUT_TOKENS", "0.0015"))
    
    # ==========================================
    # CONFIGURATION DE CONVERSATION
    # ==========================================
    MAX_CONVERSATION_HISTORY: int = int(os.environ.get("MAX_CONVERSATION_HISTORY", "20"))
    DEFAULT_SYSTEM_PROMPT: str = os.environ.get("DEFAULT_SYSTEM_PROMPT", 
        "Vous êtes Harena, un assistant financier intelligent qui aide les utilisateurs à comprendre et gérer leurs finances personnelles.")
    
    # ==========================================
    # URLS DES SERVICES
    # ==========================================
    USER_SERVICE_URL: str = os.environ.get("USER_SERVICE_URL", "")
    SYNC_SERVICE_URL: str = os.environ.get("SYNC_SERVICE_URL", "")
    TRANSACTION_VECTOR_SERVICE_URL: str = os.environ.get("TRANSACTION_VECTOR_SERVICE_URL", "")
    
    # ==========================================
    # CONFIGURATION CORS
    # ==========================================
    CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "https://app.harena.finance")
    
    # ==========================================
    # CONFIGURATION SEARCH SERVICE
    # ==========================================
    
    # Configuration générale du search service
    SEARCH_SERVICE_NAME: str = os.environ.get("SEARCH_SERVICE_NAME", "search_service")
    SEARCH_SERVICE_VERSION: str = os.environ.get("SEARCH_SERVICE_VERSION", "1.0.0")
    SEARCH_SERVICE_DEBUG: bool = os.environ.get("SEARCH_SERVICE_DEBUG", "false").lower() == "true"
    
    # Variables du search_service local qui manquaient
    TEST_USER_ID: int = int(os.environ.get("TEST_USER_ID", "34"))
    
    # Timeouts des services externes
    ELASTICSEARCH_TIMEOUT: float = float(os.environ.get("ELASTICSEARCH_TIMEOUT", "30.0"))
    QDRANT_TIMEOUT: float = float(os.environ.get("QDRANT_TIMEOUT", "8.0"))
    OPENAI_TIMEOUT: float = float(os.environ.get("OPENAI_TIMEOUT", "10.0"))
    
    # Configuration du cache de recherche
    SEARCH_CACHE_ENABLED: bool = os.environ.get("SEARCH_CACHE_ENABLED", "true").lower() == "true"
    SEARCH_CACHE_TTL: int = int(os.environ.get("SEARCH_CACHE_TTL", "300"))
    SEARCH_CACHE_MAX_SIZE: int = int(os.environ.get("SEARCH_CACHE_MAX_SIZE", "1000"))
    SEARCH_CACHE_SIZE: int = int(os.environ.get("SEARCH_CACHE_SIZE", "1000"))
    
    # Limites de recherche principales
    MAX_SEARCH_RESULTS: int = int(os.environ.get("MAX_SEARCH_RESULTS", "1000"))
    MAX_SEARCH_TIMEOUT: float = float(os.environ.get("MAX_SEARCH_TIMEOUT", "30.0"))
    MAX_SEARCH_LIMIT: int = int(os.environ.get("MAX_SEARCH_LIMIT", "100"))
    MAX_SEARCH_OFFSET: int = int(os.environ.get("MAX_SEARCH_OFFSET", "10000"))
    
    # Configuration timeouts spécialisés
    DEFAULT_SEARCH_TIMEOUT: float = float(os.environ.get("DEFAULT_SEARCH_TIMEOUT", "15.0"))
    QUICK_SEARCH_TIMEOUT: float = float(os.environ.get("QUICK_SEARCH_TIMEOUT", "3.0"))
    STANDARD_SEARCH_TIMEOUT: float = float(os.environ.get("STANDARD_SEARCH_TIMEOUT", "8.0"))
    COMPLEX_SEARCH_TIMEOUT: float = float(os.environ.get("COMPLEX_SEARCH_TIMEOUT", "15.0"))
    HEALTH_CHECK_TIMEOUT: float = float(os.environ.get("HEALTH_CHECK_TIMEOUT", "5.0"))
    
    # Configuration limites de validation
    MAX_QUERY_LENGTH: int = int(os.environ.get("MAX_QUERY_LENGTH", "1000"))
    MAX_PREVIOUS_QUERIES: int = int(os.environ.get("MAX_PREVIOUS_QUERIES", "10"))
    MAX_FILTER_VALUES: int = int(os.environ.get("MAX_FILTER_VALUES", "100"))
    MAX_FILTERS_PER_GROUP: int = int(os.environ.get("MAX_FILTERS_PER_GROUP", "50"))
    MAX_AGGREGATION_BUCKETS: int = int(os.environ.get("MAX_AGGREGATION_BUCKETS", "1000"))
    MAX_AGGREGATIONS: int = int(os.environ.get("MAX_AGGREGATIONS", "50"))
    MAX_BOOL_CLAUSES: int = int(os.environ.get("MAX_BOOL_CLAUSES", "100"))
    MAX_SEARCH_FIELDS: int = int(os.environ.get("MAX_SEARCH_FIELDS", "20"))
    
    # Configuration champs autorisés
    ALLOWED_SEARCH_FIELDS: List[str] = [
        "user_id", "transaction_id", "account_id", "amount", "amount_abs",
        "transaction_type", "operation_type", "currency_code", "date", "month_year",
        "primary_description", "merchant_name", "category_name", "clean_description",
        "searchable_text", "weekday", "is_weekend", "transaction_hash"
    ]
    ALLOWED_FILTER_FIELDS: List[str] = [
        "user_id", "account_id", "transaction_type", "operation_type", "currency_code",
        "category_name", "merchant_name", "amount", "amount_abs", "date", "month_year", "weekday"
    ]
    SENSITIVE_FIELDS: List[str] = ["user_id", "account_id", "transaction_hash"]
    
    DEFAULT_CACHE_SIZE: int = int(os.environ.get("DEFAULT_CACHE_SIZE", "1000"))
    DEFAULT_CACHE_TTL: int = int(os.environ.get("DEFAULT_CACHE_TTL", "300"))
    DEFAULT_BATCH_SIZE: int = int(os.environ.get("DEFAULT_BATCH_SIZE", "100"))
    DEFAULT_PAGE_SIZE: int = int(os.environ.get("DEFAULT_PAGE_SIZE", "20"))
    DEFAULT_TIMEOUT: int = int(os.environ.get("DEFAULT_TIMEOUT", "30"))
    
    # Configuration des highlights pour utils
    HIGHLIGHT_FIELDS: dict = {
        "searchable_text": {"fragment_size": 150, "number_of_fragments": 3},
        "merchant_name": {"fragment_size": 100, "number_of_fragments": 1},
        "clean_description": {"fragment_size": 200, "number_of_fragments": 2}
    }
    
    # ==========================================
    # CONFIGURATION DES EMBEDDINGS
    # ==========================================
    EMBEDDING_DIMENSIONS: int = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
    EMBEDDING_BATCH_SIZE: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "100"))
    EMBEDDING_CACHE_TTL: int = int(os.environ.get("EMBEDDING_CACHE_TTL", "3600"))
    EMBEDDING_CACHE_ENABLED: bool = os.environ.get("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    EMBEDDING_CACHE_MAX_SIZE: int = int(os.environ.get("EMBEDDING_CACHE_MAX_SIZE", "10000"))
    EMBEDDING_CACHE_SIZE: int = int(os.environ.get("EMBEDDING_CACHE_SIZE", "5000"))
    EMBEDDING_MAX_RETRIES: int = int(os.environ.get("EMBEDDING_MAX_RETRIES", "3"))
    
    # Configuration de pagination
    DEFAULT_SEARCH_LIMIT: int = int(os.environ.get("DEFAULT_SEARCH_LIMIT", "100"))
    DEFAULT_LIMIT: int = int(os.environ.get("DEFAULT_LIMIT", "20"))
    
    # Configuration des timeouts de recherche
    SEARCH_TIMEOUT: float = float(os.environ.get("SEARCH_TIMEOUT", "15.0"))
    
    # Configuration des retry
    MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.environ.get("RETRY_DELAY", "1.0"))
    
    # Configuration du monitoring
    METRICS_ENABLED: bool = os.environ.get("METRICS_ENABLED", "true").lower() == "true"
    DETAILED_LOGGING: bool = os.environ.get("DETAILED_LOGGING", "false").lower() == "true"
    PERFORMANCE_MONITORING: bool = os.environ.get("PERFORMANCE_MONITORING", "true").lower() == "true"
    ENABLE_DETAILED_METRICS: bool = os.environ.get("ENABLE_DETAILED_METRICS", "false").lower() == "true"
    METRICS_COLLECTION_INTERVAL: int = int(os.environ.get("METRICS_COLLECTION_INTERVAL", "60"))
    PERFORMANCE_ALERTING: bool = os.environ.get("PERFORMANCE_ALERTING", "false").lower() == "true"
    
    # ==========================================
    # CONFIGURATION RECHERCHE HYBRIDE ET OPTIMISATIONS
    # ==========================================
    
    # Fallback et résilience
    ENABLE_FALLBACK: bool = os.environ.get("ENABLE_FALLBACK", "true").lower() == "true"
    MIN_ENGINE_SUCCESS: int = int(os.environ.get("MIN_ENGINE_SUCCESS", "1"))
    
    # Optimisations de performance
    ENABLE_PARALLEL_SEARCH: bool = os.environ.get("ENABLE_PARALLEL_SEARCH", "true").lower() == "true"
    ENABLE_EARLY_TERMINATION: bool = os.environ.get("ENABLE_EARLY_TERMINATION", "false").lower() == "true"
    EARLY_TERMINATION_THRESHOLD: float = float(os.environ.get("EARLY_TERMINATION_THRESHOLD", "0.95"))
    
    # Adaptation automatique des poids
    ADAPTIVE_WEIGHTING: bool = os.environ.get("ADAPTIVE_WEIGHTING", "true").lower() == "true"
    
    # Seuils de scores minimums
    MIN_SCORE_THRESHOLD: float = float(os.environ.get("MIN_SCORE_THRESHOLD", "0.1"))
    
    # ==========================================
    # CONFIGURATION RECHERCHE LEXICALE
    # ==========================================
    
    # Boost factors
    BOOST_EXACT_PHRASE: float = float(os.environ.get("BOOST_EXACT_PHRASE", "10.0"))
    BOOST_MERCHANT_NAME: float = float(os.environ.get("BOOST_MERCHANT_NAME", "5.0"))
    BOOST_PRIMARY_DESCRIPTION: float = float(os.environ.get("BOOST_PRIMARY_DESCRIPTION", "3.0"))
    BOOST_SEARCHABLE_TEXT: float = float(os.environ.get("BOOST_SEARCHABLE_TEXT", "4.0"))
    BOOST_CLEAN_DESCRIPTION: float = float(os.environ.get("BOOST_CLEAN_DESCRIPTION", "2.5"))
    
    # Options de requête
    ENABLE_FUZZY: bool = os.environ.get("ENABLE_FUZZY", "true").lower() == "true"
    ENABLE_WILDCARDS: bool = os.environ.get("ENABLE_WILDCARDS", "true").lower() == "true"
    ENABLE_SYNONYMS: bool = os.environ.get("ENABLE_SYNONYMS", "true").lower() == "true"
    MINIMUM_SHOULD_MATCH: str = os.environ.get("MINIMUM_SHOULD_MATCH", "1")
    FUZZINESS_LEVEL: str = os.environ.get("FUZZINESS_LEVEL", "AUTO")
    
    # Configuration du highlighting
    HIGHLIGHT_ENABLED: bool = os.environ.get("HIGHLIGHT_ENABLED", "true").lower() == "true"
    HIGHLIGHT_FRAGMENT_SIZE: int = int(os.environ.get("HIGHLIGHT_FRAGMENT_SIZE", "150"))
    HIGHLIGHT_MAX_FRAGMENTS: int = int(os.environ.get("HIGHLIGHT_MAX_FRAGMENTS", "3"))
    
    # Filtres lexicaux
    LEXICAL_MIN_SCORE: float = float(os.environ.get("LEXICAL_MIN_SCORE", "1.0"))
    LEXICAL_MAX_RESULTS: int = int(os.environ.get("LEXICAL_MAX_RESULTS", "50"))
    
    # ==========================================
    # CONFIGURATION RECHERCHE SÉMANTIQUE
    # ==========================================
    
    # Seuils de similarité (noms principaux)
    SIMILARITY_THRESHOLD_DEFAULT: float = float(os.environ.get("SIMILARITY_THRESHOLD_DEFAULT", "0.1"))
    SIMILARITY_THRESHOLD_STRICT: float = float(os.environ.get("SIMILARITY_THRESHOLD_STRICT", "0.15"))
    SIMILARITY_THRESHOLD_LOOSE: float = float(os.environ.get("SIMILARITY_THRESHOLD_LOOSE", "0.05"))
    
    # Seuils de similarité (noms alternatifs pour compatibilité)
    SEMANTIC_SIMILARITY_THRESHOLD_DEFAULT: float = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD_DEFAULT", "0.1"))
    SEMANTIC_SIMILARITY_THRESHOLD_STRICT: float = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD_STRICT", "0.15"))
    SEMANTIC_SIMILARITY_THRESHOLD_LOOSE: float = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD_LOOSE", "0.05"))
    
    # Options sémantiques
    SEMANTIC_MAX_RESULTS: int = int(os.environ.get("SEMANTIC_MAX_RESULTS", "50"))
    SEMANTIC_ENABLE_FILTERING: bool = os.environ.get("SEMANTIC_ENABLE_FILTERING", "true").lower() == "true"
    SEMANTIC_FALLBACK_UNFILTERED: bool = os.environ.get("SEMANTIC_FALLBACK_UNFILTERED", "true").lower() == "true"
    SEMANTIC_FALLBACK_TO_UNFILTERED: bool = os.environ.get("SEMANTIC_FALLBACK_TO_UNFILTERED", "true").lower() == "true"
    
    # Configuration des recommandations
    RECOMMENDATION_ENABLED: bool = os.environ.get("RECOMMENDATION_ENABLED", "true").lower() == "true"
    RECOMMENDATION_THRESHOLD: float = float(os.environ.get("RECOMMENDATION_THRESHOLD", "0.6"))
    SEMANTIC_RECOMMENDATION_ENABLED: bool = os.environ.get("SEMANTIC_RECOMMENDATION_ENABLED", "true").lower() == "true"
    SEMANTIC_RECOMMENDATION_THRESHOLD: float = float(os.environ.get("SEMANTIC_RECOMMENDATION_THRESHOLD", "0.6"))
    
    # Options avancées sémantiques
    SEMANTIC_ENABLE_QUERY_EXPANSION: bool = os.environ.get("SEMANTIC_ENABLE_QUERY_EXPANSION", "false").lower() == "true"
    
    # ==========================================
    # CONFIGURATION RECHERCHE HYBRIDE ET FUSION
    # ==========================================
    
    # Stratégies de fusion
    DEFAULT_FUSION_STRATEGY: str = os.environ.get("DEFAULT_FUSION_STRATEGY", "weighted_average")
    SCORE_NORMALIZATION_METHOD: str = os.environ.get("SCORE_NORMALIZATION_METHOD", "min_max")
    
    # Facteurs RRF (Reciprocal Rank Fusion)
    RRF_K: int = int(os.environ.get("RRF_K", "60"))
    
    # Seuils adaptatifs
    ADAPTIVE_THRESHOLD: float = float(os.environ.get("ADAPTIVE_THRESHOLD", "0.1"))
    
    # Facteurs de qualité
    QUALITY_BOOST_FACTOR: float = float(os.environ.get("QUALITY_BOOST_FACTOR", "1.2"))
    
    # Déduplication et diversification
    ENABLE_DEDUPLICATION: bool = os.environ.get("ENABLE_DEDUPLICATION", "true").lower() == "true"
    DEDUP_SIMILARITY_THRESHOLD: float = float(os.environ.get("DEDUP_SIMILARITY_THRESHOLD", "0.95"))
    ENABLE_DIVERSIFICATION: bool = os.environ.get("ENABLE_DIVERSIFICATION", "true").lower() == "true"
    DIVERSITY_FACTOR: float = float(os.environ.get("DIVERSITY_FACTOR", "0.3"))
    MAX_SAME_MERCHANT: int = int(os.environ.get("MAX_SAME_MERCHANT", "3"))
    
    # ==========================================
    # CONFIGURATION ÉVALUATION DE QUALITÉ
    # ==========================================
    
    QUALITY_EXCELLENT_THRESHOLD: float = float(os.environ.get("QUALITY_EXCELLENT_THRESHOLD", "0.9"))
    QUALITY_GOOD_THRESHOLD: float = float(os.environ.get("QUALITY_GOOD_THRESHOLD", "0.7"))
    QUALITY_MEDIUM_THRESHOLD: float = float(os.environ.get("QUALITY_MEDIUM_THRESHOLD", "0.5"))
    QUALITY_POOR_THRESHOLD: float = float(os.environ.get("QUALITY_POOR_THRESHOLD", "0.3"))
    
    # Facteurs de qualité
    MIN_RESULTS_FOR_GOOD_QUALITY: int = int(os.environ.get("MIN_RESULTS_FOR_GOOD_QUALITY", "3"))
    MIN_RESULTS_FOR_FUSION: int = int(os.environ.get("MIN_RESULTS_FOR_FUSION", "2"))
    MAX_RESULTS_FOR_QUALITY_EVAL: int = int(os.environ.get("MAX_RESULTS_FOR_QUALITY_EVAL", "10"))
    DIVERSITY_THRESHOLD: float = float(os.environ.get("DIVERSITY_THRESHOLD", "0.6"))
    
    # ==========================================
    # CONFIGURATION CACHE ET OPTIMISATION
    # ==========================================
    
    # Cache d'analyse de requêtes
    QUERY_ANALYSIS_CACHE_ENABLED: bool = os.environ.get("QUERY_ANALYSIS_CACHE_ENABLED", "true").lower() == "true"
    QUERY_ANALYSIS_CACHE_TTL: int = int(os.environ.get("QUERY_ANALYSIS_CACHE_TTL", "1800"))
    QUERY_ANALYSIS_CACHE_MAX_SIZE: int = int(os.environ.get("QUERY_ANALYSIS_CACHE_MAX_SIZE", "500"))
    
    # Configuration de concurrence
    MAX_CONCURRENT_SEARCHES: int = int(os.environ.get("MAX_CONCURRENT_SEARCHES", "10"))
    MAX_CONCURRENT_EMBEDDINGS: int = int(os.environ.get("MAX_CONCURRENT_EMBEDDINGS", "5"))
    
    # Configuration du warmup
    WARMUP_ENABLED: bool = os.environ.get("WARMUP_ENABLED", "true").lower() == "true"
    WARMUP_QUERIES: str = os.environ.get("WARMUP_QUERIES", "restaurant,virement,carte bancaire,supermarché,essence,pharmacie")
    
    # Configuration de l'amélioration automatique
    AUTO_QUERY_OPTIMIZATION: bool = os.environ.get("AUTO_QUERY_OPTIMIZATION", "true").lower() == "true"
    SUGGESTION_ENABLED: bool = os.environ.get("SUGGESTION_ENABLED", "true").lower() == "true"
    MAX_SUGGESTIONS: int = int(os.environ.get("MAX_SUGGESTIONS", "5"))
    
    # ==========================================
    # CONFIGURATION CONVERSATION SERVICE
    # ==========================================

    # Configuration Conversation Service - Phase 1
    CONVERSATION_SERVICE_ENABLED: bool = os.environ.get("CONVERSATION_SERVICE_ENABLED", "True").lower() == "true"

    # Configuration service
    CONVERSATION_SERVICE_HOST: str = os.environ.get("CONVERSATION_SERVICE_HOST", "0.0.0.0")
    CONVERSATION_SERVICE_PORT: int = int(os.environ.get("CONVERSATION_SERVICE_PORT", "8001"))
    CONVERSATION_SERVICE_DEBUG: bool = os.environ.get("CONVERSATION_SERVICE_DEBUG", "false").lower() == "true"
    CONVERSATION_SERVICE_LOG_LEVEL: str = os.environ.get("CONVERSATION_SERVICE_LOG_LEVEL", "INFO")

    # Configuration DeepSeek
    DEEPSEEK_API_URL: str = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com")
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL_CHAT: str = os.environ.get("DEEPSEEK_MODEL_CHAT", "deepseek-chat")
    DEEPSEEK_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4000"))
    DEEPSEEK_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.1"))
    DEEPSEEK_TIMEOUT: int = int(os.environ.get("DEEPSEEK_TIMEOUT", "30"))

    # Secret key powering bearer token verification across services
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "")
    # Configuration Authentification JWT (utilise SECRET_KEY global)
    JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24)))

    # Configuration Cache & Performance
    REDIS_CONVERSATION_TTL: int = int(os.environ.get("REDIS_CONVERSATION_TTL", "3600"))
    MIN_CONFIDENCE_THRESHOLD: float = float(os.environ.get("MIN_CONFIDENCE_THRESHOLD", "0.7"))
    AGENT_TIMEOUT_SECONDS: int = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "30"))
    MAX_CACHE_SIZE: int = int(os.environ.get("MAX_CACHE_SIZE", "1000"))

    # Configuration Intent Classifier
    CLASSIFICATION_CACHE_TTL: int = int(os.environ.get("CLASSIFICATION_CACHE_TTL", "300"))  # Legacy
    CACHE_SIZE: int = int(os.environ.get("CACHE_SIZE", "1000"))  # Legacy
    
    # Configuration cache multi-niveaux Conversation Service
    CACHE_TTL_INTENT: int = int(os.environ.get("CACHE_TTL_INTENT", "300"))
    CACHE_TTL_ENTITY: int = int(os.environ.get("CACHE_TTL_ENTITY", "180"))
    CACHE_TTL_QUERY: int = int(os.environ.get("CACHE_TTL_QUERY", "120"))
    CACHE_TTL_RESPONSE: int = int(os.environ.get("CACHE_TTL_RESPONSE", "60"))
    CACHE_TTL_SECONDS: int = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
    USE_LLM_QUERY: bool = os.environ.get("USE_LLM_QUERY", "false").lower() == "true"
    
    # Cache L0 - Patterns pré-calculés
    PRECOMPUTED_PATTERNS_ENABLED: bool = os.environ.get("PRECOMPUTED_PATTERNS_ENABLED", "true").lower() == "true"
    PRECOMPUTED_PATTERNS_SIZE: int = int(os.environ.get("PRECOMPUTED_PATTERNS_SIZE", "100"))
    
    # Cache L2 - Redis distribué
    REDIS_CACHE_ENABLED: bool = os.environ.get("REDIS_CACHE_ENABLED", "true").lower() == "true"
    REDIS_CACHE_PREFIX: str = os.environ.get("REDIS_CACHE_PREFIX", "conversation_service")
    
    # Configuration performance
    REQUEST_TIMEOUT: int = int(os.environ.get("REQUEST_TIMEOUT", "30"))
    
    # Optimisations pipeline asynchrone
    ENABLE_ASYNC_PIPELINE: bool = os.environ.get("ENABLE_ASYNC_PIPELINE", "true").lower() == "true"
    PIPELINE_TIMEOUT: int = int(os.environ.get("PIPELINE_TIMEOUT", "10"))
    PARALLEL_PROCESSING_ENABLED: bool = os.environ.get("PARALLEL_PROCESSING_ENABLED", "true").lower() == "true"
    
    # Thread pools pour traitement asynchrone
    THREAD_POOL_SIZE: int = int(os.environ.get("THREAD_POOL_SIZE", "10"))
    MAX_CONCURRENT_REQUESTS: int = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "100"))
    MAX_CONCURRENT_QUERIES: int = int(os.environ.get("MAX_CONCURRENT_QUERIES", "10"))
    
    # Configuration monitoring avancé Conversation Service
    ENABLE_METRICS: bool = os.environ.get("ENABLE_METRICS", "true").lower() == "true"
    
    # Métriques performance détaillées
    PERFORMANCE_ALERT_THRESHOLD_MS: int = int(os.environ.get("PERFORMANCE_ALERT_THRESHOLD_MS", "2000"))
    CACHE_HIT_RATE_ALERT_THRESHOLD: float = float(os.environ.get("CACHE_HIT_RATE_ALERT_THRESHOLD", "0.8"))
    ERROR_RATE_ALERT_THRESHOLD: float = float(os.environ.get("ERROR_RATE_ALERT_THRESHOLD", "0.05"))
    
    # Configuration circuit breaker
    CIRCUIT_BREAKER_ENABLED: bool = os.environ.get("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(os.environ.get("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = int(os.environ.get("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: str = os.environ.get("CIRCUIT_BREAKER_EXPECTED_EXCEPTION", "httpx.RequestError")
    
    # Configuration batch processing
    BATCH_PROCESSING_ENABLED: bool = os.environ.get("BATCH_PROCESSING_ENABLED", "true").lower() == "true"
    BATCH_TIMEOUT_MS: int = int(os.environ.get("BATCH_TIMEOUT_MS", "100"))
    
    # Variables manquantes des logs d'erreur
    LEXICAL_CACHE_SIZE: int = int(os.environ.get("LEXICAL_CACHE_SIZE", "1000"))
    QUERY_CACHE_SIZE: int = int(os.environ.get("QUERY_CACHE_SIZE", "1000"))

    # Variables de configuration manquantes pour compatibilité

    @field_validator('DEEPSEEK_API_KEY')
    @classmethod
    def validate_deepseek_key(cls, v, info):
        if not v and info.data.get('CONVERSATION_SERVICE_ENABLED'):
            raise ValueError("DEEPSEEK_API_KEY est requis si CONVERSATION_SERVICE_ENABLED=True")
        return v

    @field_validator('SECRET_KEY')
    @classmethod
    def validate_secret_key(cls, v):
        if not v:
            raise ValueError("SECRET_KEY est requis pour l'authentification")
        if len(v) < 32:
            raise ValueError("SECRET_KEY doit faire au moins 32 caractères")
        return v


    @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info):
        # Utiliser directement DATABASE_URL (qui peut être DATABASE_URL dans Heroku)
        db_url = info.data.get("DATABASE_URL")
        if db_url:
            # Convertir postgres:// en postgresql:// si nécessaire
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            logger.debug("Utilisation de DATABASE_URL pour la connexion à la base de données")
            return db_url
        
        # Si pas de DATABASE_URL, utiliser les composants individuels
        data = info.data
        server = data.get("POSTGRES_SERVER", "")
        port = data.get("POSTGRES_PORT", "5432")
        user = quote_plus(data.get("POSTGRES_USER", ""))
        password = quote_plus(data.get("POSTGRES_PASSWORD", ""))
        db = data.get("POSTGRES_DB", "")
        
        if not server or server == "localhost":
            logger.debug("Configuration d'une connexion locale à la base de données")
        else:
            logger.debug(f"Configuration d'une connexion à la base de données sur {server}")
            
        return f"postgresql://{user}:{password}@{server}:{port}/{db}"
    
    def get_warmup_queries_list(self) -> List[str]:
        """Retourne la liste des requêtes de warmup."""
        if not self.WARMUP_QUERIES:
            return []
        return [q.strip() for q in self.WARMUP_QUERIES.split(",") if q.strip()]
    
    def get_similarity_threshold(self, mode: str = "default") -> float:
        """
        Retourne le seuil de similarité selon le mode.
        
        🎯 FONCTION CRITIQUE pour débugger les problèmes de recherche !
        
        Args:
            mode: "strict", "default", "loose"
        """
        thresholds = {
            "strict": self.SIMILARITY_THRESHOLD_STRICT,
            "default": self.SIMILARITY_THRESHOLD_DEFAULT,
            "loose": self.SIMILARITY_THRESHOLD_LOOSE
        }
        return thresholds.get(mode, self.SIMILARITY_THRESHOLD_DEFAULT)
    
    def get_search_config_summary(self):
        """Retourne un résumé de la configuration de recherche pour debug."""
        return {
            "service": {
                "name": self.SEARCH_SERVICE_NAME,
                "version": self.SEARCH_SERVICE_VERSION,
                "debug": self.SEARCH_SERVICE_DEBUG
            },
            "similarity_thresholds": {
                "strict": self.SIMILARITY_THRESHOLD_STRICT,
                "default": self.SIMILARITY_THRESHOLD_DEFAULT,
                "loose": self.SIMILARITY_THRESHOLD_LOOSE
            },
            "timeouts": {
                "elasticsearch": self.ELASTICSEARCH_TIMEOUT,
                "qdrant": self.QDRANT_TIMEOUT,
                "openai": self.OPENAI_TIMEOUT,
                "search": self.SEARCH_TIMEOUT,
                "max_search": self.MAX_SEARCH_TIMEOUT,
                "default_search": self.DEFAULT_SEARCH_TIMEOUT
            },
            "cache": {
                "search_enabled": self.SEARCH_CACHE_ENABLED,
                "search_ttl": self.SEARCH_CACHE_TTL,
                "embedding_enabled": self.EMBEDDING_CACHE_ENABLED,
                "embedding_ttl": self.EMBEDDING_CACHE_TTL
            },
            "limits": {
                "default_search": self.DEFAULT_SEARCH_LIMIT,
                "max_search": self.MAX_SEARCH_LIMIT,
                "max_search_results": self.MAX_SEARCH_RESULTS,
                "max_search_offset": self.MAX_SEARCH_OFFSET,
                "lexical_max": self.LEXICAL_MAX_RESULTS,
                "semantic_max": self.SEMANTIC_MAX_RESULTS,
                "default_cache_size": self.DEFAULT_CACHE_SIZE,
                "default_page_size": self.DEFAULT_PAGE_SIZE,
                "max_query_length": self.MAX_QUERY_LENGTH,
                "max_previous_queries": self.MAX_PREVIOUS_QUERIES
            },
            "validation_limits": {
                "max_filter_values": self.MAX_FILTER_VALUES,
                "max_filters_per_group": self.MAX_FILTERS_PER_GROUP,
                "max_aggregation_buckets": self.MAX_AGGREGATION_BUCKETS,
                "max_aggregations": self.MAX_AGGREGATIONS,
                "max_bool_clauses": self.MAX_BOOL_CLAUSES,
                "max_search_fields": self.MAX_SEARCH_FIELDS
            },
            "min_scores": {
                "threshold": self.MIN_SCORE_THRESHOLD
            },
            "quality_thresholds": {
                "excellent": self.QUALITY_EXCELLENT_THRESHOLD,
                "good": self.QUALITY_GOOD_THRESHOLD,
                "medium": self.QUALITY_MEDIUM_THRESHOLD,
                "poor": self.QUALITY_POOR_THRESHOLD
            },
            "allowed_fields": {
                "search_fields": len(self.ALLOWED_SEARCH_FIELDS),
                "filter_fields": len(self.ALLOWED_FILTER_FIELDS),
                "sensitive_fields": len(self.SENSITIVE_FIELDS)
            }
        }
    
    def validate_search_config(self) -> dict:
        """Valide la cohérence de la configuration de recherche."""
        validation = {"valid": True, "warnings": [], "errors": []}
        
        # Validation des seuils de similarité
        if self.SIMILARITY_THRESHOLD_LOOSE > self.SIMILARITY_THRESHOLD_DEFAULT:
            validation["errors"].append("Loose similarity threshold should be <= default threshold")
            validation["valid"] = False
        
        if self.SIMILARITY_THRESHOLD_DEFAULT > self.SIMILARITY_THRESHOLD_STRICT:
            validation["errors"].append("Default similarity threshold should be <= strict threshold")
            validation["valid"] = False
        
        # Validation des limites
        if self.DEFAULT_SEARCH_LIMIT > self.MAX_SEARCH_LIMIT:
            validation["errors"].append("Default limit cannot be greater than max limit")
            validation["valid"] = False
        
        if self.MAX_SEARCH_LIMIT > self.MAX_SEARCH_RESULTS:
            validation["warnings"].append("Max search limit is greater than max search results")
        
        # Validation des timeouts
        if self.ELASTICSEARCH_TIMEOUT > self.SEARCH_TIMEOUT:
            validation["warnings"].append("Elasticsearch timeout is greater than overall search timeout")
        
        if self.QDRANT_TIMEOUT > self.SEARCH_TIMEOUT:
            validation["warnings"].append("Qdrant timeout is greater than overall search timeout")
        
        if self.MAX_SEARCH_TIMEOUT < self.SEARCH_TIMEOUT:
            validation["warnings"].append("Max search timeout is less than search timeout")
        
        if self.DEFAULT_SEARCH_TIMEOUT > self.MAX_SEARCH_TIMEOUT:
            validation["warnings"].append("Default search timeout is greater than max search timeout")
        
        # Validation des caches
        if self.DEFAULT_CACHE_SIZE <= 0:
            validation["errors"].append("Default cache size must be positive")
            validation["valid"] = False
        
        if self.DEFAULT_CACHE_TTL <= 0:
            validation["errors"].append("Default cache TTL must be positive")
            validation["valid"] = False
        
        # Validation des batch sizes
        if self.DEFAULT_BATCH_SIZE <= 0:
            validation["errors"].append("Default batch size must be positive")
            validation["valid"] = False
        
        # Validation des pages
        if self.DEFAULT_PAGE_SIZE <= 0:
            validation["errors"].append("Default page size must be positive")
            validation["valid"] = False
        
        if self.DEFAULT_PAGE_SIZE > self.MAX_SEARCH_RESULTS:
            validation["warnings"].append("Default page size is greater than max search results")
        
        # Validation des limites critiques
        if self.MAX_QUERY_LENGTH <= 0:
            validation["errors"].append("Max query length must be positive")
            validation["valid"] = False
        
        if self.MAX_SEARCH_OFFSET <= 0:
            validation["errors"].append("Max search offset must be positive")
            validation["valid"] = False
        
        # Validation des champs autorisés
        if not self.ALLOWED_SEARCH_FIELDS:
            validation["warnings"].append("No allowed search fields defined")
        
        if not self.ALLOWED_FILTER_FIELDS:
            validation["warnings"].append("No allowed filter fields defined")
        
        return validation
    
    def get_search_service_constants(self) -> dict:
        """
        Retourne toutes les constantes nécessaires au Search Service.
        
        Cette méthode centralise tous les paramètres requis par search_service/utils/__init__.py
        """
        return {
            # Constantes principales
            "MAX_SEARCH_RESULTS": self.MAX_SEARCH_RESULTS,
            "MAX_SEARCH_TIMEOUT": self.MAX_SEARCH_TIMEOUT,
            "MAX_SEARCH_LIMIT": self.MAX_SEARCH_LIMIT,
            "MAX_SEARCH_OFFSET": self.MAX_SEARCH_OFFSET,
            "DEFAULT_SEARCH_TIMEOUT": self.DEFAULT_SEARCH_TIMEOUT,
            "DEFAULT_CACHE_SIZE": self.DEFAULT_CACHE_SIZE,
            "DEFAULT_CACHE_TTL": self.DEFAULT_CACHE_TTL,
            "DEFAULT_BATCH_SIZE": self.DEFAULT_BATCH_SIZE,
            "DEFAULT_PAGE_SIZE": self.DEFAULT_PAGE_SIZE,
            "DEFAULT_TIMEOUT": self.DEFAULT_TIMEOUT,
            
            # Configuration Elasticsearch
            "ELASTICSEARCH_TIMEOUT": self.ELASTICSEARCH_TIMEOUT,
            "ELASTICSEARCH_INDEX": self.ELASTICSEARCH_INDEX,
            "ELASTICSEARCH_HOST": self.ELASTICSEARCH_HOST,
            "ELASTICSEARCH_PORT": self.ELASTICSEARCH_PORT,
            
            # Configuration cache
            "SEARCH_CACHE_SIZE": self.SEARCH_CACHE_SIZE,
            "SEARCH_CACHE_TTL": self.SEARCH_CACHE_TTL,
            "SEARCH_CACHE_ENABLED": self.SEARCH_CACHE_ENABLED,
            
            # Limites de recherche
            "DEFAULT_SEARCH_LIMIT": self.DEFAULT_SEARCH_LIMIT,
            "LEXICAL_MAX_RESULTS": self.LEXICAL_MAX_RESULTS,
            "SEMANTIC_MAX_RESULTS": self.SEMANTIC_MAX_RESULTS,
            
            # Limites de validation critiques
            "MAX_QUERY_LENGTH": self.MAX_QUERY_LENGTH,
            "MAX_PREVIOUS_QUERIES": self.MAX_PREVIOUS_QUERIES,
            "MAX_FILTER_VALUES": self.MAX_FILTER_VALUES,
            "MAX_FILTERS_PER_GROUP": self.MAX_FILTERS_PER_GROUP,
            "MAX_AGGREGATION_BUCKETS": self.MAX_AGGREGATION_BUCKETS,
            "MAX_AGGREGATIONS": self.MAX_AGGREGATIONS,
            "MAX_BOOL_CLAUSES": self.MAX_BOOL_CLAUSES,
            "MAX_SEARCH_FIELDS": self.MAX_SEARCH_FIELDS,
            
            # Champs autorisés
            "ALLOWED_SEARCH_FIELDS": self.ALLOWED_SEARCH_FIELDS,
            "ALLOWED_FILTER_FIELDS": self.ALLOWED_FILTER_FIELDS,
            "SENSITIVE_FIELDS": self.SENSITIVE_FIELDS,
            
            # Highlighting (converti depuis la propriété dict)
            "HIGHLIGHT_FIELDS": self.HIGHLIGHT_FIELDS,
            "HIGHLIGHT_ENABLED": self.HIGHLIGHT_ENABLED,
            "HIGHLIGHT_FRAGMENT_SIZE": self.HIGHLIGHT_FRAGMENT_SIZE,
            "HIGHLIGHT_MAX_FRAGMENTS": self.HIGHLIGHT_MAX_FRAGMENTS,
            
            # Scores minimums
            "MIN_SCORE_THRESHOLD": self.MIN_SCORE_THRESHOLD,
            
            # Timeouts spécialisés
            "HEALTH_CHECK_TIMEOUT": self.HEALTH_CHECK_TIMEOUT,
            "QUICK_SEARCH_TIMEOUT": self.QUICK_SEARCH_TIMEOUT,
            "STANDARD_SEARCH_TIMEOUT": self.STANDARD_SEARCH_TIMEOUT,
            "COMPLEX_SEARCH_TIMEOUT": self.COMPLEX_SEARCH_TIMEOUT,
            
            # Configuration performance
            "ENABLE_PARALLEL_SEARCH": self.ENABLE_PARALLEL_SEARCH,
            "MAX_CONCURRENT_SEARCHES": self.MAX_CONCURRENT_SEARCHES,
            "PERFORMANCE_MONITORING": self.PERFORMANCE_MONITORING,
            "METRICS_ENABLED": self.METRICS_ENABLED,
            
            # Configuration boost et scoring
            "BOOST_EXACT_PHRASE": self.BOOST_EXACT_PHRASE,
            "BOOST_MERCHANT_NAME": self.BOOST_MERCHANT_NAME,
            "BOOST_PRIMARY_DESCRIPTION": self.BOOST_PRIMARY_DESCRIPTION,
            "BOOST_SEARCHABLE_TEXT": self.BOOST_SEARCHABLE_TEXT,
            "BOOST_CLEAN_DESCRIPTION": self.BOOST_CLEAN_DESCRIPTION,
            
            # Options de requête
            "ENABLE_FUZZY": self.ENABLE_FUZZY,
            "ENABLE_WILDCARDS": self.ENABLE_WILDCARDS,
            "ENABLE_SYNONYMS": self.ENABLE_SYNONYMS,
            "MINIMUM_SHOULD_MATCH": self.MINIMUM_SHOULD_MATCH,
            "FUZZINESS_LEVEL": self.FUZZINESS_LEVEL,
            
            # Seuils de similarité
            "SIMILARITY_THRESHOLD_DEFAULT": self.SIMILARITY_THRESHOLD_DEFAULT,
            "SIMILARITY_THRESHOLD_STRICT": self.SIMILARITY_THRESHOLD_STRICT,
            "SIMILARITY_THRESHOLD_LOOSE": self.SIMILARITY_THRESHOLD_LOOSE,
            
            # Configuration de qualité
            "QUALITY_EXCELLENT_THRESHOLD": self.QUALITY_EXCELLENT_THRESHOLD,
            "QUALITY_GOOD_THRESHOLD": self.QUALITY_GOOD_THRESHOLD,
            "QUALITY_MEDIUM_THRESHOLD": self.QUALITY_MEDIUM_THRESHOLD,
            "QUALITY_POOR_THRESHOLD": self.QUALITY_POOR_THRESHOLD,
            
            # Facteurs de qualité
            "MIN_RESULTS_FOR_GOOD_QUALITY": self.MIN_RESULTS_FOR_GOOD_QUALITY,
            "MIN_RESULTS_FOR_FUSION": self.MIN_RESULTS_FOR_FUSION,
            "MAX_RESULTS_FOR_QUALITY_EVAL": self.MAX_RESULTS_FOR_QUALITY_EVAL,
            "DIVERSITY_THRESHOLD": self.DIVERSITY_THRESHOLD,
            
            # Configuration déduplication et diversification
            "ENABLE_DEDUPLICATION": self.ENABLE_DEDUPLICATION,
            "DEDUP_SIMILARITY_THRESHOLD": self.DEDUP_SIMILARITY_THRESHOLD,
            "ENABLE_DIVERSIFICATION": self.ENABLE_DIVERSIFICATION,
            "DIVERSITY_FACTOR": self.DIVERSITY_FACTOR,
            "MAX_SAME_MERCHANT": self.MAX_SAME_MERCHANT,
            
            # Configuration fusion
            "DEFAULT_FUSION_STRATEGY": self.DEFAULT_FUSION_STRATEGY,
            "SCORE_NORMALIZATION_METHOD": self.SCORE_NORMALIZATION_METHOD,
            "RRF_K": self.RRF_K,
            "ADAPTIVE_THRESHOLD": self.ADAPTIVE_THRESHOLD,
            "QUALITY_BOOST_FACTOR": self.QUALITY_BOOST_FACTOR,
            
            # Configuration optimisations
            "ENABLE_FALLBACK": self.ENABLE_FALLBACK,
            "MIN_ENGINE_SUCCESS": self.MIN_ENGINE_SUCCESS,
            "ENABLE_EARLY_TERMINATION": self.ENABLE_EARLY_TERMINATION,
            "EARLY_TERMINATION_THRESHOLD": self.EARLY_TERMINATION_THRESHOLD,
            "ADAPTIVE_WEIGHTING": self.ADAPTIVE_WEIGHTING,
            
            # Configuration warmup et suggestions
            "WARMUP_ENABLED": self.WARMUP_ENABLED,
            "WARMUP_QUERIES": self.WARMUP_QUERIES,
            "AUTO_QUERY_OPTIMIZATION": self.AUTO_QUERY_OPTIMIZATION,
            "SUGGESTION_ENABLED": self.SUGGESTION_ENABLED,
            "MAX_SUGGESTIONS": self.MAX_SUGGESTIONS,
            
            # Configuration cache avancé
            "QUERY_ANALYSIS_CACHE_ENABLED": self.QUERY_ANALYSIS_CACHE_ENABLED,
            "QUERY_ANALYSIS_CACHE_TTL": self.QUERY_ANALYSIS_CACHE_TTL,
            "QUERY_ANALYSIS_CACHE_MAX_SIZE": self.QUERY_ANALYSIS_CACHE_MAX_SIZE,
            
            # Configuration embeddings
            "EMBEDDING_DIMENSIONS": self.EMBEDDING_DIMENSIONS,
            "EMBEDDING_BATCH_SIZE": self.EMBEDDING_BATCH_SIZE,
            "EMBEDDING_CACHE_TTL": self.EMBEDDING_CACHE_TTL,
            "EMBEDDING_CACHE_ENABLED": self.EMBEDDING_CACHE_ENABLED,
            "EMBEDDING_CACHE_MAX_SIZE": self.EMBEDDING_CACHE_MAX_SIZE,
            "EMBEDDING_CACHE_SIZE": self.EMBEDDING_CACHE_SIZE,
            "EMBEDDING_MAX_RETRIES": self.EMBEDDING_MAX_RETRIES
        }
    
    def get_openai_config(self, task_type: str = "default") -> dict:
        """Retourne la configuration OpenAI optimisée par tâche"""

        configs = {
            "intent": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "chat_model": self.OPENAI_CHAT_MODEL,
                "max_tokens": self.OPENAI_INTENT_MAX_TOKENS,
                "temperature": self.OPENAI_INTENT_TEMPERATURE,
                "top_p": self.OPENAI_INTENT_TOP_P,
                "timeout": self.OPENAI_INTENT_TIMEOUT,
            },
            "entity": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "chat_model": self.OPENAI_CHAT_MODEL,
                "max_tokens": self.OPENAI_ENTITY_MAX_TOKENS,
                "temperature": self.OPENAI_ENTITY_TEMPERATURE,
                "top_p": self.OPENAI_ENTITY_TOP_P,
                "timeout": self.OPENAI_ENTITY_TIMEOUT,
            },
            "query": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "chat_model": self.OPENAI_CHAT_MODEL,
                "max_tokens": self.OPENAI_QUERY_MAX_TOKENS,
                "temperature": self.OPENAI_QUERY_TEMPERATURE,
                "top_p": self.OPENAI_QUERY_TOP_P,
                "timeout": self.OPENAI_QUERY_TIMEOUT,
            },
            "response": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "chat_model": self.OPENAI_CHAT_MODEL,
                "max_tokens": self.OPENAI_RESPONSE_MAX_TOKENS,
                "temperature": self.OPENAI_RESPONSE_TEMPERATURE,
                "top_p": self.OPENAI_RESPONSE_TOP_P,
                "timeout": self.OPENAI_RESPONSE_TIMEOUT,
            },
            "default": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "chat_model": self.OPENAI_CHAT_MODEL,
                "reasoner_model": self.OPENAI_REASONER_MODEL,
                "max_tokens": self.OPENAI_MAX_TOKENS,
                "temperature": self.OPENAI_TEMPERATURE,
                "top_p": self.OPENAI_TOP_P,
                "timeout": self.OPENAI_TIMEOUT,
            },
        }

        return configs.get(task_type, configs["default"])
    
    def get_cache_config(self) -> dict:
        """Retourne la configuration cache multi-niveaux"""
        return {
            # Redis configuration
            "redis": {
                "enabled": self.REDIS_CACHE_ENABLED,
                "url": self.REDIS_URL,
                "password": self.REDIS_PASSWORD,
                "db": self.REDIS_DB,
                "max_connections": self.REDIS_MAX_CONNECTIONS,
                "retry_on_timeout": self.REDIS_RETRY_ON_TIMEOUT,
                "prefix": self.REDIS_CACHE_PREFIX
            },
            # Memory cache L1
            "memory": {
                "size": self.MEMORY_CACHE_SIZE,
                "ttl": self.MEMORY_CACHE_TTL
            },
            # Precomputed patterns L0
            "precomputed": {
                "enabled": self.PRECOMPUTED_PATTERNS_ENABLED,
                "size": self.PRECOMPUTED_PATTERNS_SIZE
            },
            # TTL par tâche
            "ttl": {
                "intent": self.CACHE_TTL_INTENT,
                "entity": self.CACHE_TTL_ENTITY,
                "query": self.CACHE_TTL_QUERY,
                "response": self.CACHE_TTL_RESPONSE
            }
        }
    
    def validate_configuration(self) -> dict:
        """Valide la configuration et retourne les erreurs"""
        errors = []
        warnings = []
        
        # Validation OpenAI
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY est requis")

        if not self.OPENAI_BASE_URL:
            errors.append("OPENAI_BASE_URL est requis")
        
        # Validation Redis (si activé)
        if self.REDIS_CACHE_ENABLED:
            if not self.REDIS_URL and not self.REDISCLOUD_URL:
                errors.append("REDIS_URL ou REDISCLOUD_URL est requis quand REDIS_CACHE_ENABLED=true")
        
        # Validation thresholds
        if self.MIN_CONFIDENCE_THRESHOLD < 0.0 or self.MIN_CONFIDENCE_THRESHOLD > 1.0:
            errors.append("MIN_CONFIDENCE_THRESHOLD doit être entre 0.0 et 1.0")
        
        if self.MIN_CONFIDENCE_THRESHOLD < 0.5:
            warnings.append("MIN_CONFIDENCE_THRESHOLD < 0.5 peut produire des résultats peu fiables")
        
        # Validation performance
        if self.REQUEST_TIMEOUT < 5:
            warnings.append("REQUEST_TIMEOUT < 5s peut causer des timeouts")
        
        if self.MEMORY_CACHE_SIZE < 100:
            warnings.append("MEMORY_CACHE_SIZE < 100 peut réduire les performances")
        
        # Validation timeouts optimisés
        if self.OPENAI_INTENT_TIMEOUT > 10:
            warnings.append("OPENAI_INTENT_TIMEOUT > 10s est trop élevé pour l'optimisation")

        if self.OPENAI_INTENT_MAX_TOKENS > 150:
            warnings.append("OPENAI_INTENT_MAX_TOKENS > 150 peut ralentir la classification")
        
        # Validation cache TTL
        if self.CACHE_TTL_INTENT < 60:
            warnings.append("CACHE_TTL_INTENT < 60s peut réduire l'efficacité du cache")
        
        # Validation rate limiting
        if self.RATE_LIMIT_REQUESTS_PER_MINUTE < 10:
            warnings.append("RATE_LIMIT_REQUESTS_PER_MINUTE très bas, peut impacter l'utilisabilité")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )


# Initialisation du singleton de configuration globale
settings = GlobalSettings()

# Afficher un message de démarrage
logger.debug(f"Configuration chargée pour l'environnement: {settings.ENVIRONMENT}")
logger.debug(f"Configuration de recherche: {settings.get_search_config_summary()}")

# Validation automatique de la configuration de recherche
search_validation = settings.validate_search_config()
if not search_validation["valid"]:
    logger.error(f"Configuration de recherche invalide: {search_validation['errors']}")
if search_validation["warnings"]:
    logger.warning(f"Avertissements configuration de recherche: {search_validation['warnings']}")

# Validation automatique de la configuration générale
config_validation = settings.validate_configuration()
if not config_validation["valid"]:
    logger.error(f"Configuration générale invalide: {config_validation['errors']}")
if config_validation["warnings"]:
    logger.warning(f"Avertissements configuration générale: {config_validation['warnings']}")

# Log des constantes Search Service ajoutées
search_constants = settings.get_search_service_constants()
logger.info(f"Constantes Search Service disponibles: {len(search_constants)} paramètres")
logger.debug(f"MAX_SEARCH_RESULTS: {search_constants['MAX_SEARCH_RESULTS']}")
logger.debug(f"MAX_SEARCH_TIMEOUT: {search_constants['MAX_SEARCH_TIMEOUT']}")
logger.debug(f"MAX_SEARCH_LIMIT: {search_constants['MAX_SEARCH_LIMIT']}")
logger.debug(f"MAX_SEARCH_OFFSET: {search_constants['MAX_SEARCH_OFFSET']}")
logger.debug(f"DEFAULT_SEARCH_TIMEOUT: {search_constants['DEFAULT_SEARCH_TIMEOUT']}")
logger.debug(f"MAX_QUERY_LENGTH: {search_constants['MAX_QUERY_LENGTH']}")
logger.debug(f"MAX_PREVIOUS_QUERIES: {search_constants['MAX_PREVIOUS_QUERIES']}")
logger.debug(f"DEFAULT_CACHE_SIZE: {search_constants['DEFAULT_CACHE_SIZE']}")
logger.debug(f"DEFAULT_CACHE_TTL: {search_constants['DEFAULT_CACHE_TTL']}")

# Log informations importantes
logger.info(f"Configuration Harena Finance Platform chargée - Mode: {settings.ENVIRONMENT}")
logger.info(f"OpenAI API: {'✅ Configuré' if settings.OPENAI_API_KEY else '❌ Manquant'}")
logger.info(f"Redis Cache: {'✅ Activé' if settings.REDIS_CACHE_ENABLED else '❌ Désactivé'}")
logger.info(f"Search Service: {settings.SEARCH_SERVICE_NAME} v{settings.SEARCH_SERVICE_VERSION}")
logger.info(
    f"Conversation Service: {'✅ Activé' if settings.CONVERSATION_SERVICE_ENABLED else '❌ Désactivé'} - "
    f"Port {settings.CONVERSATION_SERVICE_PORT}, Confidence: {settings.MIN_CONFIDENCE_THRESHOLD}"
)
logger.info(f"Elasticsearch: {'✅ Configuré' if settings.BONSAI_URL else '❌ Manquant'}")
logger.info(f"Database: {'✅ URL Configurée' if settings.DATABASE_URL else '❌ URL Manquante'}")
logger.info(f"Bridge API: {'✅ Configuré' if settings.BRIDGE_CLIENT_ID else '❌ Manquant'}")

# Avertissements de sécurité
if settings.ENVIRONMENT == "production":
    if not settings.SECRET_KEY:
        logger.warning("🔐 SECRET_KEY non défini - définir SECRET_KEY en production!")
    if not settings.OPENAI_API_KEY:
        logger.warning("🤖 OPENAI_API_KEY manquant - fonctionnalités IA désactivées")
    if not settings.DATABASE_URL:
        logger.warning("🗄️ DATABASE_URL manquant - base de données non configurée")
