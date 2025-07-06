from pydantic_settings import BaseSettings
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
    SECRET_KEY: str = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))
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
    
    # Configuration Elasticsearch (pour référence dans le search_service)
    ELASTICSEARCH_INDEX: str = os.environ.get("ELASTICSEARCH_INDEX", "harena_transactions")
    
    # ==========================================
    # CONFIGURATION QDRANT POUR LE STOCKAGE VECTORIEL
    # ==========================================
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION_NAME", "financial_transactions")
    QDRANT_VECTOR_SIZE: int = int(os.environ.get("QDRANT_VECTOR_SIZE", "1536"))
    QDRANT_DISTANCE_METRIC: str = os.environ.get("QDRANT_DISTANCE_METRIC", "cosine")
    
    # ==========================================
    # CONFIGURATION DEEPSEEK
    # ==========================================
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_CHAT_MODEL: str = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_REASONER_MODEL: str = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")
    DEEPSEEK_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "8192"))
    DEEPSEEK_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_TEMPERATURE", "1.0"))
    DEEPSEEK_TOP_P: float = float(os.environ.get("DEEPSEEK_TOP_P", "0.95"))
    DEEPSEEK_TIMEOUT: int = int(os.environ.get("DEEPSEEK_TIMEOUT", "60"))
    
    # ==========================================
    # CONFIGURATION OPENAI POUR LES EMBEDDINGS
    # ==========================================
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # ==========================================
    # CONFIGURATION COHERE POUR LE RERANKING
    # ==========================================
    COHERE_KEY: str = os.environ.get("COHERE_KEY", "")
    
    # ==========================================
    # CONFIGURATION DE PERFORMANCE ET CACHE
    # ==========================================
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))
    CACHE_TTL: int = int(os.environ.get("CACHE_TTL", "3600"))
    MEMORY_CACHE_TTL: int = int(os.environ.get("MEMORY_CACHE_TTL", "3600"))
    MEMORY_CACHE_MAX_SIZE: int = int(os.environ.get("MEMORY_CACHE_MAX_SIZE", "10000"))
    
    # ==========================================
    # CONFIGURATION DE TAUX DE LIMITE
    # ==========================================
    RATE_LIMIT_ENABLED: bool = os.environ.get("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_PERIOD: int = int(os.environ.get("RATE_LIMIT_PERIOD", "60"))
    RATE_LIMIT_REQUESTS: int = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
    
    # ==========================================
    # PARAMÈTRES DE RECHERCHE PAR DÉFAUT
    # ==========================================
    DEFAULT_LEXICAL_WEIGHT: float = float(os.environ.get("DEFAULT_LEXICAL_WEIGHT", "0.6"))
    DEFAULT_SEMANTIC_WEIGHT: float = float(os.environ.get("DEFAULT_SEMANTIC_WEIGHT", "0.4"))
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
    
    # Timeouts des services externes
    ELASTICSEARCH_TIMEOUT: float = float(os.environ.get("ELASTICSEARCH_TIMEOUT", "5.0"))
    QDRANT_TIMEOUT: float = float(os.environ.get("QDRANT_TIMEOUT", "8.0"))
    OPENAI_TIMEOUT: float = float(os.environ.get("OPENAI_TIMEOUT", "10.0"))
    
    # Configuration du cache de recherche
    SEARCH_CACHE_ENABLED: bool = os.environ.get("SEARCH_CACHE_ENABLED", "true").lower() == "true"
    SEARCH_CACHE_TTL: int = int(os.environ.get("SEARCH_CACHE_TTL", "300"))
    SEARCH_CACHE_MAX_SIZE: int = int(os.environ.get("SEARCH_CACHE_MAX_SIZE", "1000"))
    SEARCH_CACHE_SIZE: int = int(os.environ.get("SEARCH_CACHE_SIZE", "1000"))
    
    # Configuration de la recherche hybride
    DEFAULT_SEARCH_TYPE: str = os.environ.get("DEFAULT_SEARCH_TYPE", "hybrid")
    MIN_LEXICAL_SCORE: float = float(os.environ.get("MIN_LEXICAL_SCORE", "1.0"))
    MIN_SEMANTIC_SCORE: float = float(os.environ.get("MIN_SEMANTIC_SCORE", "0.5"))
    MAX_RESULTS_PER_ENGINE: int = int(os.environ.get("MAX_RESULTS_PER_ENGINE", "50"))
    
    # Configuration des embeddings
    EMBEDDING_DIMENSIONS: int = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
    EMBEDDING_BATCH_SIZE: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "100"))
    EMBEDDING_CACHE_TTL: int = int(os.environ.get("EMBEDDING_CACHE_TTL", "3600"))
    EMBEDDING_CACHE_ENABLED: bool = os.environ.get("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    EMBEDDING_CACHE_MAX_SIZE: int = int(os.environ.get("EMBEDDING_CACHE_MAX_SIZE", "10000"))
    EMBEDDING_CACHE_SIZE: int = int(os.environ.get("EMBEDDING_CACHE_SIZE", "5000"))
    EMBEDDING_MAX_RETRIES: int = int(os.environ.get("EMBEDDING_MAX_RETRIES", "3"))
    
    # Configuration de pagination
    DEFAULT_SEARCH_LIMIT: int = int(os.environ.get("DEFAULT_SEARCH_LIMIT", "20"))
    MAX_SEARCH_LIMIT: int = int(os.environ.get("MAX_SEARCH_LIMIT", "100"))
    DEFAULT_LIMIT: int = int(os.environ.get("DEFAULT_LIMIT", "20"))
    
    # Configuration des timeouts de recherche
    SEARCH_TIMEOUT: float = float(os.environ.get("SEARCH_TIMEOUT", "15.0"))
    HEALTH_CHECK_TIMEOUT: float = float(os.environ.get("HEALTH_CHECK_TIMEOUT", "5.0"))
    QUICK_SEARCH_TIMEOUT: float = float(os.environ.get("QUICK_SEARCH_TIMEOUT", "3.0"))
    STANDARD_SEARCH_TIMEOUT: float = float(os.environ.get("STANDARD_SEARCH_TIMEOUT", "8.0"))
    COMPLEX_SEARCH_TIMEOUT: float = float(os.environ.get("COMPLEX_SEARCH_TIMEOUT", "15.0"))
    
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
    SIMILARITY_THRESHOLD_DEFAULT: float = float(os.environ.get("SIMILARITY_THRESHOLD_DEFAULT", "0.5"))
    SIMILARITY_THRESHOLD_STRICT: float = float(os.environ.get("SIMILARITY_THRESHOLD_STRICT", "0.7"))
    SIMILARITY_THRESHOLD_LOOSE: float = float(os.environ.get("SIMILARITY_THRESHOLD_LOOSE", "0.3"))
    
    # Seuils de similarité (noms alternatifs pour compatibilité)
    SEMANTIC_SIMILARITY_THRESHOLD_DEFAULT: float = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD_DEFAULT", "0.5"))
    SEMANTIC_SIMILARITY_THRESHOLD_STRICT: float = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD_STRICT", "0.7"))
    SEMANTIC_SIMILARITY_THRESHOLD_LOOSE: float = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD_LOOSE", "0.3"))
    
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
    ADAPTIVE_WEIGHTING: bool = os.environ.get("ADAPTIVE_WEIGHTING", "true").lower() == "true"
    
    # Facteurs de qualité
    QUALITY_BOOST_FACTOR: float = float(os.environ.get("QUALITY_BOOST_FACTOR", "1.2"))
    MIN_SCORE_THRESHOLD: float = float(os.environ.get("MIN_SCORE_THRESHOLD", "0.1"))
    
    # Déduplication et diversification
    ENABLE_DEDUPLICATION: bool = os.environ.get("ENABLE_DEDUPLICATION", "true").lower() == "true"
    DEDUP_SIMILARITY_THRESHOLD: float = float(os.environ.get("DEDUP_SIMILARITY_THRESHOLD", "0.95"))
    ENABLE_DIVERSIFICATION: bool = os.environ.get("ENABLE_DIVERSIFICATION", "true").lower() == "true"
    DIVERSITY_FACTOR: float = float(os.environ.get("DIVERSITY_FACTOR", "0.3"))
    MAX_SAME_MERCHANT: int = int(os.environ.get("MAX_SAME_MERCHANT", "3"))
    
    # Options de fallback
    ENABLE_FALLBACK: bool = os.environ.get("ENABLE_FALLBACK", "true").lower() == "true"
    ENABLE_PARALLEL_SEARCH: bool = os.environ.get("ENABLE_PARALLEL_SEARCH", "true").lower() == "true"
    
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
                "search": self.SEARCH_TIMEOUT
            },
            "cache": {
                "search_enabled": self.SEARCH_CACHE_ENABLED,
                "search_ttl": self.SEARCH_CACHE_TTL,
                "embedding_enabled": self.EMBEDDING_CACHE_ENABLED,
                "embedding_ttl": self.EMBEDDING_CACHE_TTL
            },
            "hybrid_search": {
                "default_type": self.DEFAULT_SEARCH_TYPE,
                "lexical_weight": self.DEFAULT_LEXICAL_WEIGHT,
                "semantic_weight": self.DEFAULT_SEMANTIC_WEIGHT
            },
            "limits": {
                "default_search": self.DEFAULT_SEARCH_LIMIT,
                "max_search": self.MAX_SEARCH_LIMIT,
                "lexical_max": self.LEXICAL_MAX_RESULTS,
                "semantic_max": self.SEMANTIC_MAX_RESULTS
            },
            "min_scores": {
                "lexical": self.MIN_LEXICAL_SCORE,
                "semantic": self.MIN_SEMANTIC_SCORE
            },
            "quality_thresholds": {
                "excellent": self.QUALITY_EXCELLENT_THRESHOLD,
                "good": self.QUALITY_GOOD_THRESHOLD,
                "medium": self.QUALITY_MEDIUM_THRESHOLD,
                "poor": self.QUALITY_POOR_THRESHOLD
            }
        }
    
    def validate_search_config(self) -> dict:
        """Valide la cohérence de la configuration de recherche."""
        validation = {"valid": True, "warnings": [], "errors": []}
        
        # Validation des poids hybrides
        total_weight = self.DEFAULT_LEXICAL_WEIGHT + self.DEFAULT_SEMANTIC_WEIGHT
        if abs(total_weight - 1.0) > 0.01:
            validation["errors"].append(f"Lexical + semantic weights must equal 1.0, got {total_weight}")
            validation["valid"] = False
        
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
        
        # Validation des timeouts
        if self.ELASTICSEARCH_TIMEOUT > self.SEARCH_TIMEOUT:
            validation["warnings"].append("Elasticsearch timeout is greater than overall search timeout")
        
        if self.QDRANT_TIMEOUT > self.SEARCH_TIMEOUT:
            validation["warnings"].append("Qdrant timeout is greater than overall search timeout")
        
        return validation
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"


# Initialisation du singleton de configuration globale
settings = GlobalSettings()

# Afficher un message de démarrage
logger.debug(f"Configuration chargée pour l'environnement: {settings.ENVIRONMENT}")
logger.debug(f"Configuration de recherche: {settings.get_search_config_summary()}")