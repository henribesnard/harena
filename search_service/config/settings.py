"""
Configuration spécifique au search_service.

Ce module définit les paramètres de configuration pour le service de recherche,
séparés de la configuration globale pour permettre un réglage fin.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SearchServiceConfig:
    """Configuration principale du search_service."""
    
    # Configuration générale
    service_name: str = "search_service"
    version: str = "1.0.0"
    debug: bool = False
    
    # Configuration des clients
    elasticsearch_timeout: float = 5.0
    qdrant_timeout: float = 8.0
    openai_timeout: float = 10.0
    
    # Configuration du cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_max_size: int = 1000
    
    # Configuration de la recherche hybride
    default_search_type: str = "hybrid"
    default_lexical_weight: float = 0.6
    default_semantic_weight: float = 0.4
    
    # Seuils de qualité et performance
    min_lexical_score: float = 1.0
    min_semantic_score: float = 0.5
    max_results_per_engine: int = 50
    
    # Configuration des embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 100
    embedding_cache_ttl: int = 3600
    
    # Configuration de pagination
    default_limit: int = 20
    max_limit: int = 100
    
    # Configuration des timeouts
    search_timeout: float = 15.0
    health_check_timeout: float = 5.0
    
    # Configuration des retry
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Configuration du monitoring
    metrics_enabled: bool = True
    detailed_logging: bool = False
    performance_monitoring: bool = True


@dataclass 
class LexicalSearchConfig:
    """Configuration spécifique à la recherche lexicale."""
    
    # Boost factors pour différents champs
    boost_exact_phrase: float = 10.0
    boost_merchant_name: float = 5.0
    boost_primary_description: float = 3.0
    boost_searchable_text: float = 4.0
    boost_clean_description: float = 2.5
    
    # Configuration des requêtes
    enable_fuzzy: bool = True
    enable_wildcards: bool = True
    enable_synonyms: bool = True
    minimum_should_match: str = "1"
    
    # Configuration du highlighting
    highlight_enabled: bool = True
    highlight_fragment_size: int = 150
    highlight_max_fragments: int = 3
    
    # Filtres et seuils
    min_score_threshold: float = 1.0
    max_results: int = 50


@dataclass
class SemanticSearchConfig:
    """Configuration spécifique à la recherche sémantique."""
    
    # Seuils de similarité par type de requête
    similarity_threshold_default: float = 0.3
    similarity_threshold_strict: float = 0.55
    similarity_threshold_loose: float = 0.15
    
    # Configuration des requêtes
    max_results: int = 50
    enable_filtering: bool = True
    fallback_to_unfiltered: bool = True
    
    # Configuration des recommandations
    recommendation_enabled: bool = True
    recommendation_threshold: float = 0.6


@dataclass
class CacheConfig:
    """Configuration du système de cache."""
    
    # Cache de résultats de recherche
    search_cache_enabled: bool = True
    search_cache_ttl: int = 300
    search_cache_max_size: int = 1000
    
    # Cache d'embeddings
    embedding_cache_enabled: bool = True
    embedding_cache_ttl: int = 3600
    embedding_cache_max_size: int = 10000
    
    # Cache de requêtes analysées
    query_analysis_cache_enabled: bool = True
    query_analysis_cache_ttl: int = 1800
    query_analysis_cache_max_size: int = 500


@dataclass
class PerformanceConfig:
    """Configuration des performances et optimisations."""
    
    # Timeouts par type d'opération
    quick_search_timeout: float = 3.0
    standard_search_timeout: float = 8.0
    complex_search_timeout: float = 15.0
    
    # Limites de concurrence
    max_concurrent_searches: int = 10
    max_concurrent_embeddings: int = 5
    
    # Configuration des warmup
    warmup_enabled: bool = True
    warmup_queries: list = field(default_factory=lambda: [
        "restaurant", "virement", "carte bancaire", 
        "supermarché", "essence", "pharmacie"
    ])
    
    # Métriques et monitoring
    enable_detailed_metrics: bool = False
    metrics_collection_interval: int = 60
    performance_alerting: bool = False


@dataclass
class QualityConfig:
    """Configuration de l'évaluation de qualité."""
    
    # Seuils de qualité pour les résultats
    excellent_threshold: float = 0.9
    good_threshold: float = 0.7
    medium_threshold: float = 0.5
    poor_threshold: float = 0.3
    
    # Facteurs de qualité
    min_results_for_good_quality: int = 3
    max_results_for_quality_eval: int = 10
    diversity_threshold: float = 0.6
    
    # Configuration de l'amélioration automatique
    auto_query_optimization: bool = True
    suggestion_enabled: bool = True
    max_suggestions: int = 5


class SearchServiceSettings:
    """
    Gestionnaire des paramètres du search_service.
    
    Charge la configuration depuis les variables d'environnement
    et fournit des valeurs par défaut sensées.
    """
    
    def __init__(self):
        self.search_service = self._load_search_service_config()
        self.lexical_search = self._load_lexical_search_config()
        self.semantic_search = self._load_semantic_search_config()
        self.cache = self._load_cache_config()
        self.performance = self._load_performance_config()
        self.quality = self._load_quality_config()
    
    def _load_search_service_config(self) -> SearchServiceConfig:
        """Charge la configuration principale."""
        return SearchServiceConfig(
            service_name=os.getenv("SEARCH_SERVICE_NAME", "search_service"),
            version=os.getenv("SEARCH_SERVICE_VERSION", "1.0.0"),
            debug=os.getenv("SEARCH_SERVICE_DEBUG", "false").lower() == "true",
            
            # Timeouts
            elasticsearch_timeout=float(os.getenv("ELASTICSEARCH_TIMEOUT", "5.0")),
            qdrant_timeout=float(os.getenv("QDRANT_TIMEOUT", "8.0")),
            openai_timeout=float(os.getenv("OPENAI_TIMEOUT", "10.0")),
            
            # Cache
            cache_enabled=os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("SEARCH_CACHE_TTL", "300")),
            cache_max_size=int(os.getenv("SEARCH_CACHE_MAX_SIZE", "1000")),
            
            # Recherche hybride
            default_search_type=os.getenv("DEFAULT_SEARCH_TYPE", "hybrid"),
            default_lexical_weight=float(os.getenv("DEFAULT_LEXICAL_WEIGHT", "0.6")),
            default_semantic_weight=float(os.getenv("DEFAULT_SEMANTIC_WEIGHT", "0.4")),
            
            # Seuils
            min_lexical_score=float(os.getenv("MIN_LEXICAL_SCORE", "1.0")),
            min_semantic_score=float(os.getenv("MIN_SEMANTIC_SCORE", "0.5")),
            max_results_per_engine=int(os.getenv("MAX_RESULTS_PER_ENGINE", "50")),
            
            # Embeddings
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            embedding_cache_ttl=int(os.getenv("EMBEDDING_CACHE_TTL", "3600")),
            
            # Pagination
            default_limit=int(os.getenv("DEFAULT_SEARCH_LIMIT", "20")),
            max_limit=int(os.getenv("MAX_SEARCH_LIMIT", "100")),
            
            # Timeouts généraux
            search_timeout=float(os.getenv("SEARCH_TIMEOUT", "15.0")),
            health_check_timeout=float(os.getenv("HEALTH_CHECK_TIMEOUT", "5.0")),
            
            # Retry
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
            
            # Monitoring
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            detailed_logging=os.getenv("DETAILED_LOGGING", "false").lower() == "true",
            performance_monitoring=os.getenv("PERFORMANCE_MONITORING", "true").lower() == "true"
        )
    
    def _load_lexical_search_config(self) -> LexicalSearchConfig:
        """Charge la configuration de recherche lexicale."""
        return LexicalSearchConfig(
            # Boost factors optimisés basés sur le validateur
            boost_exact_phrase=float(os.getenv("BOOST_EXACT_PHRASE", "10.0")),
            boost_merchant_name=float(os.getenv("BOOST_MERCHANT_NAME", "5.0")),
            boost_primary_description=float(os.getenv("BOOST_PRIMARY_DESCRIPTION", "3.0")),
            boost_searchable_text=float(os.getenv("BOOST_SEARCHABLE_TEXT", "4.0")),
            boost_clean_description=float(os.getenv("BOOST_CLEAN_DESCRIPTION", "2.5")),
            
            # Options de requête
            enable_fuzzy=os.getenv("ENABLE_FUZZY", "true").lower() == "true",
            enable_wildcards=os.getenv("ENABLE_WILDCARDS", "true").lower() == "true",
            enable_synonyms=os.getenv("ENABLE_SYNONYMS", "true").lower() == "true",
            minimum_should_match=os.getenv("MINIMUM_SHOULD_MATCH", "1"),
            
            # Highlighting
            highlight_enabled=os.getenv("HIGHLIGHT_ENABLED", "true").lower() == "true",
            highlight_fragment_size=int(os.getenv("HIGHLIGHT_FRAGMENT_SIZE", "150")),
            highlight_max_fragments=int(os.getenv("HIGHLIGHT_MAX_FRAGMENTS", "3")),
            
            # Filtres
            min_score_threshold=float(os.getenv("LEXICAL_MIN_SCORE", "1.0")),
            max_results=int(os.getenv("LEXICAL_MAX_RESULTS", "50"))
        )
    
    def _load_semantic_search_config(self) -> SemanticSearchConfig:
        """Charge la configuration de recherche sémantique."""
        return SemanticSearchConfig(
            # Seuils de similarité basés sur les résultats du validateur
            similarity_threshold_default=float(os.getenv("SIMILARITY_THRESHOLD_DEFAULT", "0.3")),
            similarity_threshold_strict=float(os.getenv("SIMILARITY_THRESHOLD_STRICT", "0.55")),
            similarity_threshold_loose=float(os.getenv("SIMILARITY_THRESHOLD_LOOSE", "0.15")),
            
            # Configuration des requêtes
            max_results=int(os.getenv("SEMANTIC_MAX_RESULTS", "50")),
            enable_filtering=os.getenv("SEMANTIC_ENABLE_FILTERING", "true").lower() == "true",
            fallback_to_unfiltered=os.getenv("SEMANTIC_FALLBACK_UNFILTERED", "true").lower() == "true",
            
            # Recommandations
            recommendation_enabled=os.getenv("RECOMMENDATION_ENABLED", "true").lower() == "true",
            recommendation_threshold=float(os.getenv("RECOMMENDATION_THRESHOLD", "0.6"))
        )
    
    def _load_cache_config(self) -> CacheConfig:
        """Charge la configuration du cache."""
        return CacheConfig(
            # Cache de recherche
            search_cache_enabled=os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true",
            search_cache_ttl=int(os.getenv("SEARCH_CACHE_TTL", "300")),
            search_cache_max_size=int(os.getenv("SEARCH_CACHE_MAX_SIZE", "1000")),
            
            # Cache d'embeddings
            embedding_cache_enabled=os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true",
            embedding_cache_ttl=int(os.getenv("EMBEDDING_CACHE_TTL", "3600")),
            embedding_cache_max_size=int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "10000")),
            
            # Cache d'analyse de requêtes
            query_analysis_cache_enabled=os.getenv("QUERY_ANALYSIS_CACHE_ENABLED", "true").lower() == "true",
            query_analysis_cache_ttl=int(os.getenv("QUERY_ANALYSIS_CACHE_TTL", "1800")),
            query_analysis_cache_max_size=int(os.getenv("QUERY_ANALYSIS_CACHE_MAX_SIZE", "500"))
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Charge la configuration de performance."""
        # Parse warmup queries from env
        warmup_queries_str = os.getenv("WARMUP_QUERIES", "restaurant,virement,carte bancaire,supermarché,essence,pharmacie")
        warmup_queries = [q.strip() for q in warmup_queries_str.split(",") if q.strip()]
        
        return PerformanceConfig(
            # Timeouts par type
            quick_search_timeout=float(os.getenv("QUICK_SEARCH_TIMEOUT", "3.0")),
            standard_search_timeout=float(os.getenv("STANDARD_SEARCH_TIMEOUT", "8.0")),
            complex_search_timeout=float(os.getenv("COMPLEX_SEARCH_TIMEOUT", "15.0")),
            
            # Concurrence
            max_concurrent_searches=int(os.getenv("MAX_CONCURRENT_SEARCHES", "10")),
            max_concurrent_embeddings=int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "5")),
            
            # Warmup
            warmup_enabled=os.getenv("WARMUP_ENABLED", "true").lower() == "true",
            warmup_queries=warmup_queries,
            
            # Métriques
            enable_detailed_metrics=os.getenv("ENABLE_DETAILED_METRICS", "false").lower() == "true",
            metrics_collection_interval=int(os.getenv("METRICS_COLLECTION_INTERVAL", "60")),
            performance_alerting=os.getenv("PERFORMANCE_ALERTING", "false").lower() == "true"
        )
    
    def _load_quality_config(self) -> QualityConfig:
        """Charge la configuration de qualité."""
        return QualityConfig(
            # Seuils de qualité
            excellent_threshold=float(os.getenv("QUALITY_EXCELLENT_THRESHOLD", "0.9")),
            good_threshold=float(os.getenv("QUALITY_GOOD_THRESHOLD", "0.7")),
            medium_threshold=float(os.getenv("QUALITY_MEDIUM_THRESHOLD", "0.5")),
            poor_threshold=float(os.getenv("QUALITY_POOR_THRESHOLD", "0.3")),
            
            # Facteurs de qualité
            min_results_for_good_quality=int(os.getenv("MIN_RESULTS_FOR_GOOD_QUALITY", "3")),
            max_results_for_quality_eval=int(os.getenv("MAX_RESULTS_FOR_QUALITY_EVAL", "10")),
            diversity_threshold=float(os.getenv("DIVERSITY_THRESHOLD", "0.6")),
            
            # Optimisation automatique
            auto_query_optimization=os.getenv("AUTO_QUERY_OPTIMIZATION", "true").lower() == "true",
            suggestion_enabled=os.getenv("SUGGESTION_ENABLED", "true").lower() == "true",
            max_suggestions=int(os.getenv("MAX_SUGGESTIONS", "5"))
        )
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Retourne toute la configuration sous forme de dictionnaire."""
        return {
            "search_service": self.search_service.__dict__,
            "lexical_search": self.lexical_search.__dict__,
            "semantic_search": self.semantic_search.__dict__,
            "cache": self.cache.__dict__,
            "performance": self.performance.__dict__,
            "quality": self.quality.__dict__
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Valide la cohérence de la configuration."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validation des poids hybrides
        total_weight = self.search_service.default_lexical_weight + self.search_service.default_semantic_weight
        if abs(total_weight - 1.0) > 0.01:
            validation_result["errors"].append(
                f"Lexical + semantic weights must equal 1.0, got {total_weight}"
            )
            validation_result["valid"] = False
        
        # Validation des timeouts
        if self.search_service.elasticsearch_timeout > self.search_service.search_timeout:
            validation_result["warnings"].append(
                "Elasticsearch timeout is greater than overall search timeout"
            )
        
        if self.search_service.qdrant_timeout > self.search_service.search_timeout:
            validation_result["warnings"].append(
                "Qdrant timeout is greater than overall search timeout"
            )
        
        # Validation des seuils de similarité
        if self.semantic_search.similarity_threshold_loose > self.semantic_search.similarity_threshold_default:
            validation_result["errors"].append(
                "Loose similarity threshold should be <= default threshold"
            )
            validation_result["valid"] = False
        
        if self.semantic_search.similarity_threshold_default > self.semantic_search.similarity_threshold_strict:
            validation_result["errors"].append(
                "Default similarity threshold should be <= strict threshold"
            )
            validation_result["valid"] = False
        
        # Validation des limites
        if self.search_service.default_limit > self.search_service.max_limit:
            validation_result["errors"].append(
                "Default limit cannot be greater than max limit"
            )
            validation_result["valid"] = False
        
        # Validation du cache
        if self.cache.search_cache_ttl <= 0:
            validation_result["warnings"].append(
                "Search cache TTL should be positive"
            )
        
        if self.cache.embedding_cache_ttl <= 0:
            validation_result["warnings"].append(
                "Embedding cache TTL should be positive"
            )
        
        # Validation des seuils de qualité
        quality_thresholds = [
            self.quality.poor_threshold,
            self.quality.medium_threshold,
            self.quality.good_threshold,
            self.quality.excellent_threshold
        ]
        
        if quality_thresholds != sorted(quality_thresholds):
            validation_result["errors"].append(
                "Quality thresholds must be in ascending order"
            )
            validation_result["valid"] = False
        
        return validation_result
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'environnement de configuration."""
        env_vars = {}
        
        # Variables d'environnement liées à la recherche
        search_env_vars = [
            "SEARCH_SERVICE_NAME", "SEARCH_SERVICE_VERSION", "SEARCH_SERVICE_DEBUG",
            "ELASTICSEARCH_TIMEOUT", "QDRANT_TIMEOUT", "OPENAI_TIMEOUT",
            "SEARCH_CACHE_ENABLED", "SEARCH_CACHE_TTL", "SEARCH_CACHE_MAX_SIZE",
            "DEFAULT_SEARCH_TYPE", "DEFAULT_LEXICAL_WEIGHT", "DEFAULT_SEMANTIC_WEIGHT",
            "MIN_LEXICAL_SCORE", "MIN_SEMANTIC_SCORE", "MAX_RESULTS_PER_ENGINE",
            "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "EMBEDDING_BATCH_SIZE",
            "DEFAULT_SEARCH_LIMIT", "MAX_SEARCH_LIMIT",
            "BOOST_EXACT_PHRASE", "BOOST_MERCHANT_NAME", "BOOST_PRIMARY_DESCRIPTION",
            "SIMILARITY_THRESHOLD_DEFAULT", "SIMILARITY_THRESHOLD_STRICT",
            "WARMUP_ENABLED", "WARMUP_QUERIES",
            "QUALITY_EXCELLENT_THRESHOLD", "QUALITY_GOOD_THRESHOLD"
        ]
        
        for var in search_env_vars:
            value = os.getenv(var)
            if value is not None:
                env_vars[var] = value
        
        return {
            "environment_variables": env_vars,
            "config_source": "environment + defaults",
            "validation": self.validate_config()
        }


# Instance globale des paramètres
search_settings = SearchServiceSettings()


def get_search_settings() -> SearchServiceSettings:
    """Retourne l'instance des paramètres de recherche."""
    return search_settings


def reload_search_settings() -> SearchServiceSettings:
    """Recharge les paramètres depuis l'environnement."""
    global search_settings
    search_settings = SearchServiceSettings()
    return search_settings


# Fonctions utilitaires pour l'accès rapide

def get_elasticsearch_config() -> Dict[str, Any]:
    """Retourne la configuration Elasticsearch."""
    settings = get_search_settings()
    return {
        "timeout": settings.search_service.elasticsearch_timeout,
        "max_results": settings.lexical_search.max_results,
        "min_score": settings.lexical_search.min_score_threshold,
        "boost_config": {
            "exact_phrase": settings.lexical_search.boost_exact_phrase,
            "merchant_name": settings.lexical_search.boost_merchant_name,
            "primary_description": settings.lexical_search.boost_primary_description,
            "searchable_text": settings.lexical_search.boost_searchable_text,
            "clean_description": settings.lexical_search.boost_clean_description
        },
        "features": {
            "fuzzy": settings.lexical_search.enable_fuzzy,
            "wildcards": settings.lexical_search.enable_wildcards,
            "synonyms": settings.lexical_search.enable_synonyms,
            "highlighting": settings.lexical_search.highlight_enabled
        }
    }


def get_qdrant_config() -> Dict[str, Any]:
    """Retourne la configuration Qdrant."""
    settings = get_search_settings()
    return {
        "timeout": settings.search_service.qdrant_timeout,
        "max_results": settings.semantic_search.max_results,
        "similarity_thresholds": {
            "default": settings.semantic_search.similarity_threshold_default,
            "strict": settings.semantic_search.similarity_threshold_strict,
            "loose": settings.semantic_search.similarity_threshold_loose
        },
        "features": {
            "filtering": settings.semantic_search.enable_filtering,
            "fallback_unfiltered": settings.semantic_search.fallback_to_unfiltered,
            "recommendations": settings.semantic_search.recommendation_enabled
        }
    }


def get_embedding_config() -> Dict[str, Any]:
    """Retourne la configuration des embeddings."""
    settings = get_search_settings()
    return {
        "model": settings.search_service.embedding_model,
        "dimensions": settings.search_service.embedding_dimensions,
        "batch_size": settings.search_service.embedding_batch_size,
        "timeout": settings.search_service.openai_timeout,
        "cache": {
            "enabled": settings.cache.embedding_cache_enabled,
            "ttl": settings.cache.embedding_cache_ttl,
            "max_size": settings.cache.embedding_cache_max_size
        }
    }


def get_hybrid_search_config() -> Dict[str, Any]:
    """Retourne la configuration de la recherche hybride."""
    settings = get_search_settings()
    return {
        "default_type": settings.search_service.default_search_type,
        "weights": {
            "lexical": settings.search_service.default_lexical_weight,
            "semantic": settings.search_service.default_semantic_weight
        },
        "thresholds": {
            "lexical_min_score": settings.search_service.min_lexical_score,
            "semantic_min_score": settings.search_service.min_semantic_score
        },
        "limits": {
            "default": settings.search_service.default_limit,
            "max": settings.search_service.max_limit,
            "per_engine": settings.search_service.max_results_per_engine
        },
        "timeout": settings.search_service.search_timeout
    }


# Configuration des logs pour le search_service
def get_logging_config() -> Dict[str, Any]:
    """Retourne la configuration des logs."""
    settings = get_search_settings()
    
    level = "DEBUG" if settings.search_service.debug else "INFO"
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "detailed" if settings.search_service.detailed_logging else "simple",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "search_service": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            },
            "search_service.core": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            },
            "search_service.clients": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }