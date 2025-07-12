"""
Module utils du Search Service
Expose tous les utilitaires pour validation, cache, Elasticsearch et métriques
"""

# === VALIDATION ===
from .validators import (
    # Exceptions
    ValidationError,
    
    # Validateurs principaux
    SecurityValidator,
    ContractValidator,
    FilterValidator,
    ElasticsearchQueryValidator,
    PerformanceValidator,
    BatchValidator,
    
    # Factory
    ValidatorFactory,
    
    # Fonctions utilitaires
    sanitize_query_string,
    is_valid_user_id,
    get_field_type,
    validate_query_timeout,
    estimate_result_size
)

# === ELASTICSEARCH HELPERS ===
from .elasticsearch_helpers import (
    # Exceptions
    ElasticsearchError,
    
    # Classes principales
    QueryBuilder,
    ResponseFormatter,
    ErrorHandler,
    IndexManager,
    QueryOptimizer,
    FieldAnalyzer,
    QueryPerformanceAnalyzer,
    
    # Structures de données
    ElasticsearchResponse,
    QueryOptimization,
    
    # Fonctions utilitaires
    build_simple_user_query,
    build_text_search_query,
    validate_elasticsearch_response,
    extract_error_details,
    estimate_query_cost,
    normalize_query_for_cache,
    build_count_query,
    merge_query_filters,
    get_field_mapping_type,
    optimize_pagination,
    
    # Debugging
    explain_query_execution,
    profile_query_execution,
    log_query_for_debug
)

# === CACHE ===
from .cache import (
    # Enums
    CacheStrategy,
    CacheLevel,
    EvictionPolicy,
    
    # Classes principales
    LRUCache,
    SmartCache,
    CacheManager,
    CacheKeyGenerator,
    
    # Structures de données
    CacheEntry,
    
    # Instance globale
    cache_manager,
    
    # Décorateurs
    cached,
    
    # Fonctions utilitaires
    get_cache_stats,
    clear_all_caches,
    cleanup_expired_entries,
    periodic_cleanup
)

# Version du module utils
__version__ = "1.0.0"

# === EXPORTS ORGANISÉS ===

# Validation et sécurité
__all_validation__ = [
    "ValidationError",
    "SecurityValidator",
    "ContractValidator",
    "FilterValidator",
    "ValidatorFactory",
    "sanitize_query_string",
    "is_valid_user_id"
]

# Elasticsearch
__all_elasticsearch__ = [
    "ElasticsearchError",
    "QueryBuilder",
    "ResponseFormatter",
    "ErrorHandler",
    "IndexManager",
    "ElasticsearchResponse",
    "build_simple_user_query",
    "estimate_query_cost"
]

# Cache
__all_cache__ = [
    "CacheStrategy",
    "LRUCache",
    "SmartCache",
    "cache_manager",
    "cached",
    "get_cache_stats"
]

# Export principal
__all__ = [
    # === VALIDATION ===
    # Exceptions
    "ValidationError",
    
    # Validateurs
    "SecurityValidator",
    "ContractValidator", 
    "FilterValidator",
    "ElasticsearchQueryValidator",
    "PerformanceValidator",
    "BatchValidator",
    "ValidatorFactory",
    
    # Utilitaires validation
    "sanitize_query_string",
    "is_valid_user_id",
    "get_field_type",
    "validate_query_timeout",
    "estimate_result_size",
    
    # === ELASTICSEARCH ===
    # Exceptions
    "ElasticsearchError",
    
    # Classes ES
    "QueryBuilder",
    "ResponseFormatter",
    "ErrorHandler", 
    "IndexManager",
    "QueryOptimizer",
    "FieldAnalyzer",
    "QueryPerformanceAnalyzer",
    
    # Structures ES
    "ElasticsearchResponse",
    "QueryOptimization",
    
    # Utilitaires ES
    "build_simple_user_query",
    "build_text_search_query",
    "validate_elasticsearch_response",
    "extract_error_details",
    "estimate_query_cost",
    "normalize_query_for_cache",
    "build_count_query",
    "merge_query_filters",
    "get_field_mapping_type",
    "optimize_pagination",
    
    # Debugging ES
    "explain_query_execution",
    "profile_query_execution",
    "log_query_for_debug",
    
    # === CACHE ===
    # Enums cache
    "CacheStrategy",
    "CacheLevel",
    "EvictionPolicy",
    
    # Classes cache
    "LRUCache",
    "SmartCache", 
    "CacheManager",
    "CacheKeyGenerator",
    "CacheEntry",
    
    # Instance globale cache
    "cache_manager",
    
    # Décorateurs cache
    "cached",
    
    # Utilitaires cache
    "get_cache_stats",
    "clear_all_caches",
    "cleanup_expired_entries",
    "periodic_cleanup"
]

# === HELPERS D'IMPORT ===

def get_validation_tools():
    """Retourne les outils de validation principaux"""
    return {
        "contract_validator": ContractValidator,
        "security_validator": SecurityValidator,
        "filter_validator": FilterValidator,
        "factory": ValidatorFactory
    }

def get_elasticsearch_tools():
    """Retourne les outils Elasticsearch principaux"""
    return {
        "query_builder": QueryBuilder,
        "response_formatter": ResponseFormatter,
        "error_handler": ErrorHandler,
        "index_manager": IndexManager,
        "field_analyzer": FieldAnalyzer,
        "performance_analyzer": QueryPerformanceAnalyzer
    }

def get_cache_tools():
    """Retourne les outils de cache principaux"""
    return {
        "lru_cache": LRUCache,
        "smart_cache": SmartCache,
        "manager": cache_manager,
        "key_generator": CacheKeyGenerator
    }

# === FACTORY CENTRALISÉE ===

class UtilsFactory:
    """Factory centralisée pour tous les utilitaires"""
    
    @staticmethod
    def create_complete_validator() -> ValidatorFactory:
        """Crée un validateur complet configuré"""
        return ValidatorFactory()
    
    @staticmethod
    def create_query_builder() -> QueryBuilder:
        """Crée un constructeur de requêtes optimisé"""
        return QueryBuilder()
    
    @staticmethod
    def create_response_formatter() -> ResponseFormatter:
        """Crée un formateur de réponses"""
        return ResponseFormatter()
    
    @staticmethod
    def create_cache_manager() -> CacheManager:
        """Crée un gestionnaire de cache"""
        return CacheManager()
    
    @staticmethod
    def create_search_cache(max_size: int = 500, max_memory_mb: int = 50) -> SmartCache:
        """Crée un cache optimisé pour les recherches"""
        return SmartCache(
            strategy=CacheStrategy.MEMORY_ONLY,
            memory_cache_size=max_size,
            memory_cache_mb=max_memory_mb,
            default_ttl=300  # 5 minutes
        )
    
    @staticmethod
    def get_all_tools():
        """Retourne tous les outils organisés"""
        return {
            "validation": get_validation_tools(),
            "elasticsearch": get_elasticsearch_tools(), 
            "cache": get_cache_tools()
        }

# === CONFIGURATION PAR DÉFAUT ===

def configure_default_utils():
    """Configure les utilitaires avec les paramètres par défaut"""
    # Démarrer le nettoyage périodique du cache
    import asyncio
    try:
        # Vérifier si une boucle d'événements est déjà en cours
        loop = asyncio.get_running_loop()
        # Programmer la tâche de nettoyage
        loop.create_task(periodic_cleanup(interval_seconds=300))  # 5 minutes
    except RuntimeError:
        # Pas de boucle en cours, sera configuré plus tard
        pass

# === VALIDATION MODULE ===

def validate_utils_import():
    """Valide que tous les utilitaires sont correctement importés"""
    import sys
    current_module = sys.modules[__name__]
    
    # Vérifier les classes essentielles
    essential_classes = [
        "ValidationError", "ContractValidator", "SecurityValidator",
        "ElasticsearchError", "QueryBuilder", "ResponseFormatter",
        "LRUCache", "SmartCache", "cache_manager"
    ]
    
    missing = []
    for class_name in essential_classes:
        if not hasattr(current_module, class_name):
            missing.append(class_name)
    
    if missing:
        raise ImportError(f"Classes manquantes dans utils: {missing}")
    
    return True

# === HELPERS DE DEBUGGING ===

def get_utils_status() -> dict:
    """Retourne le statut de tous les utilitaires"""
    status = {
        "validation": {
            "available": True,
            "validators_count": 6,
            "factory_ready": hasattr(ValidatorFactory, 'validate_complete_request')
        },
        "elasticsearch": {
            "available": True,
            "query_builder_ready": hasattr(QueryBuilder, 'build_from_internal_request'),
            "response_formatter_ready": hasattr(ResponseFormatter, 'format_elasticsearch_response'),
            "error_handler_ready": hasattr(ErrorHandler, 'parse_elasticsearch_error')
        },
        "cache": {
            "available": True,
            "manager_initialized": cache_manager is not None,
            "lru_cache_ready": hasattr(LRUCache, 'get'),
            "smart_cache_ready": hasattr(SmartCache, 'get'),
            "global_stats": get_cache_stats() if cache_manager else {}
        }
    }
    
    return status

def log_utils_status():
    """Log le statut des utilitaires pour debugging"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        status = get_utils_status()
        logger.info("Utils Status:")
        for category, info in status.items():
            logger.info(f"  {category}: {info}")
    except Exception as e:
        logger.error(f"Error getting utils status: {e}")

# === INITIALISATION AUTOMATIQUE ===

# Auto-validation au chargement
try:
    validate_utils_import()
    configure_default_utils()
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Utils initialization warning: {e}")

# Helpers disponibles
__helpers__ = [
    "get_validation_tools",
    "get_elasticsearch_tools",
    "get_cache_tools",
    "UtilsFactory",
    "configure_default_utils",
    "validate_utils_import",
    "get_utils_status",
    "log_utils_status"
]

# Ajout des helpers aux exports
__all__.extend(__helpers__)