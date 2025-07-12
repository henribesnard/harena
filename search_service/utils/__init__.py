"""
Package utils du Search Service
==============================

Package centralisé exposant tous les utilitaires :
- Métriques et monitoring
- Cache et performance  
- Validation et helpers
- Utilitaires système

Architecture :
    utils/ → Exposition des classes et fonctions principales
"""

# === IMPORTS DES CLASSES PRINCIPALES ===

# Métriques
from .metrics import (
    # Classes principales
    MetricsCollector,
    AlertManager,
    MetricsDashboard,
    
    # Métriques spécialisées
    QueryMetrics,
    ResultMetrics,
    SearchMetrics,
    ElasticsearchMetrics,
    LexicalSearchMetrics,
    BusinessMetrics,
    ApiMetrics,
    PerformanceProfiler,
    
    # Instances globales
    metrics_collector,
    alert_manager,
    query_metrics,
    result_metrics,
    search_metrics,
    elasticsearch_metrics,
    lexical_search_metrics,
    business_metrics,
    api_metrics,
    performance_profiler,
    metrics_dashboard,
    
    # Fonctions utilitaires métriques
    export_metrics_to_file,
    reset_all_counters,
    get_top_slow_operations,
    get_error_metrics_summary,
    initialize_metrics_system,
    shutdown_metrics_system,
    
    # Types et enums
    MetricType,
    MetricCategory,
    AlertLevel,
    MetricDefinition,
    MetricValue,
    MetricSample,
    MetricAlert
)

# Cache
from .cache import (
    # Classes principales
    LRUCache,
    CacheManager,
    CacheEntry,
    
    # Instances globales
    cache_manager,
    global_cache_manager,
    
    # Fonctions utilitaires cache
    create_cache_key,
    serialize_cache_value,
    deserialize_cache_value,
    get_cache_statistics,
    get_cache_stats,
    clear_all_caches,
    cleanup_expired_entries,
    periodic_cleanup,
    
    # Décorateur
    cached,
    
    # Types
    CacheStats,
    CacheStrategy,
    CacheLevel,
    EvictionPolicy
)

# Fonction pour accéder au cache manager (compatibilité)
def get_cache_manager():
    """Retourne l'instance globale du cache manager"""
    return cache_manager

# Utilitaires système (NOUVEAU)
from .system_utils import (
    # Fonctions principales
    get_system_metrics,
    get_performance_summary,
    get_utils_performance,
    cleanup_old_metrics,
    get_utils_health,
    
    # Types
    HealthStatus,
    ComponentType
)

# Validation (à importer quand disponible)
try:
    from .validators import (
        ValidatorFactory,
        ValidationResult,
        SecurityValidator,
        PerformanceValidator
    )
except ImportError:
    # Validators pas encore implémentés
    ValidatorFactory = None
    ValidationResult = None
    SecurityValidator = None
    PerformanceValidator = None

# Elasticsearch helpers
from .elasticsearch_helpers import (
    # Classe principale
    ElasticsearchQueryBuilder,
    
    # Autres classes
    ResponseFormatter,
    ErrorHandler,
    IndexManager,
    QueryOptimizer,
    FieldAnalyzer,
    QueryPerformanceAnalyzer,
    
    # Exceptions
    ElasticsearchError,
    
    # Structures de données
    ElasticsearchResponse,
    FieldConfiguration,
    
    # Enums
    QueryOptimization,
    QueryType,
    FieldType,
    
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


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === MÉTRIQUES ===
    # Classes principales
    "MetricsCollector",
    "AlertManager", 
    "MetricsDashboard",
    
    # Métriques spécialisées
    "QueryMetrics",
    "ResultMetrics",
    "SearchMetrics",
    "ElasticsearchMetrics", 
    "LexicalSearchMetrics",
    "BusinessMetrics",
    "ApiMetrics",
    "PerformanceProfiler",
    
    # Instances globales métriques
    "metrics_collector",
    "alert_manager",
    "query_metrics",
    "result_metrics",
    "search_metrics",
    "elasticsearch_metrics",
    "lexical_search_metrics", 
    "business_metrics",
    "api_metrics",
    "performance_profiler",
    "metrics_dashboard",
    
    # Fonctions utilitaires métriques
    "export_metrics_to_file",
    "reset_all_counters",
    "get_top_slow_operations",
    "get_error_metrics_summary",
    "initialize_metrics_system",
    "shutdown_metrics_system",
    
    # Types métriques
    "MetricType",
    "MetricCategory",
    "AlertLevel",
    "MetricDefinition",
    "MetricValue",
    "MetricSample",
    "MetricAlert",
    
    # === CACHE ===
    # Classes cache
    "LRUCache",
    "CacheManager",
    "CacheEntry",
    "CacheStats",
    
    # Instances globales cache
    "get_cache_manager",
    "cache_manager",
    "global_cache_manager",
    
    # === UTILITAIRES SYSTÈME ===
    # Fonctions système principales
    "get_system_metrics",
    "get_performance_summary", 
    "get_utils_performance",
    "cleanup_old_metrics",
    "get_utils_health",
    
    # Types système
    "HealthStatus",
    "ComponentType",
    
    # === VALIDATION (conditionnel) ===
    "ValidatorFactory",
    "ValidationResult", 
    "SecurityValidator",
    "PerformanceValidator",
    
    # === ELASTICSEARCH HELPERS ===
    # Classe principale
    "ElasticsearchQueryBuilder",
    
    # Autres classes ES
    "ResponseFormatter",
    "ErrorHandler", 
    "IndexManager",
    "QueryOptimizer",
    "FieldAnalyzer",
    "QueryPerformanceAnalyzer",
    
    # Exceptions ES
    "ElasticsearchError",
    
    # Structures de données ES
    "ElasticsearchResponse",
    "FieldConfiguration",
    
    # Enums ES
    "QueryOptimization",
    "QueryType", 
    "FieldType",
    
    # Fonctions utilitaires ES
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
    "log_query_for_debug"
]


# === INFORMATIONS DU PACKAGE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Package utilitaires centralisé pour le Search Service"