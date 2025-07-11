"""
🛠️ Utilitaires Search Service - Boîte à Outils Spécialisée
=========================================================

Module utilitaires pour le Search Service avec des outils spécialisés
pour Elasticsearch, validation, cache et optimisations de performance.

Organisation:
- Elasticsearch helpers (priorité #1 pour le core)
- Validators (validation requêtes)
- Cache LRU (performance)
- Metrics (monitoring)
"""

# =============================================================================
# 🔍 UTILITAIRES ELASTICSEARCH (PRIORITÉ #1)
# =============================================================================

from .elasticsearch_helpers import (
    # Construction requêtes
    build_bool_query, build_text_search_query, build_filter_query, build_aggregation_query,
    # Filtres spécialisés
    build_user_filter, build_category_filter, build_merchant_filter, 
    build_amount_filter, build_date_filter, build_text_filter,
    # Agrégations financières
    build_sum_aggregation, build_count_aggregation, build_avg_aggregation,
    build_date_histogram, build_terms_aggregation, build_stats_aggregation,
    # Optimisation requêtes
    optimize_query_performance, add_query_cache, calculate_query_complexity,
    # Helpers génériques
    escape_elasticsearch_query, validate_field_name, format_elasticsearch_error,
    parse_elasticsearch_response, extract_aggregation_results,
    # Constantes
    FINANCIAL_FIELDS, SEARCHABLE_FIELDS, FILTERABLE_FIELDS, AGGREGATABLE_FIELDS,
    MAX_QUERY_SIZE, DEFAULT_TIMEOUT_MS, CACHE_TTL_SECONDS,
)

# =============================================================================
# ✅ VALIDATORS (VALIDATION REQUÊTES)
# =============================================================================

from .validators import (
    # Validators principaux
    ElasticsearchQueryValidator, SearchServiceQueryValidator, FilterValidator,
    # Validation spécialisée
    validate_user_isolation, validate_financial_query, validate_search_parameters,
    validate_filter_security, validate_aggregation_request, validate_query_complexity,
    # Validation champs
    validate_field_access, validate_field_types, validate_field_values,
    validate_date_ranges, validate_amount_ranges, validate_text_search,
    # Validation performance
    validate_query_limits, validate_timeout_settings, validate_cache_settings,
    # Sanitization
    sanitize_user_input, sanitize_elasticsearch_query, sanitize_field_values,
    # Helpers validation
    is_valid_user_id, is_valid_field_name, is_safe_query_value,
    # Exceptions
    ValidationError, SecurityValidationError, PerformanceValidationError,
    # Constantes
    MAX_QUERY_LENGTH, MAX_RESULTS_LIMIT, ALLOWED_OPERATORS, FORBIDDEN_FIELDS,
)

# =============================================================================
# 💾 CACHE LRU (PERFORMANCE)
# =============================================================================

from .cache import (
    # Cache principal
    SearchResultsCache, QueryCache, AggregationCache,
    # Cache managers
    CacheManager, CacheKey, CacheStats, CacheConfig,
    # Cache spécialisé
    UserQueryCache, FinancialDataCache, PerformanceCache,
    # Cache operations
    cache_search_results, cache_aggregations, cache_query_templates,
    invalidate_user_cache, invalidate_query_cache, clear_all_caches,
    # Cache metrics
    get_cache_hit_rate, get_cache_memory_usage, get_cache_statistics,
    # Cache strategies
    CacheStrategy, LRUCacheStrategy, TTLCacheStrategy, SizeLimitedCacheStrategy,
    # Helpers
    generate_cache_key, serialize_cache_value, deserialize_cache_value,
    # Constantes
    DEFAULT_CACHE_SIZE, DEFAULT_TTL_SECONDS, MAX_CACHE_MEMORY_MB,
)

# =============================================================================
# 📈 MÉTRIQUES (MONITORING) 
# =============================================================================

from .metrics import (
    # Métriques principales
    SearchMetrics, PerformanceMetrics, ElasticsearchMetrics,
    # Collecteurs métriques
    MetricsCollector, PerformanceCollector, CacheMetricsCollector,
    # Métriques business
    QueryComplexityMetrics, UserBehaviorMetrics, FinancialSearchMetrics,
    # Recording métriques
    record_search_execution, record_query_performance, record_cache_usage,
    record_elasticsearch_operation, record_validation_result, record_error_event,
    # Métriques temps réel
    get_real_time_metrics, get_performance_dashboard, get_health_metrics,
    # Agrégation métriques
    aggregate_hourly_metrics, aggregate_daily_metrics, get_trend_analysis,
    # Alerting
    check_performance_alerts, check_error_rate_alerts, check_resource_alerts,
    # Export métriques
    export_metrics_prometheus, export_metrics_json, export_metrics_csv,
    # Constantes
    METRIC_NAMES, ALERT_THRESHOLDS, DASHBOARD_REFRESH_SECONDS,
)

# =============================================================================
# 📋 EXPORTS GROUPÉS PAR FONCTIONNALITÉ
# =============================================================================

# Utilitaires Elasticsearch (core)
ELASTICSEARCH_UTILS = [
    "build_bool_query", "build_text_search_query", "build_filter_query",
    "optimize_query_performance", "parse_elasticsearch_response", "FINANCIAL_FIELDS"
]

# Validation et sécurité
VALIDATION_UTILS = [
    "ElasticsearchQueryValidator", "SearchServiceQueryValidator", 
    "validate_user_isolation", "sanitize_user_input", "ValidationError"
]

# Cache et performance
CACHE_UTILS = [
    "SearchResultsCache", "CacheManager", "cache_search_results",
    "get_cache_hit_rate", "CacheStrategy", "DEFAULT_CACHE_SIZE"
]

# Monitoring et métriques
METRICS_UTILS = [
    "SearchMetrics", "MetricsCollector", "record_search_execution", 
    "get_real_time_metrics", "check_performance_alerts", "METRIC_NAMES"
]

# =============================================================================
# 🔧 UTILITAIRES GÉNÉRIQUES
# =============================================================================

def get_utility_by_name(util_name: str):
    """Récupérer utilitaire par nom."""
    return globals().get(util_name)

def list_available_utilities() -> dict:
    """Lister tous les utilitaires disponibles par catégorie."""
    return {
        "elasticsearch": ELASTICSEARCH_UTILS,
        "validation": VALIDATION_UTILS,
        "cache": CACHE_UTILS,
        "metrics": METRICS_UTILS,
    }

def validate_utility_imports():
    """Valider que tous les imports fonctionnent."""
    try:
        # Test imports Elasticsearch
        assert build_bool_query is not None
        assert FINANCIAL_FIELDS is not None
        
        # Test imports validation
        assert ElasticsearchQueryValidator is not None
        assert validate_user_isolation is not None
        
        # Test imports cache
        assert SearchResultsCache is not None
        assert CacheManager is not None
        
        # Test imports métriques
        assert SearchMetrics is not None
        assert MetricsCollector is not None
        
        return True
    except (ImportError, AssertionError) as e:
        return False, str(e)

# =============================================================================
# 📋 EXPORTS FINAUX
# =============================================================================

__all__ = [
    # === ELASTICSEARCH HELPERS ===
    # Construction requêtes
    "build_bool_query", "build_text_search_query", "build_filter_query", "build_aggregation_query",
    # Filtres spécialisés
    "build_user_filter", "build_category_filter", "build_merchant_filter", 
    "build_amount_filter", "build_date_filter", "build_text_filter",
    # Agrégations financières
    "build_sum_aggregation", "build_count_aggregation", "build_avg_aggregation",
    "build_date_histogram", "build_terms_aggregation", "build_stats_aggregation",
    # Optimisation
    "optimize_query_performance", "add_query_cache", "calculate_query_complexity",
    # Helpers
    "escape_elasticsearch_query", "validate_field_name", "format_elasticsearch_error",
    "parse_elasticsearch_response", "extract_aggregation_results",
    # Constantes
    "FINANCIAL_FIELDS", "SEARCHABLE_FIELDS", "FILTERABLE_FIELDS", "AGGREGATABLE_FIELDS",
    "MAX_QUERY_SIZE", "DEFAULT_TIMEOUT_MS", "CACHE_TTL_SECONDS",
    
    # === VALIDATORS ===
    # Validators principaux
    "ElasticsearchQueryValidator", "SearchServiceQueryValidator", "FilterValidator",
    # Validation spécialisée
    "validate_user_isolation", "validate_financial_query", "validate_search_parameters",
    "validate_filter_security", "validate_aggregation_request", "validate_query_complexity",
    # Validation champs
    "validate_field_access", "validate_field_types", "validate_field_values",
    "validate_date_ranges", "validate_amount_ranges", "validate_text_search",
    # Validation performance
    "validate_query_limits", "validate_timeout_settings", "validate_cache_settings",
    # Sanitization
    "sanitize_user_input", "sanitize_elasticsearch_query", "sanitize_field_values",
    # Helpers
    "is_valid_user_id", "is_valid_field_name", "is_safe_query_value",
    # Exceptions
    "ValidationError", "SecurityValidationError", "PerformanceValidationError",
    # Constantes
    "MAX_QUERY_LENGTH", "MAX_RESULTS_LIMIT", "ALLOWED_OPERATORS", "FORBIDDEN_FIELDS",
    
    # === CACHE ===
    # Cache principal
    "SearchResultsCache", "QueryCache", "AggregationCache",
    # Cache managers
    "CacheManager", "CacheKey", "CacheStats", "CacheConfig",
    # Cache spécialisé
    "UserQueryCache", "FinancialDataCache", "PerformanceCache",
    # Cache operations
    "cache_search_results", "cache_aggregations", "cache_query_templates",
    "invalidate_user_cache", "invalidate_query_cache", "clear_all_caches",
    # Cache metrics
    "get_cache_hit_rate", "get_cache_memory_usage", "get_cache_statistics",
    # Cache strategies
    "CacheStrategy", "LRUCacheStrategy", "TTLCacheStrategy", "SizeLimitedCacheStrategy",
    # Helpers
    "generate_cache_key", "serialize_cache_value", "deserialize_cache_value",
    # Constantes
    "DEFAULT_CACHE_SIZE", "DEFAULT_TTL_SECONDS", "MAX_CACHE_MEMORY_MB",
    
    # === MÉTRIQUES ===
    # Métriques principales
    "SearchMetrics", "PerformanceMetrics", "ElasticsearchMetrics",
    # Collecteurs
    "MetricsCollector", "PerformanceCollector", "CacheMetricsCollector",
    # Business metrics
    "QueryComplexityMetrics", "UserBehaviorMetrics", "FinancialSearchMetrics",
    # Recording
    "record_search_execution", "record_query_performance", "record_cache_usage",
    "record_elasticsearch_operation", "record_validation_result", "record_error_event",
    # Temps réel
    "get_real_time_metrics", "get_performance_dashboard", "get_health_metrics",
    # Agrégation
    "aggregate_hourly_metrics", "aggregate_daily_metrics", "get_trend_analysis",
    # Alerting
    "check_performance_alerts", "check_error_rate_alerts", "check_resource_alerts",
    # Export
    "export_metrics_prometheus", "export_metrics_json", "export_metrics_csv",
    # Constantes
    "METRIC_NAMES", "ALERT_THRESHOLDS", "DASHBOARD_REFRESH_SECONDS",
    
    # === UTILITAIRES GÉNÉRIQUES ===
    "get_utility_by_name", "list_available_utilities", "validate_utility_imports",
    "ELASTICSEARCH_UTILS", "VALIDATION_UTILS", "CACHE_UTILS", "METRICS_UTILS",
]