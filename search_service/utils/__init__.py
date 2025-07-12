"""
Module utils du Search Service - Utilitaires et services transversaux
====================================================================

Ce module regroupe tous les utilitaires essentiels du Search Service :
- Métriques et monitoring spécialisés
- Cache LRU haute performance  
- Validateurs de requêtes
- Helpers Elasticsearch
- Utilitaires de performance

Architecture :
    Core Components → Utils → External Services
    
Utilisé par :
    - Tous les modules core pour les métriques
    - API routes pour la validation
    - Clients pour les helpers Elasticsearch
    - Performance optimizer pour le cache
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple

# === IMPORTS DES MODULES UTILS ===

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
    PerformanceProfiler,
    
    # Types et enums
    MetricType,
    MetricCategory,
    AlertLevel,
    MetricDefinition,
    MetricValue,
    MetricSample,
    MetricAlert,
    
    # Instances globales
    metrics_collector,
    alert_manager,
    query_metrics,
    result_metrics,
    search_metrics,
    elasticsearch_metrics,
    lexical_search_metrics,
    business_metrics,
    performance_profiler,
    metrics_dashboard,
    
    # Fonctions utilitaires
    get_system_metrics,
    get_performance_summary,
    export_metrics_to_file,
    cleanup_old_metrics,
    reset_all_counters,
    get_top_slow_operations,
    get_error_metrics_summary,
    initialize_metrics_system,
    shutdown_metrics_system,
    
    # Callbacks
    log_alert_callback,
    system_alert_callback
)

from .cache import (
    # Classes de cache
    LRUCache,
    CacheEntry,
    CacheStats,
    CacheStrategy,
    
    # Gestionnaire de cache
    CacheManager,
    
    # Instances globales  
    global_cache_manager,
    
    # Fonctions utilitaires
    create_cache_key,
    serialize_cache_value,
    deserialize_cache_value,
    cleanup_expired_entries,
    get_cache_statistics
)

from .validators import (
    # Classes de validation
    QueryValidator,
    RequestValidator,
    ResponseValidator,
    FieldValidator,
    
    # Types de validation
    ValidationRule,
    ValidationResult,
    ValidationError,
    
    # Fonctions de validation
    validate_search_request,
    validate_elasticsearch_query,
    validate_user_permissions,
    validate_date_ranges,
    validate_aggregation_request,
    sanitize_search_input,
    
    # Validateurs spécialisés
    FinancialDataValidator,
    SecurityValidator
)

from .elasticsearch_helpers import (
    # Classes d'aide Elasticsearch
    ESQueryBuilder,
    ESResponseParser,
    ESIndexManager,
    ESConnectionHelper,
    
    # Types Elasticsearch
    ESQueryType,
    ESAggregationType,
    ESFilterType,
    
    # Fonctions helper
    build_bool_query,
    build_multi_match_query,
    build_range_filter,
    build_term_filter,
    parse_es_response,
    extract_highlights,
    calculate_relevance_score,
    optimize_es_query,
    
    # Utilitaires de mapping
    get_field_mapping,
    validate_field_exists,
    get_index_settings,
    
    # Gestionnaire de templates
    QueryTemplateManager,
    template_manager
)


# === CONFIGURATION DU LOGGING ===

logger = logging.getLogger(__name__)

# Loggers spécialisés pour chaque utilitaire
metrics_logger = logging.getLogger(f"{__name__}.metrics")
cache_logger = logging.getLogger(f"{__name__}.cache")
validators_logger = logging.getLogger(f"{__name__}.validators")
elasticsearch_logger = logging.getLogger(f"{__name__}.elasticsearch_helpers")


# === GESTIONNAIRE UTILS GLOBAL ===

class UtilsManager:
    """Gestionnaire centralisé des utilitaires du Search Service"""
    
    def __init__(self):
        self._initialized = False
        self._metrics_system = None
        self._cache_manager = None
        self._validator_registry = None
        self._elasticsearch_helpers = None
        
        logger.info("UtilsManager créé")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialise tous les utilitaires du Search Service
        
        Args:
            config: Configuration optionnelle pour les utilitaires
            
        Returns:
            Dict contenant le statut d'initialisation de chaque utilitaire
        """
        if self._initialized:
            logger.warning("UtilsManager déjà initialisé")
            return await self.get_initialization_status()
        
        initialization_results = {}
        
        try:
            logger.info("Initialisation des utilitaires du Search Service...")
            
            # 1. Initialiser le système de métriques
            try:
                initialize_metrics_system()
                self._metrics_system = metrics_collector
                initialization_results["metrics"] = "initialized"
                logger.info("✅ Système de métriques initialisé")
            except Exception as e:
                logger.error(f"❌ Erreur initialisation métriques: {e}")
                initialization_results["metrics"] = f"failed: {e}"
            
            # 2. Initialiser le gestionnaire de cache
            try:
                cache_config = config.get("cache", {}) if config else {}
                await global_cache_manager.initialize(**cache_config)
                self._cache_manager = global_cache_manager
                initialization_results["cache"] = "initialized"
                logger.info("✅ Gestionnaire de cache initialisé")
            except Exception as e:
                logger.error(f"❌ Erreur initialisation cache: {e}")
                initialization_results["cache"] = f"failed: {e}"
            
            # 3. Initialiser les validateurs
            try:
                validator_config = config.get("validators", {}) if config else {}
                # Les validateurs sont stateless, pas d'initialisation spéciale nécessaire
                initialization_results["validators"] = "initialized"
                logger.info("✅ Validateurs initialisés")
            except Exception as e:
                logger.error(f"❌ Erreur initialisation validateurs: {e}")
                initialization_results["validators"] = f"failed: {e}"
            
            # 4. Initialiser les helpers Elasticsearch
            try:
                es_config = config.get("elasticsearch_helpers", {}) if config else {}
                await template_manager.initialize(**es_config)
                self._elasticsearch_helpers = template_manager
                initialization_results["elasticsearch_helpers"] = "initialized"
                logger.info("✅ Helpers Elasticsearch initialisés")
            except Exception as e:
                logger.error(f"❌ Erreur initialisation ES helpers: {e}")
                initialization_results["elasticsearch_helpers"] = f"failed: {e}"
            
            # 5. Marquer comme initialisé
            self._initialized = True
            
            logger.info("🚀 Tous les utilitaires sont initialisés avec succès")
            
            return {
                "status": "success",
                "components": initialization_results
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des utilitaires: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "components": initialization_results
            }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Arrêt propre de tous les utilitaires"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        shutdown_results = {}
        
        try:
            logger.info("Arrêt des utilitaires...")
            
            # Arrêter dans l'ordre inverse de l'initialisation
            if self._elasticsearch_helpers:
                await self._elasticsearch_helpers.shutdown()
                shutdown_results["elasticsearch_helpers"] = "shutdown"
            
            if self._cache_manager:
                await self._cache_manager.shutdown()
                shutdown_results["cache"] = "shutdown"
            
            if self._metrics_system:
                shutdown_metrics_system()
                shutdown_results["metrics"] = "shutdown"
            
            # Les validateurs n'ont pas besoin d'arrêt explicite
            shutdown_results["validators"] = "shutdown"
            
            self._initialized = False
            logger.info("✅ Tous les utilitaires arrêtés")
            
            return {
                "status": "success",
                "components": shutdown_results
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'arrêt des utilitaires: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "components": shutdown_results
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Vérification de santé des utilitaires"""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "error": "Utils not initialized"
            }
        
        try:
            health_status = {
                "system_status": "healthy",
                "components": {}
            }
            
            # Vérifier le système de métriques
            if self._metrics_system:
                metrics_health = get_system_metrics()
                health_status["components"]["metrics"] = {
                    "status": "healthy",
                    "details": metrics_health
                }
            
            # Vérifier le cache
            if self._cache_manager:
                cache_health = await self._cache_manager.get_health_status()
                health_status["components"]["cache"] = cache_health
            
            # Vérifier les validateurs (toujours healthy s'ils sont chargés)
            health_status["components"]["validators"] = {"status": "healthy"}
            
            # Vérifier les helpers Elasticsearch
            if self._elasticsearch_helpers:
                es_health = await self._elasticsearch_helpers.get_health_status()
                health_status["components"]["elasticsearch_helpers"] = es_health
            
            # Déterminer le statut global
            component_statuses = [
                comp_health.get("status", "unknown") 
                for comp_health in health_status["components"].values()
            ]
            
            if all(status == "healthy" for status in component_statuses):
                health_status["system_status"] = "healthy"
            elif any(status == "error" for status in component_statuses):
                health_status["system_status"] = "degraded"
            else:
                health_status["system_status"] = "partial"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de santé: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance des utilitaires"""
        if not self._initialized:
            return {"error": "Utils not initialized"}
        
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "components": {}
            }
            
            # Rapport métriques
            if self._metrics_system:
                report["components"]["metrics"] = get_performance_summary()
            
            # Rapport cache
            if self._cache_manager:
                report["components"]["cache"] = await self._cache_manager.get_performance_report()
            
            # Rapport validateurs
            report["components"]["validators"] = {
                "validation_rules_loaded": len(ValidationRule.__subclasses__()) if 'ValidationRule' in globals() else 0,
                "status": "active"
            }
            
            # Rapport helpers Elasticsearch
            if self._elasticsearch_helpers:
                report["components"]["elasticsearch_helpers"] = await self._elasticsearch_helpers.get_performance_report()
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")
            return {"error": str(e)}
    
    async def get_initialization_status(self) -> Dict[str, Any]:
        """Retourne le statut d'initialisation détaillé"""
        return {
            "initialized": self._initialized,
            "components": {
                "metrics": self._metrics_system is not None,
                "cache": self._cache_manager is not None,
                "validators": True,  # Toujours disponibles
                "elasticsearch_helpers": self._elasticsearch_helpers is not None
            }
        }
    
    @property
    def metrics_system(self):
        """Accès au système de métriques"""
        return self._metrics_system if self._initialized else None
    
    @property
    def cache_manager(self):
        """Accès au gestionnaire de cache"""
        return self._cache_manager if self._initialized else None
    
    @property
    def elasticsearch_helpers(self):
        """Accès aux helpers Elasticsearch"""
        return self._elasticsearch_helpers if self._initialized else None


# === INSTANCE GLOBALE ===

utils_manager = UtilsManager()


# === FONCTIONS D'INTERFACE PUBLIQUE ===

async def initialize_utils(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialise tous les utilitaires du Search Service
    
    Point d'entrée principal pour l'initialisation des utils
    """
    return await utils_manager.initialize(config)


async def shutdown_utils() -> Dict[str, Any]:
    """Arrêt propre de tous les utilitaires"""
    return await utils_manager.shutdown()


async def get_utils_health() -> Dict[str, Any]:
    """Vérification de santé globale des utilitaires"""
    return await utils_manager.get_health_status()


async def get_utils_performance() -> Dict[str, Any]:
    """Rapport de performance global des utilitaires"""
    return await utils_manager.get_performance_report()


def get_metrics_system():
    """Accès sécurisé au système de métriques"""
    return utils_manager.metrics_system


def get_cache_manager():
    """Accès sécurisé au gestionnaire de cache"""
    return utils_manager.cache_manager


def get_elasticsearch_helpers():
    """Accès sécurisé aux helpers Elasticsearch"""
    return utils_manager.elasticsearch_helpers


# === FONCTIONS UTILITAIRES INTÉGRÉES ===

def record_operation_metrics(operation_name: str, duration_ms: float, 
                           success: bool, **kwargs):
    """Enregistre des métriques pour une opération générique"""
    
    if not utils_manager._initialized:
        logger.warning("Système de métriques non initialisé")
        return
    
    tags = {
        "operation": operation_name,
        "success": str(success),
        **{k: str(v) for k, v in kwargs.items()}
    }
    
    metrics_collector.record(f"{operation_name}_duration_ms", duration_ms, tags)
    
    if success:
        metrics_collector.increment(f"{operation_name}_success_count", tags=tags)
    else:
        metrics_collector.increment(f"{operation_name}_error_count", tags=tags)


def validate_and_cache_query(query_data: Dict[str, Any], 
                            cache_key: str,
                            ttl_seconds: int = 300) -> Tuple[bool, Optional[str], Any]:
    """
    Valide une requête et vérifie le cache
    
    Returns:
        Tuple[is_valid, error_message, cached_result]
    """
    
    try:
        # Validation
        validation_result = validate_search_request(query_data)
        if not validation_result.is_valid:
            return False, validation_result.error_message, None
        
        # Vérification cache
        if utils_manager.cache_manager:
            cached_result = utils_manager.cache_manager.get(cache_key)
            if cached_result is not None:
                return True, None, cached_result
        
        return True, None, None
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation/cache: {e}")
        return False, str(e), None


def cache_search_result(cache_key: str, result: Any, ttl_seconds: int = 300):
    """Met en cache un résultat de recherche"""
    
    if utils_manager.cache_manager:
        try:
            utils_manager.cache_manager.set(cache_key, result, ttl_seconds)
        except Exception as e:
            logger.error(f"Erreur lors de la mise en cache: {e}")


def get_utils_status() -> Dict[str, str]:
    """Statut simple des utilitaires pour les health checks"""
    if not utils_manager._initialized:
        return {"status": "not_ready", "reason": "utils_not_initialized"}
    
    try:
        components_ready = all([
            utils_manager._metrics_system is not None,
            utils_manager._cache_manager is not None,
            utils_manager._elasticsearch_helpers is not None
        ])
        
        if components_ready:
            return {"status": "ready"}
        else:
            return {"status": "partial", "reason": "some_components_missing"}
            
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === GESTIONNAIRE PRINCIPAL ===
    "UtilsManager",
    "utils_manager",
    
    # === FONCTIONS D'INITIALISATION ===
    "initialize_utils",
    "shutdown_utils",
    "get_utils_health",
    "get_utils_performance",
    
    # === ACCÈS AUX COMPOSANTS ===
    "get_metrics_system",
    "get_cache_manager",
    "get_elasticsearch_helpers",
    
    # === FONCTIONS INTÉGRÉES ===
    "record_operation_metrics",
    "validate_and_cache_query",
    "cache_search_result",
    "get_utils_status",
    
    # === MÉTRIQUES ===
    # Classes principales
    "MetricsCollector",
    "AlertManager",
    "MetricsDashboard",
    
    # Métriques spécialisées (pour compatibility avec les imports existants)
    "QueryMetrics",
    "ResultMetrics", 
    "SearchMetrics",
    "ElasticsearchMetrics",
    "LexicalSearchMetrics",
    "BusinessMetrics",
    "PerformanceProfiler",
    
    # Types métriques
    "MetricType",
    "MetricCategory",
    "AlertLevel",
    "MetricDefinition",
    "MetricValue",
    "MetricSample",
    "MetricAlert",
    
    # Instances globales métriques
    "metrics_collector",
    "alert_manager",
    "query_metrics",
    "result_metrics",
    "search_metrics",
    "elasticsearch_metrics",
    "lexical_search_metrics",
    "business_metrics",
    "performance_profiler",
    "metrics_dashboard",
    
    # Fonctions métriques
    "get_system_metrics",
    "get_performance_summary",
    "export_metrics_to_file",
    "cleanup_old_metrics",
    "reset_all_counters",
    "get_top_slow_operations",
    "get_error_metrics_summary",
    "initialize_metrics_system",
    "shutdown_metrics_system",
    "log_alert_callback",
    "system_alert_callback",
    
    # === CACHE ===
    # Classes de cache
    "LRUCache",
    "CacheEntry",
    "CacheStats", 
    "CacheStrategy",
    "CacheManager",
    
    # Instance globale cache
    "global_cache_manager",
    
    # Fonctions cache
    "create_cache_key",
    "serialize_cache_value",
    "deserialize_cache_value",
    "cleanup_expired_entries",
    "get_cache_statistics",
    
    # === VALIDATEURS ===
    # Classes de validation
    "QueryValidator",
    "RequestValidator", 
    "ResponseValidator",
    "FieldValidator",
    "FinancialDataValidator",
    "SecurityValidator",
    
    # Types validation
    "ValidationRule",
    "ValidationResult",
    "ValidationError",
    
    # Fonctions validation
    "validate_search_request",
    "validate_elasticsearch_query",
    "validate_user_permissions",
    "validate_date_ranges",
    "validate_aggregation_request",
    "sanitize_search_input",
    
    # === HELPERS ELASTICSEARCH ===
    # Classes helper
    "ESQueryBuilder",
    "ESResponseParser",
    "ESIndexManager",
    "ESConnectionHelper",
    "QueryTemplateManager",
    
    # Types ES
    "ESQueryType",
    "ESAggregationType", 
    "ESFilterType",
    
    # Instance globale
    "template_manager",
    
    # Fonctions helper
    "build_bool_query",
    "build_multi_match_query",
    "build_range_filter",
    "build_term_filter",
    "parse_es_response",
    "extract_highlights",
    "calculate_relevance_score",
    "optimize_es_query",
    "get_field_mapping",
    "validate_field_exists",
    "get_index_settings"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team" 
__description__ = "Utilitaires et services transversaux du Search Service"

# Imports nécessaires pour les types
from datetime import datetime

# Logging de l'import du module
logger.info(f"Module utils initialisé - version {__version__}")