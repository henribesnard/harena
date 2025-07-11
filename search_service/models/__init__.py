"""
Modèles de données pour le Search Service - VERSION CORRIGÉE.

Ce module expose tous les modèles Pydantic pour le service de recherche lexicale
avec configuration centralisée et contrats standardisés.

ARCHITECTURE CENTRALISÉE:
- Configuration via config_service uniquement
- Modèles optimisés pour performance Elasticsearch
- Contrats standardisés avec Conversation Service
- Validation stricte des données
- Support des agrégations financières

EXPORTS PRINCIPAUX:
- Contrats d'interface (SearchServiceQuery, SearchServiceResponse)
- Modèles de requête et réponse
- Modèles Elasticsearch natifs
- Filtres et agrégations
- Validation et métadonnées

CORRECTIONS APPLIQUÉES:
- ✅ TYPOS CORRIGÉS dans les imports
- Gestion des conflits de noms
- Validation des dépendances
- Exports conditionnels sécurisés
"""

from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# ==================== IMPORTS SÉCURISÉS ====================

# 1. Contrats d'interface (priorité absolue)
try:
    from .service_contracts import (
        # Contrats principaux
        SearchServiceQuery,
        SearchServiceResponse,
        QueryMetadata,
        ResponseMetadata,
        ExecutionContext,
        AgentContext,
        
        # Filtres et paramètres
        SearchFilter,
        SearchParameters,
        FilterGroup,
        TextSearchFilter,
        
        # Agrégations
        AggregationRequest,
        AggregationResult,
        AggregationBucket,
        AggregationMetrics,
        
        # Résultats et performance
        SearchResult,
        PerformanceMetrics,
        ContextEnrichment,
        
        # Validation et erreurs
        ContractValidationError,
        validate_search_service_query,
        validate_search_service_response,
        
        # Enums et types
        QueryType,
        FilterOperator,
        AggregationType,
        IntentType
    )
    SERVICE_CONTRACTS_AVAILABLE = True
    logger.debug("✅ Service contracts imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import service contracts: {e}")
    SERVICE_CONTRACTS_AVAILABLE = False

# 2. Modèles de requête et réponse (✅ TYPO CORRIGÉ: requests au lieu de reequests)
try:
    from .requests import (
        # Modèles de requête
        LexicalSearchRequest,
        QueryValidationRequest,
        TemplateListRequest,
        HealthCheckRequest,
        MetricsRequest,
        
        # Paramètres et options
        SearchOptions,
        QueryOptions,
        ResultOptions,
        CacheOptions,
        
        # Validation
        RequestValidator,
        validate_search_request
    )
    REQUESTS_AVAILABLE = True
    logger.debug("✅ Request models imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Request models not available: {e}")
    REQUESTS_AVAILABLE = False

try:
    from .responses import (
        # Modèles de réponse
        LexicalSearchResponse,
        QueryValidationResponse,
        TemplateListResponse,
        HealthCheckResponse,
        MetricsResponse,
        
        # Enrichissement et contexte
        ResponseEnrichment,
        ResultEnrichment,
        QualityMetrics,
        
        # Erreurs et statuts
        ErrorResponse,
        SuccessResponse,
        ResponseStatus,
        
        # Validation
        ResponseValidator,
        validate_search_response
    )
    RESPONSES_AVAILABLE = True
    logger.debug("✅ Response models imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Response models not available: {e}")
    RESPONSES_AVAILABLE = False

# 3. Modèles Elasticsearch (✅ TYPO CORRIGÉ: models au lieu de modells)
try:
    from .elasticsearch_queries import (
        # Requêtes Elasticsearch
        ElasticsearchQuery,
        BoolQuery,
        MatchQuery,
        MultiMatchQuery,
        TermQuery,
        TermsQuery,
        RangeQuery,
        ExistsQuery,
        FunctionScoreQuery,
        
        # Filtres Elasticsearch (noms corrigés)
        BaseElasticsearchFilter,
        TermFilter,
        TermsFilter,
        RangeFilter as ElasticsearchRangeFilter,  # Évite conflit avec RangeFilter
        ExistsFilter,
        
        # Agrégations Elasticsearch
        BaseElasticsearchAggregation,
        TermsAggregation,
        DateHistogramAggregation,
        SumAggregation,
        AvgAggregation,
        MaxAggregation,
        MinAggregation,
        StatsAggregation,
        CardinalityAggregation,
        
        # Résultats Elasticsearch
        ElasticsearchResult,
        ElasticsearchHit,
        ElasticsearchAggregationResult,
        ElasticsearchSort,
        ElasticsearchHighlight,
        
        # Builders et helpers
        QueryBuilder,
        AggregationBuilder,
        FilterBuilder,
        
        # Templates et mapping
        QueryTemplate,
        FieldMapping,
        IndexMapping,
        
        # Fonctions utilitaires
        create_financial_search_query,
        create_financial_aggregations,
        optimize_query_for_performance,
        validate_elasticsearch_query,
        validate_query_syntax,
        
        # Enums
        QueryType as ElasticsearchQueryType,
        BoolOperator,
        MultiMatchType,
        SortOrder,
        SortMode,
        AggregationType as ElasticsearchAggregationType
    )
    ELASTICSEARCH_QUERIES_AVAILABLE = True
    logger.debug("✅ Elasticsearch query models imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Elasticsearch query models not available: {e}")
    ELASTICSEARCH_QUERIES_AVAILABLE = False

# 4. Modèles de filtres (✅ TYPO CORRIGÉ: filters au lieu de filters')
try:
    from .filters import (
        # Filtres de base
        BaseFilter,
        FilterType,
        FilterOperator as FilterOp,  # Évite conflit avec FilterOperator
        
        # Filtres spécialisés
        UserFilter,
        CategoryFilter,
        AmountFilter,
        DateFilter,
        MerchantFilter,
        TextFilter,
        
        # Groupes et combinaisons
        FilterGroup as FilterGroupModel,  # Évite conflit avec FilterGroup
        FilterCombination,
        FilterLogic,
        
        # Validation et transformation
        FilterValidator,
        FilterTransformer,
        filter_to_elasticsearch,
        
        # Constantes et enums
        FILTER_OPERATORS,
        VALID_FILTER_FIELDS,
        FINANCIAL_CATEGORIES,
        TRANSACTION_TYPES,
        
        # Factory functions
        create_user_filter,
        create_category_filter,
        create_amount_filter,
        create_date_filter
    )
    FILTERS_AVAILABLE = True
    logger.debug("✅ Filter models imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Filter models not available: {e}")
    FILTERS_AVAILABLE = False

# ==================== EXPORTS CONDITIONNELS ====================

__all__ = []

# Exports contrats (toujours prioritaires)
if SERVICE_CONTRACTS_AVAILABLE:
    __all__.extend([
        # Contrats principaux
        "SearchServiceQuery", "SearchServiceResponse",
        "QueryMetadata", "ResponseMetadata",
        "ExecutionContext", "AgentContext",
        
        # Filtres et paramètres
        "SearchFilter", "SearchParameters", "FilterGroup",
        "TextSearchFilter",
        
        # Agrégations
        "AggregationRequest", "AggregationResult",
        "AggregationBucket", "AggregationMetrics",
        
        # Résultats et performance
        "SearchResult", "PerformanceMetrics", "ContextEnrichment",
        
        # Validation
        "validate_search_service_query", "validate_search_service_response",
        "ContractValidationError",
        
        # Enums et types
        "QueryType", "FilterOperator", "AggregationType", "IntentType"
    ])

# Exports requêtes/réponses
if REQUESTS_AVAILABLE:
    __all__.extend([
        # Requêtes
        "LexicalSearchRequest", "QueryValidationRequest",
        "TemplateListRequest", "HealthCheckRequest", "MetricsRequest",
        
        # Options
        "SearchOptions", "QueryOptions", "ResultOptions", "CacheOptions",
        
        # Validation
        "RequestValidator", "validate_search_request"
    ])

if RESPONSES_AVAILABLE:
    __all__.extend([
        # Réponses
        "LexicalSearchResponse", "QueryValidationResponse",
        "TemplateListResponse", "HealthCheckResponse", "MetricsResponse",
        
        # Enrichissement
        "ResponseEnrichment", "ResultEnrichment", "QualityMetrics",
        
        # Erreurs
        "ErrorResponse", "SuccessResponse", "ResponseStatus",
        
        # Validation
        "ResponseValidator", "validate_search_response"
    ])

# Exports Elasticsearch
if ELASTICSEARCH_QUERIES_AVAILABLE:
    __all__.extend([
        # Requêtes principales
        "ElasticsearchQuery", "BoolQuery", "MatchQuery", "MultiMatchQuery",
        "TermQuery", "TermsQuery", "RangeQuery", "ExistsQuery",
        
        # Filtres Elasticsearch
        "BaseElasticsearchFilter", "TermFilter", "TermsFilter",
        "ElasticsearchRangeFilter", "ExistsFilter",
        
        # Agrégations
        "BaseElasticsearchAggregation", "TermsAggregation",
        "DateHistogramAggregation", "SumAggregation", "AvgAggregation",
        "MaxAggregation", "MinAggregation", "StatsAggregation",
        
        # Résultats
        "ElasticsearchResult", "ElasticsearchHit", "ElasticsearchAggregationResult",
        "ElasticsearchSort", "ElasticsearchHighlight",
        
        # Builders
        "QueryBuilder", "AggregationBuilder", "FilterBuilder",
        
        # Templates
        "QueryTemplate", "FieldMapping", "IndexMapping",
        
        # Utilitaires
        "create_financial_search_query", "create_financial_aggregations",
        "optimize_query_for_performance", "validate_elasticsearch_query",
        
        # Enums
        "ElasticsearchQueryType", "BoolOperator", "MultiMatchType",
        "SortOrder", "SortMode", "ElasticsearchAggregationType"
    ])

# Exports filtres
if FILTERS_AVAILABLE:
    __all__.extend([
        # Filtres de base
        "BaseFilter", "FilterType", "FilterOp",
        
        # Filtres spécialisés
        "UserFilter", "CategoryFilter", "AmountFilter",
        "DateFilter", "MerchantFilter", "TextFilter",
        
        # Groupes
        "FilterGroupModel", "FilterCombination", "FilterLogic",
        
        # Validation
        "FilterValidator", "FilterTransformer", "filter_to_elasticsearch",
        
        # Constantes
        "FILTER_OPERATORS", "VALID_FILTER_FIELDS",
        "FINANCIAL_CATEGORIES", "TRANSACTION_TYPES",
        
        # Factory functions
        "create_user_filter", "create_category_filter",
        "create_amount_filter", "create_date_filter"
    ])

# ==================== STATUS ET DIAGNOSTICS ====================

def get_models_status() -> Dict[str, Any]:
    """Retourne le statut de disponibilité de tous les modèles."""
    return {
        "service_contracts": SERVICE_CONTRACTS_AVAILABLE,
        "requests": REQUESTS_AVAILABLE,
        "responses": RESPONSES_AVAILABLE,
        "elasticsearch_queries": ELASTICSEARCH_QUERIES_AVAILABLE,
        "filters": FILTERS_AVAILABLE,
        "total_available": sum([
            SERVICE_CONTRACTS_AVAILABLE,
            REQUESTS_AVAILABLE,
            RESPONSES_AVAILABLE,
            ELASTICSEARCH_QUERIES_AVAILABLE,
            FILTERS_AVAILABLE
        ]),
        "total_modules": 5
    }

def validate_models_setup() -> Dict[str, Any]:
    """Valide la configuration des modèles."""
    status = get_models_status()
    
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "critical_missing": []
    }
    
    # Vérifications critiques
    if not SERVICE_CONTRACTS_AVAILABLE:
        validation["critical_missing"].append("service_contracts")
        validation["valid"] = False
        validation["errors"].append("Service contracts are mandatory for API contracts")
    
    # Vérifications importantes
    if not REQUESTS_AVAILABLE:
        validation["warnings"].append("Request models not available - API input validation limited")
    
    if not RESPONSES_AVAILABLE:
        validation["warnings"].append("Response models not available - API output validation limited")
    
    if not ELASTICSEARCH_QUERIES_AVAILABLE:
        validation["warnings"].append("Elasticsearch models not available - query building limited")
    
    if not FILTERS_AVAILABLE:
        validation["warnings"].append("Filter models not available - filtering capabilities limited")
    
    validation["status"] = status
    return validation

# ==================== FACTORY FUNCTIONS ====================

def create_search_service_query(**kwargs) -> Optional['SearchServiceQuery']:
    """Factory pour créer une SearchServiceQuery si disponible."""
    if not SERVICE_CONTRACTS_AVAILABLE:
        logger.error("Cannot create SearchServiceQuery - service contracts not available")
        return None
    
    try:
        return SearchServiceQuery(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create SearchServiceQuery: {e}")
        return None

def create_lexical_search_request(**kwargs) -> Optional['LexicalSearchRequest']:
    """Factory pour créer une LexicalSearchRequest si disponible."""
    if not REQUESTS_AVAILABLE:
        logger.error("Cannot create LexicalSearchRequest - request models not available")
        return None
    
    try:
        return LexicalSearchRequest(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create LexicalSearchRequest: {e}")
        return None

def create_elasticsearch_query(**kwargs) -> Optional['ElasticsearchQuery']:
    """Factory pour créer une ElasticsearchQuery si disponible."""
    if not ELASTICSEARCH_QUERIES_AVAILABLE:
        logger.error("Cannot create ElasticsearchQuery - elasticsearch models not available")
        return None
    
    try:
        return ElasticsearchQuery(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create ElasticsearchQuery: {e}")
        return None

def create_financial_query_builder() -> Optional['QueryBuilder']:
    """Factory pour créer un QueryBuilder financier si disponible."""
    if not ELASTICSEARCH_QUERIES_AVAILABLE:
        logger.error("Cannot create QueryBuilder - elasticsearch models not available")
        return None
    
    try:
        return QueryBuilder()
    except Exception as e:
        logger.error(f"Failed to create QueryBuilder: {e}")
        return None

# ==================== LOGGING ET DEBUG ====================

def log_models_status():
    """Log le statut de tous les modèles pour debugging."""
    status = get_models_status()
    logger.info(f"Models status: {status['total_available']}/{status['total_modules']} modules available")
    
    if SERVICE_CONTRACTS_AVAILABLE:
        logger.debug("✅ Service contracts available")
    else:
        logger.error("❌ Service contracts NOT available")
    
    if REQUESTS_AVAILABLE:
        logger.debug("✅ Request models available")
    else:
        logger.warning("⚠️ Request models NOT available")
    
    if RESPONSES_AVAILABLE:
        logger.debug("✅ Response models available")
    else:
        logger.warning("⚠️ Response models NOT available")
    
    if ELASTICSEARCH_QUERIES_AVAILABLE:
        logger.debug("✅ Elasticsearch query models available")
    else:
        logger.warning("⚠️ Elasticsearch query models NOT available")
    
    if FILTERS_AVAILABLE:
        logger.debug("✅ Filter models available")
    else:
        logger.warning("⚠️ Filter models NOT available")

def get_available_exports() -> List[str]:
    """Retourne la liste des exports disponibles."""
    return __all__

def debug_imports():
    """Debug les imports pour identifier les problèmes."""
    debug_info = {
        "service_contracts": SERVICE_CONTRACTS_AVAILABLE,
        "requests": REQUESTS_AVAILABLE,
        "responses": RESPONSES_AVAILABLE,
        "elasticsearch_queries": ELASTICSEARCH_QUERIES_AVAILABLE,
        "filters": FILTERS_AVAILABLE,
        "total_exports": len(__all__)
    }
    
    logger.info(f"Debug imports: {debug_info}")
    return debug_info

# Log automatique au démarrage
if __name__ != "__main__":
    log_models_status()