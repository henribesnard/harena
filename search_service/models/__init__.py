"""
Modèles de données pour le Search Service - VERSION REFACTORISÉE.

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
        RangeFilter,
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
        validate_search_service_response
    )
    SERVICE_CONTRACTS_AVAILABLE = True
    logger.debug("✅ Service contracts imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import service contracts: {e}")
    SERVICE_CONTRACTS_AVAILABLE = False

# 2. Modèles de requête et réponse
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

# 3. Modèles Elasticsearch
try:
    from .elasticsearch_queries import (
        # Requêtes Elasticsearch
        ElasticsearchQuery,
        BoolQuery,
        MatchQuery,
        TermQuery,
        RangeQuery,
        MultiMatchQuery,
        
        # Filtres Elasticsearch
        ElasticsearchFilter,
        TermFilter,
        RangeFilter as ESRangeFilter,
        ExistsFilter,
        
        # Agrégations Elasticsearch
        ElasticsearchAggregation,
        TermsAggregation,
        DateHistogramAggregation,
        SumAggregation,
        AvgAggregation,
        MaxAggregation,
        MinAggregation,
        
        # Résultats Elasticsearch
        ElasticsearchResult,
        ElasticsearchHit,
        ElasticsearchAggregationResult,
        
        # Helpers et builders
        QueryBuilder,
        AggregationBuilder,
        FilterBuilder,
        
        # Templates et mapping
        QueryTemplate,
        FieldMapping,
        IndexMapping
    )
    ELASTICSEARCH_QUERIES_AVAILABLE = True
    logger.debug("✅ Elasticsearch query models imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Elasticsearch query models not available: {e}")
    ELASTICSEARCH_QUERIES_AVAILABLE = False

# 4. Modèles de filtres
try:
    from .filters import (
        # Filtres de base
        BaseFilter,
        FilterType,
        FilterOperator,
        
        # Filtres spécialisés
        UserFilter,
        CategoryFilter,
        AmountFilter,
        DateFilter,
        MerchantFilter,
        TextFilter,
        
        # Groupes et combinaisons
        FilterGroup as FiltersGroup,
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
        TRANSACTION_TYPES
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
        "SearchServiceQuery", "SearchServiceResponse",
        "QueryMetadata", "ResponseMetadata",
        "SearchFilter", "AggregationRequest", "AggregationResult",
        "SearchResult", "PerformanceMetrics",
        "validate_search_service_query", "validate_search_service_response"
    ])

# Exports requêtes/réponses
if REQUESTS_AVAILABLE:
    __all__.extend([
        "LexicalSearchRequest", "SearchOptions", "RequestValidator"
    ])

if RESPONSES_AVAILABLE:
    __all__.extend([
        "LexicalSearchResponse", "ResponseEnrichment", "ErrorResponse"
    ])

# Exports Elasticsearch
if ELASTICSEARCH_QUERIES_AVAILABLE:
    __all__.extend([
        "ElasticsearchQuery", "BoolQuery", "MatchQuery",
        "ElasticsearchAggregation", "QueryBuilder"
    ])

# Exports filtres
if FILTERS_AVAILABLE:
    __all__.extend([
        "BaseFilter", "UserFilter", "CategoryFilter", "FilterValidator"
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

# Log automatique au démarrage
if __name__ != "__main__":
    log_models_status()