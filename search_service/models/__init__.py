"""
📋 Modèles Search Service - Exports Centralisés
===============================================

Point d'entrée centralisé pour tous les modèles du Search Service.
Organisation hiérarchique selon l'architecture hybride.

Organisation:
- Contrats interface (priorité #1)
- Modèles requêtes/réponses API
- Filtres spécialisés financiers
- Requêtes Elasticsearch optimisées
"""

# =============================================================================
# 🤝 CONTRATS INTERFACE (PRIORITÉ #1)
# =============================================================================
from typing import Dict, List
from .service_contracts import (
    # Énumérations
    QueryType, IntentType, FilterOperator, AggregationType, TextSearchOperator,
    # Modèles filtres et recherche
    SearchFilter, TextSearchQuery, AggregationRequest,
    # Contrat requête
    QueryMetadata, SearchParameters, FilterGroup, AggregationGroup, QueryOptions, SearchServiceQuery,
    # Contrat réponse
    ResponseMetadata, TransactionResult, AggregationBucket, AggregationResult, 
    PerformanceMetrics, ContextEnrichment, SearchServiceResponse,
    # Utilitaires
    ContractValidator,
)

# =============================================================================
# 📥 MODÈLES REQUÊTES API
# =============================================================================

from .requests import (
    # Requêtes recherche
    SimpleLexicalSearchRequest, CategorySearchRequest, MerchantSearchRequest, AmountRangeSearchRequest,
    # Requêtes analyse
    CategoryAnalysisRequest, MerchantAnalysisRequest, TemporalAnalysisRequest, SpendingPatternRequest,
    # Requêtes utilitaires
    ValidateQueryRequest, HealthCheckRequest, MetricsRequest,
    # Requêtes contrats
    ContractSearchRequest,
    # Requêtes batch
    BatchSearchRequest, BulkValidationRequest,
    # Requêtes spécialisées
    RecurringTransactionRequest, SuspiciousActivityRequest, BudgetAnalysisRequest,
    # Factory
    RequestFactory,
)

# =============================================================================
# 📤 MODÈLES RÉPONSES API
# =============================================================================

from .responses import (
    # Énumérations
    ResponseStatus, QueryComplexity,
    # Réponses de base
    BaseResponse, ErrorResponse,
    # Modèles support
    SearchResultSummary, SearchPerformanceMetrics, ValidationResult, HealthStatus, MetricValue,
    # Réponses recherche
    SimpleLexicalSearchResponse, CategorySearchResponse, MerchantSearchResponse,
    # Réponses analyse
    CategoryStats, MerchantStats, TemporalDataPoint, 
    CategoryAnalysisResponse, MerchantAnalysisResponse, TemporalAnalysisResponse,
    # Réponses utilitaires
    QueryValidationResponse, HealthCheckResponse, MetricsResponse,
    # Réponses contrats
    ContractSearchResponse,
    # Réponses batch
    BatchSearchResult, BatchSearchResponse, BulkValidationResult, BulkValidationResponse,
    # Réponses spécialisées
    RecurringTransaction, RecurringTransactionResponse,
    SuspiciousActivity, SuspiciousActivityResponse,
    BudgetCategory, BudgetAnalysisResponse,
    # Réponses analytics
    SpendingPattern, SpendingPatternResponse, TrendAnalysis, TrendAnalysisResponse,
    # Factory
    ResponseFactory,
)

# =============================================================================
# 🔧 MODÈLES FILTRES SPÉCIALISÉS
# =============================================================================

from .filters import (
    # Énumérations
    AmountFilterType, DateFilterType, TransactionType, OperationType, CurrencyCode,
    # Filtres de base
    UserIsolationFilter, AmountFilter, DateFilter, CategoryFilter, MerchantFilter,
    # Filtres composés
    TransactionTypeFilter, CompositeFilter,
    # Filtres spécialisés
    RecurringTransactionFilter, SuspiciousActivityFilter, BudgetFilter,
    # Builder et Factory
    FilterBuilder, FilterFactory,
)

# =============================================================================
# 🔍 MODÈLES ELASTICSEARCH
# =============================================================================

from .elasticsearch_queries import (
    # Énumérations
    ESQueryType, ESAggType, ESScoreMode, ESBoostMode,
    # Composants requête
    ESQueryClause, ESBoolQuery, ESAggregation, ElasticsearchQuery,
    # Factories
    FinancialQueryFactory, FinancialAggregationFactory,
    # Templates
    QueryTemplate, FinancialQueryTemplates,
    # Builder
    ElasticsearchQueryBuilder,
)

# =============================================================================
# 📋 EXPORTS GROUPÉS PAR FONCTIONNALITÉ
# =============================================================================

# Contrats principaux pour communication entre services
CONTRACTS = [
    "SearchServiceQuery", "SearchServiceResponse", "ContractValidator"
]

# Modèles requêtes API endpoints
REQUEST_MODELS = [
    "SimpleLexicalSearchRequest", "CategorySearchRequest", "MerchantSearchRequest",
    "CategoryAnalysisRequest", "TemporalAnalysisRequest", "ContractSearchRequest",
    "RequestFactory"
]

# Modèles réponses API endpoints
RESPONSE_MODELS = [
    "BaseResponse", "ErrorResponse", "SimpleLexicalSearchResponse",
    "CategoryAnalysisResponse", "ContractSearchResponse", "ResponseFactory"
]

# Filtres financiers spécialisés
FILTER_MODELS = [
    "CompositeFilter", "AmountFilter", "DateFilter", "CategoryFilter",
    "FilterBuilder", "FilterFactory"
]

# Requêtes Elasticsearch
ELASTICSEARCH_MODELS = [
    "ElasticsearchQuery", "ESBoolQuery", "ESQueryClause", "ESAggregation",
    "ElasticsearchQueryBuilder", "FinancialQueryFactory"
]

# =============================================================================
# 🔧 UTILITAIRES MODÈLES
# =============================================================================

def get_model_by_name(model_name: str):
    """Récupérer modèle par nom."""
    return globals().get(model_name)

def list_available_models() -> Dict[str, List[str]]:
    """Lister tous les modèles disponibles par catégorie."""
    return {
        "contracts": CONTRACTS,
        "requests": REQUEST_MODELS,
        "responses": RESPONSE_MODELS,
        "filters": FILTER_MODELS,
        "elasticsearch": ELASTICSEARCH_MODELS,
    }

def validate_model_imports():
    """Valider que tous les imports fonctionnent."""
    try:
        # Test imports contrats
        assert SearchServiceQuery is not None
        assert SearchServiceResponse is not None
        
        # Test imports requêtes
        assert SimpleLexicalSearchRequest is not None
        assert CategorySearchRequest is not None
        
        # Test imports réponses
        assert BaseResponse is not None
        assert SimpleLexicalSearchResponse is not None
        
        # Test imports filtres
        assert CompositeFilter is not None
        assert FilterBuilder is not None
        
        # Test imports Elasticsearch
        assert ElasticsearchQuery is not None
        assert ElasticsearchQueryBuilder is not None
        
        return True
    except (ImportError, AssertionError) as e:
        return False, str(e)

# =============================================================================
# 📋 EXPORTS FINAUX
# =============================================================================

__all__ = [
    # === CONTRATS INTERFACE ===
    # Énumérations contrats
    "QueryType", "IntentType", "FilterOperator", "AggregationType", "TextSearchOperator",
    # Modèles contrats
    "SearchFilter", "TextSearchQuery", "AggregationRequest",
    "QueryMetadata", "SearchParameters", "FilterGroup", "AggregationGroup", "QueryOptions",
    "SearchServiceQuery", "SearchServiceResponse",
    "ResponseMetadata", "TransactionResult", "AggregationBucket", "AggregationResult",
    "PerformanceMetrics", "ContextEnrichment",
    "ContractValidator",
    
    # === MODÈLES REQUÊTES ===
    # Requêtes recherche
    "SimpleLexicalSearchRequest", "CategorySearchRequest", "MerchantSearchRequest", "AmountRangeSearchRequest",
    # Requêtes analyse
    "CategoryAnalysisRequest", "MerchantAnalysisRequest", "TemporalAnalysisRequest", "SpendingPatternRequest",
    # Requêtes utilitaires
    "ValidateQueryRequest", "HealthCheckRequest", "MetricsRequest",
    # Requêtes contrats
    "ContractSearchRequest",
    # Requêtes batch
    "BatchSearchRequest", "BulkValidationRequest",
    # Requêtes spécialisées
    "RecurringTransactionRequest", "SuspiciousActivityRequest", "BudgetAnalysisRequest",
    "RequestFactory",
    
    # === MODÈLES RÉPONSES ===
    # Énumérations réponses
    "ResponseStatus", "QueryComplexity",
    # Réponses base
    "BaseResponse", "ErrorResponse",
    # Modèles support
    "SearchResultSummary", "SearchPerformanceMetrics", "ValidationResult", "HealthStatus", "MetricValue",
    # Réponses recherche
    "SimpleLexicalSearchResponse", "CategorySearchResponse", "MerchantSearchResponse",
    # Réponses analyse
    "CategoryStats", "MerchantStats", "TemporalDataPoint",
    "CategoryAnalysisResponse", "MerchantAnalysisResponse", "TemporalAnalysisResponse",
    # Réponses utilitaires
    "QueryValidationResponse", "HealthCheckResponse", "MetricsResponse",
    # Réponses contrats
    "ContractSearchResponse",
    # Réponses batch
    "BatchSearchResult", "BatchSearchResponse", "BulkValidationResult", "BulkValidationResponse",
    # Réponses spécialisées
    "RecurringTransaction", "RecurringTransactionResponse",
    "SuspiciousActivity", "SuspiciousActivityResponse",
    "BudgetCategory", "BudgetAnalysisResponse",
    # Réponses analytics
    "SpendingPattern", "SpendingPatternResponse", "TrendAnalysis", "TrendAnalysisResponse",
    "ResponseFactory",
    
    # === MODÈLES FILTRES ===
    # Énumérations filtres
    "AmountFilterType", "DateFilterType", "TransactionType", "OperationType", "CurrencyCode",
    # Filtres base
    "UserIsolationFilter", "AmountFilter", "DateFilter", "CategoryFilter", "MerchantFilter",
    # Filtres composés
    "TransactionTypeFilter", "CompositeFilter",
    # Filtres spécialisés
    "RecurringTransactionFilter", "SuspiciousActivityFilter", "BudgetFilter",
    # Builder et Factory
    "FilterBuilder", "FilterFactory",
    
    # === MODÈLES ELASTICSEARCH ===
    # Énumérations ES
    "ESQueryType", "ESAggType", "ESScoreMode", "ESBoostMode",
    # Composants requête
    "ESQueryClause", "ESBoolQuery", "ESAggregation", "ElasticsearchQuery",
    # Factories ES
    "FinancialQueryFactory", "FinancialAggregationFactory",
    # Templates
    "QueryTemplate", "FinancialQueryTemplates",
    # Builder
    "ElasticsearchQueryBuilder",
    
    # === UTILITAIRES ===
    "get_model_by_name", "list_available_models", "validate_model_imports",
    "CONTRACTS", "REQUEST_MODELS", "RESPONSE_MODELS", "FILTER_MODELS", "ELASTICSEARCH_MODELS",
]