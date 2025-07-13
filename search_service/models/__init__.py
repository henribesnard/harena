"""
📋 Module Models - Modèles de données et contrats

Point d'entrée simplifié pour tous les modèles de données du Search Service.
Expose les contrats d'interface, les modèles internes et les validateurs.
"""

# === IMPORTS CONTRATS D'INTERFACE ===
from .service_contracts import (
   # Enums principaux
   QueryType,
   FilterOperator,
   AggregationType,
   
   # Modèles de requête
   SearchFilter,
   SearchFilters,
   TextSearchConfig,
   AggregationRequest,
   SearchParameters,
   QueryMetadata,
   ExecutionContext,
   SearchOptions,
   
   # Contrat principal requête
   SearchServiceQuery,
   
   # Modèles de réponse
   SearchResult,
   AggregationBucket,
   AggregationResult,
   PerformanceMetrics,
   ContextEnrichment,
   ResponseMetadata,
   
   # Contrat principal réponse
   SearchServiceResponse,
   
   # Validateur
   ContractValidator
)

# === IMPORTS REQUÊTES INTERNES ===
from .requests import (
   # Enums internes
   InternalQueryType,
   ProcessingMode,
   CacheStrategy,
   
   # Structures de données
   FieldBoost,
   TermFilter,
   TextQuery,
   
   # Modèle principal interne
   InternalSearchRequest,
   
   # Transformateurs
   RequestTransformer,
   ResponseTransformer,
   
   # Validation
   RequestValidator,
   
   # Modèles API
   ValidationRequest,
   TemplateRequest,
   
   # Utilitaires
   ContractConverter,
   RequestFactory
)

# === IMPORTS RÉPONSES ===
from .responses import (
   # Enums réponses
   ExecutionStatus,
   OptimizationType,
   QualityIndicator,
   ComponentStatus,
   ValidationSeverity,
   
   # Structures internes
   RawTransaction,
   AggregationBucketInternal,
   InternalAggregationResult,
   ExecutionMetrics,
   
   # Modèle interne principal
   InternalSearchResponse,
   
   # Modèles API REST
   ValidationError,
   SecurityCheckResult,
   PerformanceAnalysis,
   ValidationResponse,
   TemplateInfo,
   TemplateListResponse,
   ComponentHealthInfo,
   SystemHealth,
   HealthResponse,
   MetricsResponse,
   
   # Builders
   ResponseBuilder,
   ValidationResponseBuilder,
   HealthResponseBuilder,
   TemplateResponseBuilder,
   
   # Utilitaires
   ResponseConverter,
   ResponseValidator
)

# === IMPORTS REQUÊTES ELASTICSEARCH ===
from .elasticsearch_queries import (
   # Enums Elasticsearch
   ESQueryType,
   ESBoolClause,
   ESSortOrder,
   ESMultiMatchType,
   ESAggregationType,
   
   # Modèles de base
   ESField,
   ESSort,
   
   # Requêtes simples
   ESTermQuery,
   ESTermsQuery,
   ESRangeQuery,
   ESMatchQuery,
   ESMultiMatchQuery,
   
   # Requêtes composées
   ESBoolQuery,
   
   # Agrégations
   ESTermsAggregation,
   ESMetricAggregation,
   ESDateHistogramAggregation,
   ESAggregationContainer,
   
   # Requête complète
   ESSearchQuery,
   
   # Builder
   FinancialTransactionQueryBuilder,
   
   # Templates
   ESQueryTemplates,
   
   # Validateur
   ESQueryValidator,
   
   # Utilitaires
   optimize_es_query,
   extract_query_metadata
)

# === IMPORTS FILTRES ===
from .filters import (
   # Enums filtres
   FieldType,
   FilterPriority,
   DatePeriod,
   
   # Configuration
   FieldConfig,
   FIELD_CONFIGURATIONS,
   
   # Modèles
   FilterValue,
   DateRangeFilter,
   AmountRangeFilter,
   ValidatedFilter,
   FilterSet,
   
   # Factory et validateurs
   FilterFactory,
   FilterValidator
)

# === CLASSE GESTIONNAIRE SIMPLIFIÉE ===
class ModelManager:
   """
   Gestionnaire unifié pour tous les modèles
   
   Centralise l'accès aux validateurs et transformateurs.
   """
   def __init__(self):
       self.contract_validator = ContractValidator()
       self.request_transformer = RequestTransformer()
       self.response_transformer = ResponseTransformer()
       self.request_validator = RequestValidator()
       self.response_validator = ResponseValidator()
       self.filter_factory = FilterFactory()
       self.filter_validator = FilterValidator()
   
   def validate_search_query(self, query: SearchServiceQuery) -> bool:
       """Valide un contrat de requête"""
       return self.contract_validator.validate_search_query(query)
   
   def validate_search_response(self, response: SearchServiceResponse) -> bool:
       """Valide un contrat de réponse"""
       return self.contract_validator.validate_search_response(response)
   
   def transform_contract_to_internal(self, contract: SearchServiceQuery) -> InternalSearchRequest:
       """Transforme un contrat en requête interne"""
       return self.request_transformer.from_contract(contract)
   
   def transform_internal_to_contract(self, internal_response: InternalSearchResponse) -> SearchServiceResponse:
       """Transforme une réponse interne en contrat"""
       return self.response_transformer.to_service_contract(internal_response, "unknown")

# === INSTANCE GLOBALE ===
model_manager = ModelManager()

# === EXPORTS ===
__all__ = [
   # === GESTIONNAIRE PRINCIPAL ===
   "ModelManager",
   "model_manager",
   
   # === CONTRATS D'INTERFACE ===
   # Enums
   "QueryType",
   "FilterOperator", 
   "AggregationType",
   
   # Modèles requête
   "SearchFilter",
   "SearchFilters",
   "TextSearchConfig",
   "AggregationRequest",
   "SearchParameters",
   "QueryMetadata",
   "ExecutionContext",
   "SearchOptions",
   
   # Contrats principaux
   "SearchServiceQuery",
   "SearchServiceResponse",
   
   # Modèles réponse
   "SearchResult",
   "AggregationBucket",
   "AggregationResult",
   "PerformanceMetrics",
   "ContextEnrichment",
   "ResponseMetadata",
   
   # Validateur contrats
   "ContractValidator",
   
   # === MODÈLES INTERNES ===
   # Enums internes
   "InternalQueryType",
   "ProcessingMode",
   "CacheStrategy",
   "ExecutionStatus",
   "OptimizationType",
   "QualityIndicator",
   
   # Structures
   "FieldBoost",
   "TermFilter",
   "TextQuery",
   "RawTransaction",
   "InternalAggregationResult",
   "ExecutionMetrics",
   
   # Modèles principaux internes
   "InternalSearchRequest",
   "InternalSearchResponse",
   
   # Transformateurs PRINCIPAUX
   "RequestTransformer",
   "ResponseTransformer",
   
   # === MODÈLES API REST ===
   "ValidationRequest",
   "TemplateRequest",
   "ValidationResponse",
   "TemplateListResponse",
   "HealthResponse",
   "MetricsResponse",
   
   # Validation API
   "ValidationError",
   "SecurityCheckResult",
   "PerformanceAnalysis",
   "ComponentStatus",
   "ValidationSeverity",
   
   # === ELASTICSEARCH ===
   # Enums ES
   "ESQueryType",
   "ESBoolClause",
   "ESSortOrder",
   "ESMultiMatchType",
   "ESAggregationType",
   
   # Modèles ES
   "ESField",
   "ESSort",
   "ESTermQuery",
   "ESBoolQuery",
   "ESSearchQuery",
   
   # Builder ES
   "FinancialTransactionQueryBuilder",
   "ESQueryTemplates",
   "ESQueryValidator",
   
   # === FILTRES ===
   # Enums filtres
   "FieldType",
   "FilterPriority",
   "DatePeriod",
   
   # Modèles filtres
   "FilterValue",
   "DateRangeFilter",
   "AmountRangeFilter",
   "ValidatedFilter",
   "FilterSet",
   "FieldConfig",
   "FIELD_CONFIGURATIONS",
   
   # Factory filtres
   "FilterFactory",
   "FilterValidator",
   
   # === BUILDERS ET UTILITAIRES ===
   "RequestValidator",
   "ResponseValidator",
   "ContractConverter",
   "RequestFactory",
   "ResponseBuilder",
   "ValidationResponseBuilder",
   "HealthResponseBuilder",
   "TemplateResponseBuilder",
   "ResponseConverter"
]