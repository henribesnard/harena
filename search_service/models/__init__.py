"""
Module models du Search Service
Expose tous les modèles, contrats et utilitaires de données
"""

# === CONTRATS D'INTERFACE ===
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

# === MODÈLES INTERNES REQUÊTES ===
from .requests import (
    # Enums internes
    InternalQueryType,
    ProcessingMode,
    CacheStrategy,
    
    # Structures de données
    FieldBoost,
    TermFilter,
    TextQuery,
    
    # Modèle principal
    InternalSearchRequest,
    
    # Transformateur
    RequestTransformer,
    
    # Validateur
    RequestValidator
)

# === MODÈLES INTERNES RÉPONSES ===
from .responses import (
    # Enums de réponse
    ExecutionStatus,
    OptimizationType,
    QualityIndicator,
    
    # Structures de données
    RawTransaction,
    AggregationBucketInternal,
    InternalAggregationResult,
    ExecutionMetrics,
    
    # Modèle principal
    InternalSearchResponse,
    
    # Transformateur
    ResponseTransformer,
    
    # Builder
    ResponseBuilder
)

# === MODÈLES DE FILTRES ===
from .filters import (
    # Enums de filtres
    FieldType,
    FilterPriority,
    DatePeriod,
    
    # Configuration
    FieldConfig,
    FIELD_CONFIGURATIONS,
    
    # Modèles de filtres
    FilterValue,
    DateRangeFilter,
    AmountRangeFilter,
    ValidatedFilter,
    FilterSet,
    
    # Factory et Validators
    FilterFactory,
    FilterValidator
)

# Version du module models
__version__ = "1.0.0"

# === EXPORTS ORGANISÉS ===

# Contrats externes (interface avec Conversation Service)
__all_contracts__ = [
    "SearchServiceQuery",
    "SearchServiceResponse", 
    "ContractValidator",
    "QueryType",
    "FilterOperator",
    "AggregationType"
]

# Modèles internes (traitement interne)
__all_internal__ = [
    "InternalSearchRequest",
    "InternalSearchResponse",
    "RequestTransformer",
    "ResponseTransformer",
    "ResponseBuilder"
]

# Système de filtres
__all_filters__ = [
    "FilterSet",
    "ValidatedFilter",
    "FilterFactory",
    "FilterValidator",
    "FIELD_CONFIGURATIONS"
]

# Export principal
__all__ = [
    # === CONTRATS D'INTERFACE ===
    # Enums
    "QueryType",
    "FilterOperator", 
    "AggregationType",
    
    # Requête externe
    "SearchFilter",
    "SearchFilters",
    "TextSearchConfig",
    "AggregationRequest",
    "SearchParameters",
    "QueryMetadata",
    "ExecutionContext",
    "SearchOptions",
    "SearchServiceQuery",
    
    # Réponse externe
    "SearchResult",
    "AggregationBucket",
    "AggregationResult",
    "PerformanceMetrics",
    "ContextEnrichment", 
    "ResponseMetadata",
    "SearchServiceResponse",
    
    # Validateur externe
    "ContractValidator",
    
    # === MODÈLES INTERNES ===
    # Enums internes
    "InternalQueryType",
    "ProcessingMode",
    "CacheStrategy",
    "ExecutionStatus",
    "OptimizationType",
    "QualityIndicator",
    
    # Requête interne
    "FieldBoost",
    "TermFilter", 
    "TextQuery",
    "InternalSearchRequest",
    "RequestTransformer",
    "RequestValidator",
    
    # Réponse interne
    "RawTransaction",
    "AggregationBucketInternal",
    "InternalAggregationResult",
    "ExecutionMetrics",
    "InternalSearchResponse",
    "ResponseTransformer",
    "ResponseBuilder",
    
    # === SYSTÈME DE FILTRES ===
    # Enums filtres
    "FieldType",
    "FilterPriority",
    "DatePeriod",
    
    # Configuration
    "FieldConfig",
    "FIELD_CONFIGURATIONS",
    
    # Modèles filtres
    "FilterValue",
    "DateRangeFilter",
    "AmountRangeFilter",
    "ValidatedFilter", 
    "FilterSet",
    
    # Factory et validation
    "FilterFactory",
    "FilterValidator"
]

# === HELPERS D'IMPORT ===

def get_contract_models():
    """Retourne les modèles de contrats externes"""
    return {
        "query": SearchServiceQuery,
        "response": SearchServiceResponse,
        "validator": ContractValidator
    }

def get_internal_models():
    """Retourne les modèles internes principaux"""
    return {
        "request": InternalSearchRequest,
        "response": InternalSearchResponse,
        "request_transformer": RequestTransformer,
        "response_transformer": ResponseTransformer,
        "response_builder": ResponseBuilder
    }

def get_filter_models():
    """Retourne les modèles de filtres"""
    return {
        "filter_set": FilterSet,
        "validated_filter": ValidatedFilter,
        "factory": FilterFactory,
        "validator": FilterValidator,
        "configurations": FIELD_CONFIGURATIONS
    }

# === VALIDATION MODULE ===

def validate_models_import():
    """Valide que tous les modèles sont correctement importés"""
    import sys
    current_module = sys.modules[__name__]
    
    # Vérifier que les classes principales sont disponibles
    essential_classes = [
        "SearchServiceQuery",
        "SearchServiceResponse", 
        "InternalSearchRequest",
        "InternalSearchResponse",
        "FilterSet",
        "ValidatedFilter"
    ]
    
    missing = []
    for class_name in essential_classes:
        if not hasattr(current_module, class_name):
            missing.append(class_name)
    
    if missing:
        raise ImportError(f"Classes manquantes dans models: {missing}")
    
    return True

# === FACTORY HELPERS ===

class ModelFactory:
    """Factory centralisée pour créer des instances de modèles"""
    
    @staticmethod
    def create_simple_query(user_id: int, category: str = None, merchant: str = None) -> SearchServiceQuery:
        """Crée une requête simple pour tests"""
        from datetime import datetime
        from uuid import uuid4
        
        # Métadonnées de base
        metadata = QueryMetadata(
            query_id=str(uuid4()),
            user_id=user_id,
            intent_type="SEARCH_BY_CATEGORY" if category else "SIMPLE_SEARCH",
            confidence=0.95,
            agent_name="test_agent",
            timestamp=datetime.utcnow()
        )
        
        # Paramètres de recherche
        search_params = SearchParameters(
            query_type=QueryType.FILTERED_SEARCH,
            limit=20,
            timeout_ms=5000
        )
        
        # Filtres obligatoires
        required_filters = [
            SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)
        ]
        
        if category:
            required_filters.append(
                SearchFilter(field="category_name.keyword", operator=FilterOperator.EQ, value=category)
            )
        
        if merchant:
            required_filters.append(
                SearchFilter(field="merchant_name.keyword", operator=FilterOperator.EQ, value=merchant)
            )
        
        filters = SearchFilters(required=required_filters)
        
        return SearchServiceQuery(
            query_metadata=metadata,
            search_parameters=search_params,
            filters=filters
        )
    
    @staticmethod
    def create_text_search_query(user_id: int, text: str) -> SearchServiceQuery:
        """Crée une requête de recherche textuelle"""
        from datetime import datetime
        from uuid import uuid4
        
        metadata = QueryMetadata(
            query_id=str(uuid4()),
            user_id=user_id,
            intent_type="TEXT_SEARCH",
            confidence=0.90,
            agent_name="test_agent",
            timestamp=datetime.utcnow()
        )
        
        search_params = SearchParameters(
            query_type=QueryType.TEXT_SEARCH,
            limit=20,
            timeout_ms=5000
        )
        
        filters = SearchFilters(
            required=[SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)]
        )
        
        text_config = TextSearchConfig(
            query=text,
            fields=["searchable_text", "primary_description", "merchant_name"]
        )
        
        return SearchServiceQuery(
            query_metadata=metadata,
            search_parameters=search_params,
            filters=filters,
            text_search=text_config
        )
    
    @staticmethod
    def create_filter_set(user_id: int) -> FilterSet:
        """Crée un FilterSet avec user_id obligatoire"""
        filter_set = FilterSet()
        filter_set.add_required_filter("user_id", "eq", user_id)
        return filter_set
    
    @staticmethod
    def create_sample_response(query_id: str, user_id: int, results_count: int = 5) -> SearchServiceResponse:
        """Crée une réponse d'exemple pour tests"""
        from datetime import datetime
        
        # Métadonnées de réponse
        metadata = ResponseMetadata(
            query_id=query_id,
            execution_time_ms=45,
            total_hits=results_count * 3,
            returned_hits=results_count,
            has_more=True,
            cache_hit=False,
            elasticsearch_took=23,
            timestamp=datetime.utcnow()
        )
        
        # Résultats d'exemple
        results = []
        for i in range(results_count):
            result = SearchResult(
                transaction_id=f"user_{user_id}_tx_{i+1}",
                user_id=user_id,
                amount=-25.50 - i * 5,
                amount_abs=25.50 + i * 5,
                transaction_type="debit",
                currency_code="EUR",
                date=f"2024-01-{15+i:02d}",
                primary_description=f"SAMPLE TRANSACTION {i+1}",
                merchant_name=f"Sample Merchant {i+1}",
                category_name="Sample Category",
                operation_type="card_payment",
                month_year="2024-01",
                score=1.0 - i * 0.1
            )
            results.append(result)
        
        # Métriques de performance
        performance = PerformanceMetrics(
            query_complexity="simple",
            optimization_applied=["user_filter"],
            index_used="harena_transactions",
            shards_queried=1,
            cache_hit=False
        )
        
        # Enrichissement contextuel
        context = ContextEnrichment(
            search_intent_matched=True,
            result_quality_score=0.85,
            suggested_followup_questions=["Voir détails", "Comparer mois précédent"]
        )
        
        return SearchServiceResponse(
            response_metadata=metadata,
            results=results,
            performance=performance,
            context_enrichment=context
        )

# Auto-validation au chargement
try:
    validate_models_import()
except ImportError as e:
    print(f"⚠️  Avertissement models: {e}")

# Helpers disponibles
__helpers__ = [
    "get_contract_models",
    "get_internal_models", 
    "get_filter_models",
    "validate_models_import",
    "ModelFactory"
]

# Ajout des helpers aux exports
__all__.extend(__helpers__)