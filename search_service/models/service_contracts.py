"""
🤝 Contrats Interface Search Service - Communication Standardisée
================================================================

Contrats standardisés pour la communication entre Conversation Service (AutoGen + DeepSeek)
et Search Service (Elasticsearch). Ces contrats définissent l'interface stable et évolutive
selon l'architecture hybride.

Responsabilités:
- Interface stable entre services
- Validation stricte des données
- Sérialisation/désérialisation cohérente
- Évolutivité versions contrats
- Sécurité et validation des champs
"""

from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, model_validator
import uuid


# =============================================================================
# 🎯 ÉNUMÉRATIONS ET TYPES
# =============================================================================

class QueryType(str, Enum):
    """Types de requêtes supportés par le Search Service."""
    SIMPLE_SEARCH = "simple_search"
    FILTERED_SEARCH = "filtered_search"
    TEXT_SEARCH = "text_search"
    TEXT_SEARCH_WITH_FILTER = "text_search_with_filter"
    FILTERED_AGGREGATION = "filtered_aggregation"
    TEMPORAL_AGGREGATION = "temporal_aggregation"
    CATEGORY_ANALYSIS = "category_analysis"
    MERCHANT_ANALYSIS = "merchant_analysis"
    AMOUNT_ANALYSIS = "amount_analysis"

class IntentType(str, Enum):
    """Types d'intentions supportés par le système."""
    # Intentions recherche basiques
    SEARCH_BY_CATEGORY = "SEARCH_BY_CATEGORY"
    SEARCH_BY_MERCHANT = "SEARCH_BY_MERCHANT"
    SEARCH_BY_AMOUNT = "SEARCH_BY_AMOUNT"
    SEARCH_BY_DATE = "SEARCH_BY_DATE"
    TEXT_SEARCH = "TEXT_SEARCH"
    
    # Intentions analyse
    COUNT_OPERATIONS = "COUNT_OPERATIONS"
    COUNT_OPERATIONS_BY_AMOUNT = "COUNT_OPERATIONS_BY_AMOUNT"
    TEMPORAL_SPENDING_ANALYSIS = "TEMPORAL_SPENDING_ANALYSIS"
    CATEGORY_SPENDING_ANALYSIS = "CATEGORY_SPENDING_ANALYSIS"
    MERCHANT_SPENDING_ANALYSIS = "MERCHANT_SPENDING_ANALYSIS"
    
    # Intentions complexes
    TEXT_SEARCH_WITH_CATEGORY = "TEXT_SEARCH_WITH_CATEGORY"
    COMPARATIVE_ANALYSIS = "COMPARATIVE_ANALYSIS"
    TREND_ANALYSIS = "TREND_ANALYSIS"

class FilterOperator(str, Enum):
    """Opérateurs de filtrage supportés."""
    EQ = "eq"           # égal
    NE = "ne"           # différent
    GT = "gt"           # supérieur
    GTE = "gte"         # supérieur ou égal
    LT = "lt"           # inférieur
    LTE = "lte"         # inférieur ou égal
    IN = "in"           # dans la liste
    NOT_IN = "not_in"   # pas dans la liste
    BETWEEN = "between" # entre deux valeurs
    EXISTS = "exists"   # champ existe
    MISSING = "missing" # champ manquant

class AggregationType(str, Enum):
    """Types d'agrégations supportés."""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    TERMS = "terms"
    DATE_HISTOGRAM = "date_histogram"
    RANGE = "range"
    STATS = "stats"

class TextSearchOperator(str, Enum):
    """Opérateurs de recherche textuelle."""
    MATCH = "match"
    MATCH_PHRASE = "match_phrase"
    MATCH_PHRASE_PREFIX = "match_phrase_prefix"
    MULTI_MATCH = "multi_match"
    QUERY_STRING = "query_string"
    SIMPLE_QUERY_STRING = "simple_query_string"


# =============================================================================
# 🔍 MODÈLES FILTRES
# =============================================================================

class SearchFilter(BaseModel):
    """Modèle filtre de recherche."""
    field: str = Field(..., description="Champ à filtrer")
    operator: FilterOperator = Field(..., description="Opérateur de filtrage")
    value: Union[str, int, float, bool, List[Union[str, int, float]], None] = Field(
        ..., description="Valeur(s) pour le filtre"
    )
    
    @validator('field')
    def validate_field_name(cls, v):
        """Validation nom de champ."""
        if not v or not isinstance(v, str):
            raise ValueError("Field name must be non-empty string")
        if len(v) > 100:
            raise ValueError("Field name too long (max 100 chars)")
        return v.strip()
    
    @validator('value')
    def validate_filter_value(cls, v, values):
        """Validation valeur selon opérateur."""
        operator = values.get('operator')
        
        if operator == FilterOperator.BETWEEN:
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError("BETWEEN operator requires list of 2 values")
        elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(v, list):
                raise ValueError(f"{operator} operator requires list of values")
            if len(v) > 100:
                raise ValueError("Too many values in filter (max 100)")
        elif operator in [FilterOperator.EXISTS, FilterOperator.MISSING]:
            if v is not None:
                raise ValueError(f"{operator} operator requires null value")
        
        return v

class TextSearchQuery(BaseModel):
    """Modèle requête recherche textuelle."""
    query: str = Field(..., min_length=1, max_length=1000, description="Texte à rechercher")
    fields: List[str] = Field(..., min_items=1, description="Champs de recherche")
    operator: TextSearchOperator = Field(default=TextSearchOperator.MATCH, description="Opérateur recherche")
    boost: Optional[float] = Field(default=1.0, ge=0.1, le=10.0, description="Boost de pertinence")
    
    @validator('query')
    def validate_query_text(cls, v):
        """Validation texte recherche."""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()
    
    @validator('fields')
    def validate_search_fields(cls, v):
        """Validation champs recherche."""
        if not v:
            raise ValueError("Search fields cannot be empty")
        
        # Validation noms champs
        for field in v:
            if not isinstance(field, str) or not field.strip():
                raise ValueError("Field names must be non-empty strings")
        
        return [field.strip() for field in v]

class AggregationRequest(BaseModel):
    """Modèle demande d'agrégation."""
    name: str = Field(..., description="Nom de l'agrégation")
    type: AggregationType = Field(..., description="Type d'agrégation")
    field: Optional[str] = Field(None, description="Champ pour l'agrégation")
    size: Optional[int] = Field(default=10, ge=1, le=1000, description="Nombre de buckets max")
    
    @validator('name')
    def validate_aggregation_name(cls, v):
        """Validation nom agrégation."""
        if not v or not isinstance(v, str):
            raise ValueError("Aggregation name must be non-empty string")
        return v.strip()


# =============================================================================
# 📥 CONTRAT REQUÊTE SEARCH SERVICE
# =============================================================================

class QueryMetadata(BaseModel):
    """Métadonnées requête pour traçabilité et contexte."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID unique requête")
    user_id: int = Field(..., ge=1, description="ID utilisateur (obligatoire sécurité)")
    intent_type: IntentType = Field(..., description="Type intention détectée")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance classification")
    agent_name: str = Field(..., description="Nom agent AutoGen source")
    team_name: Optional[str] = Field(None, description="Nom équipe AutoGen")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Contexte exécution")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp requête")
    
    @validator('agent_name')
    def validate_agent_name(cls, v):
        """Validation nom agent."""
        if not v or not isinstance(v, str):
            raise ValueError("Agent name must be non-empty string")
        return v.strip()

class SearchParameters(BaseModel):
    """Paramètres de recherche."""
    query_type: QueryType = Field(..., description="Type de requête")
    fields: List[str] = Field(default_factory=list, description="Champs à rechercher/retourner")
    limit: int = Field(default=20, ge=1, le=1000, description="Nombre max résultats")
    offset: int = Field(default=0, ge=0, le=10000, description="Offset pagination")
    timeout_ms: int = Field(default=5000, ge=100, le=30000, description="Timeout requête (ms)")
    
    @validator('fields')
    def validate_fields(cls, v):
        """Validation champs recherche."""
        if len(v) > 50:
            raise ValueError("Too many fields (max 50)")
        return [field.strip() for field in v if field and field.strip()]

class FilterGroup(BaseModel):
    """Groupe de filtres avec logique."""
    required: List[SearchFilter] = Field(default_factory=list, description="Filtres obligatoires (AND)")
    optional: List[SearchFilter] = Field(default_factory=list, description="Filtres optionnels (OR)")
    ranges: List[SearchFilter] = Field(default_factory=list, description="Filtres de plage")
    text_search: Optional[TextSearchQuery] = Field(None, description="Recherche textuelle")
    
    @validator('required')
    def validate_required_filters(cls, v):
        """Validation filtres obligatoires."""
        if len(v) > 20:
            raise ValueError("Too many required filters (max 20)")
        return v
    
    @model_validator(mode='after')
    def validate_user_id_required(self):
        """Validation user_id obligatoire pour sécurité."""
        required_filters = self.required or []
        
        # Vérifier qu'un filtre user_id existe
        user_id_filter_exists = any(
            f.field == 'user_id' for f in required_filters
        )
        
        if not user_id_filter_exists:
            raise ValueError("user_id filter is mandatory for security isolation")
        
        return self

class AggregationGroup(BaseModel):
    """Groupe d'agrégations."""
    enabled: bool = Field(default=False, description="Activation agrégations")
    types: List[AggregationType] = Field(default_factory=list, description="Types d'agrégations")
    group_by: List[str] = Field(default_factory=list, description="Champs de groupement")
    metrics: List[str] = Field(default_factory=list, description="Métriques à calculer")
    requests: List[AggregationRequest] = Field(default_factory=list, description="Demandes agrégation détaillées")
    
    @validator('requests')
    def validate_aggregation_requests(cls, v):
        """Validation demandes agrégation."""
        if len(v) > 10:
            raise ValueError("Too many aggregation requests (max 10)")
        return v

class QueryOptions(BaseModel):
    """Options requête."""
    include_highlights: bool = Field(default=False, description="Inclure highlighting")
    include_explanation: bool = Field(default=False, description="Inclure explication score")
    cache_enabled: bool = Field(default=True, description="Activation cache")
    return_raw_elasticsearch: bool = Field(default=False, description="Retourner réponse ES brute")

class SearchServiceQuery(BaseModel):
    """
    Contrat principal requête Search Service.
    
    Interface standardisée entre Conversation Service (AutoGen + DeepSeek)
    et Search Service (Elasticsearch pure).
    """
    query_metadata: QueryMetadata = Field(..., description="Métadonnées requête")
    search_parameters: SearchParameters = Field(..., description="Paramètres recherche")
    filters: FilterGroup = Field(..., description="Filtres de recherche")
    aggregations: AggregationGroup = Field(default_factory=AggregationGroup, description="Agrégations")
    options: QueryOptions = Field(default_factory=QueryOptions, description="Options requête")
    
    class Config:
        """Configuration Pydantic."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "query_metadata": {
                    "user_id": 34,
                    "intent_type": "SEARCH_BY_CATEGORY",
                    "confidence": 0.95,
                    "agent_name": "query_generator_agent",
                    "team_name": "financial_analysis_team"
                },
                "search_parameters": {
                    "query_type": "filtered_search",
                    "fields": ["searchable_text", "primary_description", "merchant_name"],
                    "limit": 20,
                    "timeout_ms": 5000
                },
                "filters": {
                    "required": [
                        {"field": "user_id", "operator": "eq", "value": 34},
                        {"field": "category_name", "operator": "eq", "value": "restaurant"}
                    ],
                    "text_search": {
                        "query": "italien",
                        "fields": ["searchable_text", "primary_description"],
                        "operator": "match"
                    }
                }
            }
        }


# =============================================================================
# 📤 CONTRAT RÉPONSE SEARCH SERVICE  
# =============================================================================

class ResponseMetadata(BaseModel):
    """Métadonnées réponse."""
    query_id: str = Field(..., description="ID requête correspondante")
    execution_time_ms: int = Field(..., ge=0, description="Temps exécution (ms)")
    total_hits: int = Field(..., ge=0, description="Nombre total résultats")
    returned_hits: int = Field(..., ge=0, description="Nombre résultats retournés")
    has_more: bool = Field(..., description="Plus de résultats disponibles")
    cache_hit: bool = Field(..., description="Résultat du cache")
    elasticsearch_took: int = Field(..., ge=0, description="Temps Elasticsearch (ms)")
    agent_context: Dict[str, Any] = Field(default_factory=dict, description="Contexte pour agents")
    
    @validator('returned_hits')
    def validate_returned_vs_total(cls, v, values):
        """Validation cohérence nombres résultats."""
        total_hits = values.get('total_hits', 0)
        if v > total_hits:
            raise ValueError("Returned hits cannot exceed total hits")
        return v

class TransactionResult(BaseModel):
    """Résultat transaction financière."""
    transaction_id: str = Field(..., description="ID unique transaction")
    user_id: int = Field(..., description="ID utilisateur")
    account_id: Optional[int] = Field(None, description="ID compte")
    amount: float = Field(..., description="Montant avec signe")
    amount_abs: float = Field(..., ge=0, description="Montant absolu")
    transaction_type: str = Field(..., description="Type transaction")
    currency_code: str = Field(..., description="Code devise")
    date: str = Field(..., description="Date transaction (YYYY-MM-DD)")
    primary_description: str = Field(..., description="Description principale")
    merchant_name: Optional[str] = Field(None, description="Nom marchand")
    category_name: Optional[str] = Field(None, description="Nom catégorie")
    operation_type: Optional[str] = Field(None, description="Type opération")
    month_year: Optional[str] = Field(None, description="Mois-année (YYYY-MM)")
    weekday: Optional[str] = Field(None, description="Jour semaine")
    score: float = Field(..., ge=0, le=10, description="Score pertinence")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Highlighting")
    
    @validator('amount_abs')
    def validate_amount_consistency(cls, v, values):
        """Validation cohérence montants."""
        amount = values.get('amount')
        if amount is not None and abs(amount) != v:
            raise ValueError("amount_abs must equal absolute value of amount")
        return v

class AggregationBucket(BaseModel):
    """Bucket d'agrégation."""
    key: Union[str, int, float] = Field(..., description="Clé bucket")
    doc_count: int = Field(..., ge=0, description="Nombre documents")
    sub_aggregations: Dict[str, Any] = Field(default_factory=dict, description="Sous-agrégations")

class AggregationResult(BaseModel):
    """Résultat d'agrégation."""
    name: str = Field(..., description="Nom agrégation")
    type: AggregationType = Field(..., description="Type agrégation")
    buckets: List[AggregationBucket] = Field(default_factory=list, description="Buckets résultats")
    stats: Dict[str, Union[int, float]] = Field(default_factory=dict, description="Statistiques")

class PerformanceMetrics(BaseModel):
    """Métriques performance."""
    query_complexity: str = Field(..., description="Complexité requête")
    optimization_applied: List[str] = Field(default_factory=list, description="Optimisations appliquées")
    index_used: str = Field(..., description="Index Elasticsearch utilisé")
    shards_queried: int = Field(..., ge=1, description="Nombre shards interrogés")

class ContextEnrichment(BaseModel):
    """Enrichissement contextuel pour agents."""
    search_intent_matched: bool = Field(..., description="Intention recherche correspondante")
    result_quality_score: float = Field(..., ge=0, le=1, description="Score qualité résultats")
    suggested_followup_questions: List[str] = Field(default_factory=list, description="Questions de suivi suggérées")
    
    @validator('suggested_followup_questions')
    def validate_followup_questions(cls, v):
        """Validation questions de suivi."""
        if len(v) > 5:
            raise ValueError("Too many followup questions (max 5)")
        return v

class SearchServiceResponse(BaseModel):
    """
    Contrat principal réponse Search Service.
    
    Réponse standardisée du Search Service vers Conversation Service
    avec enrichissement contextuel pour les agents AutoGen.
    """
    response_metadata: ResponseMetadata = Field(..., description="Métadonnées réponse")
    results: List[TransactionResult] = Field(..., description="Résultats transactions")
    aggregations: Dict[str, Any] = Field(default_factory=dict, description="Résultats agrégations")
    performance: PerformanceMetrics = Field(..., description="Métriques performance")
    context_enrichment: ContextEnrichment = Field(..., description="Enrichissement contextuel")
    debug: Dict[str, Any] = Field(default_factory=dict, description="Informations debug")
    
    @validator('results')
    def validate_results_count(cls, v, values):
        """Validation nombre résultats."""
        metadata = values.get('response_metadata')
        if metadata and len(v) != metadata.returned_hits:
            raise ValueError("Results count must match returned_hits in metadata")
        return v
    
    class Config:
        """Configuration Pydantic."""
        schema_extra = {
            "example": {
                "response_metadata": {
                    "query_id": "uuid-v4",
                    "execution_time_ms": 45,
                    "total_hits": 156,
                    "returned_hits": 20,
                    "has_more": True,
                    "cache_hit": False,
                    "elasticsearch_took": 23
                },
                "results": [
                    {
                        "transaction_id": "user_34_tx_12345",
                        "user_id": 34,
                        "amount": -45.67,
                        "amount_abs": 45.67,
                        "category_name": "restaurant",
                        "merchant_name": "RISTORANTE ITALIANO",
                        "date": "2024-01-15",
                        "score": 0.95
                    }
                ],
                "performance": {
                    "query_complexity": "simple",
                    "optimization_applied": ["user_filter", "category_filter"],
                    "index_used": "harena_transactions",
                    "shards_queried": 1
                },
                "context_enrichment": {
                    "search_intent_matched": True,
                    "result_quality_score": 0.95,
                    "suggested_followup_questions": []
                }
            }
        }


# =============================================================================
# 🛠️ UTILITAIRES CONTRATS
# =============================================================================

class ContractValidator:
    """Validateur contrats avec vérifications sécurité."""
    
    @staticmethod
    def validate_search_query(query: SearchServiceQuery) -> Dict[str, Any]:
        """Validation complète requête avec sécurité."""
        validation = {"valid": True, "errors": [], "warnings": []}
        
        try:
            # Validation métadonnées obligatoires
            if query.query_metadata.user_id <= 0:
                validation["errors"].append("user_id must be positive")
                validation["valid"] = False
            
            # Validation sécurité : user_id filter obligatoire
            user_filter_exists = any(
                f.field == "user_id" for f in query.filters.required
            )
            if not user_filter_exists:
                validation["errors"].append("user_id filter is mandatory for security")
                validation["valid"] = False
            
            # Validation limites performance
            if query.search_parameters.limit > 1000:
                validation["warnings"].append("Large limit may impact performance")
            
            if query.search_parameters.timeout_ms > 10000:
                validation["warnings"].append("High timeout may impact user experience")
            
        except Exception as e:
            validation["errors"].append(f"Validation error: {str(e)}")
            validation["valid"] = False
        
        return validation
    
    @staticmethod
    def validate_search_response(response: SearchServiceResponse) -> Dict[str, Any]:
        """Validation complète réponse."""
        validation = {"valid": True, "errors": [], "warnings": []}
        
        try:
            # Validation cohérence métadonnées
            if response.response_metadata.returned_hits != len(response.results):
                validation["errors"].append("returned_hits must match results count")
                validation["valid"] = False
            
            # Validation performance
            if response.response_metadata.execution_time_ms > 5000:
                validation["warnings"].append("Slow query execution time")
            
            # Validation qualité résultats
            if response.context_enrichment.result_quality_score < 0.5:
                validation["warnings"].append("Low result quality score")
            
        except Exception as e:
            validation["errors"].append(f"Validation error: {str(e)}")
            validation["valid"] = False
        
        return validation


# =============================================================================
# 📋 EXPORTS
# =============================================================================

__all__ = [
    # Énumérations
    "QueryType", "IntentType", "FilterOperator", "AggregationType", "TextSearchOperator",
    # Modèles filtres et recherche
    "SearchFilter", "TextSearchQuery", "AggregationRequest",
    # Contrat requête
    "QueryMetadata", "SearchParameters", "FilterGroup", "AggregationGroup", "QueryOptions", "SearchServiceQuery",
    # Contrat réponse
    "ResponseMetadata", "TransactionResult", "AggregationBucket", "AggregationResult", 
    "PerformanceMetrics", "ContextEnrichment", "SearchServiceResponse",
    # Utilitaires
    "ContractValidator",
]