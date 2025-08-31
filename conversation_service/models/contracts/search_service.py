"""
Contrats d'interface avec le search_service
Définit les structures de données pour les requêtes Elasticsearch
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from enum import Enum


class FilterOperator(str, Enum):
    """Opérateurs de filtrage supportés"""
    MATCH = "match"
    TERM = "term"
    TERMS = "terms"
    EXISTS = "exists"
    RANGE = "range"
    GTE = "gte"
    LTE = "lte"
    GT = "gt"
    LT = "lt"


class SortOrder(str, Enum):
    """Ordres de tri supportés"""
    ASC = "asc"
    DESC = "desc"


class AggregationType(str, Enum):
    """Types d'agrégations supportées"""
    TERMS = "terms"
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    VALUE_COUNT = "value_count"
    DATE_HISTOGRAM = "date_histogram"
    CARDINALITY = "cardinality"


class DateRange(BaseModel):
    """Structure pour les filtres de dates"""
    
    class Config:
        exclude_none = True
    gte: Optional[str] = Field(None, description="Date début au format ISO")
    lte: Optional[str] = Field(None, description="Date fin au format ISO")
    gt: Optional[str] = Field(None, description="Date après (exclusif)")
    lt: Optional[str] = Field(None, description="Date avant (exclusif)")


class AmountRange(BaseModel):
    """Structure pour les filtres de montants"""
    
    class Config:
        exclude_none = True
    gte: Optional[float] = Field(None, description="Montant minimum")
    lte: Optional[float] = Field(None, description="Montant maximum")
    gt: Optional[float] = Field(None, description="Montant supérieur (exclusif)")
    lt: Optional[float] = Field(None, description="Montant inférieur (exclusif)")


class TextFilter(BaseModel):
    """Structure pour les filtres textuels"""
    
    class Config:
        exclude_none = True
    match: Optional[str] = Field(None, description="Recherche textuelle floue")
    term: Optional[str] = Field(None, description="Terme exact")
    terms: Optional[List[str]] = Field(None, description="Liste de termes exacts")
    exists: Optional[bool] = Field(None, description="Vérifier existence du champ")


class AggregationConfig(BaseModel):
    """Configuration d'agrégation générale"""
    
    class Config:
        exclude_none = True
    
    field: str = Field(..., description="Champ à agréger")

class TermsAggregationConfig(AggregationConfig):
    """Configuration pour agrégation terms"""
    size: Optional[int] = Field(10, description="Nombre de buckets max")
    
    @validator('size')
    def validate_size(cls, v):
        if v is not None and (v < 1 or v > 100):
            raise ValueError("Size must be between 1 and 100")
        return v

class DateHistogramAggregationConfig(AggregationConfig):
    """Configuration pour agrégation date_histogram"""
    calendar_interval: str = Field(..., description="Intervalle pour date_histogram")


class NestedAggregation(BaseModel):
    """Agrégation imbriquée"""
    
    class Config:
        exclude_none = True
    sum: Optional[Dict[str, str]] = Field(None, description="Somme sur un champ")
    avg: Optional[Dict[str, str]] = Field(None, description="Moyenne sur un champ")
    max: Optional[Dict[str, str]] = Field(None, description="Maximum sur un champ")
    min: Optional[Dict[str, str]] = Field(None, description="Minimum sur un champ")
    value_count: Optional[Dict[str, str]] = Field(None, description="Comptage valeurs")
    cardinality: Optional[Dict[str, str]] = Field(None, description="Cardinalité")


class Aggregation(BaseModel):
    """Structure d'agrégation principale"""
    
    class Config:
        exclude_none = True
    terms: Optional[TermsAggregationConfig] = Field(None, description="Agrégation par termes")
    sum: Optional[Dict[str, str]] = Field(None, description="Somme")
    avg: Optional[Dict[str, str]] = Field(None, description="Moyenne")
    max: Optional[Dict[str, str]] = Field(None, description="Maximum")
    min: Optional[Dict[str, str]] = Field(None, description="Minimum")
    value_count: Optional[Dict[str, str]] = Field(None, description="Comptage")
    date_histogram: Optional[DateHistogramAggregationConfig] = Field(None, description="Histogramme temporel")
    cardinality: Optional[Dict[str, str]] = Field(None, description="Cardinalité")
    aggs: Optional[Dict[str, "NestedAggregation"]] = Field(None, description="Agrégations imbriquées")


class SortConfig(BaseModel):
    """Configuration de tri"""
    field: str
    order: SortOrder


class SearchFilters(BaseModel):
    """Filtres de recherche disponibles - SANS user_id et query (au niveau racine)"""
    
    class Config:
        # Exclude null values by default when serializing
        exclude_none = True
    
    # Filtres temporels
    date: Optional[DateRange] = Field(None, description="Filtres de dates")
    
    # Filtres montants
    amount: Optional[AmountRange] = Field(None, description="Filtres de montants")
    amount_abs: Optional[AmountRange] = Field(None, description="Filtres de montants absolus")
    
    # Filtres textuels
    merchant_name: Optional[TextFilter] = Field(None, description="Nom du marchand")
    primary_description: Optional[TextFilter] = Field(None, description="Description principale")
    category_name: Optional[TextFilter] = Field(None, description="Nom de catégorie")
    operation_type: Optional[TextFilter] = Field(None, description="Type d'opération")
    
    # Filtres énumérés
    transaction_type: Optional[str] = Field(None, description="Type transaction (credit/debit)")
    account_id: Optional[TextFilter] = Field(None, description="ID du compte")
    account_type: Optional[TextFilter] = Field(None, description="Type de compte")


class SearchQuery(BaseModel):
    """Structure complète d'une requête search_service"""
    # Identification
    user_id: int = Field(..., description="ID utilisateur requis")
    
    # Requête textuelle libre (optionnelle)
    query: Optional[str] = Field(None, description="Recherche textuelle libre")
    
    # Filtres structurés
    filters: Optional[SearchFilters] = Field(None, description="Filtres de recherche")
    
    # Agrégations
    aggregations: Optional[Dict[str, Aggregation]] = Field(None, description="Agrégations à calculer")
    
    # Configuration résultat
    sort: Optional[List[Dict[str, Dict[str, str]]]] = Field(None, description="Configuration tri")
    page_size: Optional[int] = Field(20, description="Taille de page")
    offset: Optional[int] = Field(0, description="Offset pagination")
    
    # Champs à inclure
    include_fields: Optional[List[str]] = Field(None, description="Champs à inclure dans la réponse")
    exclude_fields: Optional[List[str]] = Field(None, description="Champs à exclure")
    
    # Mode agrégation uniquement
    aggregation_only: Optional[bool] = Field(False, description="Retourner uniquement les agrégations")
    
    # Métadonnées
    explain: Optional[bool] = Field(False, description="Inclure explication de scoring")
    
    @validator('page_size')
    def validate_page_size(cls, v):
        if v is not None and (v < 1 or v > 1000):
            raise ValueError("page_size must be between 1 and 1000")
        return v
    
    @validator('aggregations')
    def validate_aggregations_count(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError("Maximum 10 aggregations per query")
        return v
    
    def dict(self, exclude_none=False, **kwargs):
        """Override dict to clean up null values and ensure proper formatting"""
        # First get the default dict
        data = super().dict(exclude_none=exclude_none, **kwargs)
        
        # Handle required fields that should be empty string instead of null
        if data.get('query') is None:
            data['query'] = ""
        
        # If exclude_none is True, remove null values from nested objects too
        if exclude_none:
            data = self._clean_nested_nulls(data)
        
        return data
    
    def _clean_nested_nulls(self, obj):
        """Recursively clean null values from nested objects"""
        if isinstance(obj, dict):
            return {k: self._clean_nested_nulls(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [self._clean_nested_nulls(item) for item in obj if item is not None]
        else:
            return obj
    
    def json(self, exclude_none=True, **kwargs):
        """Override json to exclude null values by default"""
        return super().json(exclude_none=exclude_none, **kwargs)


class QueryValidationResult(BaseModel):
    """Résultat de validation d'une requête"""
    schema_valid: bool = Field(..., description="Conformité au schéma")
    contract_compliant: bool = Field(..., description="Conformité aux contrats")
    estimated_performance: str = Field(..., description="Estimation performance (optimal/good/poor)")
    optimization_applied: List[str] = Field(default_factory=list, description="Optimisations appliquées")
    potential_issues: List[str] = Field(default_factory=list, description="Problèmes potentiels identifiés")
    errors: List[str] = Field(default_factory=list, description="Erreurs de validation")
    warnings: List[str] = Field(default_factory=list, description="Avertissements")


class QueryGenerationRequest(BaseModel):
    """Requête pour génération de query search_service"""
    user_id: int = Field(..., description="ID utilisateur")
    intent_type: str = Field(..., description="Type d'intention détecté")
    intent_confidence: float = Field(..., description="Confiance intention")
    entities: Dict[str, Any] = Field(..., description="Entités extraites")
    user_message: str = Field(..., description="Message utilisateur original")
    context: Optional[Dict[str, Any]] = Field(None, description="Contexte supplémentaire")


class QueryGenerationResponse(BaseModel):
    """Réponse de génération de query"""
    search_query: SearchQuery = Field(..., description="Requête générée")
    validation: QueryValidationResult = Field(..., description="Résultat validation")
    generation_confidence: float = Field(..., description="Confiance génération")
    reasoning: str = Field(..., description="Explication génération")
    query_type: str = Field(..., description="Type de requête générée")
    estimated_results_count: Optional[int] = Field(None, description="Estimation nombre résultats")


class SearchHit(BaseModel):
    """Résultat de recherche individuel"""
    id: str = Field(..., description="ID du document", alias="_id")
    score: Optional[float] = Field(None, description="Score de pertinence", alias="_score")
    source: Dict[str, Any] = Field(..., description="Contenu du document", alias="_source")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explication du scoring", alias="_explanation")
    
    class Config:
        populate_by_name = True


class SearchHits(BaseModel):
    """Container pour les résultats de recherche"""
    total: Dict[str, Any] = Field(..., description="Total des résultats")
    max_score: Optional[float] = Field(None, description="Score maximum")
    hits: List[SearchHit] = Field(..., description="Liste des résultats")


class SearchResponse(BaseModel):
    """Réponse complète du search_service"""
    hits: List[SearchHit] = Field(default_factory=list, description="Résultats de recherche")
    total_hits: int = Field(0, description="Nombre total de résultats")
    aggregations: Optional[Dict[str, Any]] = Field(None, description="Résultats d'agrégations")
    took_ms: int = Field(0, description="Temps d'exécution côté search service (ms)")
    query_id: Optional[str] = Field(None, description="ID de la requête pour tracing")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp de la réponse")
    
    @validator('total_hits')
    def validate_total_hits(cls, v):
        if v < 0:
            raise ValueError("total_hits cannot be negative")
        return v
    
    @validator('took_ms')
    def validate_took_ms(cls, v):
        if v < 0:
            raise ValueError("took_ms cannot be negative")
        return v


class SearchError(BaseModel):
    """Erreur de recherche"""
    error_type: str = Field(..., description="Type d'erreur")
    error_message: str = Field(..., description="Message d'erreur")
    query_id: Optional[str] = Field(None, description="ID de la requête")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = Field(None, description="Détails additionnels")


# Constantes pour validation
SUPPORTED_FIELD_TYPES = {
    "merchant_name": ["match", "term", "terms", "exists"],
    "primary_description": ["match", "exists"],
    "category_name": ["match", "term", "terms", "exists"],
    "operation_type": ["match", "term", "terms"],
    "transaction_type": ["term"],
    "account_id": ["term", "terms"],
    "account_type": ["term", "terms"],
    "user_id": ["term", "terms"],  # user_id can be used in filters
    "date": ["gte", "lte", "range"],
    "amount": ["gte", "lte", "range"],
    "amount_abs": ["gte", "lte", "range"],
}

ESSENTIAL_FIELDS = [
    "transaction_id",
    "amount",
    "amount_abs", 
    "merchant_name",
    "date",
    "primary_description",
    "category_name",
    "operation_type",
    "transaction_type"
]

MAX_AGGREGATION_BUCKETS = 20
MAX_NESTED_AGGREGATION_LEVELS = 3
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 1000

# Configuration par type d'intention
INTENT_QUERY_CONFIGS = {
    "SEARCH_BY_MERCHANT": {
        "required_entities": ["merchants"],
        "default_filters": ["transaction_type"],
        "recommended_aggregations": ["merchant_stats", "daily_breakdown"],
        "default_sort": [{"date": {"order": "desc"}}]
    },
    "SEARCH_BY_AMOUNT": {
        "required_entities": ["amounts"], 
        "default_filters": ["user_id"],
        "recommended_aggregations": ["amount_distribution"],
        "default_sort": [{"amount_abs": {"order": "desc"}}]
    },
    "SPENDING_ANALYSIS": {
        "required_entities": [],
        "default_filters": ["transaction_type"],
        "recommended_aggregations": ["category_breakdown", "monthly_spending"],
        "aggregation_only": True
    },
    "BALANCE_INQUIRY": {
        "required_entities": [],
        "default_filters": ["user_id"],
        "recommended_aggregations": ["balance_by_account"],
        "aggregation_only": True
    },
    "SEARCH_BY_DATE": {
        "required_entities": ["dates"],
        "default_filters": ["user_id"],
        "recommended_aggregations": ["daily_transactions"],
        "default_sort": [{"date": {"order": "desc"}}]
    },
    "SEARCH_BY_OPERATION_TYPE": {
        "required_entities": ["operation_types"],
        "default_filters": ["user_id"],
        "recommended_aggregations": ["operation_stats"],
        "default_sort": [{"date": {"order": "desc"}}]
    },
    "SEARCH_BY_CATEGORY": {
        "required_entities": ["categories"],
        "default_filters": ["user_id"],
        "recommended_aggregations": ["category_analysis"],
        "default_sort": [{"date": {"order": "desc"}}]
    },
    "TRANSACTION_HISTORY": {
        "required_entities": [],
        "default_filters": ["user_id"],
        "recommended_aggregations": [],
        "default_sort": [{"date": {"order": "desc"}}]
    },
    "COUNT_TRANSACTIONS": {
        "required_entities": [],
        "default_filters": ["user_id"],
        "recommended_aggregations": ["transaction_count"],
        "aggregation_only": True
    }
}