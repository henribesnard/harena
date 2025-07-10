"""
Modèles de requête pour l'API du Search Service.

Ces modèles définissent les structures de données pour toutes les requêtes
entrantes de l'API REST du Search Service, avec validation stricte et
configuration centralisée.

ARCHITECTURE:
- LexicalSearchRequest: Requête de recherche lexicale principale
- Validation stricte avec Pydantic v2
- Options configurables via config_service
- Support des filtres et agrégations complexes
- Gestion des timeouts et limites

CONFIGURATION CENTRALISÉE:
- Toutes les limites via config_service
- Validation basée sur les paramètres configurés
- Timeouts dynamiques selon l'environnement
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Literal
from uuid import UUID, uuid4
from enum import Enum

# CORRECTION PYDANTIC V2: Remplacer root_validator par model_validator
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveInt, NonNegativeInt, NonNegativeFloat

# Configuration centralisée
from config_service.config import settings

# Import des contrats pour réutilisation
from .service_contracts import (
    QueryType, FilterOperator, AggregationType,
    SearchFilter, FilterGroup, AggregationRequest,
    SearchOptions as BaseSearchOptions
)

# ==================== OPTIONS ET PARAMÈTRES ====================

class CacheOptions(BaseModel):
    """Options de cache pour les requêtes."""
    enabled: bool = Field(default=True, description="Activer le cache")
    ttl_seconds: Optional[PositiveInt] = Field(
        default=settings.SEARCH_CACHE_TTL,
        description="TTL du cache en secondes"
    )
    cache_key_prefix: Optional[str] = Field(None, description="Préfixe de clé de cache")
    force_refresh: bool = Field(default=False, description="Forcer le rafraîchissement")

class QueryOptions(BaseModel):
    """Options spécifiques aux requêtes Elasticsearch."""
    analyzer: Optional[str] = Field(None, description="Analyseur à utiliser")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Niveau de fuzziness")
    boost_recent: bool = Field(default=False, description="Boost pour les résultats récents")
    boost_factor: float = Field(default=1.0, ge=0.1, le=10.0, description="Facteur de boost")
    minimum_should_match: Optional[str] = Field(None, description="Minimum should match")
    tie_breaker: float = Field(default=0.3, ge=0.0, le=1.0, description="Tie breaker")

class ResultOptions(BaseModel):
    """Options de formatage des résultats."""
    include_score: bool = Field(default=True, description="Inclure le score de pertinence")
    include_highlights: bool = Field(default=False, description="Inclure les highlights")
    include_explanation: bool = Field(default=False, description="Inclure l'explication du score")
    include_source: bool = Field(default=True, description="Inclure les données source")
    source_fields: Optional[List[str]] = Field(None, description="Champs source spécifiques")

class SearchOptions(BaseModel):
    """Options de recherche étendues."""
    query_options: QueryOptions = Field(default_factory=QueryOptions, description="Options de requête")
    result_options: ResultOptions = Field(default_factory=ResultOptions, description="Options de résultat")
    cache_options: CacheOptions = Field(default_factory=CacheOptions, description="Options de cache")

# ==================== REQUÊTES PRINCIPALES ====================

class LexicalSearchRequest(BaseModel):
    """
    Requête de recherche lexicale principale.
    
    Cette requête supporte toutes les fonctionnalités de recherche
    lexicale avec Elasticsearch, incluant filtres, agrégations et options.
    """
    # Identification et sécurité
    user_id: PositiveInt = Field(..., description="ID utilisateur (sécurité obligatoire)")
    query_id: UUID = Field(default_factory=uuid4, description="ID unique de requête")
    
    # Paramètres de recherche principaux
    query_text: Optional[str] = Field(None, description="Texte de recherche libre")
    query_type: QueryType = Field(default=QueryType.FILTERED_SEARCH, description="Type de requête")
    
    # Filtres et agrégations
    filters: FilterGroup = Field(..., description="Groupe de filtres")
    aggregations: AggregationRequest = Field(default_factory=AggregationRequest, description="Agrégations")
    
    # Pagination et limites
    limit: PositiveInt = Field(default=20, description="Nombre de résultats")
    offset: NonNegativeInt = Field(default=0, description="Décalage pour pagination")
    
    # Performance et timeout
    timeout_seconds: float = Field(default=15.0, description="Timeout en secondes")
    
    # Options avancées
    options: SearchOptions = Field(default_factory=SearchOptions, description="Options de recherche")
    
    # Métadonnées contextuelles
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexte de requête")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Valide l'user_id."""
        if v <= 0:
            raise ValueError("user_id doit être positif")
        return v
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        """Valide la limite de résultats."""
        if v > settings.MAX_SEARCH_RESULTS:
            raise ValueError(f"Limite trop élevée (max {settings.MAX_SEARCH_RESULTS})")
        return v
    
    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v):
        """Valide le timeout."""
        if v > settings.MAX_SEARCH_TIMEOUT:
            raise ValueError(f"Timeout trop élevé (max {settings.MAX_SEARCH_TIMEOUT}s)")
        return v
    
    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v):
        """Valide le texte de requête."""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
            if len(v) > 1000:
                raise ValueError("query_text trop long (max 1000 caractères)")
        return v
    
    @model_validator(mode='after')
    def validate_request_coherence(self):
        """Valide la cohérence de la requête."""
        # Validation sécurité user_id en filtre
        user_filter_exists = any(
            f.field == "user_id" for f in self.filters.required
        )
        if not user_filter_exists:
            raise ValueError("Filtre user_id obligatoire dans filters.required")
        
        # Validation cohérence query_type
        if self.query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER]:
            if not self.query_text or len(self.query_text.strip()) == 0:
                raise ValueError("query_text requis pour les recherches textuelles")
        
        # Validation agrégations
        if self.query_type == QueryType.AGGREGATION_ONLY:
            if not self.aggregations.enabled:
                raise ValueError("Agrégations requises pour AGGREGATION_ONLY")
        
        # Validation pagination
        if self.offset + self.limit > settings.MAX_SEARCH_RESULTS:
            raise ValueError("offset + limit dépasse MAX_SEARCH_RESULTS")
        
        return self

class QueryValidationRequest(BaseModel):
    """Requête de validation de requête."""
    elasticsearch_query: Dict[str, Any] = Field(..., description="Requête Elasticsearch à valider")
    strict_validation: bool = Field(default=True, description="Validation stricte")
    
    @field_validator('elasticsearch_query')
    @classmethod
    def validate_es_query(cls, v):
        """Valide la structure de base de la requête ES."""
        if not isinstance(v, dict):
            raise ValueError("elasticsearch_query doit être un dictionnaire")
        if not v:
            raise ValueError("elasticsearch_query ne peut pas être vide")
        return v

class TemplateListRequest(BaseModel):
    """Requête pour lister les templates disponibles."""
    category: Optional[str] = Field(None, description="Catégorie de templates")
    include_deprecated: bool = Field(default=False, description="Inclure les templates dépréciés")

class HealthCheckRequest(BaseModel):
    """Requête de vérification de santé."""
    include_detailed: bool = Field(default=False, description="Inclure détails complets")
    check_elasticsearch: bool = Field(default=True, description="Vérifier Elasticsearch")
    check_cache: bool = Field(default=True, description="Vérifier le cache")

class MetricsRequest(BaseModel):
    """Requête de métriques."""
    time_range: Optional[str] = Field(None, description="Période (1h, 24h, 7d)")
    metric_types: List[str] = Field(default_factory=list, description="Types de métriques")
    include_performance: bool = Field(default=True, description="Inclure métriques performance")

# ==================== REQUÊTES SPÉCIALISÉES ====================

class BulkSearchRequest(BaseModel):
    """Requête de recherche en lot."""
    requests: List[LexicalSearchRequest] = Field(..., description="Liste de requêtes")
    batch_size: PositiveInt = Field(default=10, description="Taille du lot")
    parallel_execution: bool = Field(default=True, description="Exécution parallèle")
    
    @field_validator('requests')
    @classmethod
    def validate_requests(cls, v):
        """Valide la liste de requêtes."""
        if not v:
            raise ValueError("Au moins une requête requise")
        if len(v) > 100:
            raise ValueError("Maximum 100 requêtes par lot")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        """Valide la taille du lot."""
        if v > settings.DEFAULT_BATCH_SIZE:
            raise ValueError(f"batch_size trop élevé (max {settings.DEFAULT_BATCH_SIZE})")
        return v

class AggregationOnlyRequest(BaseModel):
    """Requête d'agrégation sans résultats de recherche."""
    user_id: PositiveInt = Field(..., description="ID utilisateur")
    filters: FilterGroup = Field(..., description="Filtres pour l'agrégation")
    aggregations: AggregationRequest = Field(..., description="Agrégations à calculer")
    timeout_seconds: float = Field(default=30.0, description="Timeout en secondes")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Valide l'user_id."""
        if v <= 0:
            raise ValueError("user_id doit être positif")
        return v
    
    @model_validator(mode='after')
    def validate_aggregation_request(self):
        """Valide la requête d'agrégation."""
        # Validation sécurité
        user_filter_exists = any(
            f.field == "user_id" for f in self.filters.required
        )
        if not user_filter_exists:
            raise ValueError("Filtre user_id obligatoire")
        
        # Validation agrégations
        if not self.aggregations.enabled:
            raise ValueError("Agrégations doivent être activées")
        
        if not self.aggregations.types:
            raise ValueError("Au moins un type d'agrégation requis")
        
        return self

class AutocompleteRequest(BaseModel):
    """Requête d'autocomplétion."""
    user_id: PositiveInt = Field(..., description="ID utilisateur")
    query_prefix: str = Field(..., description="Préfixe de recherche")
    field: str = Field(..., description="Champ pour autocomplétion")
    limit: PositiveInt = Field(default=10, description="Nombre de suggestions")
    
    @field_validator('query_prefix')
    @classmethod
    def validate_query_prefix(cls, v):
        """Valide le préfixe de requête."""
        if not v or len(v.strip()) < 2:
            raise ValueError("query_prefix doit faire au moins 2 caractères")
        if len(v) > 100:
            raise ValueError("query_prefix trop long (max 100 caractères)")
        return v.strip()
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        """Valide la limite de suggestions."""
        if v > 50:
            raise ValueError("Maximum 50 suggestions")
        return v

# ==================== VALIDATION ET ERREURS ====================

class RequestValidationError(Exception):
    """Erreur de validation de requête."""
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(message)

class RequestValidator:
    """Validateur pour les requêtes de recherche."""
    
    @staticmethod
    def validate_search_request(request: LexicalSearchRequest) -> bool:
        """
        Valide une requête de recherche lexicale.
        
        Args:
            request: La requête à valider
            
        Returns:
            True si valide
            
        Raises:
            RequestValidationError: Si la validation échoue
        """
        try:
            # Validation de base Pydantic
            request.model_dump()
            
            # Validations métier spécifiques
            if request.user_id <= 0:
                raise RequestValidationError("user_id doit être positif")
            
            # Validation sécurité user_id en filtre
            user_filter_exists = any(
                f.field == "user_id" for f in request.filters.required
            )
            if not user_filter_exists:
                raise RequestValidationError("Filtre user_id obligatoire")
            
            # Validation limites
            if request.limit > settings.MAX_SEARCH_RESULTS:
                raise RequestValidationError(f"Limite trop élevée (max {settings.MAX_SEARCH_RESULTS})")
            
            if request.timeout_seconds > settings.MAX_SEARCH_TIMEOUT:
                raise RequestValidationError(f"Timeout trop élevé (max {settings.MAX_SEARCH_TIMEOUT}s)")
            
            # Validation cohérence query_type
            if request.query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER]:
                if not request.query_text:
                    raise RequestValidationError("query_text requis pour les recherches textuelles")
            
            return True
            
        except Exception as e:
            raise RequestValidationError(f"Validation échouée: {str(e)}")
    
    @staticmethod
    def validate_aggregation_request(request: AggregationOnlyRequest) -> bool:
        """Valide une requête d'agrégation."""
        try:
            request.model_dump()
            
            if request.user_id <= 0:
                raise RequestValidationError("user_id doit être positif")
            
            if not request.aggregations.enabled:
                raise RequestValidationError("Agrégations doivent être activées")
            
            if not request.aggregations.types:
                raise RequestValidationError("Au moins un type d'agrégation requis")
            
            return True
            
        except Exception as e:
            raise RequestValidationError(f"Validation échouée: {str(e)}")

# ==================== FACTORY FUNCTIONS ====================

def create_lexical_search_request(
    user_id: int,
    query_text: Optional[str] = None,
    query_type: QueryType = QueryType.FILTERED_SEARCH,
    **kwargs
) -> LexicalSearchRequest:
    """
    Factory pour créer une LexicalSearchRequest avec des valeurs par défaut.
    
    Args:
        user_id: ID utilisateur
        query_text: Texte de recherche optionnel
        query_type: Type de requête
        **kwargs: Autres paramètres
        
    Returns:
        Requête de recherche lexicale configurée
    """
    # Filtre user_id obligatoire
    filters = kwargs.get('filters') or FilterGroup(
        required=[SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)]
    )
    
    # Ajouter user_id si pas présent
    user_filter_exists = any(f.field == "user_id" for f in filters.required)
    if not user_filter_exists:
        filters.required.append(
            SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)
        )
    
    return LexicalSearchRequest(
        user_id=user_id,
        query_text=query_text,
        query_type=query_type,
        filters=filters,
        **kwargs
    )

def create_aggregation_request(
    user_id: int,
    aggregation_types: List[AggregationType],
    group_by: List[str],
    **kwargs
) -> AggregationOnlyRequest:
    """Factory pour créer une requête d'agrégation."""
    filters = kwargs.get('filters') or FilterGroup(
        required=[SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)]
    )
    
    aggregations = AggregationRequest(
        enabled=True,
        types=aggregation_types,
        group_by=group_by,
        metrics=kwargs.get('metrics', [])
    )
    
    return AggregationOnlyRequest(
        user_id=user_id,
        filters=filters,
        aggregations=aggregations,
        **kwargs
    )

# ==================== UTILITAIRES ====================

def extract_user_id_from_request(request: Union[LexicalSearchRequest, AggregationOnlyRequest]) -> int:
    """Extrait l'user_id d'une requête."""
    return request.user_id

def get_request_timeout_ms(request: Union[LexicalSearchRequest, AggregationOnlyRequest]) -> int:
    """Retourne le timeout en millisecondes."""
    if hasattr(request, 'timeout_seconds'):
        return int(request.timeout_seconds * 1000)
    return settings.DEFAULT_TIMEOUT * 1000

def is_text_search_request(request: LexicalSearchRequest) -> bool:
    """Vérifie si c'est une requête de recherche textuelle."""
    return request.query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER]

def get_search_fields_for_request(request: LexicalSearchRequest) -> List[str]:
    """Retourne les champs de recherche pour une requête."""
    if request.options.result_options.source_fields:
        return request.options.result_options.source_fields
    
    # Champs par défaut pour recherche financière
    return [
        "user_id", "transaction_id", "account_id",
        "amount", "amount_abs", "currency_code",
        "date", "month_year", "weekday",
        "primary_description", "merchant_name", "category_name",
        "operation_type", "transaction_type", "searchable_text"
    ]

# ==================== CONSTANTES ET MAPPINGS ====================

QUERY_TYPE_FIELD_MAPPING = {
    QueryType.FILTERED_SEARCH: ["user_id", "category_name", "merchant_name", "amount"],
    QueryType.TEXT_SEARCH: ["primary_description", "merchant_name", "searchable_text"],
    QueryType.AGGREGATION_ONLY: ["user_id", "category_name", "month_year"],
    QueryType.FILTERED_AGGREGATION: ["user_id", "category_name", "merchant_name", "month_year"],
    QueryType.TEMPORAL_AGGREGATION: ["user_id", "date", "month_year", "weekday"],
    QueryType.TEXT_SEARCH_WITH_FILTER: ["primary_description", "merchant_name", "searchable_text", "category_name"]
}

INTENT_TO_QUERY_TYPE_MAPPING = {
    "search_by_category": QueryType.FILTERED_SEARCH,
    "search_by_merchant": QueryType.FILTERED_SEARCH,
    "search_by_description": QueryType.TEXT_SEARCH,
    "aggregate_by_category": QueryType.AGGREGATION_ONLY,
    "aggregate_by_month": QueryType.TEMPORAL_AGGREGATION
}

QUERY_TYPE_LIMITS = {
    QueryType.FILTERED_SEARCH: {
        "max_results": min(settings.MAX_SEARCH_RESULTS, 200),
        "default_timeout": settings.SEARCH_TIMEOUT,
        "max_filters": 20
    },
    QueryType.TEXT_SEARCH: {
        "max_results": min(settings.MAX_SEARCH_RESULTS, 100),
        "default_timeout": settings.SEARCH_TIMEOUT + 2,
        "max_filters": 10
    },
    QueryType.AGGREGATION_ONLY: {
        "max_results": 0,
        "default_timeout": settings.SEARCH_TIMEOUT + 3,
        "max_filters": 30
    }
}

__all__ = [
    # Requêtes principales
    "LexicalSearchRequest",
    "QueryValidationRequest",
    "TemplateListRequest",
    "HealthCheckRequest",
    "MetricsRequest",
    
    # Requêtes spécialisées
    "BulkSearchRequest",
    "AggregationOnlyRequest",
    "AutocompleteRequest",
    
    # Options
    "SearchOptions",
    "QueryOptions",
    "ResultOptions",
    "CacheOptions",
    
    # Validation
    "RequestValidationError",
    "RequestValidator",
    "validate_search_request",
    
    # Factory functions
    "create_lexical_search_request",
    "create_aggregation_request",
    
    # Utilitaires
    "extract_user_id_from_request",
    "get_request_timeout_ms",
    "is_text_search_request",
    "get_search_fields_for_request",
    
    # Constantes et mappings
    "QUERY_TYPE_FIELD_MAPPING",
    "INTENT_TO_QUERY_TYPE_MAPPING",
    "QUERY_TYPE_LIMITS"
]