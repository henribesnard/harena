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

from pydantic import BaseModel, Field, validator, root_validator
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
    include_raw_score: bool = Field(default=False, description="Inclure le score brut")
    include_sort_values: bool = Field(default=False, description="Inclure les valeurs de tri")
    include_version: bool = Field(default=False, description="Inclure la version du document")
    highlight_fragment_size: PositiveInt = Field(default=150, description="Taille des fragments highlight")
    highlight_max_fragments: PositiveInt = Field(default=3, le=10, description="Nombre max de fragments")

class SearchOptions(BaseSearchOptions):
    """Options de recherche étendues pour les requêtes API."""
    query_options: QueryOptions = Field(default_factory=QueryOptions, description="Options de requête")
    result_options: ResultOptions = Field(default_factory=ResultOptions, description="Options de résultat")
    cache_options: CacheOptions = Field(default_factory=CacheOptions, description="Options de cache")

# ==================== REQUÊTES DE BASE ====================

class BaseSearchRequest(BaseModel):
    """Classe de base pour toutes les requêtes de recherche."""
    request_id: UUID = Field(default_factory=uuid4, description="ID unique de la requête")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de création")
    user_agent: Optional[str] = Field(None, description="User agent du client")
    source: str = Field(default="api", description="Source de la requête")
    
    class Config:
        use_enum_values = True

class LexicalSearchRequest(BaseSearchRequest):
    """
    Requête de recherche lexicale principale.
    
    Cette requête est utilisée pour toutes les recherches lexicales
    via l'endpoint POST /search/lexical.
    """
    # Paramètres obligatoires
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    query_text: Optional[str] = Field(None, min_length=1, max_length=1000, description="Texte de recherche")
    
    # Paramètres de recherche
    query_type: QueryType = Field(default=QueryType.FILTERED_SEARCH, description="Type de requête")
    fields: List[str] = Field(
        default=["user_id", "category_name", "merchant_name", "primary_description", "amount", "date"],
        min_items=1,
        description="Champs à récupérer"
    )
    
    # Pagination et limites
    limit: PositiveInt = Field(
        default=settings.DEFAULT_LIMIT,
        le=settings.MAX_SEARCH_RESULTS,
        description="Nombre maximum de résultats"
    )
    offset: NonNegativeInt = Field(default=0, description="Décalage pour pagination")
    
    # Timeout
    timeout_seconds: PositiveInt = Field(
        default=settings.SEARCH_TIMEOUT,
        le=settings.MAX_SEARCH_TIMEOUT,
        description="Timeout en secondes"
    )
    
    # Filtres
    filters: FilterGroup = Field(default_factory=FilterGroup, description="Filtres de recherche")
    
    # Agrégations
    aggregations: Optional[AggregationRequest] = Field(None, description="Demandes d'agrégation")
    
    # Options
    options: SearchOptions = Field(default_factory=SearchOptions, description="Options de recherche")
    
    # Scoring et pertinence
    min_score: Optional[float] = Field(None, ge=0.0, description="Score minimum requis")
    boost_query: Optional[str] = Field(None, description="Requête de boost personnalisée")
    
    @validator('fields')
    def validate_fields(cls, v):
        """Valide les champs demandés."""
        valid_fields = [
            "user_id", "account_id", "transaction_id",
            "amount", "amount_abs", "currency_code",
            "transaction_type", "operation_type", "date",
            "primary_description", "merchant_name", "category_name",
            "month_year", "weekday", "searchable_text"
        ]
        
        for field in v:
            if field != "*" and field not in valid_fields:
                raise ValueError(f"Champ invalide: {field}")
        
        return v
    
    @root_validator
    def validate_request_consistency(cls, values):
        """Valide la cohérence globale de la requête."""
        query_type = values.get('query_type')
        query_text = values.get('query_text')
        filters = values.get('filters')
        
        # Validation cohérence query_type et query_text
        if query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER]:
            if not query_text:
                raise ValueError("query_text requis pour les recherches textuelles")
        
        # Validation sécurité: user_id obligatoire en filtre
        if filters:
            user_filter_exists = any(
                f.field == "user_id" and f.operator == FilterOperator.EQ
                for f in filters.required
            )
            if not user_filter_exists:
                # Ajouter automatiquement le filtre user_id
                filters.required.append(
                    SearchFilter(
                        field="user_id",
                        operator=FilterOperator.EQ,
                        value=values.get('user_id')
                    )
                )
        
        return values

class QueryValidationRequest(BaseSearchRequest):
    """Requête de validation d'une requête Elasticsearch."""
    query: Dict[str, Any] = Field(..., description="Requête Elasticsearch à valider")
    explain: bool = Field(default=False, description="Inclure l'explication")
    rewrite: bool = Field(default=False, description="Inclure la réécriture de requête")

class TemplateListRequest(BaseSearchRequest):
    """Requête pour lister les templates disponibles."""
    category: Optional[str] = Field(None, description="Catégorie de templates")
    intent_type: Optional[str] = Field(None, description="Type d'intention")
    include_examples: bool = Field(default=False, description="Inclure des exemples")

class HealthCheckRequest(BaseSearchRequest):
    """Requête de vérification de santé."""
    include_elasticsearch: bool = Field(default=True, description="Inclure le statut Elasticsearch")
    include_cache: bool = Field(default=True, description="Inclure le statut du cache")
    include_metrics: bool = Field(default=False, description="Inclure les métriques")
    detailed: bool = Field(default=False, description="Informations détaillées")

class MetricsRequest(BaseSearchRequest):
    """Requête pour récupérer les métriques."""
    time_range: Optional[str] = Field(None, description="Plage temporelle (1h, 24h, 7d)")
    metric_types: List[str] = Field(
        default=["performance", "usage", "errors"],
        description="Types de métriques"
    )
    aggregation_interval: Optional[str] = Field(None, description="Intervalle d'agrégation")
    include_raw_data: bool = Field(default=False, description="Inclure les données brutes")

# ==================== REQUÊTES SPÉCIALISÉES ====================

class BulkSearchRequest(BaseSearchRequest):
    """Requête de recherche en lot."""
    searches: List[LexicalSearchRequest] = Field(
        ..., min_items=1, max_items=10, description="Recherches à exécuter"
    )
    parallel: bool = Field(default=True, description="Exécution en parallèle")
    fail_fast: bool = Field(default=False, description="Arrêter au premier échec")
    
    @validator('searches')
    def validate_searches_consistency(cls, v):
        """Valide la cohérence des recherches en lot."""
        user_ids = {search.user_id for search in v}
        if len(user_ids) > 1:
            raise ValueError("Toutes les recherches doivent avoir le même user_id")
        return v

class AggregationOnlyRequest(BaseSearchRequest):
    """Requête d'agrégation sans résultats détaillés."""
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    filters: FilterGroup = Field(default_factory=FilterGroup, description="Filtres")
    aggregations: AggregationRequest = Field(..., description="Agrégations à calculer")
    
    timeout_seconds: PositiveInt = Field(
        default=settings.SEARCH_TIMEOUT,
        le=settings.MAX_SEARCH_TIMEOUT,
        description="Timeout en secondes"
    )

class AutocompleteRequest(BaseSearchRequest):
    """Requête d'autocomplétion."""
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    query_text: str = Field(..., min_length=1, max_length=100, description="Texte partiel")
    field: str = Field(..., description="Champ à compléter")
    limit: PositiveInt = Field(default=10, le=50, description="Nombre de suggestions")
    
    @validator('field')
    def validate_autocomplete_field(cls, v):
        """Valide que le champ supporte l'autocomplétion."""
        valid_fields = ["merchant_name", "category_name", "primary_description"]
        if v not in valid_fields:
            raise ValueError(f"Autocomplétion non supportée pour le champ: {v}")
        return v

# ==================== VALIDATION ET HELPERS ====================

class RequestValidationError(Exception):
    """Exception pour les erreurs de validation de requête."""
    pass

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
            request.dict()
            
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
            request.dict()
            
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
        user_id: ID de l'utilisateur
        query_text: Texte de recherche optionnel
        query_type: Type de requête
        **kwargs: Paramètres additionnels
        
    Returns:
        LexicalSearchRequest configurée
    """
    # Filtres avec user_id obligatoire
    filters = FilterGroup(
        required=[
            SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)
        ]
    )
    
    # Ajout des filtres personnalisés
    if 'filters' in kwargs:
        custom_filters = kwargs.pop('filters')
        if 'required' in custom_filters:
            filters.required.extend(custom_filters['required'])
        if 'optional' in custom_filters:
            filters.optional.extend(custom_filters['optional'])
        if 'ranges' in custom_filters:
            filters.ranges.extend(custom_filters['ranges'])
        if 'text_search' in custom_filters:
            filters.text_search = custom_filters['text_search']
    
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
    """
    Factory pour créer une AggregationOnlyRequest.
    
    Args:
        user_id: ID de l'utilisateur
        aggregation_types: Types d'agrégation
        group_by: Champs de groupement
        **kwargs: Paramètres additionnels
        
    Returns:
        AggregationOnlyRequest configurée
    """
    # Filtres avec user_id obligatoire
    filters = FilterGroup(
        required=[
            SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)
        ]
    )
    
    # Agrégations
    aggregations = AggregationRequest(
        enabled=True,
        types=aggregation_types,
        group_by=group_by,
        metrics=kwargs.get('metrics', ['amount_abs', 'transaction_id']),
        size=kwargs.get('size', 10)
    )
    
    return AggregationOnlyRequest(
        user_id=user_id,
        filters=filters,
        aggregations=aggregations,
        timeout_seconds=kwargs.get('timeout_seconds', settings.SEARCH_TIMEOUT)
    )

# ==================== UTILITAIRES ====================

def validate_search_request(request: LexicalSearchRequest) -> bool:
    """
    Fonction utilitaire pour valider une requête de recherche.
    
    Args:
        request: La requête à valider
        
    Returns:
        True si valide
        
    Raises:
        RequestValidationError: Si la validation échoue
    """
    return RequestValidator.validate_search_request(request)

def extract_user_id_from_request(request: BaseSearchRequest) -> Optional[int]:
    """
    Extrait l'user_id d'une requête.
    
    Args:
        request: La requête
        
    Returns:
        L'user_id si trouvé, None sinon
    """
    if hasattr(request, 'user_id'):
        return request.user_id
    
    # Chercher dans les filtres
    if hasattr(request, 'filters') and request.filters:
        for filter_item in request.filters.required:
            if filter_item.field == "user_id":
                return filter_item.value
    
    return None

def get_request_timeout_ms(request: BaseSearchRequest) -> int:
    """
    Récupère le timeout en millisecondes d'une requête.
    
    Args:
        request: La requête
        
    Returns:
        Timeout en millisecondes
    """
    if hasattr(request, 'timeout_seconds'):
        return request.timeout_seconds * 1000
    
    return settings.SEARCH_TIMEOUT * 1000

def is_text_search_request(request: LexicalSearchRequest) -> bool:
    """
    Détermine si une requête est une recherche textuelle.
    
    Args:
        request: La requête
        
    Returns:
        True si c'est une recherche textuelle
    """
    return (
        request.query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER] or
        (request.query_text is not None and len(request.query_text.strip()) > 0) or
        (request.filters.text_search is not None)
    )

def get_search_fields_for_request(request: LexicalSearchRequest) -> List[str]:
    """
    Détermine les champs de recherche appropriés pour une requête.
    
    Args:
        request: La requête
        
    Returns:
        Liste des champs de recherche
    """
    if request.fields and request.fields != ["*"]:
        return request.fields
    
    # Champs par défaut selon le type de requête
    if is_text_search_request(request):
        return [
            "searchable_text", "primary_description", "merchant_name",
            "category_name", "amount", "date", "user_id"
        ]
    
    return [
        "user_id", "transaction_id", "amount", "amount_abs",
        "date", "category_name", "merchant_name", "primary_description"
    ]

# ==================== CONSTANTES ET MAPPINGS ====================

# Mapping des types de requête vers les champs recommandés
QUERY_TYPE_FIELD_MAPPING = {
    QueryType.FILTERED_SEARCH: [
        "user_id", "transaction_id", "amount", "date",
        "category_name", "merchant_name", "primary_description"
    ],
    QueryType.TEXT_SEARCH: [
        "searchable_text", "primary_description", "merchant_name",
        "user_id", "amount", "date", "score"
    ],
    QueryType.AGGREGATION_ONLY: [
        "user_id", "amount", "amount_abs", "category_name",
        "merchant_name", "month_year", "transaction_id"
    ],
    QueryType.FILTERED_AGGREGATION: [
        "user_id", "amount", "amount_abs", "category_name",
        "merchant_name", "date", "month_year"
    ],
    QueryType.TEMPORAL_AGGREGATION: [
        "user_id", "date", "month_year", "weekday",
        "amount", "amount_abs", "transaction_id"
    ],
    QueryType.TEXT_SEARCH_WITH_FILTER: [
        "searchable_text", "primary_description", "merchant_name",
        "category_name", "user_id", "amount", "date"
    ]
}

# Mapping des intentions vers les types de requête recommandés
INTENT_TO_QUERY_TYPE_MAPPING = {
    "SEARCH_BY_CATEGORY": QueryType.FILTERED_SEARCH,
    "SEARCH_BY_MERCHANT": QueryType.FILTERED_SEARCH,
    "SEARCH_BY_AMOUNT": QueryType.FILTERED_SEARCH,
    "SEARCH_BY_DATE": QueryType.FILTERED_SEARCH,
    "TEXT_SEARCH": QueryType.TEXT_SEARCH,
    "COUNT_OPERATIONS": QueryType.AGGREGATION_ONLY,
    "TEMPORAL_ANALYSIS": QueryType.TEMPORAL_AGGREGATION,
    "CATEGORY_BREAKDOWN": QueryType.FILTERED_AGGREGATION,
    "TEXT_SEARCH_WITH_CATEGORY": QueryType.TEXT_SEARCH_WITH_FILTER
}

# Limites par type de requête
QUERY_TYPE_LIMITS = {
    QueryType.FILTERED_SEARCH: {
        "max_results": settings.MAX_SEARCH_RESULTS,
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