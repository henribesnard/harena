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
    include_source: bool = Field(default=True, description="Inclure le document source")
    include_highlights: bool = Field(default=True, description="Inclure les highlights")
    highlight_fragment_size: int = Field(default=150, description="Taille des fragments highlight")
    highlight_max_fragments: int = Field(default=3, description="Nombre max de fragments")
    source_includes: Optional[List[str]] = Field(None, description="Champs source à inclure")
    source_excludes: Optional[List[str]] = Field(None, description="Champs source à exclure")

class SearchOptions(BaseSearchOptions):
    """Options de recherche étendues pour l'API."""
    # Hérite des options de base des contrats
    
    # Options spécifiques à l'API
    cache: CacheOptions = Field(default_factory=CacheOptions, description="Options de cache")
    query: QueryOptions = Field(default_factory=QueryOptions, description="Options de requête")
    results: ResultOptions = Field(default_factory=ResultOptions, description="Options de résultats")
    
    # Performance et monitoring
    enable_profiling: bool = Field(default=False, description="Activer le profiling")
    track_total_hits: bool = Field(default=True, description="Tracker le nombre total")
    request_cache: bool = Field(default=True, description="Utiliser le cache de requête ES")

# ==================== MODÈLES DE REQUÊTE ====================

class BaseLexicalRequest(BaseModel):
    """Modèle de base pour toutes les requêtes lexicales."""
    # Identification
    request_id: UUID = Field(default_factory=uuid4, description="ID unique de la requête")
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    
    # Configuration
    timeout_seconds: int = Field(
        default=settings.DEFAULT_SEARCH_TIMEOUT,
        le=settings.MAX_SEARCH_TIMEOUT,
        description="Timeout en secondes"
    )
    
    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v):
        """Valide le timeout."""
        if v <= 0:
            raise ValueError("Le timeout doit être positif")
        return v

class LexicalSearchRequest(BaseLexicalRequest):
    """
    Requête de recherche lexicale principale.
    
    Modèle complet pour toutes les recherches lexicales avec support
    des filtres, agrégations et options avancées.
    """
    # Contenu de recherche
    query_type: QueryType = Field(default=QueryType.FILTERED_SEARCH, description="Type de requête")
    query_text: Optional[str] = Field(None, description="Texte de recherche")
    
    # Filtres et agrégations
    filters: FilterGroup = Field(default_factory=FilterGroup, description="Filtres de recherche")
    aggregations: AggregationRequest = Field(default_factory=AggregationRequest, description="Agrégations")
    
    # Pagination
    offset: NonNegativeInt = Field(default=0, description="Offset pour pagination")
    limit: PositiveInt = Field(
        default=settings.DEFAULT_SEARCH_LIMIT,
        le=settings.MAX_SEARCH_RESULTS,
        description="Nombre de résultats"
    )
    
    # Options
    options: SearchOptions = Field(default_factory=SearchOptions, description="Options de recherche")
    
    # Contexte
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexte additionnel")
    
    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v):
        """Valide le texte de requête."""
        if v is not None:
            if len(v.strip()) == 0:
                raise ValueError("Le texte de requête ne peut pas être vide")
            if len(v) > settings.MAX_QUERY_LENGTH:
                raise ValueError(f"Texte trop long (max {settings.MAX_QUERY_LENGTH} caractères)")
        return v
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        """Valide la limite de résultats."""
        if v <= 0:
            raise ValueError("La limite doit être positive")
        return v
    
    @model_validator(mode='after')
    def validate_search_request(self):
        """Valide la cohérence de la requête de recherche."""
        # Validation selon le type de requête
        if self.query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER]:
            if not self.query_text:
                raise ValueError("query_text requis pour les recherches textuelles")
        
        if self.query_type == QueryType.AGGREGATION_ONLY:
            if not self.aggregations.enabled:
                raise ValueError("Agrégations requises pour AGGREGATION_ONLY")
        
        # Validation sécurité: user_id obligatoire en filtre
        user_filter_exists = any(
            f.field == "user_id" for f in self.filters.required
        )
        if not user_filter_exists:
            # Ajouter automatiquement le filtre user_id pour la sécurité
            user_filter = SearchFilter(
                field="user_id",
                operator=FilterOperator.EQ,
                value=self.user_id
            )
            self.filters.required.append(user_filter)
        
        # Validation pagination
        if self.offset + self.limit > settings.MAX_SEARCH_OFFSET:
            raise ValueError(f"Pagination trop élevée (max {settings.MAX_SEARCH_OFFSET})")
        
        return self

class AggregationOnlyRequest(BaseLexicalRequest):
    """
    Requête d'agrégation uniquement sans résultats de recherche.
    
    Optimisée pour les analyses et statistiques pures sans récupération
    de documents.
    """
    # Filtres pour limiter l'agrégation
    filters: FilterGroup = Field(default_factory=FilterGroup, description="Filtres pour agrégation")
    
    # Agrégations obligatoires
    aggregations: AggregationRequest = Field(..., description="Configuration d'agrégation")
    
    # Options spécifiques
    include_empty_buckets: bool = Field(default=False, description="Inclure les buckets vides")
    bucket_sort: Optional[Dict[str, str]] = Field(None, description="Tri des buckets")
    
    @model_validator(mode='after')
    def validate_aggregation_request(self):
        """Valide la requête d'agrégation."""
        if not self.aggregations.enabled:
            raise ValueError("Agrégations doivent être activées")
        
        if not self.aggregations.types:
            raise ValueError("Au moins un type d'agrégation requis")
        
        # Validation sécurité user_id
        user_filter_exists = any(
            f.field == "user_id" for f in self.filters.required
        )
        if not user_filter_exists:
            user_filter = SearchFilter(
                field="user_id",
                operator=FilterOperator.EQ,
                value=self.user_id
            )
            self.filters.required.append(user_filter)
        
        return self

class HealthCheckRequest(BaseModel):
    """Requête de vérification de santé du service."""
    check_elasticsearch: bool = Field(default=True, description="Vérifier Elasticsearch")
    check_cache: bool = Field(default=True, description="Vérifier le cache")
    include_stats: bool = Field(default=False, description="Inclure les statistiques")
    timeout_seconds: int = Field(default=5, le=30, description="Timeout de vérification")

class QueryValidationRequest(BaseModel):
    """Requête de validation de requête de recherche."""
    query_text: str = Field(..., description="Texte de la requête à valider")
    query_type: QueryType = Field(default=QueryType.FILTERED_SEARCH, description="Type de requête")
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    strict_validation: bool = Field(default=False, description="Validation stricte")
    include_suggestions: bool = Field(default=True, description="Inclure des suggestions")
    
    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v):
        """Valide le texte de requête."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Le texte de requête ne peut pas être vide")
        if len(v) > settings.MAX_QUERY_LENGTH:
            raise ValueError(f"Texte trop long (max {settings.MAX_QUERY_LENGTH} caractères)")
        return v.strip()

class MetricsRequest(BaseModel):
    """Requête de métriques du service."""
    period: Literal["1h", "24h", "7d", "30d"] = Field(default="1h", description="Période des métriques")
    include_detailed: bool = Field(default=False, description="Inclure les détails")
    metrics_types: List[str] = Field(
        default_factory=lambda: ["search", "performance", "errors"],
        description="Types de métriques"
    )

# ==================== VALIDATION ET UTILITAIRES ====================

class RequestValidationError(ValueError):
    """Erreur de validation de requête."""
    def __init__(self, message: str, field: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.field = field
        self.details = details or {}

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
        user_id: ID de l'utilisateur
        query_text: Texte de recherche optionnel
        query_type: Type de requête
        **kwargs: Autres paramètres
    
    Returns:
        LexicalSearchRequest configurée
    """
    # Filtres par défaut avec user_id pour sécurité
    default_filters = FilterGroup(
        required=[SearchFilter(
            field="user_id",
            operator=FilterOperator.EQ,
            value=user_id
        )]
    )
    
    return LexicalSearchRequest(
        user_id=user_id,
        query_text=query_text,
        query_type=query_type,
        filters=kwargs.get('filters', default_filters),
        aggregations=kwargs.get('aggregations', AggregationRequest()),
        options=kwargs.get('options', SearchOptions()),
        **{k: v for k, v in kwargs.items() if k not in ['filters', 'aggregations', 'options']}
    )

def create_aggregation_request(
    user_id: int,
    aggregation_types: List[AggregationType],
    aggregation_fields: List[str],
    **kwargs
) -> AggregationOnlyRequest:
    """
    Factory pour créer une AggregationOnlyRequest.
    
    Args:
        user_id: ID de l'utilisateur
        aggregation_types: Types d'agrégation
        aggregation_fields: Champs d'agrégation
        **kwargs: Autres paramètres
    
    Returns:
        AggregationOnlyRequest configurée
    """
    # Agrégations activées par défaut
    aggregations = AggregationRequest(
        enabled=True,
        types=aggregation_types,
        fields=aggregation_fields,
        bucket_size=kwargs.get('bucket_size', 10)
    )
    
    # Filtres avec user_id
    default_filters = FilterGroup(
        required=[SearchFilter(
            field="user_id",
            operator=FilterOperator.EQ,
            value=user_id
        )]
    )
    
    return AggregationOnlyRequest(
        user_id=user_id,
        aggregations=aggregations,
        filters=kwargs.get('filters', default_filters),
        **{k: v for k, v in kwargs.items() if k not in ['aggregations', 'filters', 'bucket_size']}
    )

def create_text_search_request(
    user_id: int,
    query_text: str,
    include_aggregations: bool = False,
    **kwargs
) -> LexicalSearchRequest:
    """
    Factory pour créer une requête de recherche textuelle simple.
    
    Args:
        user_id: ID de l'utilisateur
        query_text: Texte à rechercher
        include_aggregations: Inclure des agrégations
        **kwargs: Autres paramètres
    
    Returns:
        LexicalSearchRequest pour recherche textuelle
    """
    query_type = QueryType.TEXT_SEARCH_WITH_FILTER if kwargs.get('filters') else QueryType.TEXT_SEARCH
    
    # Options optimisées pour la recherche textuelle
    options = SearchOptions(
        include_highlights=True,
        include_aggregations=include_aggregations,
        query=QueryOptions(
            fuzziness="AUTO",
            boost_recent=True,
            minimum_should_match="75%"
        )
    )
    
    return create_lexical_search_request(
        user_id=user_id,
        query_text=query_text,
        query_type=query_type,
        options=options,
        **kwargs
    )

# ==================== EXPORTS ====================

__all__ = [
    # Modèles principaux
    'LexicalSearchRequest',
    'AggregationOnlyRequest',
    'HealthCheckRequest',
    'MetricsRequest',
    
    # Options et configurations
    'CacheOptions',
    'QueryOptions', 
    'ResultOptions',
    'SearchOptions',
    
    # Validation
    'RequestValidator',
    'RequestValidationError',
    
    # Factory functions
    'create_lexical_search_request',
    'create_aggregation_request',
    'create_text_search_request',
    
    # Base
    'BaseLexicalRequest'
]