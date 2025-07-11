"""
Contrats d'interface standardisés pour la communication entre services.

Ces contrats définissent l'interface stable entre le Conversation Service (AutoGen)
et le Search Service (Elasticsearch), garantissant une communication fiable
et une évolution contrôlée de l'API.

ARCHITECTURE:
- SearchServiceQuery: Format standard des requêtes du Conversation Service
- SearchServiceResponse: Format standard des réponses du Search Service
- Validation stricte avec Pydantic v2
- Métadonnées complètes pour observabilité
- Support des agrégations financières complexes
- Gestion des contextes conversationnels multi-tours

CONFIGURATION CENTRALISÉE:
- Toutes les constantes via config_service
- Validation basée sur les paramètres configurés
- Limites dynamiques selon l'environnement
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Literal
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveInt, NonNegativeInt, NonNegativeFloat

# Configuration centralisée
from config_service.config import settings

# ==================== ENUMS ET CONSTANTES ====================

class QueryType(str, Enum):
    """Types de requêtes supportées."""
    FILTERED_SEARCH = "filtered_search"
    TEXT_SEARCH = "text_search"
    AGGREGATION_ONLY = "aggregation_only"
    FILTERED_AGGREGATION = "filtered_aggregation"
    TEMPORAL_AGGREGATION = "temporal_aggregation"
    TEXT_SEARCH_WITH_FILTER = "text_search_with_filter"

class FilterOperator(str, Enum):
    """Opérateurs de filtrage supportés."""
    EQ = "eq"           # Égalité exacte
    NE = "ne"           # Différent de
    GT = "gt"           # Supérieur à
    GTE = "gte"         # Supérieur ou égal
    LT = "lt"           # Inférieur à
    LTE = "lte"         # Inférieur ou égal
    IN = "in"           # Dans la liste
    NOT_IN = "not_in"   # Pas dans la liste
    BETWEEN = "between" # Entre deux valeurs
    MATCH = "match"     # Recherche textuelle
    REGEX = "regex"     # Expression régulière

class AggregationType(str, Enum):
    """Types d'agrégation supportés."""
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    TERMS = "terms"
    DATE_HISTOGRAM = "date_histogram"
    HISTOGRAM = "histogram"
    STATS = "stats"

class IntentType(str, Enum):
    """Types d'intention financière (taxonomie complète)."""
    # Recherche de base
    TRANSACTION_SEARCH = "transaction_search"
    ACCOUNT_INQUIRY = "account_inquiry"
    BALANCE_CHECK = "balance_check"
    
    # Analyses financières
    SPENDING_ANALYSIS = "spending_analysis"
    CATEGORY_BREAKDOWN = "category_breakdown"
    TREND_ANALYSIS = "trend_analysis"
    BUDGET_TRACKING = "budget_tracking"
    
    # Agrégations temporelles
    MONTHLY_SUMMARY = "monthly_summary"
    YEARLY_OVERVIEW = "yearly_overview"
    DAILY_TRANSACTIONS = "daily_transactions"
    
    # Recherche avancée
    COMPLEX_FILTER = "complex_filter"
    MULTI_CRITERIA = "multi_criteria"
    CONTEXTUAL_SEARCH = "contextual_search"

class ResponseFormat(str, Enum):
    """Formats de réponse supportés."""
    STANDARD = "standard"
    DETAILED = "detailed"
    SUMMARY = "summary"
    AGGREGATION = "aggregation"
    CONVERSATIONAL = "conversational"

# ==================== MODÈLES DE BASE ====================

class QueryMetadata(BaseModel):
    """Métadonnées d'une requête de recherche."""
    query_id: UUID = Field(default_factory=uuid4, description="ID unique de la requête")
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    conversation_id: Optional[UUID] = Field(None, description="ID de la conversation")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de la requête")
    intent_type: Optional[IntentType] = Field(None, description="Type d'intention détectée")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confiance de l'intention")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Valide le niveau de confiance."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("La confiance doit être entre 0.0 et 1.0")
        return v

class AggregationMetrics(BaseModel):
    """Métriques d'agrégation standardisées."""
    total_count: int = Field(default=0, description="Nombre total d'éléments")
    sum_amount: Optional[float] = Field(None, description="Somme des montants")
    avg_amount: Optional[float] = Field(None, description="Montant moyen")
    min_amount: Optional[float] = Field(None, description="Montant minimum")
    max_amount: Optional[float] = Field(None, description="Montant maximum")
    unique_merchants: Optional[int] = Field(None, description="Nombre de commerçants uniques")
    unique_categories: Optional[int] = Field(None, description="Nombre de catégories uniques")
    date_range: Optional[Dict[str, str]] = Field(None, description="Plage de dates")
    
    @field_validator('total_count')
    @classmethod
    def validate_count(cls, v):
        """Valide le comptage total."""
        if v < 0:
            raise ValueError("Le comptage total ne peut pas être négatif")
        return v

class SearchFilter(BaseModel):
    """Filtre de recherche standardisé."""
    field: str = Field(..., description="Champ à filtrer")
    operator: FilterOperator = Field(..., description="Opérateur de filtrage")
    value: Union[str, int, float, List[Any]] = Field(..., description="Valeur du filtre")
    boost: Optional[float] = Field(default=1.0, description="Boost pour ce filtre")
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Valide que le champ est autorisé."""
        allowed_fields = settings.ALLOWED_SEARCH_FIELDS
        if v not in allowed_fields:
            raise ValueError(f"Champ non autorisé: {v}")
        return v
    
    @field_validator('boost')
    @classmethod
    def validate_boost(cls, v):
        """Valide le facteur de boost."""
        if v <= 0 or v > 10:
            raise ValueError("Le boost doit être entre 0 et 10")
        return v

class FilterGroup(BaseModel):
    """Groupe de filtres avec logique."""
    logic: Literal["AND", "OR"] = Field(default="AND", description="Logique de combinaison")
    required: List[SearchFilter] = Field(default_factory=list, description="Filtres obligatoires")
    optional: List[SearchFilter] = Field(default_factory=list, description="Filtres optionnels")
    exclusions: List[SearchFilter] = Field(default_factory=list, description="Filtres d'exclusion")
    
    @model_validator(mode='after')
    def validate_filter_group(self):
        """Valide la cohérence du groupe de filtres."""
        total_filters = len(self.required) + len(self.optional) + len(self.exclusions)
        if total_filters == 0:
            raise ValueError("Un groupe de filtres doit contenir au moins un filtre")
        
        if total_filters > settings.MAX_FILTERS_PER_GROUP:
            raise ValueError(f"Trop de filtres dans le groupe (max {settings.MAX_FILTERS_PER_GROUP})")
        
        return self

class AggregationRequest(BaseModel):
    """Requête d'agrégation standardisée."""
    enabled: bool = Field(default=False, description="Activer les agrégations")
    types: List[AggregationType] = Field(default_factory=list, description="Types d'agrégation")
    fields: List[str] = Field(default_factory=list, description="Champs d'agrégation")
    bucket_size: Optional[int] = Field(default=10, description="Taille des buckets")
    
    @field_validator('bucket_size')
    @classmethod
    def validate_bucket_size(cls, v):
        """Valide la taille des buckets."""
        if v and (v <= 0 or v > settings.MAX_AGGREGATION_BUCKETS):
            raise ValueError(f"Taille bucket invalide (max {settings.MAX_AGGREGATION_BUCKETS})")
        return v
    
    @model_validator(mode='after')
    def validate_aggregation_consistency(self):
        """Valide la cohérence de l'agrégation."""
        if self.enabled and not self.types:
            raise ValueError("Types d'agrégation requis si activées")
        
        if self.types and not self.fields:
            # Certains types d'agrégation ne nécessitent pas de champs spécifiques
            count_only_types = {AggregationType.COUNT}
            if not all(t in count_only_types for t in self.types):
                raise ValueError("Champs requis pour les types d'agrégation sélectionnés")
        
        return self

class SearchOptions(BaseModel):
    """Options de recherche configurables."""
    timeout_seconds: int = Field(
        default=settings.DEFAULT_SEARCH_TIMEOUT,
        description="Timeout de recherche en secondes"
    )
    max_results: int = Field(
        default=settings.DEFAULT_SEARCH_LIMIT,
        description="Nombre maximum de résultats"
    )
    include_highlights: bool = Field(default=True, description="Inclure les highlights")
    include_aggregations: bool = Field(default=False, description="Inclure les agrégations")
    explain_score: bool = Field(default=False, description="Expliquer le score")
    
    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v):
        """Valide le timeout."""
        if v <= 0 or v > settings.MAX_SEARCH_TIMEOUT:
            raise ValueError(f"Timeout invalide (max {settings.MAX_SEARCH_TIMEOUT}s)")
        return v
    
    @field_validator('max_results')
    @classmethod
    def validate_max_results(cls, v):
        """Valide le nombre maximum de résultats."""
        if v <= 0 or v > settings.MAX_SEARCH_RESULTS:
            raise ValueError(f"Nombre de résultats invalide (max {settings.MAX_SEARCH_RESULTS})")
        return v

# ==================== CONTRATS PRINCIPAUX ====================

class SearchServiceQuery(BaseModel):
    """
    Requête standardisée du Conversation Service vers le Search Service.
    
    Format stable pour toutes les interactions entre services, garantissant
    une interface cohérente et évolutive.
    """
    # Métadonnées de requête
    query_id: UUID = Field(default_factory=uuid4, description="ID unique de la requête")
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    conversation_id: Optional[UUID] = Field(None, description="ID de la conversation")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de la requête")
    
    # Configuration de recherche
    query_type: QueryType = Field(..., description="Type de requête")
    intent_type: Optional[IntentType] = Field(None, description="Type d'intention détectée")
    
    # Contenu de recherche
    query_text: Optional[str] = Field(None, description="Texte de recherche")
    filters: FilterGroup = Field(default_factory=FilterGroup, description="Filtres de recherche")
    aggregations: AggregationRequest = Field(default_factory=AggregationRequest, description="Agrégations")
    options: SearchOptions = Field(default_factory=SearchOptions, description="Options de recherche")
    
    # Contexte conversationnel
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexte conversationnel")
    previous_queries: List[str] = Field(default_factory=list, description="Requêtes précédentes")
    
    # Métadonnées techniques
    response_format: ResponseFormat = Field(default=ResponseFormat.STANDARD, description="Format de réponse")
    
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
    
    @field_validator('previous_queries')
    @classmethod
    def validate_previous_queries(cls, v):
        """Valide les requêtes précédentes."""
        if len(v) > settings.MAX_PREVIOUS_QUERIES:
            raise ValueError(f"Trop de requêtes précédentes (max {settings.MAX_PREVIOUS_QUERIES})")
        return v
    
    @model_validator(mode='after')
    def validate_query_consistency(self):
        """Valide la cohérence de la requête."""
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
            # Ajouter automatiquement le filtre user_id
            from .service_contracts import SearchFilter, FilterOperator
            user_filter = SearchFilter(
                field="user_id",
                operator=FilterOperator.EQ,
                value=self.user_id
            )
            self.filters.required.append(user_filter)
        
        return self

class SearchResult(BaseModel):
    """Résultat de recherche standardisé."""
    id: str = Field(..., description="ID unique du résultat")
    score: float = Field(..., description="Score de pertinence")
    source: Dict[str, Any] = Field(..., description="Document source")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Highlights")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explication du score")

class AggregationResult(BaseModel):
    """Résultat d'agrégation standardisé."""
    name: str = Field(..., description="Nom de l'agrégation")
    type: AggregationType = Field(..., description="Type d'agrégation")
    value: Union[int, float, Dict[str, Any]] = Field(..., description="Valeur d'agrégation")
    buckets: Optional[List[Dict[str, Any]]] = Field(None, description="Buckets pour agrégations")

class SearchServiceResponse(BaseModel):
    """
    Réponse standardisée du Search Service vers le Conversation Service.
    
    Format stable pour toutes les réponses, avec métadonnées complètes
    pour l'observabilité et le debugging.
    """
    # Métadonnées de réponse
    query_id: UUID = Field(..., description="ID de la requête correspondante")
    response_id: UUID = Field(default_factory=uuid4, description="ID unique de la réponse")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de la réponse")
    
    # Résultats
    success: bool = Field(..., description="Succès de la requête")
    results: List[SearchResult] = Field(default_factory=list, description="Résultats de recherche")
    aggregations: List[AggregationResult] = Field(default_factory=list, description="Résultats d'agrégation")
    
    # Métadonnées de performance
    total_hits: int = Field(default=0, description="Nombre total de résultats")
    execution_time_ms: float = Field(..., description="Temps d'exécution en ms")
    elasticsearch_time_ms: Optional[float] = Field(None, description="Temps Elasticsearch")
    
    # Informations de pagination
    offset: int = Field(default=0, description="Offset des résultats")
    limit: int = Field(..., description="Limite des résultats")
    has_more: bool = Field(default=False, description="Plus de résultats disponibles")
    
    # Contexte et suggestions
    suggestions: List[str] = Field(default_factory=list, description="Suggestions de requêtes")
    related_queries: List[str] = Field(default_factory=list, description="Requêtes liées")
    
    # Gestion d'erreurs
    errors: List[str] = Field(default_factory=list, description="Erreurs rencontrées")
    warnings: List[str] = Field(default_factory=list, description="Avertissements")
    
    # Métadonnées de debugging
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Informations de debug")
    
    @model_validator(mode='after')
    def validate_response_consistency(self):
        """Valide la cohérence de la réponse."""
        if self.success:
            if self.errors:
                raise ValueError("Pas d'erreurs attendues si succès")
        else:
            if not self.errors:
                raise ValueError("Erreurs requises si échec")
        
        if self.total_hits < len(self.results):
            raise ValueError("total_hits ne peut pas être inférieur au nombre de résultats")
        
        if self.offset < 0:
            raise ValueError("offset ne peut pas être négatif")
        
        if self.limit <= 0:
            raise ValueError("limit doit être positif")
        
        return self

# ==================== FACTORY FUNCTIONS ====================

def create_search_query(
    user_id: int,
    query_type: QueryType,
    query_text: Optional[str] = None,
    **kwargs
) -> SearchServiceQuery:
    """
    Factory pour créer une SearchServiceQuery avec des valeurs par défaut.
    
    Args:
        user_id: ID de l'utilisateur
        query_type: Type de requête
        query_text: Texte de recherche optionnel
        **kwargs: Autres paramètres
    
    Returns:
        SearchServiceQuery configurée
    """
    # Filtres par défaut avec user_id
    default_filters = FilterGroup(
        required=[SearchFilter(
            field="user_id",
            operator=FilterOperator.EQ,
            value=user_id
        )]
    )
    
    return SearchServiceQuery(
        user_id=user_id,
        query_type=query_type,
        query_text=query_text,
        filters=kwargs.get('filters', default_filters),
        aggregations=kwargs.get('aggregations', AggregationRequest()),
        options=kwargs.get('options', SearchOptions()),
        **{k: v for k, v in kwargs.items() if k not in ['filters', 'aggregations', 'options']}
    )

def create_success_response(
    query_id: UUID,
    results: List[SearchResult],
    execution_time_ms: float,
    **kwargs
) -> SearchServiceResponse:
    """
    Factory pour créer une réponse de succès.
    
    Args:
        query_id: ID de la requête
        results: Résultats de recherche
        execution_time_ms: Temps d'exécution
        **kwargs: Autres paramètres
    
    Returns:
        SearchServiceResponse de succès
    """
    return SearchServiceResponse(
        query_id=query_id,
        success=True,
        results=results,
        execution_time_ms=execution_time_ms,
        total_hits=kwargs.get('total_hits', len(results)),
        limit=kwargs.get('limit', len(results)),
        **{k: v for k, v in kwargs.items() if k not in ['total_hits', 'limit']}
    )

def create_error_response(
    query_id: UUID,
    errors: List[str],
    execution_time_ms: float,
    **kwargs
) -> SearchServiceResponse:
    """
    Factory pour créer une réponse d'erreur.
    
    Args:
        query_id: ID de la requête
        errors: Liste des erreurs
        execution_time_ms: Temps d'exécution
        **kwargs: Autres paramètres
    
    Returns:
        SearchServiceResponse d'erreur
    """
    return SearchServiceResponse(
        query_id=query_id,
        success=False,
        errors=errors,
        execution_time_ms=execution_time_ms,
        limit=0,
        **kwargs
    )

# ==================== UTILITAIRES DE VALIDATION ====================

def validate_query_format(query: SearchServiceQuery) -> List[str]:
    """
    Valide le format d'une requête sans lever d'exception.
    
    Args:
        query: Requête à valider
    
    Returns:
        Liste des erreurs de validation (vide si OK)
    """
    errors = []
    
    try:
        # Validation Pydantic
        query.model_validate(query.model_dump())
    except Exception as e:
        errors.append(f"Erreur de validation Pydantic: {str(e)}")
    
    # Validations métier supplémentaires
    if query.query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER]:
        if not query.query_text or len(query.query_text.strip()) == 0:
            errors.append("query_text requis pour les recherches textuelles")
    
    if query.query_type == QueryType.AGGREGATION_ONLY:
        if not query.aggregations.enabled:
            errors.append("Agrégations requises pour AGGREGATION_ONLY")
    
    # Validation sécurité
    user_filter_exists = any(
        f.field == "user_id" for f in query.filters.required
    )
    if not user_filter_exists:
        errors.append("Filtre user_id obligatoire pour la sécurité")
    
    return errors

def validate_response_format(response: SearchServiceResponse) -> List[str]:
    """
    Valide le format d'une réponse sans lever d'exception.
    
    Args:
        response: Réponse à valider
    
    Returns:
        Liste des erreurs de validation (vide si OK)
    """
    errors = []
    
    try:
        # Validation Pydantic
        response.model_validate(response.model_dump())
    except Exception as e:
        errors.append(f"Erreur de validation Pydantic: {str(e)}")
    
    # Validations métier
    if response.success and response.errors:
        errors.append("Pas d'erreurs attendues pour une réponse de succès")
    
    if not response.success and not response.errors:
        errors.append("Erreurs requises pour une réponse d'échec")
    
    if response.total_hits < len(response.results):
        errors.append("total_hits incohérent avec le nombre de résultats")
    
    return errors