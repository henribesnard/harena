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

# CORRECTION PYDANTIC V2: Remplacer root_validator par model_validator
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
    # Intentions de recherche (12 principales)
    SEARCH_BY_CATEGORY = "search_by_category"
    SEARCH_BY_MERCHANT = "search_by_merchant"
    SEARCH_BY_AMOUNT = "search_by_amount"
    SEARCH_BY_DATE = "search_by_date"
    SEARCH_BY_PERIOD = "search_by_period"
    SEARCH_BY_DESCRIPTION = "search_by_description"
    SEARCH_BY_TYPE = "search_by_type"
    SEARCH_BY_ACCOUNT = "search_by_account"
    SEARCH_RECENT = "search_recent"
    SEARCH_LARGEST = "search_largest"
    SEARCH_SMALLEST = "search_smallest"
    SEARCH_RECURRING = "search_recurring"
    
    # Intentions d'agrégation (8 principales)
    AGGREGATE_BY_CATEGORY = "aggregate_by_category"
    AGGREGATE_BY_MERCHANT = "aggregate_by_merchant"
    AGGREGATE_BY_MONTH = "aggregate_by_month"
    AGGREGATE_BY_WEEK = "aggregate_by_week"
    AGGREGATE_BY_ACCOUNT = "aggregate_by_account"
    AGGREGATE_TOTAL = "aggregate_total"
    AGGREGATE_AVERAGE = "aggregate_average"
    AGGREGATE_COUNT = "aggregate_count"

# ==================== MODÈLES DE BASE ====================

class SearchFilter(BaseModel):
    """Filtre de recherche standard."""
    field: str = Field(..., description="Champ à filtrer")
    operator: FilterOperator = Field(..., description="Opérateur de filtrage")
    value: Union[str, int, float, List[Any], Dict[str, Any]] = Field(..., description="Valeur du filtre")
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Valide le nom du champ."""
        if not v or not isinstance(v, str):
            raise ValueError("Field doit être une chaîne non vide")
        return v

class FilterGroup(BaseModel):
    """Groupe de filtres avec logique."""
    required: List[SearchFilter] = Field(default_factory=list, description="Filtres obligatoires (AND)")
    optional: List[SearchFilter] = Field(default_factory=list, description="Filtres optionnels (OR)")
    ranges: List[SearchFilter] = Field(default_factory=list, description="Filtres de plage")
    text_search: Optional[Dict[str, Any]] = Field(None, description="Recherche textuelle")
    
    @model_validator(mode='after')
    def validate_filter_groups(self):
        """Valide la cohérence des groupes de filtres."""
        # Au moins un filtre requis
        if not self.required and not self.optional and not self.ranges and not self.text_search:
            raise ValueError("Au moins un filtre requis")
        
        # Validation sécurité - user_id obligatoire dans required
        user_filter_exists = any(f.field == "user_id" for f in self.required)
        if not user_filter_exists:
            raise ValueError("Filtre user_id obligatoire dans required pour la sécurité")
        
        return self

class AggregationRequest(BaseModel):
    """Requête d'agrégation."""
    enabled: bool = Field(default=False, description="Activer les agrégations")
    types: List[AggregationType] = Field(default_factory=list, description="Types d'agrégation")
    group_by: List[str] = Field(default_factory=list, description="Champs de regroupement")
    metrics: List[str] = Field(default_factory=list, description="Métriques à calculer")
    
    @model_validator(mode='after')
    def validate_aggregations(self):
        """Valide la cohérence des agrégations."""
        if self.enabled:
            if not self.types:
                raise ValueError("Types d'agrégation requis quand enabled=True")
            if not self.group_by and AggregationType.TERMS in self.types:
                raise ValueError("group_by requis pour les agrégations TERMS")
        
        return self

class SearchOptions(BaseModel):
    """Options de recherche avancées."""
    include_highlights: bool = Field(default=False, description="Inclure les highlights")
    include_explanation: bool = Field(default=False, description="Inclure l'explication des scores")
    cache_enabled: bool = Field(default=True, description="Activer le cache")
    return_raw_elasticsearch: bool = Field(default=False, description="Retourner la réponse Elasticsearch brute")

# ==================== MÉTADONNÉES ====================

class QueryMetadata(BaseModel):
    """Métadonnées de la requête pour observabilité."""
    query_id: UUID = Field(default_factory=uuid4, description="ID unique de la requête")
    user_id: PositiveInt = Field(..., description="ID utilisateur")
    intent_type: IntentType = Field(..., description="Type d'intention détectée")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de l'intention")
    agent_name: str = Field(..., description="Nom de l'agent AutoGen")
    team_name: str = Field(..., description="Nom de l'équipe AutoGen")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Contexte d'exécution")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de création")

class SearchParameters(BaseModel):
    """Paramètres de recherche configurables."""
    query_type: QueryType = Field(default=QueryType.FILTERED_SEARCH, description="Type de requête")
    fields: List[str] = Field(default_factory=list, description="Champs à rechercher")
    limit: PositiveInt = Field(default=20, description="Nombre de résultats")
    offset: NonNegativeInt = Field(default=0, description="Décalage pour pagination")
    timeout_ms: PositiveInt = Field(default=5000, description="Timeout en millisecondes")
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        """Valide la limite de résultats."""
        if v > settings.MAX_SEARCH_RESULTS:
            raise ValueError(f"Limite trop élevée (max {settings.MAX_SEARCH_RESULTS})")
        return v
    
    @field_validator('timeout_ms')
    @classmethod
    def validate_timeout(cls, v):
        """Valide le timeout."""
        max_timeout_ms = settings.MAX_SEARCH_TIMEOUT * 1000
        if v > max_timeout_ms:
            raise ValueError(f"Timeout trop élevé (max {settings.MAX_SEARCH_TIMEOUT}s)")
        return v

# ==================== CONTRATS PRINCIPAUX ====================

class SearchServiceQuery(BaseModel):
    """
    Contrat de requête standardisé du Conversation Service vers Search Service.
    
    Ce modèle définit l'interface stable pour toutes les requêtes envoyées
    par le Conversation Service via les agents AutoGen.
    """
    query_metadata: QueryMetadata = Field(..., description="Métadonnées de la requête")
    search_parameters: SearchParameters = Field(..., description="Paramètres de recherche")
    filters: FilterGroup = Field(..., description="Filtres de recherche")
    aggregations: AggregationRequest = Field(default_factory=AggregationRequest, description="Agrégations demandées")
    options: SearchOptions = Field(default_factory=SearchOptions, description="Options avancées")
    
    # Champs optionnels pour compatibilité
    query_text: Optional[str] = Field(None, description="Texte de recherche libre")
    
    @model_validator(mode='after')
    def validate_query_coherence(self):
        """Valide la cohérence des paramètres de requête."""
        # Validation logique métier
        if self.search_parameters.query_type in ["text_search", "text_search_with_filter"]:
            if not self.query_text or len(self.query_text.strip()) == 0:
                raise ValueError("query_text requis pour les recherches textuelles")
        
        # Validation sécurité - user_id cohérent
        query_user_id = self.query_metadata.user_id
        filter_user_ids = [
            f.value for f in self.filters.required 
            if f.field == "user_id" and f.operator == FilterOperator.EQ
        ]
        
        if filter_user_ids and filter_user_ids[0] != query_user_id:
            raise ValueError("user_id incohérent entre metadata et filtres")
        
        # Validation limites configurables
        if self.search_parameters.limit > settings.MAX_SEARCH_RESULTS:
            raise ValueError(f"Limite trop élevée (max {settings.MAX_SEARCH_RESULTS})")
        
        return self

class ResponseMetadata(BaseModel):
    """Métadonnées de la réponse pour observabilité."""
    query_id: UUID = Field(..., description="ID de la requête originale")
    execution_time_ms: NonNegativeInt = Field(..., description="Temps d'exécution en ms")
    total_hits: NonNegativeInt = Field(..., description="Nombre total de résultats")
    returned_hits: NonNegativeInt = Field(..., description="Nombre de résultats retournés")
    has_more: bool = Field(..., description="Plus de résultats disponibles")
    cache_hit: bool = Field(..., description="Résultat du cache")
    elasticsearch_took: NonNegativeInt = Field(..., description="Temps Elasticsearch en ms")
    agent_context: Dict[str, Any] = Field(default_factory=dict, description="Contexte pour agents")

class TransactionResult(BaseModel):
    """Résultat de transaction financière."""
    transaction_id: str = Field(..., description="ID unique transaction")
    user_id: PositiveInt = Field(..., description="ID utilisateur")
    account_id: PositiveInt = Field(..., description="ID compte")
    amount: float = Field(..., description="Montant transaction")
    amount_abs: NonNegativeFloat = Field(..., description="Montant absolu")
    transaction_type: str = Field(..., description="Type transaction")
    currency_code: str = Field(..., description="Code devise")
    date: str = Field(..., description="Date transaction (YYYY-MM-DD)")
    primary_description: str = Field(..., description="Description principale")
    merchant_name: Optional[str] = Field(None, description="Nom marchand")
    category_name: Optional[str] = Field(None, description="Nom catégorie")
    operation_type: Optional[str] = Field(None, description="Type opération")
    month_year: str = Field(..., description="Mois-année (YYYY-MM)")
    weekday: str = Field(..., description="Jour semaine")
    score: NonNegativeFloat = Field(..., description="Score de pertinence")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Highlights de recherche")

class AggregationResult(BaseModel):
    """Résultat d'agrégation."""
    total_amount: float = Field(..., description="Montant total")
    transaction_count: NonNegativeInt = Field(..., description="Nombre de transactions")
    average_amount: float = Field(..., description="Montant moyen")
    by_period: List[Dict[str, Any]] = Field(default_factory=list, description="Agrégation par période")
    statistics: Dict[str, float] = Field(default_factory=dict, description="Statistiques avancées")

class PerformanceMetrics(BaseModel):
    """Métriques de performance."""
    query_complexity: str = Field(..., description="Complexité de la requête")
    optimization_applied: List[str] = Field(default_factory=list, description="Optimisations appliquées")
    index_used: str = Field(..., description="Index Elasticsearch utilisé")
    shards_queried: PositiveInt = Field(..., description="Nombre de shards interrogés")

class ContextEnrichment(BaseModel):
    """Enrichissement contextuel pour les agents."""
    search_intent_matched: bool = Field(..., description="Intention correctement matchée")
    result_quality_score: float = Field(..., ge=0.0, le=1.0, description="Score qualité résultats")
    suggested_followup_questions: List[str] = Field(default_factory=list, description="Questions de suivi suggérées")

class SearchServiceResponse(BaseModel):
    """
    Contrat de réponse standardisé du Search Service vers Conversation Service.
    
    Ce modèle définit l'interface stable pour toutes les réponses retournées
    aux agents AutoGen du Conversation Service.
    """
    response_metadata: ResponseMetadata = Field(..., description="Métadonnées de réponse")
    results: List[TransactionResult] = Field(default_factory=list, description="Résultats de transaction")
    aggregations: Optional[AggregationResult] = Field(None, description="Résultats d'agrégation")
    performance: PerformanceMetrics = Field(..., description="Métriques de performance")
    context_enrichment: ContextEnrichment = Field(..., description="Enrichissement contextuel")
    debug: Optional[Dict[str, Any]] = Field(None, description="Informations de debug")
    
    @model_validator(mode='after')
    def validate_response_coherence(self):
        """Valide la cohérence de la réponse."""
        # Cohérence des compteurs
        if len(self.results) != self.response_metadata.returned_hits:
            raise ValueError("Incohérence entre nombre de résultats et metadata")
        
        if self.response_metadata.returned_hits > self.response_metadata.total_hits:
            raise ValueError("returned_hits ne peut pas être > total_hits")
        
        # Validation qualité
        if not (0.0 <= self.context_enrichment.result_quality_score <= 1.0):
            raise ValueError("result_quality_score doit être entre 0.0 et 1.0")
        
        return self

# ==================== EXPORTS ====================

__all__ = [
    # Enums
    "QueryType", "FilterOperator", "AggregationType", "IntentType",
    
    # Modèles de base
    "SearchFilter", "FilterGroup", "AggregationRequest", "SearchOptions",
    
    # Métadonnées
    "QueryMetadata", "SearchParameters", "ResponseMetadata",
    
    # Résultats
    "TransactionResult", "AggregationResult", "PerformanceMetrics", "ContextEnrichment",
    
    # Contrats principaux
    "SearchServiceQuery", "SearchServiceResponse"
]