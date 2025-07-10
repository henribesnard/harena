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

from pydantic import BaseModel, Field, validator, root_validator
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
    SEARCH_BY_CATEGORY = "SEARCH_BY_CATEGORY"
    SEARCH_BY_MERCHANT = "SEARCH_BY_MERCHANT"
    SEARCH_BY_AMOUNT = "SEARCH_BY_AMOUNT"
    SEARCH_BY_DATE = "SEARCH_BY_DATE"
    TEXT_SEARCH = "TEXT_SEARCH"
    
    # Analyses temporelles
    TEMPORAL_ANALYSIS = "TEMPORAL_ANALYSIS"
    SPENDING_EVOLUTION = "SPENDING_EVOLUTION"
    MONTHLY_SUMMARY = "MONTHLY_SUMMARY"
    WEEKLY_PATTERN = "WEEKLY_PATTERN"
    
    # Analyses catégorielles
    CATEGORY_BREAKDOWN = "CATEGORY_BREAKDOWN"
    MERCHANT_ANALYSIS = "MERCHANT_ANALYSIS"
    TOP_MERCHANTS = "TOP_MERCHANTS"
    TOP_CATEGORIES = "TOP_CATEGORIES"
    
    # Opérations et comptages
    COUNT_OPERATIONS = "COUNT_OPERATIONS"
    COUNT_OPERATIONS_BY_CATEGORY = "COUNT_OPERATIONS_BY_CATEGORY"
    COUNT_OPERATIONS_BY_AMOUNT = "COUNT_OPERATIONS_BY_AMOUNT"
    
    # Analyses avancées
    BUDGET_ANALYSIS = "BUDGET_ANALYSIS"
    SPENDING_COMPARISON = "SPENDING_COMPARISON"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    
    # Recherches complexes
    TEXT_SEARCH_WITH_CATEGORY = "TEXT_SEARCH_WITH_CATEGORY"
    MULTI_CRITERIA_SEARCH = "MULTI_CRITERIA_SEARCH"
    
    # Fallback
    GENERAL_QUERY = "GENERAL_QUERY"

# ==================== MODÈLES DE BASE ====================

class SearchFilter(BaseModel):
    """Modèle pour un filtre de recherche."""
    field: str = Field(..., description="Nom du champ à filtrer")
    operator: FilterOperator = Field(..., description="Opérateur de filtrage")
    value: Union[str, int, float, List[Union[str, int, float]]] = Field(
        ..., description="Valeur(s) à filtrer"
    )
    boost: Optional[float] = Field(None, ge=0.0, le=10.0, description="Boost pour ce filtre")
    
    class Config:
        use_enum_values = True

class RangeFilter(BaseModel):
    """Filtre pour les plages de valeurs."""
    field: str = Field(..., description="Nom du champ")
    operator: Literal["between"] = Field("between", description="Opérateur de plage")
    value: List[Union[str, int, float]] = Field(
        ..., min_items=2, max_items=2, description="Valeurs min et max"
    )

class TextSearchFilter(BaseModel):
    """Filtre pour la recherche textuelle."""
    query: str = Field(..., min_length=1, max_length=1000, description="Texte à rechercher")
    fields: List[str] = Field(..., min_items=1, description="Champs à rechercher")
    operator: Literal["match", "match_phrase", "multi_match"] = Field(
        "match", description="Type de recherche textuelle"
    )
    boost: Optional[float] = Field(None, ge=0.0, le=10.0, description="Boost pour cette recherche")

class FilterGroup(BaseModel):
    """Groupe de filtres avec logique AND/OR."""
    required: List[SearchFilter] = Field(default=[], description="Filtres obligatoires (AND)")
    optional: List[SearchFilter] = Field(default=[], description="Filtres optionnels (OR)")
    ranges: List[RangeFilter] = Field(default=[], description="Filtres de plage")
    text_search: Optional[TextSearchFilter] = Field(None, description="Recherche textuelle")
    
    @validator('required', 'optional', 'ranges')
    def validate_filter_lists(cls, v):
        """Valide que les listes de filtres ne sont pas trop longues."""
        if len(v) > settings.MAX_FILTERS_PER_GROUP:
            raise ValueError(f"Trop de filtres: max {settings.MAX_FILTERS_PER_GROUP}")
        return v

# ==================== MÉTADONNÉES ====================

class ExecutionContext(BaseModel):
    """Contexte d'exécution de la requête."""
    conversation_id: Optional[str] = Field(None, description="ID de la conversation")
    turn_number: Optional[int] = Field(None, ge=1, description="Numéro du tour de conversation")
    agent_chain: List[str] = Field(default=[], description="Chaîne d'agents exécutés")
    team_name: Optional[str] = Field(None, description="Nom de l'équipe AutoGen")
    workflow_id: Optional[str] = Field(None, description="ID du workflow")

class AgentContext(BaseModel):
    """Contexte de l'agent AutoGen."""
    requesting_agent: str = Field(..., description="Agent ayant émis la requête")
    requesting_team: Optional[str] = Field(None, description="Équipe de l'agent")
    next_suggested_agent: Optional[str] = Field(None, description="Agent suivant suggéré")
    agent_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confiance de l'agent")

class QueryMetadata(BaseModel):
    """Métadonnées complètes de la requête."""
    query_id: UUID = Field(default_factory=uuid4, description="ID unique de la requête")
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    intent_type: IntentType = Field(..., description="Type d'intention détecté")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de classification")
    agent_name: str = Field(..., description="Nom de l'agent AutoGen")
    team_name: Optional[str] = Field(None, description="Nom de l'équipe AutoGen")
    execution_context: Optional[ExecutionContext] = Field(None, description="Contexte d'exécution")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de création")
    original_query: Optional[str] = Field(None, description="Requête originale utilisateur")
    
    class Config:
        use_enum_values = True

# ==================== PARAMÈTRES DE RECHERCHE ====================

class SearchParameters(BaseModel):
    """Paramètres de recherche Elasticsearch."""
    query_type: QueryType = Field(..., description="Type de requête")
    fields: List[str] = Field(..., min_items=1, description="Champs à récupérer")
    limit: PositiveInt = Field(
        default=settings.DEFAULT_LIMIT, 
        le=settings.MAX_SEARCH_RESULTS,
        description="Nombre maximum de résultats"
    )
    offset: NonNegativeInt = Field(default=0, description="Décalage pour pagination")
    timeout_ms: PositiveInt = Field(
        default=settings.SEARCH_TIMEOUT * 1000,
        le=settings.MAX_SEARCH_TIMEOUT * 1000,
        description="Timeout en millisecondes"
    )
    min_score: Optional[float] = Field(None, ge=0.0, description="Score minimum requis")
    
    class Config:
        use_enum_values = True

# ==================== AGRÉGATIONS ====================

class AggregationRequest(BaseModel):
    """Demande d'agrégation."""
    enabled: bool = Field(default=True, description="Activer les agrégations")
    types: List[AggregationType] = Field(default=[], description="Types d'agrégation")
    group_by: List[str] = Field(default=[], description="Champs de groupement")
    metrics: List[str] = Field(default=[], description="Champs métriques")
    date_interval: Optional[str] = Field(None, description="Intervalle pour date_histogram")
    size: PositiveInt = Field(default=10, le=100, description="Nombre de buckets max")
    
    class Config:
        use_enum_values = True
    
    @validator('types')
    def validate_aggregation_types(cls, v):
        """Valide les types d'agrégation."""
        if len(v) > 10:
            raise ValueError("Trop de types d'agrégation: max 10")
        return v

class AggregationBucket(BaseModel):
    """Bucket d'agrégation."""
    key: Union[str, int, float] = Field(..., description="Clé du bucket")
    doc_count: NonNegativeInt = Field(..., description="Nombre de documents")
    key_as_string: Optional[str] = Field(None, description="Clé formatée")
    metrics: Dict[str, Union[int, float]] = Field(default={}, description="Métriques calculées")

class AggregationResult(BaseModel):
    """Résultat d'agrégation."""
    name: str = Field(..., description="Nom de l'agrégation")
    type: AggregationType = Field(..., description="Type d'agrégation")
    buckets: List[AggregationBucket] = Field(default=[], description="Buckets de résultats")
    value: Optional[Union[int, float]] = Field(None, description="Valeur pour métriques simples")
    doc_count: NonNegativeInt = Field(default=0, description="Nombre total de documents")
    
    class Config:
        use_enum_values = True

class AggregationMetrics(BaseModel):
    """Métriques d'agrégation globales."""
    total_amount: Optional[float] = Field(None, description="Montant total")
    transaction_count: NonNegativeInt = Field(default=0, description="Nombre de transactions")
    average_amount: Optional[float] = Field(None, description="Montant moyen")
    by_month: List[AggregationBucket] = Field(default=[], description="Répartition mensuelle")
    by_category: List[AggregationBucket] = Field(default=[], description="Répartition par catégorie")
    by_merchant: List[AggregationBucket] = Field(default=[], description="Répartition par marchand")
    statistics: Dict[str, float] = Field(default={}, description="Statistiques avancées")

# ==================== OPTIONS ET ENRICHISSEMENT ====================

class SearchOptions(BaseModel):
    """Options de recherche avancées."""
    include_highlights: bool = Field(default=False, description="Inclure le highlighting")
    include_explanation: bool = Field(default=False, description="Inclure l'explication du score")
    cache_enabled: bool = Field(default=True, description="Activer le cache")
    return_raw_elasticsearch: bool = Field(default=False, description="Retourner la réponse ES brute")
    enable_fuzzy: bool = Field(default=False, description="Activer la recherche floue")
    fuzziness: Optional[str] = Field(None, description="Niveau de fuzziness")

class ContextEnrichment(BaseModel):
    """Enrichissement contextuel des résultats."""
    search_intent_matched: bool = Field(default=True, description="Intention de recherche trouvée")
    result_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score qualité")
    suggested_followup_questions: List[str] = Field(default=[], description="Questions de suivi")
    related_categories: List[str] = Field(default=[], description="Catégories liées")
    confidence_indicators: Dict[str, float] = Field(default={}, description="Indicateurs de confiance")

# ==================== RÉSULTATS ====================

class SearchResult(BaseModel):
    """Résultat de recherche individuel."""
    # Champs obligatoires
    transaction_id: str = Field(..., description="ID unique de la transaction")
    user_id: PositiveInt = Field(..., description="ID de l'utilisateur")
    account_id: PositiveInt = Field(..., description="ID du compte")
    
    # Montants
    amount: float = Field(..., description="Montant avec signe")
    amount_abs: NonNegativeFloat = Field(..., description="Montant en valeur absolue")
    currency_code: str = Field(..., description="Code devise")
    
    # Informations transaction
    transaction_type: Literal["debit", "credit"] = Field(..., description="Type de transaction")
    operation_type: str = Field(..., description="Type d'opération")
    date: str = Field(..., description="Date de la transaction (YYYY-MM-DD)")
    
    # Descriptions et catégories
    primary_description: str = Field(..., description="Description principale")
    merchant_name: Optional[str] = Field(None, description="Nom du marchand")
    category_name: str = Field(..., description="Nom de la catégorie")
    
    # Champs calculés
    month_year: str = Field(..., description="Mois-année (YYYY-MM)")
    weekday: str = Field(..., description="Jour de la semaine")
    
    # Métadonnées de recherche
    score: float = Field(default=1.0, ge=0.0, description="Score de pertinence")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Highlights de recherche")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explication du score")

# ==================== PERFORMANCE ET MONITORING ====================

class PerformanceMetrics(BaseModel):
    """Métriques de performance."""
    query_complexity: Literal["simple", "medium", "complex"] = Field(
        "simple", description="Complexité de la requête"
    )
    optimization_applied: List[str] = Field(default=[], description="Optimisations appliquées")
    index_used: str = Field(..., description="Index Elasticsearch utilisé")
    shards_queried: PositiveInt = Field(default=1, description="Nombre de shards interrogés")
    cache_hit: bool = Field(default=False, description="Résultat en cache")
    elasticsearch_took: NonNegativeInt = Field(..., description="Temps Elasticsearch (ms)")

class ResponseMetadata(BaseModel):
    """Métadonnées de la réponse."""
    query_id: UUID = Field(..., description="ID de la requête")
    execution_time_ms: NonNegativeInt = Field(..., description="Temps d'exécution total (ms)")
    total_hits: NonNegativeInt = Field(..., description="Nombre total de résultats")
    returned_hits: NonNegativeInt = Field(..., description="Nombre de résultats retournés")
    has_more: bool = Field(default=False, description="Plus de résultats disponibles")
    cache_hit: bool = Field(default=False, description="Résultat en cache")
    elasticsearch_took: NonNegativeInt = Field(..., description="Temps Elasticsearch (ms)")
    agent_context: Optional[AgentContext] = Field(None, description="Contexte agent")
    
    @validator('returned_hits')
    def validate_returned_hits(cls, v, values):
        """Valide que returned_hits <= total_hits."""
        if 'total_hits' in values and v > values['total_hits']:
            raise ValueError("returned_hits ne peut pas être supérieur à total_hits")
        return v

# ==================== CONTRATS PRINCIPAUX ====================

class SearchServiceQuery(BaseModel):
    """
    Contrat principal de requête du Conversation Service vers le Search Service.
    
    Ce modèle définit l'interface standardisée pour toutes les requêtes
    émises par les agents AutoGen vers le moteur de recherche Elasticsearch.
    """
    query_metadata: QueryMetadata = Field(..., description="Métadonnées de la requête")
    search_parameters: SearchParameters = Field(..., description="Paramètres de recherche")
    filters: FilterGroup = Field(..., description="Filtres de recherche")
    aggregations: Optional[AggregationRequest] = Field(None, description="Demandes d'agrégation")
    options: SearchOptions = Field(default_factory=SearchOptions, description="Options avancées")
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "query_metadata": {
                    "user_id": 34,
                    "intent_type": "SEARCH_BY_CATEGORY",
                    "confidence": 0.94,
                    "agent_name": "query_generator_agent",
                    "team_name": "financial_analysis_team",
                    "original_query": "mes restaurants du mois"
                },
                "search_parameters": {
                    "query_type": "filtered_search",
                    "fields": ["user_id", "category_name", "merchant_name", "amount", "date"],
                    "limit": 20,
                    "timeout_ms": 5000
                },
                "filters": {
                    "required": [
                        {"field": "user_id", "operator": "eq", "value": 34},
                        {"field": "category_name", "operator": "eq", "value": "restaurant"}
                    ]
                },
                "aggregations": {
                    "enabled": True,
                    "types": ["sum", "count"],
                    "metrics": ["amount_abs", "transaction_id"]
                }
            }
        }
    
    @root_validator
    def validate_query_consistency(cls, values):
        """Valide la cohérence globale de la requête."""
        metadata = values.get('query_metadata')
        parameters = values.get('search_parameters')
        filters = values.get('filters')
        
        if not metadata or not parameters or not filters:
            return values
        
        # Validation sécurité: user_id obligatoire
        user_filter_exists = any(
            f.field == "user_id" and f.operator == FilterOperator.EQ
            for f in filters.required
        )
        if not user_filter_exists:
            raise ValueError("Filtre user_id obligatoire pour la sécurité")
        
        # Validation cohérence user_id
        user_filter = next(
            (f for f in filters.required if f.field == "user_id"), None
        )
        if user_filter and user_filter.value != metadata.user_id:
            raise ValueError("user_id incohérent entre metadata et filtres")
        
        # Validation types d'agrégation selon le query_type
        aggregations = values.get('aggregations')
        if aggregations and aggregations.enabled:
            if parameters.query_type == QueryType.TEXT_SEARCH and not aggregations.types:
                raise ValueError("Types d'agrégation requis pour les recherches textuelles")
        
        return values

class SearchServiceResponse(BaseModel):
    """
    Contrat principal de réponse du Search Service vers le Conversation Service.
    
    Ce modèle définit l'interface standardisée pour toutes les réponses
    retournées par le moteur de recherche vers les agents AutoGen.
    """
    response_metadata: ResponseMetadata = Field(..., description="Métadonnées de la réponse")
    results: List[SearchResult] = Field(default=[], description="Résultats de recherche")
    aggregations: Optional[AggregationMetrics] = Field(None, description="Résultats d'agrégation")
    performance: PerformanceMetrics = Field(..., description="Métriques de performance")
    context_enrichment: Optional[ContextEnrichment] = Field(None, description="Enrichissement contextuel")
    debug: Optional[Dict[str, Any]] = Field(None, description="Informations de debug")
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "response_metadata": {
                    "query_id": "uuid-v4",
                    "execution_time_ms": 45,
                    "total_hits": 156,
                    "returned_hits": 20,
                    "has_more": True,
                    "elasticsearch_took": 23
                },
                "results": [
                    {
                        "transaction_id": "user_34_tx_12345",
                        "user_id": 34,
                        "amount": -45.67,
                        "amount_abs": 45.67,
                        "transaction_type": "debit",
                        "category_name": "Restaurant",
                        "merchant_name": "Le Bistrot",
                        "date": "2024-01-15",
                        "score": 1.0
                    }
                ],
                "performance": {
                    "query_complexity": "simple",
                    "index_used": "harena_transactions",
                    "elasticsearch_took": 23
                }
            }
        }
    
    @validator('results')
    def validate_results_count(cls, v, values):
        """Valide que le nombre de résultats est cohérent."""
        metadata = values.get('response_metadata')
        if metadata and len(v) != metadata.returned_hits:
            raise ValueError("Incohérence entre returned_hits et nombre de résultats")
        return v

# ==================== VALIDATION ET HELPERS ====================

class ContractValidationError(Exception):
    """Exception pour les erreurs de validation de contrat."""
    pass

def validate_search_service_query(query: SearchServiceQuery) -> bool:
    """
    Valide un contrat SearchServiceQuery.
    
    Args:
        query: Le contrat à valider
        
    Returns:
        True si valide
        
    Raises:
        ContractValidationError: Si la validation échoue
    """
    try:
        # Validation de base Pydantic
        query.dict()
        
        # Validations métier spécifiques
        if query.query_metadata.user_id <= 0:
            raise ContractValidationError("user_id doit être positif")
        
        if query.query_metadata.confidence < 0.5:
            raise ContractValidationError("Confiance trop faible (< 0.5)")
        
        # Validation sécurité user_id
        user_filter_exists = any(
            f.field == "user_id" for f in query.filters.required
        )
        if not user_filter_exists:
            raise ContractValidationError("Filtre user_id obligatoire")
        
        # Validation limites performance
        if query.search_parameters.limit > settings.MAX_SEARCH_RESULTS:
            raise ContractValidationError(f"Limite trop élevée (max {settings.MAX_SEARCH_RESULTS})")
        
        if query.search_parameters.timeout_ms > settings.MAX_SEARCH_TIMEOUT * 1000:
            raise ContractValidationError(f"Timeout trop élevé (max {settings.MAX_SEARCH_TIMEOUT}s)")
        
        return True
        
    except Exception as e:
        raise ContractValidationError(f"Validation échouée: {str(e)}")

def validate_search_service_response(response: SearchServiceResponse) -> bool:
    """
    Valide un contrat SearchServiceResponse.
    
    Args:
        response: Le contrat à valider
        
    Returns:
        True si valide
        
    Raises:
        ContractValidationError: Si la validation échoue
    """
    try:
        # Validation de base Pydantic
        response.dict()
        
        # Validations métier spécifiques
        if response.response_metadata.execution_time_ms < 0:
            raise ContractValidationError("execution_time_ms ne peut pas être négatif")
        
        if response.response_metadata.returned_hits > response.response_metadata.total_hits:
            raise ContractValidationError("returned_hits > total_hits impossible")
        
        if len(response.results) != response.response_metadata.returned_hits:
            raise ContractValidationError("Incohérence nombre de résultats")
        
        # Validation cohérence données
        for result in response.results:
            if result.amount_abs < 0:
                raise ContractValidationError("amount_abs ne peut pas être négatif")
            
            if abs(result.amount) != result.amount_abs:
                raise ContractValidationError("Incohérence amount/amount_abs")
        
        return True
        
    except Exception as e:
        raise ContractValidationError(f"Validation échouée: {str(e)}")

# ==================== FACTORY FUNCTIONS ====================

def create_search_service_query(
    user_id: int,
    intent_type: IntentType,
    agent_name: str,
    filters: Dict[str, Any],
    **kwargs
) -> SearchServiceQuery:
    """
    Factory pour créer une SearchServiceQuery avec des valeurs par défaut.
    
    Args:
        user_id: ID de l'utilisateur
        intent_type: Type d'intention
        agent_name: Nom de l'agent
        filters: Filtres à appliquer
        **kwargs: Paramètres additionnels
        
    Returns:
        SearchServiceQuery configurée
    """
    # Métadonnées par défaut
    query_metadata = QueryMetadata(
        user_id=user_id,
        intent_type=intent_type,
        confidence=kwargs.get('confidence', 0.8),
        agent_name=agent_name,
        team_name=kwargs.get('team_name'),
        original_query=kwargs.get('original_query')
    )
    
    # Paramètres par défaut
    search_parameters = SearchParameters(
        query_type=kwargs.get('query_type', QueryType.FILTERED_SEARCH),
        fields=kwargs.get('fields', ['*']),
        limit=kwargs.get('limit', settings.DEFAULT_LIMIT),
        timeout_ms=kwargs.get('timeout_ms', settings.SEARCH_TIMEOUT * 1000)
    )
    
    # Filtres avec user_id obligatoire
    filter_group = FilterGroup(
        required=[
            SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)
        ]
    )
    
    # Ajout des filtres personnalisés
    if 'required_filters' in filters:
        filter_group.required.extend(filters['required_filters'])
    if 'optional_filters' in filters:
        filter_group.optional.extend(filters['optional_filters'])
    if 'range_filters' in filters:
        filter_group.ranges.extend(filters['range_filters'])
    if 'text_search' in filters:
        filter_group.text_search = filters['text_search']
    
    return SearchServiceQuery(
        query_metadata=query_metadata,
        search_parameters=search_parameters,
        filters=filter_group,
        aggregations=kwargs.get('aggregations'),
        options=kwargs.get('options', SearchOptions())
    )

def create_search_service_response(
    query_id: UUID,
    results: List[Dict[str, Any]],
    total_hits: int,
    execution_time_ms: int,
    elasticsearch_took: int,
    **kwargs
) -> SearchServiceResponse:
    """
    Factory pour créer une SearchServiceResponse.
    
    Args:
        query_id: ID de la requête
        results: Résultats de recherche
        total_hits: Nombre total de résultats
        execution_time_ms: Temps d'exécution total
        elasticsearch_took: Temps Elasticsearch
        **kwargs: Paramètres additionnels
        
    Returns:
        SearchServiceResponse configurée
    """
    # Conversion des résultats
    search_results = [SearchResult(**result) for result in results]
    
    # Métadonnées de réponse
    response_metadata = ResponseMetadata(
        query_id=query_id,
        execution_time_ms=execution_time_ms,
        total_hits=total_hits,
        returned_hits=len(search_results),
        has_more=total_hits > len(search_results),
        elasticsearch_took=elasticsearch_took,
        agent_context=kwargs.get('agent_context')
    )
    
    # Métriques de performance
    performance = PerformanceMetrics(
        query_complexity=kwargs.get('query_complexity', 'simple'),
        optimization_applied=kwargs.get('optimization_applied', []),
        index_used=kwargs.get('index_used', settings.ELASTICSEARCH_INDEX),
        elasticsearch_took=elasticsearch_took
    )
    
    return SearchServiceResponse(
        response_metadata=response_metadata,
        results=search_results,
        aggregations=kwargs.get('aggregations'),
        performance=performance,
        context_enrichment=kwargs.get('context_enrichment'),
        debug=kwargs.get('debug')
    )

# ==================== CONSTANTES ET EXPORTS ====================

# Champs de recherche financiers valides
FINANCIAL_SEARCH_FIELDS = [
    "user_id", "account_id", "transaction_id",
    "amount", "amount_abs", "currency_code",
    "transaction_type", "operation_type", "date",
    "primary_description", "merchant_name", "category_name",
    "month_year", "weekday", "searchable_text"
]

# Champs d'agrégation valides
AGGREGATION_FIELDS = [
    "category_name", "merchant_name", "transaction_type",
    "month_year", "weekday", "amount", "amount_abs"
]

# Intentions nécessitant des agrégations
AGGREGATION_REQUIRED_INTENTS = [
    IntentType.COUNT_OPERATIONS,
    IntentType.TEMPORAL_ANALYSIS,
    IntentType.CATEGORY_BREAKDOWN,
    IntentType.SPENDING_EVOLUTION
]

__all__ = [
    # Contrats principaux
    "SearchServiceQuery",
    "SearchServiceResponse",
    
    # Métadonnées
    "QueryMetadata",
    "ResponseMetadata",
    "ExecutionContext",
    "AgentContext",
    
    # Filtres et paramètres
    "SearchFilter",
    "RangeFilter",
    "TextSearchFilter",
    "FilterGroup",
    "SearchParameters",
    
    # Agrégations
    "AggregationRequest",
    "AggregationResult",
    "AggregationBucket",
    "AggregationMetrics",
    
    # Résultats et enrichissement
    "SearchResult",
    "PerformanceMetrics",
    "ContextEnrichment",
    
    # Options
    "SearchOptions",
    
    # Enums
    "QueryType",
    "FilterOperator",
    "AggregationType",
    "IntentType",
    
    # Validation
    "ContractValidationError",
    "validate_search_service_query",
    "validate_search_service_response",
    
    # Factory functions
    "create_search_service_query",
    "create_search_service_response",
    
    # Constantes
    "FINANCIAL_SEARCH_FIELDS",
    "AGGREGATION_FIELDS",
    "AGGREGATION_REQUIRED_INTENTS"
]