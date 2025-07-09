"""
Contrats d'interface standardisés pour le Search Service.

Ces modèles définissent l'interface stable entre le Conversation Service
et le Search Service, permettant une évolution indépendante des deux services.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator


# === ENUMS ===

class QueryType(str, Enum):
    """Types de requêtes supportées."""
    SIMPLE_SEARCH = "simple_search"
    FILTERED_SEARCH = "filtered_search"
    TEXT_SEARCH = "text_search"
    AGGREGATION = "aggregation"
    FILTERED_AGGREGATION = "filtered_aggregation"
    TEXT_SEARCH_WITH_FILTER = "text_search_with_filter"
    TEMPORAL_AGGREGATION = "temporal_aggregation"


class FilterOperator(str, Enum):
    """Opérateurs de filtrage supportés."""
    EQ = "eq"                    # Égal
    NE = "ne"                    # Différent
    GT = "gt"                    # Supérieur
    GTE = "gte"                  # Supérieur ou égal
    LT = "lt"                    # Inférieur
    LTE = "lte"                  # Inférieur ou égal
    IN = "in"                    # Dans la liste
    NOT_IN = "not_in"            # Pas dans la liste
    BETWEEN = "between"          # Entre deux valeurs
    MATCH = "match"              # Recherche textuelle
    MATCH_PHRASE = "match_phrase" # Phrase exacte
    WILDCARD = "wildcard"        # Avec wildcards
    REGEX = "regex"              # Expression régulière


class AggregationType(str, Enum):
    """Types d'agrégations supportées."""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    TERMS = "terms"
    DATE_HISTOGRAM = "date_histogram"
    STATS = "stats"


class SortOrder(str, Enum):
    """Ordres de tri supportés."""
    ASC = "asc"
    DESC = "desc"


class IntentType(str, Enum):
    """Types d'intentions détectées par le Conversation Service."""
    SEARCH_BY_CATEGORY = "SEARCH_BY_CATEGORY"
    SEARCH_BY_MERCHANT = "SEARCH_BY_MERCHANT"
    SEARCH_BY_AMOUNT = "SEARCH_BY_AMOUNT"
    SEARCH_BY_DATE = "SEARCH_BY_DATE"
    TEXT_SEARCH = "TEXT_SEARCH"
    COUNT_OPERATIONS = "COUNT_OPERATIONS"
    TEMPORAL_ANALYSIS = "TEMPORAL_ANALYSIS"
    COUNT_OPERATIONS_BY_AMOUNT = "COUNT_OPERATIONS_BY_AMOUNT"
    TEXT_SEARCH_WITH_CATEGORY = "TEXT_SEARCH_WITH_CATEGORY"
    TEMPORAL_SPENDING_ANALYSIS = "TEMPORAL_SPENDING_ANALYSIS"


# === MODÈLES DE BASE ===

class SearchFilter(BaseModel):
    """
    Filtre de recherche standardisé.
    
    Permet de filtrer les résultats selon différents critères
    avec des opérateurs flexibles.
    """
    field: str = Field(..., description="Nom du champ à filtrer")
    operator: FilterOperator = Field(..., description="Opérateur de filtrage")
    value: Union[str, int, float, List[Any], Dict[str, Any]] = Field(
        ..., description="Valeur de filtrage"
    )
    
    @validator('field')
    def validate_field_name(cls, v):
        """Valide que le nom du champ n'est pas vide."""
        if not v or not v.strip():
            raise ValueError("Le nom du champ ne peut pas être vide")
        return v.strip()
    
    @validator('value')
    def validate_value(cls, v, values):
        """Valide la valeur selon l'opérateur."""
        operator = values.get('operator')
        
        if operator == FilterOperator.BETWEEN:
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError("L'opérateur 'between' requiert une liste de 2 valeurs")
        
        elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(v, list):
                raise ValueError(f"L'opérateur '{operator}' requiert une liste de valeurs")
        
        return v


class TextSearchConfig(BaseModel):
    """Configuration pour la recherche textuelle."""
    query: str = Field(..., description="Texte à rechercher")
    fields: List[str] = Field(..., description="Champs dans lesquels rechercher")
    operator: FilterOperator = Field(
        default=FilterOperator.MATCH, 
        description="Type de recherche textuelle"
    )
    boost: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Boost par champ"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Valide que la requête n'est pas vide."""
        if not v or not v.strip():
            raise ValueError("La requête de recherche ne peut pas être vide")
        return v.strip()
    
    @validator('fields')
    def validate_fields(cls, v):
        """Valide que la liste des champs n'est pas vide."""
        if not v:
            raise ValueError("Au moins un champ de recherche doit être spécifié")
        return v


class AggregationRequest(BaseModel):
    """Demande d'agrégation."""
    enabled: bool = Field(default=False, description="Activer les agrégations")
    types: List[AggregationType] = Field(
        default_factory=list, 
        description="Types d'agrégations à effectuer"
    )
    group_by: List[str] = Field(
        default_factory=list, 
        description="Champs de regroupement"
    )
    metrics: List[str] = Field(
        default_factory=list, 
        description="Champs pour les métriques"
    )
    
    @root_validator
    def validate_aggregation(cls, values):
        """Valide la cohérence de la demande d'agrégation."""
        enabled = values.get('enabled', False)
        types = values.get('types', [])
        
        if enabled and not types:
            raise ValueError("Si les agrégations sont activées, au moins un type doit être spécifié")
        
        return values


class SearchOptions(BaseModel):
    """Options de recherche."""
    include_highlights: bool = Field(default=False, description="Inclure le highlighting")
    include_explanation: bool = Field(default=False, description="Inclure l'explication du score")
    cache_enabled: bool = Field(default=True, description="Utiliser le cache")
    return_raw_elasticsearch: bool = Field(default=False, description="Retourner la requête ES brute")


class QueryMetadata(BaseModel):
    """Métadonnées de la requête."""
    query_id: str = Field(default_factory=lambda: str(uuid4()), description="ID unique de la requête")
    user_id: int = Field(..., description="ID de l'utilisateur")
    intent_type: IntentType = Field(..., description="Type d'intention détectée")
    confidence: float = Field(default=1.0, description="Confiance dans l'intention", ge=0.0, le=1.0)
    agent_name: Optional[str] = Field(default=None, description="Nom de l'agent AutoGen")
    team_name: Optional[str] = Field(default=None, description="Nom de l'équipe AutoGen")
    execution_context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Contexte d'exécution"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de la requête")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Valide que l'user_id est positif."""
        if v <= 0:
            raise ValueError("L'user_id doit être positif")
        return v


class SearchParameters(BaseModel):
    """Paramètres de recherche."""
    query_type: QueryType = Field(..., description="Type de requête")
    fields: List[str] = Field(default_factory=list, description="Champs à retourner")
    limit: int = Field(default=20, description="Nombre max de résultats", ge=1, le=100)
    offset: int = Field(default=0, description="Décalage pour la pagination", ge=0)
    timeout_ms: int = Field(default=5000, description="Timeout en millisecondes", ge=100, le=10000)
    sort: Optional[List[Dict[str, SortOrder]]] = Field(
        default=None, 
        description="Critères de tri"
    )


class FilterSet(BaseModel):
    """Ensemble de filtres organisés par type."""
    required: List[SearchFilter] = Field(
        default_factory=list, 
        description="Filtres obligatoires (AND)"
    )
    optional: List[SearchFilter] = Field(
        default_factory=list, 
        description="Filtres optionnels (SHOULD)"
    )
    ranges: List[SearchFilter] = Field(
        default_factory=list, 
        description="Filtres de plage (dates, montants)"
    )
    text_search: Optional[TextSearchConfig] = Field(
        default=None, 
        description="Configuration recherche textuelle"
    )
    
    @root_validator
    def validate_user_id_filter(cls, values):
        """Valide qu'un filtre user_id est toujours présent pour la sécurité."""
        required = values.get('required', [])
        
        has_user_filter = any(
            f.field == "user_id" and f.operator == FilterOperator.EQ
            for f in required
        )
        
        if not has_user_filter:
            raise ValueError("Un filtre user_id obligatoire est requis pour la sécurité")
        
        return values


# === CONTRAT DE REQUÊTE ===

class SearchServiceQuery(BaseModel):
    """
    Contrat de requête standardisé du Search Service.
    
    Ce modèle définit l'interface stable entre le Conversation Service
    et le Search Service pour toutes les requêtes de recherche.
    """
    query_metadata: QueryMetadata = Field(..., description="Métadonnées de la requête")
    search_parameters: SearchParameters = Field(..., description="Paramètres de recherche")
    filters: FilterSet = Field(..., description="Ensemble de filtres")
    aggregations: AggregationRequest = Field(
        default_factory=AggregationRequest, 
        description="Configuration des agrégations"
    )
    options: SearchOptions = Field(
        default_factory=SearchOptions, 
        description="Options de recherche"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# === MODÈLES DE RÉPONSE ===

class SearchResultItem(BaseModel):
    """Item de résultat de recherche."""
    # Identifiants
    transaction_id: str = Field(..., description="ID unique de la transaction")
    user_id: int = Field(..., description="ID de l'utilisateur")
    account_id: int = Field(..., description="ID du compte")
    
    # Données financières
    amount: float = Field(..., description="Montant avec signe")
    amount_abs: float = Field(..., description="Valeur absolue du montant")
    transaction_type: str = Field(..., description="Type de transaction (debit/credit)")
    currency_code: str = Field(..., description="Code devise")
    
    # Dates
    date: str = Field(..., description="Date de la transaction (YYYY-MM-DD)")
    month_year: str = Field(..., description="Mois-année (YYYY-MM)")
    weekday: Optional[str] = Field(default=None, description="Jour de la semaine")
    
    # Descriptions
    primary_description: str = Field(..., description="Description principale")
    merchant_name: Optional[str] = Field(default=None, description="Nom du marchand")
    category_name: Optional[str] = Field(default=None, description="Nom de la catégorie")
    operation_type: Optional[str] = Field(default=None, description="Type d'opération")
    
    # Métadonnées de recherche
    score: float = Field(..., description="Score de pertinence", ge=0.0)
    highlights: Optional[Dict[str, List[str]]] = Field(
        default=None, 
        description="Passages surlignés"
    )
    
    # Champs optionnels pour l'enrichissement
    searchable_text: Optional[str] = Field(default=None, description="Texte enrichi pour recherche")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "user_34_tx_12345",
                "user_id": 34,
                "account_id": 101,
                "amount": -45.67,
                "amount_abs": 45.67,
                "transaction_type": "debit",
                "currency_code": "EUR",
                "date": "2024-01-15",
                "month_year": "2024-01",
                "weekday": "Monday",
                "primary_description": "RESTAURANT LE BISTROT",
                "merchant_name": "Le Bistrot",
                "category_name": "Restaurant",
                "operation_type": "card_payment",
                "score": 1.0,
                "highlights": {
                    "primary_description": ["RESTAURANT <em>LE BISTROT</em>"]
                }
            }
        }


class AggregationBucket(BaseModel):
    """Bucket d'agrégation."""
    key: Union[str, int, float] = Field(..., description="Clé du bucket")
    doc_count: int = Field(..., description="Nombre de documents")
    total_amount: Optional[float] = Field(default=None, description="Somme des montants")
    avg_amount: Optional[float] = Field(default=None, description="Moyenne des montants")
    min_amount: Optional[float] = Field(default=None, description="Montant minimum")
    max_amount: Optional[float] = Field(default=None, description="Montant maximum")


class AggregationResult(BaseModel):
    """Résultat d'agrégation."""
    # Métriques globales
    total_amount: Optional[float] = Field(default=None, description="Somme totale")
    transaction_count: int = Field(..., description="Nombre total de transactions")
    average_amount: Optional[float] = Field(default=None, description="Montant moyen")
    
    # Buckets par groupement
    by_month: Optional[List[AggregationBucket]] = Field(
        default=None, 
        description="Agrégation par mois"
    )
    by_category: Optional[List[AggregationBucket]] = Field(
        default=None, 
        description="Agrégation par catégorie"
    )
    by_merchant: Optional[List[AggregationBucket]] = Field(
        default=None, 
        description="Agrégation par marchand"
    )
    
    # Statistiques détaillées
    statistics: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Statistiques détaillées (min, max, std_dev, etc.)"
    )


class SearchPerformance(BaseModel):
    """Métriques de performance de la recherche."""
    query_complexity: str = Field(..., description="Complexité de la requête (simple/complex)")
    optimization_applied: List[str] = Field(
        default_factory=list, 
        description="Optimisations appliquées"
    )
    index_used: str = Field(..., description="Index Elasticsearch utilisé")
    shards_queried: int = Field(..., description="Nombre de shards interrogés")
    cache_hit: bool = Field(..., description="Résultat du cache")


class ContextEnrichment(BaseModel):
    """Enrichissement contextuel pour le Conversation Service."""
    search_intent_matched: bool = Field(..., description="Intention de recherche correspondante")
    result_quality_score: float = Field(..., description="Score de qualité des résultats", ge=0.0, le=1.0)
    suggested_followup_questions: List[str] = Field(
        default_factory=list, 
        description="Questions de suivi suggérées"
    )
    detected_patterns: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Patterns détectés dans les résultats"
    )


class ResponseMetadata(BaseModel):
    """Métadonnées de la réponse."""
    query_id: str = Field(..., description="ID de la requête correspondante")
    execution_time_ms: int = Field(..., description="Temps d'exécution en millisecondes")
    total_hits: int = Field(..., description="Nombre total de résultats trouvés")
    returned_hits: int = Field(..., description="Nombre de résultats retournés")
    has_more: bool = Field(..., description="Y a-t-il plus de résultats")
    cache_hit: bool = Field(..., description="Résultat du cache")
    elasticsearch_took: int = Field(..., description="Temps Elasticsearch en millisecondes")
    
    # Contexte pour le Conversation Service
    agent_context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Contexte pour les agents AutoGen"
    )


class SearchServiceDebug(BaseModel):
    """Informations de debug optionnelles."""
    elasticsearch_query: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Requête Elasticsearch brute"
    )
    query_explanation: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Explication du scoring"
    )
    template_used: Optional[str] = Field(
        default=None, 
        description="Template de requête utilisé"
    )
    optimization_details: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Détails des optimisations"
    )


# === CONTRAT DE RÉPONSE ===

class SearchServiceResponse(BaseModel):
    """
    Contrat de réponse standardisé du Search Service.
    
    Ce modèle définit l'interface stable pour toutes les réponses
    du Search Service vers le Conversation Service.
    """
    response_metadata: ResponseMetadata = Field(..., description="Métadonnées de la réponse")
    results: List[SearchResultItem] = Field(
        default_factory=list, 
        description="Résultats de recherche"
    )
    aggregations: Optional[AggregationResult] = Field(
        default=None, 
        description="Résultats d'agrégation"
    )
    performance: SearchPerformance = Field(..., description="Métriques de performance")
    context_enrichment: ContextEnrichment = Field(..., description="Enrichissement contextuel")
    debug: Optional[SearchServiceDebug] = Field(
        default=None, 
        description="Informations de debug"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "response_metadata": {
                    "query_id": "uuid-v4",
                    "execution_time_ms": 45,
                    "total_hits": 156,
                    "returned_hits": 20,
                    "has_more": True,
                    "cache_hit": False,
                    "elasticsearch_took": 23,
                    "agent_context": {
                        "requesting_agent": "query_generator_agent",
                        "requesting_team": "financial_analysis_team",
                        "next_suggested_agent": "response_generator_agent"
                    }
                },
                "results": [
                    {
                        "transaction_id": "user_34_tx_12345",
                        "user_id": 34,
                        "account_id": 101,
                        "amount": -45.67,
                        "amount_abs": 45.67,
                        "transaction_type": "debit",
                        "currency_code": "EUR",
                        "date": "2024-01-15",
                        "primary_description": "RESTAURANT LE BISTROT",
                        "merchant_name": "Le Bistrot",
                        "category_name": "Restaurant",
                        "operation_type": "card_payment",
                        "month_year": "2024-01",
                        "weekday": "Monday",
                        "score": 1.0
                    }
                ],
                "aggregations": {
                    "total_amount": -1247.89,
                    "transaction_count": 156,
                    "average_amount": -7.99,
                    "by_month": [
                        {"key": "2024-01", "doc_count": 45, "total_amount": -567.89}
                    ]
                },
                "performance": {
                    "query_complexity": "simple",
                    "optimization_applied": ["user_filter", "category_filter"],
                    "index_used": "harena_transactions",
                    "shards_queried": 1,
                    "cache_hit": False
                },
                "context_enrichment": {
                    "search_intent_matched": True,
                    "result_quality_score": 0.92,
                    "suggested_followup_questions": [
                        "Voir détails de ces restaurants",
                        "Comparer avec mois précédent"
                    ]
                }
            }
        }


# === MODÈLES D'ERREUR ===

class SearchServiceError(BaseModel):
    """Erreur du Search Service."""
    error_code: str = Field(..., description="Code d'erreur")
    error_message: str = Field(..., description="Message d'erreur")
    query_id: Optional[str] = Field(default=None, description="ID de la requête en erreur")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Détails supplémentaires")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de l'erreur")


# === VALIDATORS DE CONTRATS ===

class ContractValidator:
    """Validateur pour les contrats d'interface."""
    
    @staticmethod
    def validate_search_query(query: SearchServiceQuery) -> None:
        """
        Valide un contrat de requête.
        
        Args:
            query: Requête à valider
            
        Raises:
            ValueError: Si la validation échoue
        """
        # Validation métadonnées obligatoires
        if query.query_metadata.user_id <= 0:
            raise ValueError("user_id doit être positif")
        
        if not query.query_metadata.intent_type:
            raise ValueError("intent_type est obligatoire")
        
        # Validation sécurité : user_id filter obligatoire
        user_filter_exists = any(
            f.field == "user_id" and f.operator == FilterOperator.EQ
            for f in query.filters.required
        )
        
        if not user_filter_exists:
            raise ValueError("Un filtre user_id est obligatoire pour la sécurité")
        
        # Validation limites performance
        if query.search_parameters.limit > 100:
            raise ValueError("La limite ne peut pas dépasser 100")
        
        if query.search_parameters.timeout_ms > 10000:
            raise ValueError("Le timeout ne peut pas dépasser 10 secondes")
    
    @staticmethod
    def validate_search_response(response: SearchServiceResponse) -> None:
        """
        Valide un contrat de réponse.
        
        Args:
            response: Réponse à valider
            
        Raises:
            ValueError: Si la validation échoue
        """
        # Validation structure réponse
        if not response.response_metadata.query_id:
            raise ValueError("query_id est obligatoire dans la réponse")
        
        if response.response_metadata.execution_time_ms < 0:
            raise ValueError("execution_time_ms doit être positif")
        
        if len(response.results) > response.response_metadata.returned_hits:
            raise ValueError("Le nombre de résultats dépasse returned_hits")
        
        # Validation cohérence données
        if response.aggregations and response.aggregations.transaction_count < 0:
            raise ValueError("transaction_count ne peut pas être négatif")
        
        # Validation score qualité
        if not (0.0 <= response.context_enrichment.result_quality_score <= 1.0):
            raise ValueError("result_quality_score doit être entre 0.0 et 1.0")