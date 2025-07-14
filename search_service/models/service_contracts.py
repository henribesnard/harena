"""
Contrats d'interface standardisés entre Conversation Service et Search Service
Définit les formats exacts de communication avec validation Pydantic stricte
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from uuid import uuid4


class QueryType(str, Enum):
    """Types de requêtes supportés"""
    SIMPLE_SEARCH = "simple_search"
    FILTERED_SEARCH = "filtered_search"
    TEXT_SEARCH = "text_search"
    TEXT_SEARCH_WITH_FILTER = "text_search_with_filter"
    FILTERED_AGGREGATION = "filtered_aggregation"
    TEMPORAL_AGGREGATION = "temporal_aggregation"
    COMPLEX_QUERY = "complex_query"


class FilterOperator(str, Enum):
    """Opérateurs de filtrage supportés"""
    EQ = "eq"           # égal
    NE = "ne"           # différent
    GT = "gt"           # supérieur
    GTE = "gte"         # supérieur ou égal
    LT = "lt"           # inférieur
    LTE = "lte"         # inférieur ou égal
    BETWEEN = "between" # entre deux valeurs
    IN = "in"           # dans une liste
    NOT_IN = "not_in"   # pas dans une liste
    EXISTS = "exists"   # champ existe
    MATCH = "match"     # recherche textuelle
    PREFIX = "prefix"   # préfixe


class AggregationType(str, Enum):
    """Types d'agrégation disponibles"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    CARDINALITY = "cardinality"
    PERCENTILES = "percentiles"
    STATS = "stats"
    TERMS = "terms"
    DATE_HISTOGRAM = "date_histogram"
    HISTOGRAM = "histogram"


# === MODÈLES DE REQUÊTE ===

class SearchFilter(BaseModel):
    """Modèle d'un filtre de recherche"""
    field: str = Field(..., description="Nom du champ à filtrer")
    operator: FilterOperator = Field(..., description="Opérateur de filtrage")
    value: Union[str, int, float, List[Any], bool] = Field(..., description="Valeur de filtrage")
    
    @field_validator("field")
    @classmethod
    def validate_field_name(cls, v):
        """Valide que le nom de champ est autorisé"""
        from search_service.config import INDEXED_FIELDS
        allowed_fields = list(INDEXED_FIELDS.keys())
        if v not in allowed_fields:
            raise ValueError(f"Champ non autorisé: {v}. Champs disponibles: {allowed_fields}")
        return v
    
    @field_validator("value")
    @classmethod
    def validate_filter_value(cls, v, info):
        """Valide la valeur selon l'opérateur"""
        # Récupérer l'opérateur depuis les autres valeurs
        data = info.data if hasattr(info, 'data') else {}
        operator = data.get("operator")
        
        if operator == FilterOperator.BETWEEN:
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError("BETWEEN nécessite une liste de 2 valeurs [min, max]")
        
        elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(v, list):
                raise ValueError(f"{operator.value} nécessite une liste de valeurs")
        
        elif operator == FilterOperator.EXISTS:
            if not isinstance(v, bool):
                raise ValueError("EXISTS nécessite une valeur booléenne")
        
        return v


class SearchFilters(BaseModel):
    """Ensemble des filtres de recherche"""
    required: List[SearchFilter] = Field(
        default_factory=list,
        description="Filtres obligatoires (AND)"
    )
    optional: List[SearchFilter] = Field(
        default_factory=list,
        description="Filtres optionnels (OR)"
    )
    ranges: List[SearchFilter] = Field(
        default_factory=list,
        description="Filtres de plage (dates, montants)"
    )
    text_search: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration recherche textuelle"
    )
    
    @field_validator("required")
    @classmethod
    def ensure_user_id_filter(cls, v):
        """S'assure qu'un filtre user_id est présent pour la sécurité"""
        if not v:
            v = []
        
        # Vérifier si user_id est déjà présent
        has_user_id = any(f.field == "user_id" if hasattr(f, 'field') else f.get('field') == "user_id" for f in v)
        
        if not has_user_id:
            raise ValueError("Un filtre user_id est obligatoire pour des raisons de sécurité")
        
        return v


class TextSearchConfig(BaseModel):
    """Configuration pour la recherche textuelle"""
    query: str = Field(..., description="Terme de recherche")
    fields: List[str] = Field(
        default_factory=lambda: ["searchable_text", "primary_description", "merchant_name"],
        description="Champs où chercher"
    )
    operator: str = Field(default="match", description="Opérateur de recherche")
    boost: Optional[Dict[str, float]] = Field(
        default=None,
        description="Boost par champ"
    )
    fuzziness: Optional[str] = Field(
        default=None,
        description="Fuzziness pour typos (AUTO, 1, 2)"
    )
    minimum_should_match: Optional[str] = Field(
        default=None,
        description="Pourcentage de termes qui doivent matcher"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v):
        """Valide la longueur de la requête"""
        if len(v) > 500:
            raise ValueError("La requête ne peut pas dépasser 500 caractères")
        if len(v.strip()) == 0:
            raise ValueError("La requête ne peut pas être vide")
        return v.strip()


class AggregationRequest(BaseModel):
    """Configuration pour les agrégations"""
    enabled: bool = Field(default=True, description="Activer les agrégations")
    types: List[AggregationType] = Field(
        default_factory=lambda: [AggregationType.COUNT, AggregationType.SUM],
        description="Types d'agrégation à calculer"
    )
    group_by: List[str] = Field(
        default_factory=list,
        description="Champs de groupement"
    )
    metrics: List[str] = Field(
        default_factory=lambda: ["amount_abs", "transaction_id"],
        description="Champs sur lesquels calculer les métriques"
    )
    date_histogram_interval: Optional[str] = Field(
        default="month",
        description="Intervalle pour histogramme temporel"
    )
    terms_size: int = Field(
        default=10,
        description="Nombre de termes à retourner"
    )
    
    @field_validator("terms_size")
    @classmethod
    def validate_terms_size(cls, v):
        """Valide la taille des agrégations terms"""
        if v < 1 or v > 1000:
            raise ValueError("terms_size doit être entre 1 et 1000")
        return v


class SearchParameters(BaseModel):
    """Paramètres de recherche"""
    query_type: QueryType = Field(..., description="Type de requête")
    fields: List[str] = Field(
        default_factory=list,
        description="Champs à retourner (vide = tous)"
    )
    limit: int = Field(default=20, description="Nombre de résultats")
    offset: int = Field(default=0, description="Décalage pour pagination")
    timeout_ms: int = Field(default=5000, description="Timeout en millisecondes")
    sort: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Critères de tri"
    )
    
    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        """Valide la limite de résultats"""
        if v < 1 or v > 100:
            raise ValueError("limit doit être entre 1 et 100")
        return v
    
    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v):
        """Valide l'offset de pagination"""
        if v < 0 or v > 10000:
            raise ValueError("offset doit être entre 0 et 10000")
        return v
    
    @field_validator("timeout_ms")
    @classmethod
    def validate_timeout(cls, v):
        """Valide le timeout"""
        if v < 100 or v > 10000:
            raise ValueError("timeout_ms doit être entre 100 et 10000")
        return v


class ExecutionContext(BaseModel):
    """Contexte d'exécution de la requête pour traçabilité"""
    conversation_id: Optional[str] = Field(default=None, description="ID de conversation")
    turn_number: Optional[int] = Field(default=None, description="Numéro de tour")
    agent_chain: List[str] = Field(
        default_factory=list,
        description="Chaîne d'agents ayant traité la requête"
    )
    original_query: Optional[str] = Field(default=None, description="Requête utilisateur originale")
    processing_time_ms: Optional[int] = Field(default=None, description="Temps de traitement agents")


class QueryMetadata(BaseModel):
    """Métadonnées de la requête"""
    query_id: str = Field(default_factory=lambda: str(uuid4()), description="ID unique de requête")
    user_id: int = Field(..., description="ID utilisateur")
    intent_type: str = Field(..., description="Type d'intention détectée")
    confidence: float = Field(..., description="Confiance de classification")
    agent_name: str = Field(..., description="Nom de l'agent générateur")
    team_name: Optional[str] = Field(default=None, description="Nom de l'équipe AutoGen")
    execution_context: Optional[ExecutionContext] = Field(default=None, description="Contexte d'exécution")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de création")
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        """Valide l'ID utilisateur"""
        if v <= 0:
            raise ValueError("user_id doit être positif")
        return v
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Valide le score de confiance"""
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence doit être entre 0.0 et 1.0")
        return v


class SearchOptions(BaseModel):
    """Options avancées de recherche"""
    include_highlights: bool = Field(default=False, description="Inclure surlignage des termes")
    include_explanation: bool = Field(default=False, description="Inclure explication du score")
    cache_enabled: bool = Field(default=True, description="Utiliser le cache")
    return_raw_elasticsearch: bool = Field(default=False, description="Retourner réponse Elasticsearch brute")
    preference: Optional[str] = Field(default="_local", description="Préférence de routing")
    min_score: Optional[float] = Field(default=None, description="Score minimum")


# === CONTRAT DE REQUÊTE PRINCIPAL ===

class SearchServiceQuery(BaseModel):
    """
    Contrat principal de requête envoyé par le Conversation Service
    Format standardisé pour toutes les communications
    """
    query_metadata: QueryMetadata = Field(..., description="Métadonnées de requête")
    search_parameters: SearchParameters = Field(..., description="Paramètres de recherche")
    filters: SearchFilters = Field(..., description="Filtres de recherche")
    aggregations: Optional[AggregationRequest] = Field(default=None, description="Configuration agrégations")
    text_search: Optional[TextSearchConfig] = Field(default=None, description="Configuration recherche textuelle")
    options: SearchOptions = Field(default_factory=SearchOptions, description="Options avancées")
    
    @model_validator(mode='after')
    def validate_query_consistency(self):
        """Valide la cohérence globale de la requête"""
        query_type = self.search_parameters.query_type if self.search_parameters else None
        text_search = self.text_search
        
        # Si c'est une recherche textuelle, text_search doit être défini
        if query_type in [QueryType.TEXT_SEARCH, QueryType.TEXT_SEARCH_WITH_FILTER]:
            if not text_search:
                raise ValueError(f"text_search requis pour query_type {query_type}")
        
        # Si aggregations demandées, vérifier cohérence
        aggregations = self.aggregations
        if aggregations and aggregations.enabled:
            if query_type not in [QueryType.FILTERED_AGGREGATION, QueryType.TEMPORAL_AGGREGATION]:
                # Autoriser agrégations sur autres types mais avec avertissement
                pass
        
        return self

    class Config:
        """Configuration Pydantic"""
        validate_assignment = True
        extra = "forbid"  # Interdit champs supplémentaires


# === MODÈLES DE RÉPONSE ===

class SearchResult(BaseModel):
    """Un résultat de recherche individuel"""
    transaction_id: str = Field(..., description="ID unique de transaction")
    user_id: int = Field(..., description="ID utilisateur")
    account_id: Optional[int] = Field(default=None, description="ID compte")
    amount: float = Field(..., description="Montant avec signe")
    amount_abs: float = Field(..., description="Montant absolu")
    transaction_type: str = Field(..., description="Type transaction (debit/credit)")
    currency_code: str = Field(..., description="Code devise")
    date: str = Field(..., description="Date transaction")
    primary_description: str = Field(..., description="Description principale")
    merchant_name: Optional[str] = Field(default=None, description="Nom marchand")
    category_name: Optional[str] = Field(default=None, description="Nom catégorie")
    operation_type: str = Field(..., description="Type opération")
    month_year: str = Field(..., description="Mois-année")
    weekday: Optional[str] = Field(default=None, description="Jour de la semaine")
    score: float = Field(..., description="Score de pertinence")
    highlights: Optional[Dict[str, List[str]]] = Field(default=None, description="Surlignage termes")


class AggregationBucket(BaseModel):
    """Bucket d'agrégation"""
    key: Union[str, int, float] = Field(..., description="Clé du bucket")
    doc_count: int = Field(..., description="Nombre de documents")
    total_amount: Optional[float] = Field(default=None, description="Somme des montants")
    avg_amount: Optional[float] = Field(default=None, description="Moyenne des montants")
    min_amount: Optional[float] = Field(default=None, description="Montant minimum")
    max_amount: Optional[float] = Field(default=None, description="Montant maximum")


class AggregationResult(BaseModel):
    """Résultats d'agrégation"""
    total_amount: Optional[float] = Field(default=None, description="Montant total")
    transaction_count: int = Field(..., description="Nombre total de transactions")
    average_amount: Optional[float] = Field(default=None, description="Montant moyen")
    by_month: List[AggregationBucket] = Field(default_factory=list, description="Agrégation par mois")
    by_category: List[AggregationBucket] = Field(default_factory=list, description="Agrégation par catégorie")
    by_merchant: List[AggregationBucket] = Field(default_factory=list, description="Agrégation par marchand")
    statistics: Optional[Dict[str, float]] = Field(default=None, description="Statistiques générales")


class PerformanceMetrics(BaseModel):
    """Métriques de performance"""
    query_complexity: str = Field(..., description="Complexité de la requête")
    optimization_applied: List[str] = Field(default_factory=list, description="Optimisations appliquées")
    index_used: str = Field(..., description="Index utilisé")
    shards_queried: int = Field(..., description="Nombre de shards interrogés")
    cache_hit: bool = Field(..., description="Cache utilisé")


class ContextEnrichment(BaseModel):
    """Enrichissement contextuel pour les agents"""
    search_intent_matched: bool = Field(..., description="Intention de recherche matched")
    result_quality_score: float = Field(..., description="Score qualité des résultats")
    suggested_followup_questions: List[str] = Field(
        default_factory=list,
        description="Questions de suivi suggérées"
    )
    next_suggested_agent: Optional[str] = Field(
        default=None,
        description="Agent suggéré pour la suite"
    )


class ResponseMetadata(BaseModel):
    """Métadonnées de la réponse"""
    query_id: str = Field(..., description="ID de la requête originale")
    execution_time_ms: int = Field(..., description="Temps d'exécution total")
    total_hits: int = Field(..., description="Nombre total de résultats")
    returned_hits: int = Field(..., description="Nombre de résultats retournés")
    has_more: bool = Field(..., description="Plus de résultats disponibles")
    cache_hit: bool = Field(..., description="Résultat servi depuis le cache")
    elasticsearch_took: int = Field(..., description="Temps Elasticsearch en ms")
    agent_context: Optional[Dict[str, str]] = Field(
        default=None,
        description="Contexte pour les agents AutoGen"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de réponse")


# === CONTRAT DE RÉPONSE PRINCIPAL ===

class SearchServiceResponse(BaseModel):
    """
    Contrat principal de réponse retourné par le Search Service
    Format standardisé pour toutes les communications
    """
    response_metadata: ResponseMetadata = Field(..., description="Métadonnées de réponse")
    results: List[SearchResult] = Field(default_factory=list, description="Résultats de recherche")
    aggregations: Optional[AggregationResult] = Field(default=None, description="Résultats d'agrégation")
    performance: PerformanceMetrics = Field(..., description="Métriques de performance")
    context_enrichment: ContextEnrichment = Field(..., description="Enrichissement contextuel")
    debug: Optional[Dict[str, Any]] = Field(default=None, description="Informations de debug")
    
    class Config:
        """Configuration Pydantic"""
        validate_assignment = True


# === HELPERS ET VALIDATEURS ===

class ContractValidator:
    """Validateur de contrats pour garantir la cohérence"""
    
    @staticmethod
    def validate_search_query(query: SearchServiceQuery) -> bool:
        """Valide un contrat de requête"""
        try:
            # Validation sécurité : user_id obligatoire
            user_filters = [f for f in query.filters.required if f.field == "user_id"]
            if not user_filters:
                raise ValueError("Filtre user_id obligatoire manquant")
            
            # Validation cohérence metadata/filters
            metadata_user_id = query.query_metadata.user_id
            filter_user_id = user_filters[0].value
            if metadata_user_id != filter_user_id:
                raise ValueError("user_id incohérent entre metadata et filters")
            
            # Validation limites
            if query.search_parameters.limit > 100:
                raise ValueError("limit ne peut pas dépasser 100")
            
            return True
        except Exception as e:
            raise ValueError(f"Validation échec: {e}")
    
    @staticmethod
    def validate_search_response(response: SearchServiceResponse) -> bool:
        """Valide un contrat de réponse"""
        try:
            # Validation cohérence counts
            if len(response.results) != response.response_metadata.returned_hits:
                raise ValueError("Incohérence entre results et returned_hits")
            
            # Validation métriques
            if response.response_metadata.execution_time_ms < 0:
                raise ValueError("execution_time_ms ne peut pas être négatif")
            
            # Validation agrégations si présentes
            if response.aggregations:
                if response.aggregations.transaction_count < 0:
                    raise ValueError("transaction_count ne peut pas être négatif")
            
            return True
        except Exception as e:
            raise ValueError(f"Validation réponse échec: {e}")


# === EXPORTS ===

__all__ = [
    # Enums
    "QueryType",
    "FilterOperator", 
    "AggregationType",
    
    # Modèles de requête
    "SearchFilter",
    "SearchFilters",
    "TextSearchConfig",
    "AggregationRequest",
    "SearchParameters",
    "QueryMetadata",
    "ExecutionContext",
    "SearchOptions",
    
    # Contrat principal requête
    "SearchServiceQuery",
    
    # Modèles de réponse
    "SearchResult",
    "AggregationBucket",
    "AggregationResult", 
    "PerformanceMetrics",
    "ContextEnrichment",
    "ResponseMetadata",
    
    # Contrat principal réponse
    "SearchServiceResponse",
    
    # Validateur
    "ContractValidator"
]