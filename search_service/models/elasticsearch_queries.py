"""
Modèles de requêtes Elasticsearch pour le Search Service.

Ces modèles définissent les structures de données pour toutes les requêtes
Elasticsearch supportées par le service de recherche, avec validation stricte
et optimisation pour le domaine financier.

ARCHITECTURE:
- Requêtes typées par cas d'usage financier
- Validation stricte des paramètres Elasticsearch
- Builders et factories pour construction simplifiée
- Support des requêtes complexes et agrégations
- Optimisation performance pour transactions

CONFIGURATION CENTRALISÉE:
- Timeouts et limites via config_service
- Champs et index configurables
- Scoring et boost personnalisables
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union, Literal, Type
from enum import Enum
from abc import ABC, abstractmethod
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveInt, NonNegativeInt, NonNegativeFloat

# Configuration centralisée
from config_service.config import settings

# ==================== ENUMS ET CONSTANTES ====================

class QueryType(str, Enum):
    """Types de requêtes Elasticsearch supportées."""
    BOOL = "bool"
    MATCH = "match"
    MATCH_ALL = "match_all"
    MATCH_PHRASE = "match_phrase"
    MATCH_PHRASE_PREFIX = "match_phrase_prefix"
    MULTI_MATCH = "multi_match"
    TERM = "term"
    TERMS = "terms"
    RANGE = "range"
    EXISTS = "exists"
    WILDCARD = "wildcard"
    REGEX = "regexp"
    FUZZY = "fuzzy"
    PREFIX = "prefix"
    NESTED = "nested"
    FUNCTION_SCORE = "function_score"

class BoolOperator(str, Enum):
    """Opérateurs pour les requêtes bool."""
    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"
    FILTER = "filter"

class MultiMatchType(str, Enum):
    """Types de requêtes multi_match."""
    BEST_FIELDS = "best_fields"
    MOST_FIELDS = "most_fields"
    CROSS_FIELDS = "cross_fields"
    PHRASE = "phrase"
    PHRASE_PREFIX = "phrase_prefix"
    BOOL_PREFIX = "bool_prefix"

class SortOrder(str, Enum):
    """Ordres de tri."""
    ASC = "asc"
    DESC = "desc"

class SortMode(str, Enum):
    """Modes de tri."""
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    AVG = "avg"
    MEDIAN = "median"

class AggregationType(str, Enum):
    """Types d'agrégations Elasticsearch."""
    TERMS = "terms"
    DATE_HISTOGRAM = "date_histogram"
    HISTOGRAM = "histogram"
    RANGE = "range"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "value_count"
    STATS = "stats"
    EXTENDED_STATS = "extended_stats"
    CARDINALITY = "cardinality"
    PERCENTILES = "percentiles"

class ScoreFunctionType(str, Enum):
    """Types de fonctions de score."""
    SCRIPT_SCORE = "script_score"
    WEIGHT = "weight"
    RANDOM_SCORE = "random_score"
    FIELD_VALUE_FACTOR = "field_value_factor"
    DECAY = "decay"

# ==================== MODÈLES DE BASE ====================

class BaseElasticsearchQuery(BaseModel, ABC):
    """
    Classe de base pour toutes les requêtes Elasticsearch.
    
    Définit l'interface commune et les validations de base.
    """
    query_type: QueryType = Field(..., description="Type de requête Elasticsearch")
    boost: Optional[float] = Field(default=1.0, ge=0.1, le=100.0, description="Facteur de boost")
    
    @field_validator('boost')
    @classmethod
    def validate_boost(cls, v):
        """Valide le facteur de boost."""
        if v is not None and (v <= 0 or v > 100):
            raise ValueError("Le boost doit être entre 0.1 et 100")
        return v
    
    @abstractmethod
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        pass
    
    @abstractmethod
    def estimate_performance_impact(self) -> int:
        """Estime l'impact performance (1-10, 10 = très lourd)."""
        pass

# ==================== REQUÊTES SIMPLES ====================

class MatchQuery(BaseElasticsearchQuery):
    """Requête match pour recherche textuelle."""
    query_type: Literal[QueryType.MATCH] = QueryType.MATCH
    field: str = Field(..., description="Champ à rechercher")
    query: str = Field(..., description="Texte à rechercher")
    operator: Literal["and", "or"] = Field(default="or", description="Opérateur logique")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Niveau de fuzziness")
    prefix_length: Optional[int] = Field(None, ge=0, description="Longueur de préfixe")
    max_expansions: Optional[int] = Field(None, ge=1, le=1000, description="Expansions maximales")
    analyzer: Optional[str] = Field(None, description="Analyseur à utiliser")
    
    @field_validator('query')
    @classmethod
    def validate_query_text(cls, v):
        """Valide le texte de requête."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Le texte de requête ne peut pas être vide")
        if len(v) > settings.MAX_QUERY_LENGTH:
            raise ValueError(f"Texte trop long (max {settings.MAX_QUERY_LENGTH})")
        return v.strip()
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Valide que le champ est autorisé."""
        allowed_fields = getattr(settings, 'ALLOWED_SEARCH_FIELDS', [])
        if allowed_fields and v not in allowed_fields:
            raise ValueError(f"Champ non autorisé: {v}")
        return v
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        match_params = {
            "query": self.query,
            "operator": self.operator
        }
        
        if self.boost and self.boost != 1.0:
            match_params["boost"] = self.boost
        if self.fuzziness is not None:
            match_params["fuzziness"] = self.fuzziness
        if self.prefix_length is not None:
            match_params["prefix_length"] = self.prefix_length
        if self.max_expansions is not None:
            match_params["max_expansions"] = self.max_expansions
        if self.analyzer:
            match_params["analyzer"] = self.analyzer
        
        return {"match": {self.field: match_params}}
    
    def estimate_performance_impact(self) -> int:
        """Estime l'impact performance."""
        impact = 3  # Base pour match
        
        if self.fuzziness:
            impact += 2  # Fuzziness augmente la complexité
        if self.operator == "and":
            impact += 1  # AND plus restrictif
        if len(self.query) > 50:
            impact += 1  # Texte long
        
        return min(impact, 10)

class TermQuery(BaseElasticsearchQuery):
    """Requête term pour correspondance exacte."""
    query_type: Literal[QueryType.TERM] = QueryType.TERM
    field: str = Field(..., description="Champ à filtrer")
    value: Union[str, int, float, bool] = Field(..., description="Valeur exacte")
    case_insensitive: Optional[bool] = Field(None, description="Insensible à la casse")
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Valide le champ."""
        allowed_fields = getattr(settings, 'ALLOWED_SEARCH_FIELDS', [])
        if allowed_fields and v not in allowed_fields:
            raise ValueError(f"Champ non autorisé: {v}")
        return v
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        term_params = {"value": self.value}
        
        if self.boost and self.boost != 1.0:
            term_params["boost"] = self.boost
        if self.case_insensitive is not None:
            term_params["case_insensitive"] = self.case_insensitive
        
        return {"term": {self.field: term_params}}
    
    def estimate_performance_impact(self) -> int:
        """Estime l'impact performance."""
        return 1  # Term est très performant

class TermsQuery(BaseElasticsearchQuery):
    """Requête terms pour correspondance multiple."""
    query_type: Literal[QueryType.TERMS] = QueryType.TERMS
    field: str = Field(..., description="Champ à filtrer")
    values: List[Union[str, int, float, bool]] = Field(..., min_length=1, description="Liste de valeurs")
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Valide le champ."""
        allowed_fields = getattr(settings, 'ALLOWED_SEARCH_FIELDS', [])
        if allowed_fields and v not in allowed_fields:
            raise ValueError(f"Champ non autorisé: {v}")
        return v
    
    @field_validator('values')
    @classmethod
    def validate_values(cls, v):
        """Valide la liste de valeurs."""
        if len(v) > settings.MAX_FILTER_VALUES:
            raise ValueError(f"Trop de valeurs (max {settings.MAX_FILTER_VALUES})")
        return v
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        terms_params = {"values": self.values}
        
        if self.boost and self.boost != 1.0:
            terms_params["boost"] = self.boost
        
        return {"terms": {self.field: terms_params}}
    
    def estimate_performance_impact(self) -> int:
        """Estime l'impact performance."""
        impact = 2  # Base pour terms
        if len(self.values) > 10:
            impact += 1  # Plus de valeurs = plus lourd
        return min(impact, 10)

class RangeQuery(BaseElasticsearchQuery):
    """Requête range pour plages de valeurs."""
    query_type: Literal[QueryType.RANGE] = QueryType.RANGE
    field: str = Field(..., description="Champ de plage")
    gte: Optional[Union[int, float, datetime, str]] = Field(None, description="Supérieur ou égal")
    gt: Optional[Union[int, float, datetime, str]] = Field(None, description="Supérieur à")
    lte: Optional[Union[int, float, datetime, str]] = Field(None, description="Inférieur ou égal")
    lt: Optional[Union[int, float, datetime, str]] = Field(None, description="Inférieur à")
    format: Optional[str] = Field(None, description="Format pour les dates")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    
    @model_validator(mode='after')
    def validate_range_bounds(self):
        """Valide les bornes de la plage."""
        bounds = [self.gte, self.gt, self.lte, self.lt]
        if all(b is None for b in bounds):
            raise ValueError("Au moins une borne doit être définie")
        
        # Vérifier la cohérence des bornes si définies
        if self.gte is not None and self.gt is not None:
            if self.gte <= self.gt:
                raise ValueError("gte doit être > gt si les deux sont définis")
        
        if self.lte is not None and self.lt is not None:
            if self.lte >= self.lt:
                raise ValueError("lte doit être < lt si les deux sont définis")
        
        return self
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        range_params = {}
        
        for param in ['gte', 'gt', 'lte', 'lt']:
            value = getattr(self, param)
            if value is not None:
                range_params[param] = value
        
        if self.boost and self.boost != 1.0:
            range_params["boost"] = self.boost
        if self.format:
            range_params["format"] = self.format
        if self.time_zone:
            range_params["time_zone"] = self.time_zone
        
        return {"range": {self.field: range_params}}
    
    def estimate_performance_impact(self) -> int:
        """Estime l'impact performance."""
        impact = 2  # Base pour range
        
        # Les plages ouvertes sont plus lourdes
        if self.gte is None and self.gt is None:
            impact += 1
        if self.lte is None and self.lt is None:
            impact += 1
        
        return min(impact, 10)

# ==================== REQUÊTES COMPLEXES ====================

class BoolQuery(BaseElasticsearchQuery):
    """Requête bool pour combinaisons logiques."""
    query_type: Literal[QueryType.BOOL] = QueryType.BOOL
    must: List[BaseElasticsearchQuery] = Field(default_factory=list, description="Clauses obligatoires")
    should: List[BaseElasticsearchQuery] = Field(default_factory=list, description="Clauses optionnelles")
    must_not: List[BaseElasticsearchQuery] = Field(default_factory=list, description="Clauses d'exclusion")
    filter: List[BaseElasticsearchQuery] = Field(default_factory=list, description="Clauses de filtrage")
    minimum_should_match: Optional[Union[int, str]] = Field(None, description="Minimum should match")
    
    @model_validator(mode='after')
    def validate_bool_query(self):
        """Valide la requête bool."""
        total_clauses = len(self.must) + len(self.should) + len(self.must_not) + len(self.filter)
        
        if total_clauses == 0:
            raise ValueError("Une requête bool doit contenir au moins une clause")
        
        max_clauses = getattr(settings, 'MAX_BOOL_CLAUSES', 100)
        if total_clauses > max_clauses:
            raise ValueError(f"Trop de clauses bool (max {max_clauses})")
        
        # Validation du minimum_should_match
        if self.minimum_should_match is not None and not self.should:
            raise ValueError("minimum_should_match requis uniquement avec des clauses should")
        
        return self
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        bool_params = {}
        
        if self.must:
            bool_params["must"] = [q.to_elasticsearch_dict() for q in self.must]
        if self.should:
            bool_params["should"] = [q.to_elasticsearch_dict() for q in self.should]
        if self.must_not:
            bool_params["must_not"] = [q.to_elasticsearch_dict() for q in self.must_not]
        if self.filter:
            bool_params["filter"] = [q.to_elasticsearch_dict() for q in self.filter]
        
        if self.minimum_should_match is not None:
            bool_params["minimum_should_match"] = self.minimum_should_match
        
        if self.boost and self.boost != 1.0:
            bool_params["boost"] = self.boost
        
        return {"bool": bool_params}
    
    def estimate_performance_impact(self) -> int:
        """Estime l'impact performance."""
        total_impact = 1  # Base pour bool
        
        # Additionner l'impact de toutes les clauses
        for clause_list in [self.must, self.should, self.must_not, self.filter]:
            for clause in clause_list:
                total_impact += clause.estimate_performance_impact()
        
        # Les clauses filter sont plus performantes
        filter_bonus = len(self.filter) * 0.5
        total_impact -= filter_bonus
        
        return min(int(total_impact), 10)

class MultiMatchQuery(BaseElasticsearchQuery):
    """Requête multi_match pour recherche sur plusieurs champs."""
    query_type: Literal[QueryType.MULTI_MATCH] = QueryType.MULTI_MATCH
    query: str = Field(..., description="Texte à rechercher")
    fields: List[str] = Field(..., min_length=1, description="Champs à rechercher")
    type: MultiMatchType = Field(default=MultiMatchType.BEST_FIELDS, description="Type de multi_match")
    operator: Literal["and", "or"] = Field(default="or", description="Opérateur logique")
    minimum_should_match: Optional[str] = Field(None, description="Minimum should match")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Niveau de fuzziness")
    tie_breaker: Optional[float] = Field(None, ge=0.0, le=1.0, description="Tie breaker")
    
    @field_validator('fields')
    @classmethod
    def validate_fields(cls, v):
        """Valide la liste des champs."""
        if len(v) > settings.MAX_SEARCH_FIELDS:
            raise ValueError(f"Trop de champs (max {settings.MAX_SEARCH_FIELDS})")
        
        allowed_fields = getattr(settings, 'ALLOWED_SEARCH_FIELDS', [])
        if allowed_fields:
            invalid_fields = [f for f in v if f not in allowed_fields]
            if invalid_fields:
                raise ValueError(f"Champs non autorisés: {invalid_fields}")
        
        return v
    
    @field_validator('query')
    @classmethod
    def validate_query_text(cls, v):
        """Valide le texte de requête."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Le texte de requête ne peut pas être vide")
        if len(v) > settings.MAX_QUERY_LENGTH:
            raise ValueError(f"Texte trop long (max {settings.MAX_QUERY_LENGTH})")
        return v.strip()
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        multi_match_params = {
            "query": self.query,
            "fields": self.fields,
            "type": self.type.value,
            "operator": self.operator
        }
        
        if self.boost and self.boost != 1.0:
            multi_match_params["boost"] = self.boost
        if self.minimum_should_match:
            multi_match_params["minimum_should_match"] = self.minimum_should_match
        if self.fuzziness is not None:
            multi_match_params["fuzziness"] = self.fuzziness
        if self.tie_breaker is not None:
            multi_match_params["tie_breaker"] = self.tie_breaker
        
        return {"multi_match": multi_match_params}
    
    def estimate_performance_impact(self) -> int:
        """Estime l'impact performance."""
        impact = 4  # Base pour multi_match
        
        impact += len(self.fields) * 0.5  # Plus de champs = plus lourd
        
        if self.fuzziness:
            impact += 2  # Fuzziness augmente la complexité
        if self.type in [MultiMatchType.CROSS_FIELDS, MultiMatchType.PHRASE]:
            impact += 1  # Types plus complexes
        
        return min(int(impact), 10)

# ==================== AGRÉGATIONS ====================

class BaseAggregation(BaseModel, ABC):
    """Classe de base pour les agrégations."""
    name: str = Field(..., description="Nom de l'agrégation")
    agg_type: AggregationType = Field(..., description="Type d'agrégation")
    
    @abstractmethod
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        pass

class TermsAggregation(BaseAggregation):
    """Agrégation terms pour grouper par valeurs."""
    agg_type: Literal[AggregationType.TERMS] = AggregationType.TERMS
    field: str = Field(..., description="Champ à agréger")
    size: int = Field(default=10, ge=1, le=1000, description="Nombre de buckets")
    order: Optional[Dict[str, str]] = Field(None, description="Ordre des buckets")
    min_doc_count: int = Field(default=1, ge=0, description="Nombre minimum de documents")
    include: Optional[Union[str, List[str]]] = Field(None, description="Valeurs à inclure")
    exclude: Optional[Union[str, List[str]]] = Field(None, description="Valeurs à exclure")
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v):
        """Valide la taille."""
        max_buckets = getattr(settings, 'MAX_AGGREGATION_BUCKETS', 1000)
        if v > max_buckets:
            raise ValueError(f"Trop de buckets (max {max_buckets})")
        return v
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        terms_params = {
            "field": self.field,
            "size": self.size,
            "min_doc_count": self.min_doc_count
        }
        
        if self.order:
            terms_params["order"] = self.order
        if self.include is not None:
            terms_params["include"] = self.include
        if self.exclude is not None:
            terms_params["exclude"] = self.exclude
        
        return {self.name: {"terms": terms_params}}

class DateHistogramAggregation(BaseAggregation):
    """Agrégation date_histogram pour grouper par périodes."""
    agg_type: Literal[AggregationType.DATE_HISTOGRAM] = AggregationType.DATE_HISTOGRAM
    field: str = Field(..., description="Champ de date")
    calendar_interval: Optional[str] = Field(None, description="Intervalle calendaire")
    fixed_interval: Optional[str] = Field(None, description="Intervalle fixe")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    min_doc_count: int = Field(default=0, ge=0, description="Nombre minimum de documents")
    extended_bounds: Optional[Dict[str, str]] = Field(None, description="Bornes étendues")
    
    @model_validator(mode='after')
    def validate_interval(self):
        """Valide qu'un intervalle est défini."""
        if not self.calendar_interval and not self.fixed_interval:
            raise ValueError("calendar_interval ou fixed_interval requis")
        
        if self.calendar_interval and self.fixed_interval:
            raise ValueError("Seul un type d'intervalle peut être défini")
        
        return self
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        date_histogram_params = {
            "field": self.field,
            "min_doc_count": self.min_doc_count
        }
        
        if self.calendar_interval:
            date_histogram_params["calendar_interval"] = self.calendar_interval
        if self.fixed_interval:
            date_histogram_params["fixed_interval"] = self.fixed_interval
        if self.time_zone:
            date_histogram_params["time_zone"] = self.time_zone
        if self.extended_bounds:
            date_histogram_params["extended_bounds"] = self.extended_bounds
        
        return {self.name: {"date_histogram": date_histogram_params}}

class MetricAggregation(BaseAggregation):
    """Agrégation métrique (sum, avg, min, max, etc.)."""
    agg_type: AggregationType = Field(..., description="Type d'agrégation métrique")
    field: str = Field(..., description="Champ à agréger")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur par défaut")
    script: Optional[str] = Field(None, description="Script personnalisé")
    
    @model_validator(mode='after')
    def validate_metric_type(self):
        """Valide le type d'agrégation métrique."""
        metric_types = {
            AggregationType.SUM, AggregationType.AVG, AggregationType.MIN,
            AggregationType.MAX, AggregationType.COUNT, AggregationType.STATS,
            AggregationType.EXTENDED_STATS, AggregationType.CARDINALITY
        }
        
        if self.agg_type not in metric_types:
            raise ValueError(f"Type d'agrégation métrique invalide: {self.agg_type}")
        
        return self
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        metric_params = {}
        
        if self.script:
            metric_params["script"] = self.script
        else:
            metric_params["field"] = self.field
        
        if self.missing is not None:
            metric_params["missing"] = self.missing
        
        agg_name = self.agg_type.value
        return {self.name: {agg_name: metric_params}}

# ==================== REQUÊTE COMPLÈTE ====================

class ElasticsearchQuery(BaseModel):
    """
    Requête Elasticsearch complète avec tous les paramètres.
    
    Modèle principal pour construire des requêtes complexes avec
    query, agrégations, tri, pagination, etc.
    """
    # Requête principale
    query: Optional[BaseElasticsearchQuery] = Field(None, description="Requête principale")
    
    # Agrégations
    aggregations: List[BaseAggregation] = Field(default_factory=list, description="Agrégations")
    
    # Tri
    sort: List[Dict[str, Any]] = Field(default_factory=list, description="Critères de tri")
    
    # Pagination
    from_: int = Field(default=0, alias="from", ge=0, description="Offset")
    size: int = Field(default=10, ge=0, le=10000, description="Taille")
    
    # Options
    track_total_hits: bool = Field(default=True, description="Tracker le total")
    track_scores: bool = Field(default=True, description="Calculer les scores")
    explain: bool = Field(default=False, description="Expliquer le score")
    version: bool = Field(default=False, description="Inclure la version")
    
    # Source et highlights
    source: Optional[Union[bool, List[str], Dict[str, List[str]]]] = Field(
        None, description="Champs source à inclure/exclure"
    )
    highlight: Optional[Dict[str, Any]] = Field(None, description="Configuration highlight")
    
    # Performance
    timeout: Optional[str] = Field(None, description="Timeout de la requête")
    terminate_after: Optional[int] = Field(None, description="Terminer après N docs")
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v):
        """Valide la taille de pagination."""
        max_results = getattr(settings, 'MAX_SEARCH_RESULTS', 10000)
        if v > max_results:
            raise ValueError(f"Taille trop grande (max {max_results})")
        return v
    
    @field_validator('from_')
    @classmethod
    def validate_from(cls, v):
        """Valide l'offset de pagination."""
        max_offset = getattr(settings, 'MAX_SEARCH_OFFSET', 100000)
        if v > max_offset:
            raise ValueError(f"Offset trop grand (max {max_offset})")
        return v
    
    @model_validator(mode='after')
    def validate_complete_query(self):
        """Valide la cohérence de la requête complète."""
        # Validation pagination
        if self.from_ + self.size > getattr(settings, 'MAX_SEARCH_OFFSET', 100000):
            raise ValueError("Pagination trop profonde")
        
        # Validation agrégations
        if len(self.aggregations) > getattr(settings, 'MAX_AGGREGATIONS', 50):
            raise ValueError("Trop d'agrégations")
        
        # Validation performance
        total_impact = 0
        if self.query:
            total_impact += self.query.estimate_performance_impact()
        
        total_impact += len(self.aggregations) * 2
        total_impact += len(self.sort) * 1
        
        if total_impact > 50:
            raise ValueError("Requête trop complexe, impact performance élevé")
        
        return self
    
    def to_elasticsearch_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch complet."""
        es_query = {}
        
        # Requête principale
        if self.query:
            es_query["query"] = self.query.to_elasticsearch_dict()
        else:
            es_query["query"] = {"match_all": {}}
        
        # Agrégations
        if self.aggregations:
            aggs = {}
            for agg in self.aggregations:
                agg_dict = agg.to_elasticsearch_dict()
                aggs.update(agg_dict)
            es_query["aggs"] = aggs
        
        # Tri
        if self.sort:
            es_query["sort"] = self.sort
        
        # Pagination
        es_query["from"] = self.from_
        es_query["size"] = self.size
        
        # Options
        if not self.track_total_hits:
            es_query["track_total_hits"] = False
        if not self.track_scores:
            es_query["track_scores"] = False
        if self.explain:
            es_query["explain"] = True
        if self.version:
            es_query["version"] = True
        
        # Source
        if self.source is not None:
            es_query["_source"] = self.source
        
        # Highlights
        if self.highlight:
            es_query["highlight"] = self.highlight
        
        # Performance
        if self.timeout:
            es_query["timeout"] = self.timeout
        if self.terminate_after:
            es_query["terminate_after"] = self.terminate_after
        
        return es_query

# ==================== BUILDERS ET FACTORIES ====================

class QueryBuilder:
    """Builder pour construire des requêtes Elasticsearch complexes."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'QueryBuilder':
        """Remet à zéro le builder."""
        self._query = None
        self._aggregations = []
        self._sort = []
        self._from = 0
        self._size = 10
        self._source = None
        self._highlight = None
        self._options = {}
        return self
    
    def query(self, query: BaseElasticsearchQuery) -> 'QueryBuilder':
        """Définit la requête principale."""
        self._query = query
        return self
    
    def match(self, field: str, query: str, **kwargs) -> 'QueryBuilder':
        """Ajoute une requête match."""
        self._query = MatchQuery(field=field, query=query, **kwargs)
        return self
    
    def term(self, field: str, value: Any, **kwargs) -> 'QueryBuilder':
        """Ajoute une requête term."""
        self._query = TermQuery(field=field, value=value, **kwargs)
        return self
    
    def range(self, field: str, **kwargs) -> 'QueryBuilder':
        """Ajoute une requête range."""
        self._query = RangeQuery(field=field, **kwargs)
        return self
    
    def bool_query(self) -> 'BoolQueryBuilder':
        """Démarre un builder de requête bool."""
        return BoolQueryBuilder(self)
    
    def add_aggregation(self, agg: BaseAggregation) -> 'QueryBuilder':
        """Ajoute une agrégation."""
        self._aggregations.append(agg)
        return self
    
    def terms_agg(self, name: str, field: str, **kwargs) -> 'QueryBuilder':
        """Ajoute une agrégation terms."""
        agg = TermsAggregation(name=name, field=field, **kwargs)
        return self.add_aggregation(agg)
    
    def date_histogram_agg(self, name: str, field: str, **kwargs) -> 'QueryBuilder':
        """Ajoute une agrégation date_histogram."""
        agg = DateHistogramAggregation(name=name, field=field, **kwargs)
        return self.add_aggregation(agg)
    
    def metric_agg(self, name: str, agg_type: AggregationType, field: str, **kwargs) -> 'QueryBuilder':
        """Ajoute une agrégation métrique."""
        agg = MetricAggregation(name=name, agg_type=agg_type, field=field, **kwargs)
        return self.add_aggregation(agg)
    
    def sort_by(self, field: str, order: SortOrder = SortOrder.DESC, **kwargs) -> 'QueryBuilder':
        """Ajoute un critère de tri."""
        sort_spec = {field: {"order": order.value}}
        sort_spec[field].update(kwargs)
        self._sort.append(sort_spec)
        return self
    
    def paginate(self, from_: int = 0, size: int = 10) -> 'QueryBuilder':
        """Configure la pagination."""
        self._from = from_
        self._size = size
        return self
    
    def source(self, includes: Optional[List[str]] = None, excludes: Optional[List[str]] = None) -> 'QueryBuilder':
        """Configure les champs source."""
        if includes or excludes:
            self._source = {}
            if includes:
                self._source["includes"] = includes
            if excludes:
                self._source["excludes"] = excludes
        return self
    
    def highlight(self, fields: List[str], **kwargs) -> 'QueryBuilder':
        """Configure les highlights."""
        highlight_fields = {field: {} for field in fields}
        self._highlight = {"fields": highlight_fields}
        self._highlight.update(kwargs)
        return self
    
    def timeout(self, timeout: str) -> 'QueryBuilder':
        """Configure le timeout."""
        self._options["timeout"] = timeout
        return self
    
    def build(self) -> ElasticsearchQuery:
        """Construit la requête finale."""
        return ElasticsearchQuery(
            query=self._query,
            aggregations=self._aggregations,
            sort=self._sort,
            from_=self._from,
            size=self._size,
            source=self._source,
            highlight=self._highlight,
            **self._options
        )

class BoolQueryBuilder:
    """Builder spécialisé pour les requêtes bool."""
    
    def __init__(self, parent_builder: QueryBuilder):
        self.parent = parent_builder
        self._must = []
        self._should = []
        self._must_not = []
        self._filter = []
        self._minimum_should_match = None
    
    def must(self, query: BaseElasticsearchQuery) -> 'BoolQueryBuilder':
        """Ajoute une clause must."""
        self._must.append(query)
        return self
    
    def should(self, query: BaseElasticsearchQuery) -> 'BoolQueryBuilder':
        """Ajoute une clause should."""
        self._should.append(query)
        return self
    
    def must_not(self, query: BaseElasticsearchQuery) -> 'BoolQueryBuilder':
        """Ajoute une clause must_not."""
        self._must_not.append(query)
        return self
    
    def filter(self, query: BaseElasticsearchQuery) -> 'BoolQueryBuilder':
        """Ajoute une clause filter."""
        self._filter.append(query)
        return self
    
    def minimum_should_match(self, value: Union[int, str]) -> 'BoolQueryBuilder':
        """Configure minimum_should_match."""
        self._minimum_should_match = value
        return self
    
    def end_bool(self) -> QueryBuilder:
        """Termine le builder bool et retourne au parent."""
        bool_query = BoolQuery(
            must=self._must,
            should=self._should,
            must_not=self._must_not,
            filter=self._filter,
            minimum_should_match=self._minimum_should_match
        )
        self.parent._query = bool_query
        return self.parent

# ==================== FACTORY FUNCTIONS ====================

def create_simple_search(
    text: str,
    fields: List[str],
    user_id: int,
    size: int = 10
) -> ElasticsearchQuery:
    """Crée une recherche textuelle simple."""
    # Requête multi_match
    query = MultiMatchQuery(
        query=text,
        fields=fields,
        type=MultiMatchType.BEST_FIELDS,
        fuzziness="AUTO"
    )
    
    # Filtre utilisateur pour sécurité
    user_filter = TermQuery(field="user_id", value=user_id)
    
    # Combiner avec bool
    bool_query = BoolQuery(
        must=[query],
        filter=[user_filter]
    )
    
    return ElasticsearchQuery(
        query=bool_query,
        size=size,
        highlight={"fields": {field: {} for field in fields}}
    )

def create_filtered_search(
    filters: List[BaseElasticsearchQuery],
    aggregations: Optional[List[BaseAggregation]] = None,
    sort_field: Optional[str] = None,
    size: int = 10
) -> ElasticsearchQuery:
    """Crée une recherche avec filtres."""
    bool_query = BoolQuery(filter=filters)
    
    query = ElasticsearchQuery(
        query=bool_query,
        aggregations=aggregations or [],
        size=size
    )
    
    if sort_field:
        query.sort = [{sort_field: {"order": "desc"}}]
    
    return query

# ==================== EXPORTS ====================

__all__ = [
    # Enums
    'QueryType',
    'BoolOperator',
    'MultiMatchType',
    'SortOrder',
    'SortMode',
    'AggregationType',
    'ScoreFunctionType',
    
    # Classes de base
    'BaseElasticsearchQuery',
    'BaseAggregation',
    
    # Requêtes simples
    'MatchQuery',
    'TermQuery',
    'TermsQuery',
    'RangeQuery',
    
    # Requêtes complexes
    'BoolQuery',
    'MultiMatchQuery',
    
    # Agrégations
    'TermsAggregation',
    'DateHistogramAggregation',
    'MetricAggregation',
    
    # Requête complète
    'ElasticsearchQuery',
    
    # Builders
    'QueryBuilder',
    'BoolQueryBuilder',
    
    # Factory functions
    'create_simple_search',
    'create_filtered_search'
]