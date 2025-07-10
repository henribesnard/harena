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

from pydantic import BaseModel, Field, validator, root_validator
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
    DATE_RANGE = "date_range"
    FILTER = "filter"
    FILTERS = "filters"
    NESTED = "nested"
    REVERSE_NESTED = "reverse_nested"
    CARDINALITY = "cardinality"
    VALUE_COUNT = "value_count"
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STATS = "stats"
    EXTENDED_STATS = "extended_stats"
    PERCENTILES = "percentiles"
    PERCENTILE_RANKS = "percentile_ranks"
    TOP_HITS = "top_hits"

# ==================== CLASSES DE BASE ====================

class BaseElasticsearchQuery(BaseModel, ABC):
    """Classe de base pour toutes les requêtes Elasticsearch."""
    
    @abstractmethod
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        pass
    
    @abstractmethod
    def get_query_type(self) -> QueryType:
        """Retourne le type de requête."""
        pass
    
    def validate_elasticsearch_syntax(self) -> bool:
        """Valide la syntaxe Elasticsearch."""
        try:
            query_dict = self.to_elasticsearch()
            return isinstance(query_dict, dict) and bool(query_dict)
        except Exception:
            return False

class BaseElasticsearchFilter(BaseModel, ABC):
    """Classe de base pour les filtres Elasticsearch."""
    
    @abstractmethod
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        pass

class BaseElasticsearchAggregation(BaseModel, ABC):
    """Classe de base pour les agrégations Elasticsearch."""
    
    @abstractmethod
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        pass
    
    @abstractmethod
    def get_aggregation_type(self) -> AggregationType:
        """Retourne le type d'agrégation."""
        pass

# ==================== REQUÊTES PRINCIPALES ====================

class MatchQuery(BaseElasticsearchQuery):
    """Requête match pour recherche textuelle."""
    field: str = Field(..., description="Champ à rechercher")
    query: str = Field(..., description="Texte à rechercher")
    analyzer: Optional[str] = Field(None, description="Analyseur à utiliser")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Niveau de fuzziness")
    operator: Optional[Literal["and", "or"]] = Field(None, description="Opérateur logique")
    minimum_should_match: Optional[str] = Field(None, description="Minimum should match")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de scoring")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        query_params = {"query": self.query}
        
        if self.analyzer:
            query_params["analyzer"] = self.analyzer
        if self.fuzziness:
            query_params["fuzziness"] = self.fuzziness
        if self.operator:
            query_params["operator"] = self.operator
        if self.minimum_should_match:
            query_params["minimum_should_match"] = self.minimum_should_match
        if self.boost:
            query_params["boost"] = self.boost
        
        return {"match": {self.field: query_params}}
    
    def get_query_type(self) -> QueryType:
        return QueryType.MATCH

class MultiMatchQuery(BaseElasticsearchQuery):
    """Requête multi_match pour recherche sur plusieurs champs."""
    query: str = Field(..., description="Texte à rechercher")
    fields: List[str] = Field(..., min_items=1, description="Champs à rechercher")
    type: MultiMatchType = Field(default=MultiMatchType.BEST_FIELDS, description="Type de multi_match")
    analyzer: Optional[str] = Field(None, description="Analyseur à utiliser")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Niveau de fuzziness")
    operator: Optional[Literal["and", "or"]] = Field(None, description="Opérateur logique")
    minimum_should_match: Optional[str] = Field(None, description="Minimum should match")
    tie_breaker: Optional[float] = Field(None, ge=0.0, le=1.0, description="Tie breaker")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de scoring")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        query_params = {
            "query": self.query,
            "fields": self.fields,
            "type": self.type.value
        }
        
        if self.analyzer:
            query_params["analyzer"] = self.analyzer
        if self.fuzziness:
            query_params["fuzziness"] = self.fuzziness
        if self.operator:
            query_params["operator"] = self.operator
        if self.minimum_should_match:
            query_params["minimum_should_match"] = self.minimum_should_match
        if self.tie_breaker:
            query_params["tie_breaker"] = self.tie_breaker
        if self.boost:
            query_params["boost"] = self.boost
        
        return {"multi_match": query_params}
    
    def get_query_type(self) -> QueryType:
        return QueryType.MULTI_MATCH

class TermQuery(BaseElasticsearchQuery):
    """Requête term pour recherche exacte."""
    field: str = Field(..., description="Champ à rechercher")
    value: Union[str, int, float, bool] = Field(..., description="Valeur exacte")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de scoring")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        if self.boost:
            return {"term": {self.field: {"value": self.value, "boost": self.boost}}}
        return {"term": {self.field: self.value}}
    
    def get_query_type(self) -> QueryType:
        return QueryType.TERM

class TermsQuery(BaseElasticsearchQuery):
    """Requête terms pour recherche multiple exacte."""
    field: str = Field(..., description="Champ à rechercher")
    values: List[Union[str, int, float, bool]] = Field(..., min_items=1, description="Valeurs exactes")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de scoring")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        query_params = {self.field: self.values}
        if self.boost:
            query_params["boost"] = self.boost
        return {"terms": query_params}
    
    def get_query_type(self) -> QueryType:
        return QueryType.TERMS

class RangeQuery(BaseElasticsearchQuery):
    """Requête range pour recherche par plage."""
    field: str = Field(..., description="Champ à rechercher")
    gte: Optional[Union[str, int, float, datetime]] = Field(None, description="Supérieur ou égal")
    gt: Optional[Union[str, int, float, datetime]] = Field(None, description="Supérieur")
    lte: Optional[Union[str, int, float, datetime]] = Field(None, description="Inférieur ou égal")
    lt: Optional[Union[str, int, float, datetime]] = Field(None, description="Inférieur")
    format: Optional[str] = Field(None, description="Format pour les dates")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de scoring")
    
    @root_validator
    def validate_range_params(cls, values):
        """Valide les paramètres de plage."""
        gte, gt, lte, lt = values.get('gte'), values.get('gt'), values.get('lte'), values.get('lt')
        
        if not any([gte, gt, lte, lt]):
            raise ValueError("Au moins un paramètre de plage doit être défini")
        
        if gte is not None and gt is not None:
            raise ValueError("gte et gt ne peuvent pas être définis simultanément")
        
        if lte is not None and lt is not None:
            raise ValueError("lte et lt ne peuvent pas être définis simultanément")
        
        return values
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        range_params = {}
        
        if self.gte is not None:
            range_params["gte"] = self.gte
        if self.gt is not None:
            range_params["gt"] = self.gt
        if self.lte is not None:
            range_params["lte"] = self.lte
        if self.lt is not None:
            range_params["lt"] = self.lt
        if self.format:
            range_params["format"] = self.format
        if self.time_zone:
            range_params["time_zone"] = self.time_zone
        if self.boost:
            range_params["boost"] = self.boost
        
        return {"range": {self.field: range_params}}
    
    def get_query_type(self) -> QueryType:
        return QueryType.RANGE

class ExistsQuery(BaseElasticsearchQuery):
    """Requête exists pour vérifier l'existence d'un champ."""
    field: str = Field(..., description="Champ à vérifier")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        return {"exists": {"field": self.field}}
    
    def get_query_type(self) -> QueryType:
        return QueryType.EXISTS

class BoolQuery(BaseElasticsearchQuery):
    """Requête bool pour combiner plusieurs requêtes."""
    must: List[BaseElasticsearchQuery] = Field(default=[], description="Clauses must")
    should: List[BaseElasticsearchQuery] = Field(default=[], description="Clauses should")
    must_not: List[BaseElasticsearchQuery] = Field(default=[], description="Clauses must_not")
    filter: List[BaseElasticsearchQuery] = Field(default=[], description="Clauses filter")
    minimum_should_match: Optional[Union[int, str]] = Field(None, description="Minimum should match")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de scoring")
    
    @root_validator
    def validate_bool_clauses(cls, values):
        """Valide les clauses bool."""
        must, should, must_not, filter_clauses = (
            values.get('must', []),
            values.get('should', []),
            values.get('must_not', []),
            values.get('filter', [])
        )
        
        if not any([must, should, must_not, filter_clauses]):
            raise ValueError("Au moins une clause bool doit être définie")
        
        return values
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        bool_params = {}
        
        if self.must:
            bool_params["must"] = [q.to_elasticsearch() for q in self.must]
        if self.should:
            bool_params["should"] = [q.to_elasticsearch() for q in self.should]
        if self.must_not:
            bool_params["must_not"] = [q.to_elasticsearch() for q in self.must_not]
        if self.filter:
            bool_params["filter"] = [q.to_elasticsearch() for q in self.filter]
        if self.minimum_should_match:
            bool_params["minimum_should_match"] = self.minimum_should_match
        if self.boost:
            bool_params["boost"] = self.boost
        
        return {"bool": bool_params}
    
    def get_query_type(self) -> QueryType:
        return QueryType.BOOL

class FunctionScoreQuery(BaseElasticsearchQuery):
    """Requête function_score pour scoring personnalisé."""
    query: BaseElasticsearchQuery = Field(..., description="Requête de base")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost global")
    boost_mode: Optional[Literal["multiply", "sum", "avg", "first", "max", "min"]] = Field(None, description="Mode de boost")
    score_mode: Optional[Literal["multiply", "sum", "avg", "first", "max", "min"]] = Field(None, description="Mode de score")
    max_boost: Optional[float] = Field(None, ge=0.0, description="Boost maximum")
    min_score: Optional[float] = Field(None, ge=0.0, description="Score minimum")
    functions: List[Dict[str, Any]] = Field(default=[], description="Fonctions de scoring")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        function_score_params = {
            "query": self.query.to_elasticsearch()
        }
        
        if self.boost:
            function_score_params["boost"] = self.boost
        if self.boost_mode:
            function_score_params["boost_mode"] = self.boost_mode
        if self.score_mode:
            function_score_params["score_mode"] = self.score_mode
        if self.max_boost:
            function_score_params["max_boost"] = self.max_boost
        if self.min_score:
            function_score_params["min_score"] = self.min_score
        if self.functions:
            function_score_params["functions"] = self.functions
        
        return {"function_score": function_score_params}
    
    def get_query_type(self) -> QueryType:
        return QueryType.FUNCTION_SCORE

# ==================== FILTRES ELASTICSEARCH ====================

class TermFilter(BaseElasticsearchFilter):
    """Filtre term pour valeur exacte."""
    field: str = Field(..., description="Champ à filtrer")
    value: Union[str, int, float, bool] = Field(..., description="Valeur exacte")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        return {"term": {self.field: self.value}}

class TermsFilter(BaseElasticsearchFilter):
    """Filtre terms pour valeurs multiples exactes."""
    field: str = Field(..., description="Champ à filtrer")
    values: List[Union[str, int, float, bool]] = Field(..., min_items=1, description="Valeurs exactes")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        return {"terms": {self.field: self.values}}

class RangeFilter(BaseElasticsearchFilter):
    """Filtre range pour plage de valeurs."""
    field: str = Field(..., description="Champ à filtrer")
    gte: Optional[Union[str, int, float, datetime]] = Field(None, description="Supérieur ou égal")
    gt: Optional[Union[str, int, float, datetime]] = Field(None, description="Supérieur")
    lte: Optional[Union[str, int, float, datetime]] = Field(None, description="Inférieur ou égal")
    lt: Optional[Union[str, int, float, datetime]] = Field(None, description="Inférieur")
    format: Optional[str] = Field(None, description="Format pour les dates")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        range_params = {}
        
        if self.gte is not None:
            range_params["gte"] = self.gte
        if self.gt is not None:
            range_params["gt"] = self.gt
        if self.lte is not None:
            range_params["lte"] = self.lte
        if self.lt is not None:
            range_params["lt"] = self.lt
        if self.format:
            range_params["format"] = self.format
        if self.time_zone:
            range_params["time_zone"] = self.time_zone
        
        return {"range": {self.field: range_params}}

class ExistsFilter(BaseElasticsearchFilter):
    """Filtre exists pour vérifier l'existence d'un champ."""
    field: str = Field(..., description="Champ à vérifier")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        return {"exists": {"field": self.field}}

# ==================== AGRÉGATIONS ELASTICSEARCH ====================

class TermsAggregation(BaseElasticsearchAggregation):
    """Agrégation terms pour grouper par valeurs."""
    field: str = Field(..., description="Champ d'agrégation")
    size: PositiveInt = Field(default=10, description="Nombre de buckets")
    min_doc_count: Optional[NonNegativeInt] = Field(None, description="Nombre minimum de documents")
    order: Optional[Dict[str, str]] = Field(None, description="Ordre des buckets")
    include: Optional[Union[str, List[str]]] = Field(None, description="Valeurs à inclure")
    exclude: Optional[Union[str, List[str]]] = Field(None, description="Valeurs à exclure")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        agg_params = {
            "field": self.field,
            "size": self.size
        }
        
        if self.min_doc_count is not None:
            agg_params["min_doc_count"] = self.min_doc_count
        if self.order:
            agg_params["order"] = self.order
        if self.include:
            agg_params["include"] = self.include
        if self.exclude:
            agg_params["exclude"] = self.exclude
        
        return {"terms": agg_params}
    
    def get_aggregation_type(self) -> AggregationType:
        return AggregationType.TERMS

class DateHistogramAggregation(BaseElasticsearchAggregation):
    """Agrégation date_histogram pour grouper par intervalles de date."""
    field: str = Field(..., description="Champ de date")
    calendar_interval: Optional[str] = Field(None, description="Intervalle calendaire")
    fixed_interval: Optional[str] = Field(None, description="Intervalle fixe")
    format: Optional[str] = Field(None, description="Format de date")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    offset: Optional[str] = Field(None, description="Décalage")
    min_doc_count: Optional[NonNegativeInt] = Field(None, description="Nombre minimum de documents")
    
    @root_validator
    def validate_interval(cls, values):
        """Valide l'intervalle."""
        calendar_interval = values.get('calendar_interval')
        fixed_interval = values.get('fixed_interval')
        
        if not calendar_interval and not fixed_interval:
            raise ValueError("calendar_interval ou fixed_interval doit être défini")
        
        if calendar_interval and fixed_interval:
            raise ValueError("calendar_interval et fixed_interval ne peuvent pas être définis simultanément")
        
        return values
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        agg_params = {"field": self.field}
        
        if self.calendar_interval:
            agg_params["calendar_interval"] = self.calendar_interval
        if self.fixed_interval:
            agg_params["fixed_interval"] = self.fixed_interval
        if self.format:
            agg_params["format"] = self.format
        if self.time_zone:
            agg_params["time_zone"] = self.time_zone
        if self.offset:
            agg_params["offset"] = self.offset
        if self.min_doc_count is not None:
            agg_params["min_doc_count"] = self.min_doc_count
        
        return {"date_histogram": agg_params}
    
    def get_aggregation_type(self) -> AggregationType:
        return AggregationType.DATE_HISTOGRAM

class SumAggregation(BaseElasticsearchAggregation):
    """Agrégation sum pour calculer la somme."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur par défaut")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        agg_params = {"field": self.field}
        if self.missing is not None:
            agg_params["missing"] = self.missing
        return {"sum": agg_params}
    
    def get_aggregation_type(self) -> AggregationType:
        return AggregationType.SUM

class AvgAggregation(BaseElasticsearchAggregation):
    """Agrégation avg pour calculer la moyenne."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur par défaut")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        agg_params = {"field": self.field}
        if self.missing is not None:
            agg_params["missing"] = self.missing
        return {"avg": agg_params}
    
    def get_aggregation_type(self) -> AggregationType:
        return AggregationType.AVG

class MaxAggregation(BaseElasticsearchAggregation):
    """Agrégation max pour calculer le maximum."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur par défaut")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        agg_params = {"field": self.field}
        if self.missing is not None:
            agg_params["missing"] = self.missing
        return {"max": agg_params}
    
    def get_aggregation_type(self) -> AggregationType:
        return AggregationType.MAX

class MinAggregation(BaseElasticsearchAggregation):
    """Agrégation min pour calculer le minimum."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur par défaut")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        agg_params = {"field": self.field}
        if self.missing is not None:
            agg_params["missing"] = self.missing
        return {"min": agg_params}
    
    def get_aggregation_type(self) -> AggregationType:
        return AggregationType.MIN

class StatsAggregation(BaseElasticsearchAggregation):
    """Agrégation stats pour calculer les statistiques."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur par défaut")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        agg_params = {"field": self.field}
        if self.missing is not None:
            agg_params["missing"] = self.missing
        return {"stats": agg_params}
    
    def get_aggregation_type(self) -> AggregationType:
        return AggregationType.STATS

class CardinalityAggregation(BaseElasticsearchAggregation):
    """Agrégation cardinality pour compter les valeurs uniques."""
    field: str = Field(..., description="Champ à compter")
    precision_threshold: Optional[PositiveInt] = Field(None, description="Seuil de précision")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        agg_params = {"field": self.field}
        if self.precision_threshold:
            agg_params["precision_threshold"] = self.precision_threshold
        return {"cardinality": agg_params}
    
    def get_aggregation_type(self) -> AggregationType:
        return AggregationType.CARDINALITY

# ==================== STRUCTURES COMPLEXES ====================

class ElasticsearchSort(BaseModel):
    """Configuration de tri Elasticsearch."""
    field: str = Field(..., description="Champ de tri")
    order: SortOrder = Field(default=SortOrder.ASC, description="Ordre de tri")
    mode: Optional[SortMode] = Field(None, description="Mode de tri")
    missing: Optional[Union[str, int, float]] = Field(None, description="Valeur pour champs manquants")
    unmapped_type: Optional[str] = Field(None, description="Type pour champs non mappés")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        sort_params = {"order": self.order.value}
        
        if self.mode:
            sort_params["mode"] = self.mode.value
        if self.missing is not None:
            sort_params["missing"] = self.missing
        if self.unmapped_type:
            sort_params["unmapped_type"] = self.unmapped_type
        
        return {self.field: sort_params}

class ElasticsearchHighlight(BaseModel):
    """Configuration de highlight Elasticsearch."""
    fields: Dict[str, Dict[str, Any]] = Field(..., description="Champs à highlighter")
    pre_tags: List[str] = Field(default=["<em>"], description="Tags d'ouverture")
    post_tags: List[str] = Field(default=["</em>"], description="Tags de fermeture")
    fragment_size: Optional[PositiveInt] = Field(None, description="Taille des fragments")
    number_of_fragments: Optional[PositiveInt] = Field(None, description="Nombre de fragments")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        highlight_params = {
            "fields": self.fields,
            "pre_tags": self.pre_tags,
            "post_tags": self.post_tags
        }
        
        if self.fragment_size:
            highlight_params["fragment_size"] = self.fragment_size
        if self.number_of_fragments:
            highlight_params["number_of_fragments"] = self.number_of_fragments
        
        return highlight_params

# ==================== RÉSULTATS ET HITS ====================

class ElasticsearchHit(BaseModel):
    """Représentation d'un hit Elasticsearch."""
    index: str = Field(..., description="Index source")
    type: Optional[str] = Field(None, description="Type de document")
    id: str = Field(..., description="ID du document")
    score: Optional[float] = Field(None, description="Score de pertinence")
    source: Dict[str, Any] = Field(..., description="Document source")
    highlight: Optional[Dict[str, List[str]]] = Field(None, description="Highlights")
    sort: Optional[List[Union[str, int, float]]] = Field(None, description="Valeurs de tri")
    inner_hits: Optional[Dict[str, Any]] = Field(None, description="Inner hits")
    
    class Config:
        extra = "allow"

class ElasticsearchResult(BaseModel):
    """Résultat complet d'une requête Elasticsearch."""
    took: NonNegativeInt = Field(..., description="Temps d'exécution en ms")
    timed_out: bool = Field(..., description="Timeout atteint")
    shards: Dict[str, Any] = Field(..., description="Informations sur les shards")
    hits: Dict[str, Any] = Field(..., description="Hits et métadonnées")
    aggregations: Optional[Dict[str, Any]] = Field(None, description="Résultats d'agrégation")
    suggest: Optional[Dict[str, Any]] = Field(None, description="Suggestions")
    
    class Config:
        extra = "allow"
    
    @property
    def total_hits(self) -> int:
        """Nombre total de hits."""
        if isinstance(self.hits.get("total"), dict):
            return self.hits["total"].get("value", 0)
        return self.hits.get("total", 0)
    
    @property
    def max_score(self) -> Optional[float]:
        """Score maximum."""
        return self.hits.get("max_score")
    
    @property
    def documents(self) -> List[ElasticsearchHit]:
        """Liste des documents."""
        return [ElasticsearchHit(**hit) for hit in self.hits.get("hits", [])]

class ElasticsearchAggregationResult(BaseModel):
    """Résultat d'agrégation Elasticsearch."""
    doc_count_error_upper_bound: Optional[int] = Field(None, description="Erreur upper bound")
    sum_other_doc_count: Optional[int] = Field(None, description="Somme autres documents")
    buckets: List[Dict[str, Any]] = Field(default=[], description="Buckets d'agrégation")
    value: Optional[Union[int, float]] = Field(None, description="Valeur métrique")
    values: Optional[Dict[str, float]] = Field(None, description="Valeurs multiples")
    
    class Config:
        extra = "allow"

# ==================== REQUÊTE COMPLÈTE ====================

class ElasticsearchQuery(BaseModel):
    """Requête Elasticsearch complète."""
    query: Optional[BaseElasticsearchQuery] = Field(None, description="Requête principale")
    size: Optional[NonNegativeInt] = Field(None, description="Nombre de résultats")
    from_: Optional[NonNegativeInt] = Field(None, alias="from", description="Offset")
    sort: Optional[List[ElasticsearchSort]] = Field(None, description="Configuration de tri")
    source: Optional[Union[bool, List[str], Dict[str, Any]]] = Field(None, alias="_source", description="Champs source")
    highlight: Optional[ElasticsearchHighlight] = Field(None, description="Configuration highlight")
    aggregations: Optional[Dict[str, BaseElasticsearchAggregation]] = Field(None, alias="aggs", description="Agrégations")
    post_filter: Optional[BaseElasticsearchQuery] = Field(None, description="Post-filtre")
    min_score: Optional[float] = Field(None, description="Score minimum")
    timeout: Optional[str] = Field(None, description="Timeout")
    terminate_after: Optional[PositiveInt] = Field(None, description="Terminer après N docs")
    track_total_hits: Optional[Union[bool, int]] = Field(None, description="Tracker total hits")
    
    class Config:
        allow_population_by_field_name = True
        extra = "forbid"
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        query_dict = {}
        
        if self.query:
            query_dict["query"] = self.query.to_elasticsearch()
        
        if self.size is not None:
            query_dict["size"] = self.size
        
        if self.from_ is not None:
            query_dict["from"] = self.from_
        
        if self.sort:
            query_dict["sort"] = [sort.to_elasticsearch() for sort in self.sort]
        
        if self.source is not None:
            query_dict["_source"] = self.source
        
        if self.highlight:
            query_dict["highlight"] = self.highlight.to_elasticsearch()
        
        if self.aggregations:
            query_dict["aggs"] = {
                name: agg.to_elasticsearch() 
                for name, agg in self.aggregations.items()
            }
        
        if self.post_filter:
            query_dict["post_filter"] = self.post_filter.to_elasticsearch()
        
        if self.min_score is not None:
            query_dict["min_score"] = self.min_score
        
        if self.timeout:
            query_dict["timeout"] = self.timeout
        
        if self.terminate_after:
            query_dict["terminate_after"] = self.terminate_after
        
        if self.track_total_hits is not None:
            query_dict["track_total_hits"] = self.track_total_hits
        
        return query_dict

# ==================== BUILDERS ET FACTORIES ====================

class QueryBuilder:
    """Builder pour construire des requêtes Elasticsearch."""
    
    def __init__(self):
        self.query = None
        self.size = None
        self.from_ = None
        self.sort = []
        self.source = None
        self.highlight = None
        self.aggregations = {}
        self.post_filter = None
        self.min_score = None
        self.timeout = None
        self.terminate_after = None
        self.track_total_hits = None
    
    def set_query(self, query: BaseElasticsearchQuery) -> 'QueryBuilder':
        """Définit la requête principale."""
        self.query = query
        return self
    
    def set_size(self, size: int) -> 'QueryBuilder':
        """Définit le nombre de résultats."""
        self.size = size
        return self
    
    def set_from(self, from_: int) -> 'QueryBuilder':
        """Définit l'offset."""
        self.from_ = from_
        return self
    
    def add_sort(self, field: str, order: SortOrder = SortOrder.ASC, **kwargs) -> 'QueryBuilder':
        """Ajoute un tri."""
        sort_config = ElasticsearchSort(field=field, order=order, **kwargs)
        self.sort.append(sort_config)
        return self
    
    def set_source(self, source: Union[bool, List[str], Dict[str, Any]]) -> 'QueryBuilder':
        """Définit les champs source."""
        self.source = source
        return self
    
    def set_highlight(self, fields: Dict[str, Dict[str, Any]], **kwargs) -> 'QueryBuilder':
        """Définit le highlight."""
        self.highlight = ElasticsearchHighlight(fields=fields, **kwargs)
        return self
    
    def add_aggregation(self, name: str, aggregation: BaseElasticsearchAggregation) -> 'QueryBuilder':
        """Ajoute une agrégation."""
        self.aggregations[name] = aggregation
        return self
    
    def set_post_filter(self, post_filter: BaseElasticsearchQuery) -> 'QueryBuilder':
        """Définit le post-filtre."""
        self.post_filter = post_filter
        return self
    
    def set_min_score(self, min_score: float) -> 'QueryBuilder':
        """Définit le score minimum."""
        self.min_score = min_score
        return self
    
    def set_timeout(self, timeout: str) -> 'QueryBuilder':
        """Définit le timeout."""
        self.timeout = timeout
        return self
    
    def set_terminate_after(self, terminate_after: int) -> 'QueryBuilder':
        """Définit terminate_after."""
        self.terminate_after = terminate_after
        return self
    
    def set_track_total_hits(self, track_total_hits: Union[bool, int]) -> 'QueryBuilder':
        """Définit track_total_hits."""
        self.track_total_hits = track_total_hits
        return self
    
    def build(self) -> ElasticsearchQuery:
        """Construit la requête finale."""
        return ElasticsearchQuery(
            query=self.query,
            size=self.size,
            from_=self.from_,
            sort=self.sort if self.sort else None,
            source=self.source,
            highlight=self.highlight,
            aggregations=self.aggregations if self.aggregations else None,
            post_filter=self.post_filter,
            min_score=self.min_score,
            timeout=self.timeout,
            terminate_after=self.terminate_after,
            track_total_hits=self.track_total_hits
        )

class AggregationBuilder:
    """Builder pour construire des agrégations."""
    
    @staticmethod
    def terms(field: str, size: int = 10, **kwargs) -> TermsAggregation:
        """Crée une agrégation terms."""
        return TermsAggregation(field=field, size=size, **kwargs)
    
    @staticmethod
    def date_histogram(field: str, interval: str, calendar: bool = True, **kwargs) -> DateHistogramAggregation:
        """Crée une agrégation date_histogram."""
        if calendar:
            return DateHistogramAggregation(field=field, calendar_interval=interval, **kwargs)
        else:
            return DateHistogramAggregation(field=field, fixed_interval=interval, **kwargs)
    
    @staticmethod
    def sum(field: str, **kwargs) -> SumAggregation:
        """Crée une agrégation sum."""
        return SumAggregation(field=field, **kwargs)
    
    @staticmethod
    def avg(field: str, **kwargs) -> AvgAggregation:
        """Crée une agrégation avg."""
        return AvgAggregation(field=field, **kwargs)
    
    @staticmethod
    def max(field: str, **kwargs) -> MaxAggregation:
        """Crée une agrégation max."""
        return MaxAggregation(field=field, **kwargs)
    
    @staticmethod
    def min(field: str, **kwargs) -> MinAggregation:
        """Crée une agrégation min."""
        return MinAggregation(field=field, **kwargs)
    
    @staticmethod
    def stats(field: str, **kwargs) -> StatsAggregation:
        """Crée une agrégation stats."""
        return StatsAggregation(field=field, **kwargs)
    
    @staticmethod
    def cardinality(field: str, **kwargs) -> CardinalityAggregation:
        """Crée une agrégation cardinality."""
        return CardinalityAggregation(field=field, **kwargs)

class FilterBuilder:
    """Builder pour construire des filtres."""
    
    @staticmethod
    def term(field: str, value: Union[str, int, float, bool]) -> TermFilter:
        """Crée un filtre term."""
        return TermFilter(field=field, value=value)
    
    @staticmethod
    def terms(field: str, values: List[Union[str, int, float, bool]]) -> TermsFilter:
        """Crée un filtre terms."""
        return TermsFilter(field=field, values=values)
    
    @staticmethod
    def range(field: str, **kwargs) -> RangeFilter:
        """Crée un filtre range."""
        return RangeFilter(field=field, **kwargs)
    
    @staticmethod
    def exists(field: str) -> ExistsFilter:
        """Crée un filtre exists."""
        return ExistsFilter(field=field)

# ==================== TEMPLATES ET MAPPING ====================

class QueryTemplate(BaseModel):
    """Template de requête réutilisable."""
    name: str = Field(..., description="Nom du template")
    description: str = Field(..., description="Description du template")
    intent_types: List[str] = Field(..., description="Types d'intention supportés")
    template: Dict[str, Any] = Field(..., description="Template de requête")
    parameters: Dict[str, Any] = Field(default={}, description="Paramètres du template")
    examples: List[Dict[str, Any]] = Field(default=[], description="Exemples d'utilisation")
    
    def render(self, **kwargs) -> ElasticsearchQuery:
        """Rend le template avec les paramètres."""
        import copy
        rendered_template = copy.deepcopy(self.template)
        
        # Remplacer les paramètres dans le template
        def replace_params(obj, params):
            if isinstance(obj, dict):
                return {k: replace_params(v, params) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_params(item, params) for item in obj]
            elif isinstance(obj, str) and obj.startswith("{{") and obj.endswith("}}"):
                param_name = obj[2:-2].strip()
                return params.get(param_name, obj)
            return obj
        
        rendered = replace_params(rendered_template, kwargs)
        return ElasticsearchQuery(**rendered)

class FieldMapping(BaseModel):
    """Mapping d'un champ Elasticsearch."""
    type: str = Field(..., description="Type du champ")
    analyzer: Optional[str] = Field(None, description="Analyseur")
    search_analyzer: Optional[str] = Field(None, description="Analyseur de recherche")
    index: Optional[bool] = Field(None, description="Indexer le champ")
    store: Optional[bool] = Field(None, description="Stocker le champ")
    doc_values: Optional[bool] = Field(None, description="Doc values")
    norms: Optional[bool] = Field(None, description="Norms")
    boost: Optional[float] = Field(None, description="Boost du champ")
    copy_to: Optional[Union[str, List[str]]] = Field(None, description="Copy to")
    fields: Optional[Dict[str, 'FieldMapping']] = Field(None, description="Sous-champs")
    properties: Optional[Dict[str, 'FieldMapping']] = Field(None, description="Propriétés objet")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en mapping Elasticsearch."""
        mapping = {"type": self.type}
        
        if self.analyzer:
            mapping["analyzer"] = self.analyzer
        if self.search_analyzer:
            mapping["search_analyzer"] = self.search_analyzer
        if self.index is not None:
            mapping["index"] = self.index
        if self.store is not None:
            mapping["store"] = self.store
        if self.doc_values is not None:
            mapping["doc_values"] = self.doc_values
        if self.norms is not None:
            mapping["norms"] = self.norms
        if self.boost is not None:
            mapping["boost"] = self.boost
        if self.copy_to:
            mapping["copy_to"] = self.copy_to
        if self.fields:
            mapping["fields"] = {k: v.to_elasticsearch() for k, v in self.fields.items()}
        if self.properties:
            mapping["properties"] = {k: v.to_elasticsearch() for k, v in self.properties.items()}
        
        return mapping

class IndexMapping(BaseModel):
    """Mapping complet d'un index Elasticsearch."""
    properties: Dict[str, FieldMapping] = Field(..., description="Propriétés des champs")
    dynamic: Optional[Union[bool, str]] = Field(None, description="Mapping dynamique")
    dynamic_templates: Optional[List[Dict[str, Any]]] = Field(None, description="Templates dynamiques")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en mapping Elasticsearch."""
        mapping = {
            "properties": {k: v.to_elasticsearch() for k, v in self.properties.items()}
        }
        
        if self.dynamic is not None:
            mapping["dynamic"] = self.dynamic
        if self.dynamic_templates:
            mapping["dynamic_templates"] = self.dynamic_templates
        
        return mapping

# ==================== FONCTIONS UTILITAIRES ====================

def create_financial_search_query(
    user_id: int,
    query_text: Optional[str] = None,
    filters: Optional[List[BaseElasticsearchFilter]] = None,
    aggregations: Optional[Dict[str, BaseElasticsearchAggregation]] = None,
    size: int = 20,
    from_: int = 0,
    sort_by: Optional[str] = None,
    sort_order: SortOrder = SortOrder.DESC
) -> ElasticsearchQuery:
    """
    Crée une requête de recherche financière optimisée.
    
    Args:
        user_id: ID de l'utilisateur
        query_text: Texte de recherche
        filters: Filtres à appliquer
        aggregations: Agrégations à calculer
        size: Nombre de résultats
        from_: Offset
        sort_by: Champ de tri
        sort_order: Ordre de tri
    
    Returns:
        Requête Elasticsearch configurée
    """
    builder = QueryBuilder()
    
    # Requête principale
    bool_query = BoolQuery()
    
    # Filtre utilisateur obligatoire
    bool_query.filter.append(TermQuery(field="user_id", value=user_id))
    
    # Recherche textuelle si fournie
    if query_text:
        text_query = MultiMatchQuery(
            query=query_text,
            fields=[
                "searchable_text^2.0",
                "primary_description^1.5",
                "merchant_name^1.8",
                "category_name^1.2"
            ],
            type=MultiMatchType.BEST_FIELDS,
            fuzziness="AUTO",
            tie_breaker=0.3
        )
        bool_query.must.append(text_query)
    
    # Filtres additionnels
    if filters:
        for filter_obj in filters:
            bool_query.filter.append(filter_obj)
    
    builder.set_query(bool_query)
    builder.set_size(size)
    builder.set_from(from_)
    
    # Tri
    if sort_by:
        builder.add_sort(sort_by, sort_order)
    else:
        # Tri par défaut: score puis date
        if query_text:
            builder.add_sort("_score", SortOrder.DESC)
        builder.add_sort("date", SortOrder.DESC)
    
    # Agrégations
    if aggregations:
        for name, agg in aggregations.items():
            builder.add_aggregation(name, agg)
    
    # Configuration par défaut
    builder.set_timeout(f"{settings.SEARCH_TIMEOUT}s")
    builder.set_track_total_hits(True)
    
    return builder.build()

def create_financial_aggregations() -> Dict[str, BaseElasticsearchAggregation]:
    """Crée les agrégations financières standard."""
    return {
        "by_category": AggregationBuilder.terms("category_name.keyword", size=20),
        "by_merchant": AggregationBuilder.terms("merchant_name.keyword", size=10),
        "by_month": AggregationBuilder.date_histogram("date", "month", calendar=True),
        "total_amount": AggregationBuilder.sum("amount"),
        "total_spending": AggregationBuilder.sum("amount_abs"),
        "avg_amount": AggregationBuilder.avg("amount_abs"),
        "transaction_count": AggregationBuilder.cardinality("transaction_id"),
        "amount_stats": AggregationBuilder.stats("amount_abs")
    }

def optimize_query_for_performance(query: ElasticsearchQuery) -> ElasticsearchQuery:
    """Optimise une requête pour les performances."""
    # Copier la requête
    import copy
    optimized = copy.deepcopy(query)
    
    # Optimisations de performance
    if not optimized.track_total_hits:
        optimized.track_total_hits = 10000  # Limite pour éviter les calculs coûteux
    
    if not optimized.timeout:
        optimized.timeout = f"{settings.SEARCH_TIMEOUT}s"
    
    # Optimiser les agrégations
    if optimized.aggregations:
        for agg_name, agg in optimized.aggregations.items():
            if isinstance(agg, TermsAggregation) and agg.size > 100:
                agg.size = 100  # Limiter les buckets
    
    return optimized

# ==================== FACTORY FUNCTIONS ====================

def create_match_query(field: str, query: str, **kwargs) -> MatchQuery:
    """Factory pour créer une requête match."""
    return MatchQuery(field=field, query=query, **kwargs)

def create_bool_query(
    must: List[BaseElasticsearchQuery] = None,
    should: List[BaseElasticsearchQuery] = None,
    must_not: List[BaseElasticsearchQuery] = None,
    filter: List[BaseElasticsearchQuery] = None,
    **kwargs
) -> BoolQuery:
    """Factory pour créer une requête bool."""
    return BoolQuery(
        must=must or [],
        should=should or [],
        must_not=must_not or [],
        filter=filter or [],
        **kwargs
    )

def create_range_query(field: str, **kwargs) -> RangeQuery:
    """Factory pour créer une requête range."""
    return RangeQuery(field=field, **kwargs)

def create_terms_query(field: str, values: List[Union[str, int, float, bool]], **kwargs) -> TermsQuery:
    """Factory pour créer une requête terms."""
    return TermsQuery(field=field, values=values, **kwargs)

# ==================== VALIDATION ====================

def validate_elasticsearch_query(query: ElasticsearchQuery) -> bool:
    """Valide une requête Elasticsearch."""
    try:
        query_dict = query.to_elasticsearch()
        return isinstance(query_dict, dict) and bool(query_dict)
    except Exception:
        return False

def validate_query_syntax(query_dict: Dict[str, Any]) -> bool:
    """Valide la syntaxe d'une requête Elasticsearch."""
    # Validation basique de la structure
    if not isinstance(query_dict, dict):
        return False
    
    # Vérifier les clés autorisées
    allowed_keys = {
        "query", "size", "from", "sort", "_source", "highlight", 
        "aggs", "aggregations", "post_filter", "min_score", 
        "timeout", "terminate_after", "track_total_hits"
    }
    
    for key in query_dict.keys():
        if key not in allowed_keys:
            return False
    
    return True

# ==================== EXPORTS ====================

__all__ = [
    # Enums
    "QueryType", "BoolOperator", "MultiMatchType", "SortOrder", "SortMode", "AggregationType",
    
    # Classes de base
    "BaseElasticsearchQuery", "BaseElasticsearchFilter", "BaseElasticsearchAggregation",
    
    # Requêtes
    "MatchQuery", "MultiMatchQuery", "TermQuery", "TermsQuery", "RangeQuery", "ExistsQuery",
    "BoolQuery", "FunctionScoreQuery", "ElasticsearchQuery",
    
    # Filtres
    "TermFilter", "TermsFilter", "RangeFilter", "ExistsFilter",
    
    # Agrégations
    "TermsAggregation", "DateHistogramAggregation", "SumAggregation", "AvgAggregation",
    "MaxAggregation", "MinAggregation", "StatsAggregation", "CardinalityAggregation",
    
    # Structures
    "ElasticsearchSort", "ElasticsearchHighlight", "ElasticsearchHit", "ElasticsearchResult",
    "ElasticsearchAggregationResult",
    
    # Builders
    "QueryBuilder", "AggregationBuilder", "FilterBuilder",
    
    # Templates
    "QueryTemplate", "FieldMapping", "IndexMapping",
    
    # Fonctions utilitaires
    "create_financial_search_query", "create_financial_aggregations", "optimize_query_for_performance",
    "create_match_query", "create_bool_query", "create_range_query", "create_terms_query",
    "validate_elasticsearch_query", "validate_query_syntax"
]