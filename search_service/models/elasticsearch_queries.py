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

# CORRECTION PYDANTIC V2: Remplacer root_validator par model_validator
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
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "value_count"
    STATS = "stats"
    CARDINALITY = "cardinality"

# ==================== MODÈLES DE BASE ====================

class BaseElasticsearchQuery(BaseModel, ABC):
    """Classe de base pour toutes les requêtes Elasticsearch."""
    
    @abstractmethod
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch native."""
        pass
    
    @field_validator('*', mode='before')
    @classmethod
    def validate_not_empty(cls, v):
        """Valide que les valeurs ne sont pas vides."""
        if isinstance(v, str) and v.strip() == '':
            return None
        return v

class ElasticsearchQuery(BaseModel):
    """Requête Elasticsearch complète."""
    query: Optional[Dict[str, Any]] = Field(None, description="Clause query")
    aggs: Optional[Dict[str, Any]] = Field(None, description="Agrégations")
    sort: Optional[List[Dict[str, Any]]] = Field(None, description="Tri")
    size: Optional[PositiveInt] = Field(None, description="Nombre de résultats")
    from_: Optional[NonNegativeInt] = Field(None, alias="from", description="Offset")
    source_fields: Optional[List[str]] = Field(None, alias="_source", description="Champs source")
    highlight: Optional[Dict[str, Any]] = Field(None, description="Configuration highlight")
    timeout: Optional[str] = Field(None, description="Timeout (ex: '30s')")
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v):
        """Valide la taille des résultats."""
        if v is not None and v > settings.MAX_SEARCH_RESULTS:
            raise ValueError(f"Size trop élevé (max {settings.MAX_SEARCH_RESULTS})")
        return v
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        """Valide le timeout."""
        if v and isinstance(v, str):
            # Extraire la valeur numérique du timeout (ex: "30s" -> 30)
            try:
                timeout_val = float(v.rstrip('s'))
                if timeout_val > settings.MAX_SEARCH_TIMEOUT:
                    raise ValueError(f"Timeout trop élevé (max {settings.MAX_SEARCH_TIMEOUT}s)")
            except (ValueError, AttributeError):
                pass  # Laisser Elasticsearch valider le format
        return v
    
    @field_validator('source_fields')
    @classmethod
    def validate_source_fields(cls, v):
        """Valide les champs source."""
        if v and len(v) > 50:
            raise ValueError("Trop de champs source demandés (max 50)")
        return v
    
    @model_validator(mode='after') 
    def validate_query_structure(self):
        """Valide la structure de la requête."""
        # Validation que la requête n'est pas vide
        if not self.query and not self.aggs:
            raise ValueError("Requête ou agrégations requis")
        
        # Validation timeout cohérent avec size
        if self.timeout and self.size:
            try:
                timeout_val = float(self.timeout.rstrip('s'))
                if self.size > 100 and timeout_val < 10:
                    raise ValueError("Timeout trop court pour une requête volumineuse")
            except (ValueError, AttributeError):
                pass
        
        return self

# ==================== REQUÊTES SPÉCIALISÉES ====================

class BoolQuery(BaseElasticsearchQuery):
    """Requête bool Elasticsearch."""
    must: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses must")
    should: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses should")
    must_not: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses must_not")
    filter: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses filter")
    minimum_should_match: Optional[Union[str, int]] = Field(None, description="Minimum should match")
    boost: Optional[float] = Field(None, description="Boost de la requête")
    
    @model_validator(mode='after')
    def validate_bool_query(self):
        """Valide la structure de la requête bool."""
        # Au moins une clause requise
        if not any([self.must, self.should, self.must_not, self.filter]):
            raise ValueError("Au moins une clause bool requise")
        
        # Validation minimum_should_match
        if self.minimum_should_match and not self.should:
            raise ValueError("minimum_should_match require des clauses should")
        
        return self
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        query = {"bool": {}}
        
        if self.must:
            query["bool"]["must"] = self.must
        if self.should:
            query["bool"]["should"] = self.should
        if self.must_not:
            query["bool"]["must_not"] = self.must_not
        if self.filter:
            query["bool"]["filter"] = self.filter
        if self.minimum_should_match:
            query["bool"]["minimum_should_match"] = self.minimum_should_match
        if self.boost:
            query["bool"]["boost"] = self.boost
        
        return query

class MatchQuery(BaseElasticsearchQuery):
    """Requête match Elasticsearch."""
    field: str = Field(..., description="Champ à rechercher")
    query: str = Field(..., description="Texte à rechercher")
    analyzer: Optional[str] = Field(None, description="Analyseur")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Fuzziness")
    minimum_should_match: Optional[str] = Field(None, description="Minimum should match")
    boost: Optional[float] = Field(None, description="Boost")
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Valide le nom du champ."""
        if not v or not isinstance(v, str):
            raise ValueError("Field doit être une chaîne non vide")
        return v
    
    @field_validator('query')
    @classmethod
    def validate_query_text(cls, v):
        """Valide le texte de recherche."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Query text ne peut pas être vide")
        if len(v) > 1000:
            raise ValueError("Query text trop long (max 1000 caractères)")
        return v.strip()
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        match_query = {
            "match": {
                self.field: {
                    "query": self.query
                }
            }
        }
        
        if self.analyzer:
            match_query["match"][self.field]["analyzer"] = self.analyzer
        if self.fuzziness:
            match_query["match"][self.field]["fuzziness"] = self.fuzziness
        if self.minimum_should_match:
            match_query["match"][self.field]["minimum_should_match"] = self.minimum_should_match
        if self.boost:
            match_query["match"][self.field]["boost"] = self.boost
        
        return match_query

class MultiMatchQuery(BaseElasticsearchQuery):
    """Requête multi_match Elasticsearch."""
    query: str = Field(..., description="Texte à rechercher")
    fields: List[str] = Field(..., description="Champs à rechercher")
    type: MultiMatchType = Field(default=MultiMatchType.BEST_FIELDS, description="Type de multi_match")
    analyzer: Optional[str] = Field(None, description="Analyseur")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Fuzziness")
    minimum_should_match: Optional[str] = Field(None, description="Minimum should match")
    boost: Optional[float] = Field(None, description="Boost")
    tie_breaker: Optional[float] = Field(None, description="Tie breaker")
    
    @field_validator('fields')
    @classmethod
    def validate_fields(cls, v):
        """Valide la liste des champs."""
        if not v:
            raise ValueError("Au moins un champ requis")
        if len(v) > 20:
            raise ValueError("Trop de champs (max 20)")
        return v
    
    @field_validator('tie_breaker')
    @classmethod
    def validate_tie_breaker(cls, v):
        """Valide le tie breaker."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("tie_breaker doit être entre 0.0 et 1.0")
        return v
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        multi_match = {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                "type": self.type
            }
        }
        
        if self.analyzer:
            multi_match["multi_match"]["analyzer"] = self.analyzer
        if self.fuzziness:
            multi_match["multi_match"]["fuzziness"] = self.fuzziness
        if self.minimum_should_match:
            multi_match["multi_match"]["minimum_should_match"] = self.minimum_should_match
        if self.boost:
            multi_match["multi_match"]["boost"] = self.boost
        if self.tie_breaker:
            multi_match["multi_match"]["tie_breaker"] = self.tie_breaker
        
        return multi_match

class TermQuery(BaseElasticsearchQuery):
    """Requête term Elasticsearch."""
    field: str = Field(..., description="Champ")
    value: Union[str, int, float, bool] = Field(..., description="Valeur exacte")
    boost: Optional[float] = Field(None, description="Boost")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        term_query = {
            "term": {
                self.field: {"value": self.value}
            }
        }
        
        if self.boost:
            term_query["term"][self.field]["boost"] = self.boost
        
        return term_query

class RangeQuery(BaseElasticsearchQuery):
    """Requête range Elasticsearch."""
    field: str = Field(..., description="Champ")
    gte: Optional[Union[str, int, float, datetime]] = Field(None, description="Supérieur ou égal")
    gt: Optional[Union[str, int, float, datetime]] = Field(None, description="Supérieur")
    lte: Optional[Union[str, int, float, datetime]] = Field(None, description="Inférieur ou égal")
    lt: Optional[Union[str, int, float, datetime]] = Field(None, description="Inférieur")
    format: Optional[str] = Field(None, description="Format pour les dates")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    boost: Optional[float] = Field(None, description="Boost")
    
    @model_validator(mode='after')
    def validate_range_bounds(self):
        """Valide les bornes de la plage."""
        # Au moins une borne requise
        if not any([self.gte, self.gt, self.lte, self.lt]):
            raise ValueError("Au moins une borne de plage requise")
        
        # Validation cohérence des bornes
        if self.gte is not None and self.gt is not None:
            raise ValueError("gte et gt ne peuvent pas être utilisés ensemble")
        
        if self.lte is not None and self.lt is not None:
            raise ValueError("lte et lt ne peuvent pas être utilisés ensemble")
        
        return self
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch."""
        range_query = {"range": {self.field: {}}}
        
        if self.gte is not None:
            range_query["range"][self.field]["gte"] = self.gte
        if self.gt is not None:
            range_query["range"][self.field]["gt"] = self.gt
        if self.lte is not None:
            range_query["range"][self.field]["lte"] = self.lte
        if self.lt is not None:
            range_query["range"][self.field]["lt"] = self.lt
        if self.format:
            range_query["range"][self.field]["format"] = self.format
        if self.time_zone:
            range_query["range"][self.field]["time_zone"] = self.time_zone
        if self.boost:
            range_query["range"][self.field]["boost"] = self.boost
        
        return range_query

# ==================== AGRÉGATIONS ====================

class TermsAggregation(BaseModel):
    """Agrégation terms Elasticsearch."""
    field: str = Field(..., description="Champ d'agrégation")
    size: PositiveInt = Field(default=10, description="Nombre de buckets")
    order: Optional[Dict[str, str]] = Field(None, description="Ordre de tri")
    include: Optional[Union[str, List[str]]] = Field(None, description="Valeurs à inclure")
    exclude: Optional[Union[str, List[str]]] = Field(None, description="Valeurs à exclure")
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v):
        """Valide la taille de l'agrégation."""
        if v > 1000:
            raise ValueError("Size trop élevé pour agrégation (max 1000)")
        return v
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en agrégation Elasticsearch."""
        agg = {
            "terms": {
                "field": self.field,
                "size": self.size
            }
        }
        
        if self.order:
            agg["terms"]["order"] = self.order
        if self.include:
            agg["terms"]["include"] = self.include
        if self.exclude:
            agg["terms"]["exclude"] = self.exclude
        
        return agg

class DateHistogramAggregation(BaseModel):
    """Agrégation date_histogram Elasticsearch."""
    field: str = Field(..., description="Champ date")
    calendar_interval: Optional[str] = Field(None, description="Intervalle calendaire")
    fixed_interval: Optional[str] = Field(None, description="Intervalle fixe")
    format: Optional[str] = Field(None, description="Format de date")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    min_doc_count: Optional[int] = Field(None, description="Minimum de documents")
    
    @model_validator(mode='after')
    def validate_intervals(self):
        """Valide les intervalles."""
        if not self.calendar_interval and not self.fixed_interval:
            raise ValueError("calendar_interval ou fixed_interval requis")
        
        if self.calendar_interval and self.fixed_interval:
            raise ValueError("calendar_interval et fixed_interval sont mutuellement exclusifs")
        
        return self
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en agrégation Elasticsearch."""
        agg = {
            "date_histogram": {
                "field": self.field
            }
        }
        
        if self.calendar_interval:
            agg["date_histogram"]["calendar_interval"] = self.calendar_interval
        if self.fixed_interval:
            agg["date_histogram"]["fixed_interval"] = self.fixed_interval
        if self.format:
            agg["date_histogram"]["format"] = self.format
        if self.time_zone:
            agg["date_histogram"]["time_zone"] = self.time_zone
        if self.min_doc_count is not None:
            agg["date_histogram"]["min_doc_count"] = self.min_doc_count
        
        return agg

class SumAggregation(BaseModel):
    """Agrégation sum Elasticsearch."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur par défaut")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Convertit en agrégation Elasticsearch."""
        agg = {"sum": {"field": self.field}}
        
        if self.missing is not None:
            agg["sum"]["missing"] = self.missing
        
        return agg

# ==================== FACTORIES ET BUILDERS ====================

class QueryBuilder:
    """Builder pour construire des requêtes Elasticsearch complexes."""
    
    def __init__(self):
        self.query = ElasticsearchQuery()
        self._bool_query = BoolQuery()
    
    def match(self, field: str, query: str, **kwargs) -> 'QueryBuilder':
        """Ajoute une clause match."""
        match_query = MatchQuery(field=field, query=query, **kwargs)
        self._bool_query.must.append(match_query.to_elasticsearch())
        return self
    
    def term(self, field: str, value: Any, **kwargs) -> 'QueryBuilder':
        """Ajoute une clause term."""
        term_query = TermQuery(field=field, value=value, **kwargs)
        self._bool_query.filter.append(term_query.to_elasticsearch())
        return self
    
    def range(self, field: str, **kwargs) -> 'QueryBuilder':
        """Ajoute une clause range."""
        range_query = RangeQuery(field=field, **kwargs)
        self._bool_query.filter.append(range_query.to_elasticsearch())
        return self
    
    def should(self, queries: List[BaseElasticsearchQuery]) -> 'QueryBuilder':
        """Ajoute des clauses should."""
        for query in queries:
            self._bool_query.should.append(query.to_elasticsearch())
        return self
    
    def size(self, size: int) -> 'QueryBuilder':
        """Définit la taille des résultats."""
        self.query.size = size
        return self
    
    def from_(self, from_: int) -> 'QueryBuilder':
        """Définit l'offset."""
        self.query.from_ = from_
        return self
    
    def sort(self, field: str, order: SortOrder = SortOrder.ASC) -> 'QueryBuilder':
        """Ajoute un tri."""
        if not self.query.sort:
            self.query.sort = []
        self.query.sort.append({field: {"order": order}})
        return self
    
    def source(self, fields: List[str]) -> 'QueryBuilder':
        """Définit les champs source."""
        self.query.source_fields = fields
        return self
    
    def timeout(self, timeout: str) -> 'QueryBuilder':
        """Définit le timeout."""
        self.query.timeout = timeout
        return self
    
    def agg_terms(self, name: str, field: str, size: int = 10) -> 'QueryBuilder':
        """Ajoute une agrégation terms."""
        if not self.query.aggs:
            self.query.aggs = {}
        
        terms_agg = TermsAggregation(field=field, size=size)
        self.query.aggs[name] = terms_agg.to_elasticsearch()
        return self
    
    def agg_date_histogram(self, name: str, field: str, calendar_interval: str) -> 'QueryBuilder':
        """Ajoute une agrégation date_histogram."""
        if not self.query.aggs:
            self.query.aggs = {}
        
        date_agg = DateHistogramAggregation(
            field=field, 
            calendar_interval=calendar_interval
        )
        self.query.aggs[name] = date_agg.to_elasticsearch()
        return self
    
    def agg_sum(self, name: str, field: str) -> 'QueryBuilder':
        """Ajoute une agrégation sum."""
        if not self.query.aggs:
            self.query.aggs = {}
        
        sum_agg = SumAggregation(field=field)
        self.query.aggs[name] = sum_agg.to_elasticsearch()
        return self
    
    def build(self) -> ElasticsearchQuery:
        """Construit la requête finale."""
        # Ajouter la bool query si elle contient des clauses
        if any([self._bool_query.must, self._bool_query.should, 
                self._bool_query.must_not, self._bool_query.filter]):
            self.query.query = self._bool_query.to_elasticsearch()
        
        return self.query

def create_financial_search_query(
    user_id: int,
    text_query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    date_range: Optional[Dict[str, str]] = None,
    amount_range: Optional[Dict[str, float]] = None,
    categories: Optional[List[str]] = None,
    merchants: Optional[List[str]] = None,
    limit: int = 20,
    offset: int = 0
) -> ElasticsearchQuery:
    """
    Factory pour créer des requêtes de recherche financière optimisées.
    
    Args:
        user_id: ID utilisateur (obligatoire pour sécurité)
        text_query: Recherche textuelle libre
        filters: Filtres additionnels
        date_range: Plage de dates {from, to}
        amount_range: Plage de montants {min, max}
        categories: Liste des catégories
        merchants: Liste des marchands
        limit: Nombre de résultats
        offset: Décalage
        
    Returns:
        Requête Elasticsearch optimisée
    """
    builder = QueryBuilder()
    
    # Filtre sécurité user_id obligatoire
    builder.term("user_id", user_id)
    
    # Recherche textuelle
    if text_query:
        builder.match("searchable_text", text_query, fuzziness="AUTO")
    
    # Filtres par catégorie
    if categories:
        builder.term("category_name", categories[0] if len(categories) == 1 else categories)
    
    # Filtres par marchand
    if merchants:
        builder.term("merchant_name", merchants[0] if len(merchants) == 1 else merchants)
    
    # Plage de dates
    if date_range:
        builder.range("date", **date_range)
    
    # Plage de montants
    if amount_range:
        builder.range("amount_abs", **amount_range)
    
    # Filtres additionnels
    if filters:
        for field, value in filters.items():
            builder.term(field, value)
    
    # Configuration résultats
    builder.size(limit).from_(offset)
    
    # Tri par défaut par pertinence puis date
    builder.sort("_score", SortOrder.DESC)
    builder.sort("date", SortOrder.DESC)
    
    # Champs source optimisés pour les transactions
    builder.source([
        "transaction_id", "user_id", "account_id",
        "amount", "amount_abs", "currency_code",
        "date", "month_year", "weekday",
        "primary_description", "merchant_name", "category_name",
        "operation_type", "transaction_type"
    ])
    
    # Timeout adaptatif
    timeout = "10s" if limit <= 50 else "20s"
    builder.timeout(timeout)
    
    return builder.build()

# ==================== EXPORTS ====================

__all__ = [
    # Enums
    "QueryType", "BoolOperator", "MultiMatchType", "SortOrder", "SortMode", "AggregationType",
    
    # Classes de base
    "BaseElasticsearchQuery", "ElasticsearchQuery",
    
    # Requêtes spécialisées
    "BoolQuery", "MatchQuery", "MultiMatchQuery", "TermQuery", "RangeQuery",
    
    # Agrégations
    "TermsAggregation", "DateHistogramAggregation", "SumAggregation",
    
    # Builder et factories
    "QueryBuilder", "create_financial_search_query"
]