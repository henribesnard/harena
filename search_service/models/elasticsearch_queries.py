"""
Modèles Elasticsearch natifs pour le Search Service - Partie 1.

Ces modèles définissent les structures de données pour construire et exécuter
des requêtes Elasticsearch optimisées, avec support des agrégations complexes
et des filtres financiers spécialisés.

ARCHITECTURE:
- Requêtes Elasticsearch typées avec Pydantic
- Filtres et agrégations spécialisés
- Validation stricte des paramètres
- Conversion automatique vers dictionnaires Elasticsearch

CONFIGURATION CENTRALISÉE:
- Index et settings via config_service
- Validation basée sur les paramètres configurés
- Optimisations d'index financières
"""

from typing import Dict, List, Any, Optional, Union, Literal
from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, NonNegativeInt, NonNegativeFloat

# Configuration centralisée
from config_service.config import settings

# ==================== ENUMS ET TYPES ====================

class QueryType(str, Enum):
    """Types de requêtes Elasticsearch."""
    BOOL = "bool"
    MATCH = "match"
    MATCH_PHRASE = "match_phrase"
    MULTI_MATCH = "multi_match"
    TERM = "term"
    TERMS = "terms"
    RANGE = "range"
    EXISTS = "exists"
    WILDCARD = "wildcard"
    FUZZY = "fuzzy"
    QUERY_STRING = "query_string"

class BoolClause(str, Enum):
    """Clauses bool Elasticsearch."""
    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"
    FILTER = "filter"

class AggregationType(str, Enum):
    """Types d'agrégation Elasticsearch."""
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

class SortOrder(str, Enum):
    """Ordres de tri."""
    ASC = "asc"
    DESC = "desc"

# ==================== REQUÊTES DE BASE ====================

class ElasticsearchQuery(BaseModel, ABC):
    """Classe de base pour toutes les requêtes Elasticsearch."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la requête en dictionnaire Elasticsearch."""
        pass
    
    class Config:
        use_enum_values = True

class MatchQuery(ElasticsearchQuery):
    """Requête match Elasticsearch."""
    field: str = Field(..., description="Champ à rechercher")
    query: str = Field(..., description="Texte à rechercher")
    operator: Optional[Literal["and", "or"]] = Field(None, description="Opérateur logique")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Niveau de fuzziness")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de pertinence")
    analyzer: Optional[str] = Field(None, description="Analyseur à utiliser")
    minimum_should_match: Optional[str] = Field(None, description="Minimum should match")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        match_params = {"query": self.query}
        
        if self.operator:
            match_params["operator"] = self.operator
        if self.fuzziness is not None:
            match_params["fuzziness"] = self.fuzziness
        if self.boost is not None:
            match_params["boost"] = self.boost
        if self.analyzer:
            match_params["analyzer"] = self.analyzer
        if self.minimum_should_match:
            match_params["minimum_should_match"] = self.minimum_should_match
        
        return {"match": {self.field: match_params}}

class MultiMatchQuery(ElasticsearchQuery):
    """Requête multi_match Elasticsearch."""
    query: str = Field(..., description="Texte à rechercher")
    fields: List[str] = Field(..., min_items=1, description="Champs à rechercher")
    type: Optional[Literal["best_fields", "most_fields", "cross_fields", "phrase", "phrase_prefix"]] = Field(
        "best_fields", description="Type de multi_match"
    )
    operator: Optional[Literal["and", "or"]] = Field(None, description="Opérateur logique")
    fuzziness: Optional[Union[str, int]] = Field(None, description="Niveau de fuzziness")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de pertinence")
    tie_breaker: Optional[float] = Field(None, ge=0.0, le=1.0, description="Tie breaker")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        multi_match_params = {
            "query": self.query,
            "fields": self.fields
        }
        
        if self.type:
            multi_match_params["type"] = self.type
        if self.operator:
            multi_match_params["operator"] = self.operator
        if self.fuzziness is not None:
            multi_match_params["fuzziness"] = self.fuzziness
        if self.boost is not None:
            multi_match_params["boost"] = self.boost
        if self.tie_breaker is not None:
            multi_match_params["tie_breaker"] = self.tie_breaker
        
        return {"multi_match": multi_match_params}

class TermQuery(ElasticsearchQuery):
    """Requête term Elasticsearch."""
    field: str = Field(..., description="Champ exact")
    value: Union[str, int, float, bool] = Field(..., description="Valeur exacte")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de pertinence")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        if self.boost is not None:
            return {"term": {self.field: {"value": self.value, "boost": self.boost}}}
        return {"term": {self.field: self.value}}

class TermsQuery(ElasticsearchQuery):
    """Requête terms Elasticsearch."""
    field: str = Field(..., description="Champ exact")
    values: List[Union[str, int, float, bool]] = Field(..., min_items=1, description="Valeurs exactes")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de pertinence")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        terms_params = {self.field: self.values}
        if self.boost is not None:
            terms_params["boost"] = self.boost
        return {"terms": terms_params}

class RangeQuery(ElasticsearchQuery):
    """Requête range Elasticsearch."""
    field: str = Field(..., description="Champ numérique/date")
    gte: Optional[Union[str, int, float]] = Field(None, description="Supérieur ou égal")
    gt: Optional[Union[str, int, float]] = Field(None, description="Supérieur")
    lte: Optional[Union[str, int, float]] = Field(None, description="Inférieur ou égal")
    lt: Optional[Union[str, int, float]] = Field(None, description="Inférieur")
    format: Optional[str] = Field(None, description="Format pour les dates")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de pertinence")
    
    def to_dict(self) -> Dict[str, Any]:
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
        if self.boost is not None:
            range_params["boost"] = self.boost
        
        return {"range": {self.field: range_params}}

class ExistsQuery(ElasticsearchQuery):
    """Requête exists Elasticsearch."""
    field: str = Field(..., description="Champ à vérifier")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        return {"exists": {"field": self.field}}

class BoolQuery(ElasticsearchQuery):
    """Requête bool Elasticsearch."""
    must: List[ElasticsearchQuery] = Field(default=[], description="Clauses obligatoires")
    should: List[ElasticsearchQuery] = Field(default=[], description="Clauses optionnelles")
    must_not: List[ElasticsearchQuery] = Field(default=[], description="Clauses interdites")
    filter: List[ElasticsearchQuery] = Field(default=[], description="Filtres (sans score)")
    
    minimum_should_match: Optional[Union[int, str]] = Field(None, description="Minimum should match")
    boost: Optional[float] = Field(None, ge=0.0, description="Boost de pertinence")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        bool_params = {}
        
        if self.must:
            bool_params["must"] = [q.to_dict() for q in self.must]
        if self.should:
            bool_params["should"] = [q.to_dict() for q in self.should]
        if self.must_not:
            bool_params["must_not"] = [q.to_dict() for q in self.must_not]
        if self.filter:
            bool_params["filter"] = [q.to_dict() for q in self.filter]
        
        if self.minimum_should_match is not None:
            bool_params["minimum_should_match"] = self.minimum_should_match
        if self.boost is not None:
            bool_params["boost"] = self.boost
        
        return {"bool": bool_params}

# ==================== FILTRES ELASTICSEARCH ====================

class ElasticsearchFilter(BaseModel, ABC):
    """Classe de base pour les filtres Elasticsearch."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le filtre en dictionnaire Elasticsearch."""
        pass
    
    class Config:
        use_enum_values = True

class TermFilter(ElasticsearchFilter):
    """Filtre term pour valeurs exactes."""
    field: str = Field(..., description="Champ à filtrer")
    value: Union[str, int, float, bool] = Field(..., description="Valeur exacte")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        return {"term": {self.field: self.value}}

class TermsFilter(ElasticsearchFilter):
    """Filtre terms pour valeurs multiples."""
    field: str = Field(..., description="Champ à filtrer")
    values: List[Union[str, int, float, bool]] = Field(..., min_items=1, description="Valeurs exactes")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        return {"terms": {self.field: self.values}}

class RangeFilter(ElasticsearchFilter):
    """Filtre range pour plages de valeurs."""
    field: str = Field(..., description="Champ numérique/date")
    gte: Optional[Union[str, int, float]] = Field(None, description="Supérieur ou égal")
    gt: Optional[Union[str, int, float]] = Field(None, description="Supérieur")
    lte: Optional[Union[str, int, float]] = Field(None, description="Inférieur ou égal")
    lt: Optional[Union[str, int, float]] = Field(None, description="Inférieur")
    format: Optional[str] = Field(None, description="Format pour les dates")
    
    def to_dict(self) -> Dict[str, Any]:
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
        
        return {"range": {self.field: range_params}}

class ExistsFilter(ElasticsearchFilter):
    """Filtre exists pour champs non-null."""
    field: str = Field(..., description="Champ à vérifier")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        return {"exists": {"field": self.field}}

# ==================== AGRÉGATIONS ELASTICSEARCH ====================

class ElasticsearchAggregation(BaseModel, ABC):
    """Classe de base pour les agrégations Elasticsearch."""
    name: str = Field(..., description="Nom de l'agrégation")
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'agrégation en dictionnaire Elasticsearch."""
        pass
    
    class Config:
        use_enum_values = True

class TermsAggregation(ElasticsearchAggregation):
    """Agrégation terms pour groupements."""
    field: str = Field(..., description="Champ de groupement")
    size: PositiveInt = Field(default=10, le=1000, description="Nombre de buckets")
    order: Dict[str, str] = Field(default={"_count": "desc"}, description="Ordre de tri")
    min_doc_count: NonNegativeInt = Field(default=1, description="Nombre minimum de documents")
    missing: Optional[Union[str, int, float]] = Field(None, description="Valeur pour champs manquants")
    include: Optional[Union[str, List[str]]] = Field(None, description="Valeurs à inclure")
    exclude: Optional[Union[str, List[str]]] = Field(None, description="Valeurs à exclure")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        terms_params = {
            "field": self.field,
            "size": self.size,
            "order": self.order,
            "min_doc_count": self.min_doc_count
        }
        
        if self.missing is not None:
            terms_params["missing"] = self.missing
        if self.include is not None:
            terms_params["include"] = self.include
        if self.exclude is not None:
            terms_params["exclude"] = self.exclude
        
        return {self.name: {"terms": terms_params}}

class DateHistogramAggregation(ElasticsearchAggregation):
    """Agrégation date_histogram pour analyses temporelles."""
    field: str = Field(..., description="Champ date")
    calendar_interval: Optional[str] = Field(None, description="Intervalle calendaire")
    fixed_interval: Optional[str] = Field(None, description="Intervalle fixe")
    format: Optional[str] = Field(None, description="Format de date")
    time_zone: Optional[str] = Field(None, description="Fuseau horaire")
    min_doc_count: NonNegativeInt = Field(default=0, description="Nombre minimum de documents")
    extended_bounds: Optional[Dict[str, str]] = Field(None, description="Bornes étendues")
    
    @root_validator
    def validate_interval(cls, values):
        """Valide qu'un seul type d'intervalle est spécifié."""
        calendar = values.get('calendar_interval')
        fixed = values.get('fixed_interval')
        
        if not calendar and not fixed:
            raise ValueError("calendar_interval ou fixed_interval requis")
        if calendar and fixed:
            raise ValueError("Spécifiez soit calendar_interval soit fixed_interval")
        
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        date_histogram_params = {
            "field": self.field,
            "min_doc_count": self.min_doc_count
        }
        
        if self.calendar_interval:
            date_histogram_params["calendar_interval"] = self.calendar_interval
        if self.fixed_interval:
            date_histogram_params["fixed_interval"] = self.fixed_interval
        if self.format:
            date_histogram_params["format"] = self.format
        if self.time_zone:
            date_histogram_params["time_zone"] = self.time_zone
        if self.extended_bounds:
            date_histogram_params["extended_bounds"] = self.extended_bounds
        
        return {self.name: {"date_histogram": date_histogram_params}}

class SumAggregation(ElasticsearchAggregation):
    """Agrégation sum pour sommes."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur pour champs manquants")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        sum_params = {"field": self.field}
        
        if self.missing is not None:
            sum_params["missing"] = self.missing
        
        return {self.name: {"sum": sum_params}}

class AvgAggregation(ElasticsearchAggregation):
    """Agrégation avg pour moyennes."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur pour champs manquants")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        avg_params = {"field": self.field}
        
        if self.missing is not None:
            avg_params["missing"] = self.missing
        
        return {self.name: {"avg": avg_params}}

class MaxAggregation(ElasticsearchAggregation):
    """Agrégation max pour valeurs maximales."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur pour champs manquants")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        max_params = {"field": self.field}
        
        if self.missing is not None:
            max_params["missing"] = self.missing
        
        return {self.name: {"max": max_params}}

class MinAggregation(ElasticsearchAggregation):
    """Agrégation min pour valeurs minimales."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur pour champs manquants")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        min_params = {"field": self.field}
        
        if self.missing is not None:
            min_params["missing"] = self.missing
        
        return {self.name: {"min": min_params}}

class StatsAggregation(ElasticsearchAggregation):
    """Agrégation stats pour statistiques complètes."""
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(None, description="Valeur pour champs manquants")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        stats_params = {"field": self.field}
        
        if self.missing is not None:
            stats_params["missing"] = self.missing
        
        return {self.name: {"stats": stats_params}}

# ==================== RÉSULTATS ELASTICSEARCH ====================

class ElasticsearchHit(BaseModel):
    """Résultat individuel Elasticsearch."""
    index: str = Field(..., description="Index source")
    id: str = Field(..., description="ID du document")
    score: Optional[float] = Field(None, description="Score de pertinence")
    source: Dict[str, Any] = Field(..., description="Document source")
    highlight: Optional[Dict[str, List[str]]] = Field(None, description="Highlights")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explication du score")
    sort: Optional[List[Any]] = Field(None, description="Valeurs de tri")
    
    class Config:
        # Permet l'utilisation de _id, _score, etc.
        allow_population_by_field_name = True
        fields = {
            "index": {"alias": "_index"},
            "id": {"alias": "_id"},
            "score": {"alias": "_score"},
            "source": {"alias": "_source"}
        }

class ElasticsearchAggregationBucket(BaseModel):
    """Bucket d'agrégation Elasticsearch."""
    key: Union[str, int, float] = Field(..., description="Clé du bucket")
    doc_count: NonNegativeInt = Field(..., description="Nombre de documents")
    key_as_string: Optional[str] = Field(None, description="Clé formatée")
    sub_aggregations: Dict[str, Any] = Field(default={}, description="Sous-agrégations")

class ElasticsearchAggregationResult(BaseModel):
    """Résultat d'agrégation Elasticsearch."""
    doc_count_error_upper_bound: Optional[int] = Field(None, description="Erreur borne supérieure")
    sum_other_doc_count: Optional[int] = Field(None, description="Somme autres documents")
    buckets: List[ElasticsearchAggregationBucket] = Field(default=[], description="Buckets")
    value: Optional[Union[int, float]] = Field(None, description="Valeur pour métriques simples")
    
    # Statistiques pour l'agrégation stats
    count: Optional[int] = Field(None, description="Nombre de valeurs")
    min: Optional[float] = Field(None, description="Valeur minimale")
    max: Optional[float] = Field(None, description="Valeur maximale")
    avg: Optional[float] = Field(None, description="Moyenne")
    sum: Optional[float] = Field(None, description="Somme")

class ElasticsearchResult(BaseModel):
    """Résultat complet Elasticsearch."""
    took: NonNegativeInt = Field(..., description="Temps d'exécution (ms)")
    timed_out: bool = Field(..., description="Timeout dépassé")
    shards: Dict[str, int] = Field(..., description="Informations shards")
    hits: Dict[str, Any] = Field(..., description="Résultats de recherche")
    aggregations: Optional[Dict[str, ElasticsearchAggregationResult]] = Field(
        None, description="Résultats d'agrégation"
    )

# ==================== EXPORTS PARTIE 1 ====================

__all__ = [
    # Enums
    "QueryType",
    "BoolClause", 
    "AggregationType",
    "SortOrder",
    
    # Requêtes de base
    "ElasticsearchQuery",
    "MatchQuery",
    "MultiMatchQuery",
    "TermQuery",
    "TermsQuery",
    "RangeQuery",
    "ExistsQuery",
    "BoolQuery",
    
    # Filtres
    "ElasticsearchFilter",
    "TermFilter",
    "TermsFilter",
    "RangeFilter",
    "ExistsFilter",
    
    # Agrégations
    "ElasticsearchAggregation",
    "TermsAggregation",
    "DateHistogramAggregation",
    "SumAggregation",
    "AvgAggregation",
    "MaxAggregation",
    "MinAggregation",
    "StatsAggregation",
    
    # Résultats
    "ElasticsearchHit",
    "ElasticsearchAggregationBucket",
    "ElasticsearchAggregationResult",
    "ElasticsearchResult"
]