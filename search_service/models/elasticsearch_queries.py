"""
🔍 Modèles Requêtes Elasticsearch - Requêtes Optimisées
======================================================

Modèles pour la génération et validation de requêtes Elasticsearch optimisées
pour les transactions financières. Mapping direct vers l'API Elasticsearch.

Responsabilités:
- Génération requêtes Elasticsearch valides
- Optimisations performance spécifiques
- Validation syntaxe et sémantique
- Templates requêtes réutilisables
- Mapping filtres vers clauses ES
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import json

from .service_contracts import SearchFilter, FilterOperator, AggregationType
from .filters import CompositeFilter


# =============================================================================
# 🎯 ÉNUMÉRATIONS ELASTICSEARCH
# =============================================================================

class ESQueryType(str, Enum):
    """Types de requêtes Elasticsearch."""
    MATCH = "match"
    MATCH_ALL = "match_all"
    MATCH_PHRASE = "match_phrase"
    MULTI_MATCH = "multi_match"
    TERM = "term"
    TERMS = "terms"
    RANGE = "range"
    BOOL = "bool"
    QUERY_STRING = "query_string"
    SIMPLE_QUERY_STRING = "simple_query_string"

class ESAggType(str, Enum):
    """Types d'agrégations Elasticsearch."""
    TERMS = "terms"
    DATE_HISTOGRAM = "date_histogram"
    HISTOGRAM = "histogram"
    RANGE = "range"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    STATS = "stats"
    CARDINALITY = "cardinality"

class ESScoreMode(str, Enum):
    """Modes de scoring Elasticsearch."""
    MULTIPLY = "multiply"
    SUM = "sum"
    AVG = "avg"
    FIRST = "first"
    MAX = "max"
    MIN = "min"

class ESBoostMode(str, Enum):
    """Modes de boost Elasticsearch."""
    MULTIPLY = "multiply"
    REPLACE = "replace"
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"


# =============================================================================
# 🏗️ COMPOSANTS REQUÊTE ELASTICSEARCH
# =============================================================================

class ESQueryClause(BaseModel):
    """Clause de requête Elasticsearch."""
    query_type: ESQueryType = Field(..., description="Type de requête")
    field: Optional[str] = Field(None, description="Champ cible")
    value: Optional[Union[str, int, float, bool, List, Dict]] = Field(None, description="Valeur requête")
    boost: Optional[float] = Field(None, ge=0.1, le=100.0, description="Boost scoring")
    options: Dict[str, Any] = Field(default_factory=dict, description="Options spécifiques")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Conversion vers clause Elasticsearch."""
        if self.query_type == ESQueryType.MATCH_ALL:
            query = {"match_all": {}}
            if self.boost:
                query["match_all"]["boost"] = self.boost
            return query
        
        elif self.query_type == ESQueryType.MATCH:
            query_body = {"query": self.value}
            if self.boost:
                query_body["boost"] = self.boost
            query_body.update(self.options)
            return {"match": {self.field: query_body}}
        
        elif self.query_type == ESQueryType.MATCH_PHRASE:
            query_body = {"query": self.value}
            if self.boost:
                query_body["boost"] = self.boost
            query_body.update(self.options)
            return {"match_phrase": {self.field: query_body}}
        
        elif self.query_type == ESQueryType.MULTI_MATCH:
            query_body = {
                "query": self.value,
                "fields": self.options.get("fields", [])
            }
            if self.boost:
                query_body["boost"] = self.boost
            
            # Options multi_match spécifiques
            for option in ["type", "tie_breaker", "minimum_should_match"]:
                if option in self.options:
                    query_body[option] = self.options[option]
            
            return {"multi_match": query_body}
        
        elif self.query_type == ESQueryType.TERM:
            query_body = {"value": self.value}
            if self.boost:
                query_body["boost"] = self.boost
            return {"term": {self.field: query_body}}
        
        elif self.query_type == ESQueryType.TERMS:
            return {"terms": {self.field: self.value}}
        
        elif self.query_type == ESQueryType.RANGE:
            range_body = self.value if isinstance(self.value, dict) else {}
            if self.boost:
                range_body["boost"] = self.boost
            return {"range": {self.field: range_body}}
        
        elif self.query_type == ESQueryType.QUERY_STRING:
            query_body = {"query": self.value}
            if self.boost:
                query_body["boost"] = self.boost
            query_body.update(self.options)
            return {"query_string": query_body}
        
        else:
            raise ValueError(f"Unsupported query type: {self.query_type}")

class ESBoolQuery(BaseModel):
    """Requête bool Elasticsearch."""
    must: List[ESQueryClause] = Field(default_factory=list, description="Clauses must (AND)")
    should: List[ESQueryClause] = Field(default_factory=list, description="Clauses should (OR)")
    must_not: List[ESQueryClause] = Field(default_factory=list, description="Clauses must_not (NOT)")
    filter: List[ESQueryClause] = Field(default_factory=list, description="Clauses filter (AND sans score)")
    minimum_should_match: Optional[Union[int, str]] = Field(None, description="Minimum should clauses")
    boost: Optional[float] = Field(None, ge=0.1, le=100.0, description="Boost global")
    
    @validator('must', 'should', 'must_not', 'filter')
    def validate_clause_lists(cls, v):
        """Validation listes clauses."""
        if len(v) > 100:  # Limite raisonnable
            raise ValueError("Too many clauses in bool query (max 100)")
        return v
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Conversion vers requête bool Elasticsearch."""
        bool_query = {}
        
        if self.must:
            bool_query["must"] = [clause.to_elasticsearch() for clause in self.must]
        
        if self.should:
            bool_query["should"] = [clause.to_elasticsearch() for clause in self.should]
            
        if self.must_not:
            bool_query["must_not"] = [clause.to_elasticsearch() for clause in self.must_not]
        
        if self.filter:
            bool_query["filter"] = [clause.to_elasticsearch() for clause in self.filter]
        
        if self.minimum_should_match is not None:
            bool_query["minimum_should_match"] = self.minimum_should_match
        
        if self.boost:
            bool_query["boost"] = self.boost
        
        return {"bool": bool_query}

class ESAggregation(BaseModel):
    """Agrégation Elasticsearch."""
    name: str = Field(..., description="Nom agrégation")
    agg_type: ESAggType = Field(..., description="Type agrégation")
    field: Optional[str] = Field(None, description="Champ agrégation")
    size: Optional[int] = Field(None, ge=1, le=10000, description="Taille résultats")
    options: Dict[str, Any] = Field(default_factory=dict, description="Options spécifiques")
    sub_aggregations: List['ESAggregation'] = Field(default_factory=list, description="Sous-agrégations")
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Conversion vers agrégation Elasticsearch."""
        agg_body = {}
        
        if self.agg_type == ESAggType.TERMS:
            agg_body = {"field": self.field}
            if self.size:
                agg_body["size"] = self.size
            
            # Options terms spécifiques
            for option in ["order", "min_doc_count", "shard_min_doc_count"]:
                if option in self.options:
                    agg_body[option] = self.options[option]
        
        elif self.agg_type == ESAggType.DATE_HISTOGRAM:
            agg_body = {
                "field": self.field,
                "calendar_interval": self.options.get("interval", "1M")
            }
            
            # Options date_histogram spécifiques
            for option in ["format", "time_zone", "offset", "min_doc_count"]:
                if option in self.options:
                    agg_body[option] = self.options[option]
        
        elif self.agg_type == ESAggType.RANGE:
            agg_body = {
                "field": self.field,
                "ranges": self.options.get("ranges", [])
            }
        
        elif self.agg_type in [ESAggType.SUM, ESAggType.AVG, ESAggType.MIN, ESAggType.MAX]:
            agg_body = {"field": self.field}
        
        elif self.agg_type == ESAggType.STATS:
            agg_body = {"field": self.field}
        
        elif self.agg_type == ESAggType.CARDINALITY:
            agg_body = {"field": self.field}
            if "precision_threshold" in self.options:
                agg_body["precision_threshold"] = self.options["precision_threshold"]
        
        # Construire agrégation complète
        aggregation = {self.agg_type.value: agg_body}
        
        # Ajouter sous-agrégations
        if self.sub_aggregations:
            aggs = {}
            for sub_agg in self.sub_aggregations:
                sub_agg_dict = sub_agg.to_elasticsearch()
                aggs[sub_agg.name] = sub_agg_dict
            aggregation["aggs"] = aggs
        
        return aggregation


# =============================================================================
# 🔍 REQUÊTE ELASTICSEARCH COMPLÈTE
# =============================================================================

class ElasticsearchQuery(BaseModel):
    """Requête Elasticsearch complète."""
    query: Optional[Union[ESQueryClause, ESBoolQuery]] = Field(None, description="Clause requête principale")
    aggregations: List[ESAggregation] = Field(default_factory=list, description="Agrégations")
    size: int = Field(default=20, ge=0, le=10000, description="Nombre résultats")
    from_: int = Field(default=0, ge=0, alias="from", description="Offset pagination")
    sort: List[Dict[str, Any]] = Field(default_factory=list, description="Tri résultats")
    source: Optional[Union[bool, List[str], Dict[str, List[str]]]] = Field(None, description="Champs source")
    highlight: Optional[Dict[str, Any]] = Field(None, description="Configuration highlighting")
    timeout: Optional[str] = Field(None, description="Timeout requête")
    track_total_hits: bool = Field(default=True, description="Tracker nombre total")
    
    class Config:
        """Configuration Pydantic."""
        allow_population_by_field_name = True
    
    @validator('sort')
    def validate_sort(cls, v):
        """Validation configuration tri."""
        if len(v) > 10:  # Limite raisonnable
            raise ValueError("Too many sort criteria (max 10)")
        return v
    
    def to_elasticsearch(self) -> Dict[str, Any]:
        """Conversion vers requête Elasticsearch complète."""
        es_query = {}
        
        # Requête principale
        if self.query:
            if isinstance(self.query, ESBoolQuery):
                es_query["query"] = self.query.to_elasticsearch()
            else:
                es_query["query"] = self.query.to_elasticsearch()
        else:
            # Requête par défaut match_all
            es_query["query"] = {"match_all": {}}
        
        # Agrégations
        if self.aggregations:
            aggs = {}
            for agg in self.aggregations:
                aggs[agg.name] = agg.to_elasticsearch()
            es_query["aggs"] = aggs
        
        # Pagination
        es_query["size"] = self.size
        if self.from_ > 0:
            es_query["from"] = self.from_
        
        # Tri
        if self.sort:
            es_query["sort"] = self.sort
        
        # Champs source
        if self.source is not None:
            es_query["_source"] = self.source
        
        # Highlighting
        if self.highlight:
            es_query["highlight"] = self.highlight
        
        # Options performance
        if self.timeout:
            es_query["timeout"] = self.timeout
        
        es_query["track_total_hits"] = self.track_total_hits
        
        return es_query
    
    def validate_query(self) -> Dict[str, Any]:
        """Validation complète de la requête."""
        validation = {"valid": True, "errors": [], "warnings": []}
        
        try:
            # Test sérialisation
            es_dict = self.to_elasticsearch()
            
            # Validation taille
            query_size = len(json.dumps(es_dict))
            if query_size > 1024 * 1024:  # 1MB
                validation["warnings"].append("Query size very large, may impact performance")
            
            # Validation pagination profonde
            if self.from_ > 10000:
                validation["warnings"].append("Deep pagination may impact performance")
            
            # Validation nombre agrégations
            if len(self.aggregations) > 10:
                validation["warnings"].append("Many aggregations may impact performance")
            
        except Exception as e:
            validation["errors"].append(f"Query serialization error: {str(e)}")
            validation["valid"] = False
        
        return validation


# =============================================================================
# 🏭 FACTORY REQUÊTES SPÉCIALISÉES
# =============================================================================

class FinancialQueryFactory:
    """Factory pour requêtes financières optimisées."""
    
    @staticmethod
    def create_user_isolation_filter(user_id: int) -> ESQueryClause:
        """Créer filtre isolation utilisateur."""
        return ESQueryClause(
            query_type=ESQueryType.TERM,
            field="user_id",
            value=user_id
        )
    
    @staticmethod
    def create_text_search_query(query_text: str, fields: List[str], boost: float = 1.0) -> ESQueryClause:
        """Créer requête recherche textuelle."""
        return ESQueryClause(
            query_type=ESQueryType.MULTI_MATCH,
            value=query_text,
            boost=boost,
            options={
                "fields": fields,
                "type": "best_fields",
                "tie_breaker": 0.3,
                "minimum_should_match": "75%"
            }
        )
    
    @staticmethod
    def create_category_filter(categories: List[str]) -> ESQueryClause:
        """Créer filtre catégories."""
        if len(categories) == 1:
            return ESQueryClause(
                query_type=ESQueryType.TERM,
                field="category_name.keyword",
                value=categories[0]
            )
        else:
            return ESQueryClause(
                query_type=ESQueryType.TERMS,
                field="category_name.keyword",
                value=categories
            )
    
    @staticmethod
    def create_amount_range_filter(min_amount: float = None, max_amount: float = None, 
                                  field: str = "amount_abs") -> ESQueryClause:
        """Créer filtre plage montants."""
        range_conditions = {}
        
        if min_amount is not None:
            range_conditions["gte"] = min_amount
        
        if max_amount is not None:
            range_conditions["lte"] = max_amount
        
        return ESQueryClause(
            query_type=ESQueryType.RANGE,
            field=field,
            value=range_conditions
        )
    
    @staticmethod
    def create_date_range_filter(start_date: str, end_date: str) -> ESQueryClause:
        """Créer filtre plage dates."""
        return ESQueryClause(
            query_type=ESQueryType.RANGE,
            field="date",
            value={
                "gte": start_date,
                "lte": end_date,
                "format": "yyyy-MM-dd"
            }
        )
    
    @staticmethod
    def create_merchant_filter(merchants: List[str], exact_match: bool = True) -> ESQueryClause:
        """Créer filtre marchands."""
        field = "merchant_name.keyword" if exact_match else "merchant_name"
        
        if len(merchants) == 1:
            query_type = ESQueryType.TERM if exact_match else ESQueryType.MATCH
            return ESQueryClause(
                query_type=query_type,
                field=field,
                value=merchants[0]
            )
        else:
            return ESQueryClause(
                query_type=ESQueryType.TERMS,
                field=field,
                value=merchants
            )


# =============================================================================
# 📊 AGRÉGATIONS FINANCIÈRES SPÉCIALISÉES
# =============================================================================

class FinancialAggregationFactory:
    """Factory pour agrégations financières."""
    
    @staticmethod
    def create_category_stats() -> ESAggregation:
        """Créer agrégation statistiques par catégorie."""
        return ESAggregation(
            name="category_stats",
            agg_type=ESAggType.TERMS,
            field="category_name.keyword",
            size=50,
            options={"order": {"total_amount": "desc"}},
            sub_aggregations=[
                ESAggregation(
                    name="total_amount",
                    agg_type=ESAggType.SUM,
                    field="amount_abs"
                ),
                ESAggregation(
                    name="avg_amount",
                    agg_type=ESAggType.AVG,
                    field="amount_abs"
                ),
                ESAggregation(
                    name="transaction_count",
                    agg_type=ESAggType.CARDINALITY,
                    field="transaction_id"
                )
            ]
        )
    
    @staticmethod
    def create_merchant_stats() -> ESAggregation:
        """Créer agrégation statistiques par marchand."""
        return ESAggregation(
            name="merchant_stats",
            agg_type=ESAggType.TERMS,
            field="merchant_name.keyword",
            size=20,
            options={"min_doc_count": 2},
            sub_aggregations=[
                ESAggregation(
                    name="total_spent",
                    agg_type=ESAggType.SUM,
                    field="amount_abs"
                ),
                ESAggregation(
                    name="avg_transaction",
                    agg_type=ESAggType.AVG,
                    field="amount_abs"
                )
            ]
        )
    
    @staticmethod
    def create_temporal_analysis(interval: str = "1M") -> ESAggregation:
        """Créer agrégation analyse temporelle."""
        return ESAggregation(
            name="temporal_analysis",
            agg_type=ESAggType.DATE_HISTOGRAM,
            field="date",
            options={
                "calendar_interval": interval,
                "format": "yyyy-MM",
                "min_doc_count": 1
            },
            sub_aggregations=[
                ESAggregation(
                    name="monthly_total",
                    agg_type=ESAggType.SUM,
                    field="amount_abs"
                ),
                ESAggregation(
                    name="monthly_avg",
                    agg_type=ESAggType.AVG,
                    field="amount_abs"
                ),
                ESAggregation(
                    name="transaction_count",
                    agg_type=ESAggType.CARDINALITY,
                    field="transaction_id"
                )
            ]
        )
    
    @staticmethod
    def create_amount_ranges() -> ESAggregation:
        """Créer agrégation plages montants."""
        return ESAggregation(
            name="amount_ranges",
            agg_type=ESAggType.RANGE,
            field="amount_abs",
            options={
                "ranges": [
                    {"to": 10},
                    {"from": 10, "to": 50},
                    {"from": 50, "to": 100},
                    {"from": 100, "to": 500},
                    {"from": 500}
                ]
            }
        )


# =============================================================================
# 🎯 TEMPLATES REQUÊTES RÉUTILISABLES
# =============================================================================

class QueryTemplate(BaseModel):
    """Template de requête réutilisable."""
    name: str = Field(..., description="Nom template")
    description: str = Field(..., description="Description template")
    parameters: Dict[str, Any] = Field(..., description="Paramètres template")
    query_template: Dict[str, Any] = Field(..., description="Structure requête")
    
    def render(self, **kwargs) -> ElasticsearchQuery:
        """Rendre template avec paramètres."""
        # Substitution simple des paramètres
        rendered_template = self._substitute_parameters(self.query_template, kwargs)
        
        # Conversion vers ElasticsearchQuery
        # Note: Implémentation simplifiée, en production utiliser Jinja2
        return ElasticsearchQuery.parse_obj(rendered_template)
    
    def _substitute_parameters(self, template: Any, params: Dict[str, Any]) -> Any:
        """Substitution récursive paramètres."""
        if isinstance(template, dict):
            return {k: self._substitute_parameters(v, params) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._substitute_parameters(item, params) for item in template]
        elif isinstance(template, str) and template.startswith("{{") and template.endswith("}}"):
            param_name = template[2:-2].strip()
            return params.get(param_name, template)
        else:
            return template

class FinancialQueryTemplates:
    """Templates prédéfinis pour requêtes financières."""
    
    CATEGORY_SEARCH = QueryTemplate(
        name="category_search",
        description="Recherche par catégorie avec statistiques",
        parameters={
            "user_id": int,
            "category": str,
            "size": 20,
            "include_stats": True
        },
        query_template={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"user_id": "{{user_id}}"}},
                        {"term": {"category_name.keyword": "{{category}}"}}
                    ]
                }
            },
            "size": "{{size}}",
            "sort": [{"date": {"order": "desc"}}],
            "aggs": {
                "category_stats": {
                    "terms": {"field": "category_name.keyword"},
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount_abs"}},
                        "avg_amount": {"avg": {"field": "amount_abs"}}
                    }
                }
            }
        }
    )
    
    TEXT_SEARCH = QueryTemplate(
        name="text_search",
        description="Recherche textuelle multi-champs",
        parameters={
            "user_id": int,
            "query_text": str,
            "fields": ["searchable_text", "primary_description", "merchant_name"],
            "size": 20
        },
        query_template={
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": "{{query_text}}",
                                "fields": "{{fields}}",
                                "type": "best_fields",
                                "tie_breaker": 0.3
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"user_id": "{{user_id}}"}}
                    ]
                }
            },
            "size": "{{size}}",
            "highlight": {
                "fields": {
                    "searchable_text": {},
                    "primary_description": {}
                }
            }
        }
    )
    
    TEMPORAL_ANALYSIS = QueryTemplate(
        name="temporal_analysis",
        description="Analyse temporelle avec tendances",
        parameters={
            "user_id": int,
            "start_date": str,
            "end_date": str,
            "interval": "1M"
        },
        query_template={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"user_id": "{{user_id}}"}},
                        {
                            "range": {
                                "date": {
                                    "gte": "{{start_date}}",
                                    "lte": "{{end_date}}",
                                    "format": "yyyy-MM-dd"
                                }
                            }
                        }
                    ]
                }
            },
            "size": 0,
            "aggs": {
                "timeline": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "{{interval}}",
                        "format": "yyyy-MM"
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}},
                        "avg_amount": {"avg": {"field": "amount_abs"}}
                    }
                }
            }
        }
    )


# =============================================================================
# 🛠️ BUILDERS REQUÊTES COMPLEXES
# =============================================================================

class ElasticsearchQueryBuilder:
    """Builder pour construire requêtes Elasticsearch complexes."""
    
    def __init__(self):
        """Initialisation builder."""
        self._bool_query = ESBoolQuery()
        self._aggregations = []
        self._size = 20
        self._from = 0
        self._sort = []
        self._source = None
        self._highlight = None
    
    def add_must(self, clause: ESQueryClause) -> 'ElasticsearchQueryBuilder':
        """Ajouter clause must."""
        self._bool_query.must.append(clause)
        return self
    
    def add_filter(self, clause: ESQueryClause) -> 'ElasticsearchQueryBuilder':
        """Ajouter clause filter."""
        self._bool_query.filter.append(clause)
        return self
    
    def add_should(self, clause: ESQueryClause) -> 'ElasticsearchQueryBuilder':
        """Ajouter clause should."""
        self._bool_query.should.append(clause)
        return self
    
    def add_must_not(self, clause: ESQueryClause) -> 'ElasticsearchQueryBuilder':
        """Ajouter clause must_not."""
        self._bool_query.must_not.append(clause)
        return self
    
    def add_aggregation(self, aggregation: ESAggregation) -> 'ElasticsearchQueryBuilder':
        """Ajouter agrégation."""
        self._aggregations.append(aggregation)
        return self
    
    def set_size(self, size: int) -> 'ElasticsearchQueryBuilder':
        """Définir taille résultats."""
        self._size = size
        return self
    
    def set_from(self, from_: int) -> 'ElasticsearchQueryBuilder':
        """Définir offset pagination."""
        self._from = from_
        return self
    
    def add_sort(self, field: str, order: str = "desc") -> 'ElasticsearchQueryBuilder':
        """Ajouter tri."""
        self._sort.append({field: {"order": order}})
        return self
    
    def set_source(self, source: Union[bool, List[str]]) -> 'ElasticsearchQueryBuilder':
        """Définir champs source."""
        self._source = source
        return self
    
    def enable_highlighting(self, fields: List[str]) -> 'ElasticsearchQueryBuilder':
        """Activer highlighting."""
        self._highlight = {
            "fields": {field: {} for field in fields}
        }
        return self
    
    def apply_composite_filter(self, composite_filter: CompositeFilter) -> 'ElasticsearchQueryBuilder':
        """Appliquer filtre composite."""
        filter_groups = composite_filter.to_search_filters()
        
        # Convertir SearchFilters vers ESQueryClauses
        for search_filter in filter_groups["required"]:
            es_clause = self._convert_search_filter(search_filter)
            self.add_filter(es_clause)
        
        for search_filter in filter_groups["ranges"]:
            es_clause = self._convert_search_filter(search_filter)
            self.add_filter(es_clause)
        
        return self
    
    def _convert_search_filter(self, search_filter: SearchFilter) -> ESQueryClause:
        """Convertir SearchFilter vers ESQueryClause."""
        if search_filter.operator == FilterOperator.EQ:
            return ESQueryClause(
                query_type=ESQueryType.TERM,
                field=search_filter.field,
                value=search_filter.value
            )
        elif search_filter.operator == FilterOperator.IN:
            return ESQueryClause(
                query_type=ESQueryType.TERMS,
                field=search_filter.field,
                value=search_filter.value
            )
        elif search_filter.operator == FilterOperator.BETWEEN:
            return ESQueryClause(
                query_type=ESQueryType.RANGE,
                field=search_filter.field,
                value={
                    "gte": search_filter.value[0],
                    "lte": search_filter.value[1]
                }
            )
        elif search_filter.operator == FilterOperator.GTE:
            return ESQueryClause(
                query_type=ESQueryType.RANGE,
                field=search_filter.field,
                value={"gte": search_filter.value}
            )
        elif search_filter.operator == FilterOperator.LTE:
            return ESQueryClause(
                query_type=ESQueryType.RANGE,
                field=search_filter.field,
                value={"lte": search_filter.value}
            )
        else:
            raise ValueError(f"Unsupported filter operator: {search_filter.operator}")
    
    def build(self) -> ElasticsearchQuery:
        """Construire requête finale."""
        return ElasticsearchQuery(
            query=self._bool_query,
            aggregations=self._aggregations,
            size=self._size,
            from_=self._from,
            sort=self._sort,
            source=self._source,
            highlight=self._highlight
        )


# =============================================================================
# 📋 EXPORTS
# =============================================================================

__all__ = [
    # Énumérations
    "ESQueryType", "ESAggType", "ESScoreMode", "ESBoostMode",
    # Composants requête
    "ESQueryClause", "ESBoolQuery", "ESAggregation", "ElasticsearchQuery",
    # Factories
    "FinancialQueryFactory", "FinancialAggregationFactory",
    # Templates
    "QueryTemplate", "FinancialQueryTemplates",
    # Builder
    "ElasticsearchQueryBuilder",
]