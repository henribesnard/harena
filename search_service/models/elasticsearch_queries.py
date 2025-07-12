"""
Modèles de requêtes Elasticsearch pour le Search Service
Structures spécialisées pour la construction de requêtes ES optimisées
"""

import json
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

from config import INDEXED_FIELDS, settings


class ESQueryType(str, Enum):
    """Types de requêtes Elasticsearch supportés"""
    MATCH = "match"
    MULTI_MATCH = "multi_match"
    MATCH_PHRASE = "match_phrase"
    MATCH_PHRASE_PREFIX = "match_phrase_prefix"
    TERM = "term"
    TERMS = "terms"
    RANGE = "range"
    EXISTS = "exists"
    PREFIX = "prefix"
    WILDCARD = "wildcard"
    FUZZY = "fuzzy"
    BOOL = "bool"
    CONSTANT_SCORE = "constant_score"
    FUNCTION_SCORE = "function_score"
    SIMPLE_QUERY_STRING = "simple_query_string"


class ESBoolClause(str, Enum):
    """Clauses bool Elasticsearch"""
    MUST = "must"
    FILTER = "filter"
    SHOULD = "should"
    MUST_NOT = "must_not"


class ESSortOrder(str, Enum):
    """Ordres de tri Elasticsearch"""
    ASC = "asc"
    DESC = "desc"


class ESMultiMatchType(str, Enum):
    """Types de multi_match Elasticsearch"""
    BEST_FIELDS = "best_fields"
    MOST_FIELDS = "most_fields"
    CROSS_FIELDS = "cross_fields"
    PHRASE = "phrase"
    PHRASE_PREFIX = "phrase_prefix"
    BOOL_PREFIX = "bool_prefix"


class ESAggregationType(str, Enum):
    """Types d'agrégations Elasticsearch"""
    TERMS = "terms"
    RANGE = "range"
    DATE_HISTOGRAM = "date_histogram"
    HISTOGRAM = "histogram"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "value_count"
    CARDINALITY = "cardinality"
    STATS = "stats"
    EXTENDED_STATS = "extended_stats"
    PERCENTILES = "percentiles"


# === MODÈLES DE BASE ===

class ESField(BaseModel):
    """Modèle pour un champ Elasticsearch avec boost"""
    name: str = Field(..., description="Nom du champ")
    boost: float = Field(default=1.0, description="Facteur de boost")
    
    @field_validator("name")
    @classmethod
    def validate_field_name(cls, v):
        """Valide que le champ existe dans la configuration"""
        if v not in INDEXED_FIELDS:
            raise ValueError(f"Unknown field: {v}")
        return v
    
    @field_validator("boost")
    @classmethod
    def validate_boost(cls, v):
        """Valide le facteur de boost"""
        if v <= 0 or v > 10:
            raise ValueError("Boost must be between 0 and 10")
        return v
    
    def to_es_string(self) -> str:
        """Convertit en string Elasticsearch (field^boost)"""
        if self.boost == 1.0:
            return self.name
        return f"{self.name}^{self.boost}"


class ESSort(BaseModel):
    """Modèle pour le tri Elasticsearch"""
    field: str = Field(..., description="Champ de tri")
    order: ESSortOrder = Field(default=ESSortOrder.DESC, description="Ordre de tri")
    missing: Optional[str] = Field(default=None, description="Valeur pour champs manquants")
    unmapped_type: Optional[str] = Field(default=None, description="Type pour champs non mappés")
    
    @field_validator("field")
    @classmethod
    def validate_sort_field(cls, v):
        """Valide le champ de tri"""
        # Autoriser _score et _id
        if v in ["_score", "_id"]:
            return v
        
        # Vérifier les champs indexés
        if v not in INDEXED_FIELDS:
            raise ValueError(f"Cannot sort on unknown field: {v}")
        
        return v
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        sort_config = {"order": self.order.value}
        
        if self.missing is not None:
            sort_config["missing"] = self.missing
        
        if self.unmapped_type is not None:
            sort_config["unmapped_type"] = self.unmapped_type
        
        return {self.field: sort_config}


# === REQUÊTES SIMPLES ===

class ESTermQuery(BaseModel):
    """Requête term Elasticsearch"""
    field: str = Field(..., description="Champ à filtrer")
    value: Union[str, int, float, bool] = Field(..., description="Valeur exacte")
    boost: float = Field(default=1.0, description="Facteur de boost")
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        if self.boost == 1.0:
            return {"term": {self.field: self.value}}
        else:
            return {"term": {self.field: {"value": self.value, "boost": self.boost}}}


class ESTermsQuery(BaseModel):
    """Requête terms Elasticsearch"""
    field: str = Field(..., description="Champ à filtrer")
    values: List[Union[str, int, float]] = Field(..., description="Liste de valeurs")
    boost: float = Field(default=1.0, description="Facteur de boost")
    
    @field_validator("values")
    @classmethod
    def validate_values_count(cls, v):
        """Valide le nombre de valeurs"""
        if len(v) == 0:
            raise ValueError("Values list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Too many values (max 1000)")
        return v
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        query = {"terms": {self.field: self.values}}
        if self.boost != 1.0:
            query["terms"]["boost"] = self.boost
        return query


class ESRangeQuery(BaseModel):
    """Requête range Elasticsearch"""
    field: str = Field(..., description="Champ numérique/date")
    gte: Optional[Union[int, float, str]] = Field(default=None, description="Supérieur ou égal")
    gt: Optional[Union[int, float, str]] = Field(default=None, description="Supérieur")
    lte: Optional[Union[int, float, str]] = Field(default=None, description="Inférieur ou égal")
    lt: Optional[Union[int, float, str]] = Field(default=None, description="Inférieur")
    boost: float = Field(default=1.0, description="Facteur de boost")
    
    @model_validator(mode='after')
    def validate_range_values(self):
        """Valide qu'au moins une condition de range est fournie"""
        if not any([self.gte, self.gt, self.lte, self.lt]):
            raise ValueError("At least one range condition must be provided")
        return self
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        range_conditions = {}
        
        if self.gte is not None:
            range_conditions["gte"] = self.gte
        if self.gt is not None:
            range_conditions["gt"] = self.gt
        if self.lte is not None:
            range_conditions["lte"] = self.lte
        if self.lt is not None:
            range_conditions["lt"] = self.lt
        
        if self.boost != 1.0:
            range_conditions["boost"] = self.boost
        
        return {"range": {self.field: range_conditions}}


class ESMatchQuery(BaseModel):
    """Requête match Elasticsearch"""
    field: str = Field(..., description="Champ textuel")
    query: str = Field(..., description="Terme de recherche")
    operator: str = Field(default="or", description="Opérateur (and/or)")
    fuzziness: Optional[str] = Field(default=None, description="Fuzziness (AUTO, 0, 1, 2)")
    boost: float = Field(default=1.0, description="Facteur de boost")
    minimum_should_match: Optional[str] = Field(default=None, description="Pourcentage minimum de matching")
    
    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v):
        """Valide l'opérateur"""
        if v not in ["and", "or"]:
            raise ValueError("Operator must be 'and' or 'or'")
        return v
    
    @field_validator("fuzziness")
    @classmethod
    def validate_fuzziness(cls, v):
        """Valide la fuzziness"""
        if v is not None and v not in ["AUTO", "0", "1", "2"]:
            raise ValueError("Fuzziness must be AUTO, 0, 1, or 2")
        return v
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        match_config = {
            "query": self.query,
            "operator": self.operator
        }
        
        if self.fuzziness is not None:
            match_config["fuzziness"] = self.fuzziness
        
        if self.boost != 1.0:
            match_config["boost"] = self.boost
        
        if self.minimum_should_match is not None:
            match_config["minimum_should_match"] = self.minimum_should_match
        
        return {"match": {self.field: match_config}}


class ESMultiMatchQuery(BaseModel):
    """Requête multi_match Elasticsearch"""
    query: str = Field(..., description="Terme de recherche")
    fields: List[ESField] = Field(..., description="Champs avec boost")
    type: ESMultiMatchType = Field(default=ESMultiMatchType.BEST_FIELDS, description="Type de multi_match")
    operator: str = Field(default="or", description="Opérateur")
    fuzziness: Optional[str] = Field(default=None, description="Fuzziness")
    boost: float = Field(default=1.0, description="Facteur de boost global")
    minimum_should_match: Optional[str] = Field(default=None, description="Pourcentage minimum")
    
    @field_validator("fields")
    @classmethod
    def validate_fields_not_empty(cls, v):
        """Valide que la liste des champs n'est pas vide"""
        if not v:
            raise ValueError("Fields list cannot be empty")
        return v
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        # Convertir les champs en strings ES
        fields_strings = [field.to_es_string() for field in self.fields]
        
        multi_match_config = {
            "query": self.query,
            "fields": fields_strings,
            "type": self.type.value,
            "operator": self.operator
        }
        
        if self.fuzziness is not None:
            multi_match_config["fuzziness"] = self.fuzziness
        
        if self.boost != 1.0:
            multi_match_config["boost"] = self.boost
        
        if self.minimum_should_match is not None:
            multi_match_config["minimum_should_match"] = self.minimum_should_match
        
        return {"multi_match": multi_match_config}


# === REQUÊTES COMPOSÉES ===

class ESBoolQuery(BaseModel):
    """Requête bool Elasticsearch"""
    must: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses obligatoires")
    filter: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses de filtrage")
    should: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses optionnelles")
    must_not: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses d'exclusion")
    minimum_should_match: Optional[Union[int, str]] = Field(default=None, description="Minimum should match")
    boost: float = Field(default=1.0, description="Facteur de boost")
    
    def add_must(self, query: Dict[str, Any]):
        """Ajoute une clause must"""
        self.must.append(query)
    
    def add_filter(self, query: Dict[str, Any]):
        """Ajoute une clause filter"""
        self.filter.append(query)
    
    def add_should(self, query: Dict[str, Any]):
        """Ajoute une clause should"""
        self.should.append(query)
    
    def add_must_not(self, query: Dict[str, Any]):
        """Ajoute une clause must_not"""
        self.must_not.append(query)
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        bool_query = {"bool": {}}
        
        if self.must:
            bool_query["bool"]["must"] = self.must
        
        if self.filter:
            bool_query["bool"]["filter"] = self.filter
        
        if self.should:
            bool_query["bool"]["should"] = self.should
        
        if self.must_not:
            bool_query["bool"]["must_not"] = self.must_not
        
        if self.minimum_should_match is not None:
            bool_query["bool"]["minimum_should_match"] = self.minimum_should_match
        
        if self.boost != 1.0:
            bool_query["bool"]["boost"] = self.boost
        
        return bool_query


# === AGRÉGATIONS ===

class ESTermsAggregation(BaseModel):
    """Agrégation terms Elasticsearch"""
    field: str = Field(..., description="Champ d'agrégation")
    size: int = Field(default=10, description="Nombre de buckets")
    order: Optional[Dict[str, str]] = Field(default=None, description="Ordre des buckets")
    include: Optional[str] = Field(default=None, description="Pattern d'inclusion")
    exclude: Optional[str] = Field(default=None, description="Pattern d'exclusion")
    
    @field_validator("size")
    @classmethod
    def validate_size(cls, v):
        """Valide la taille"""
        if v < 1 or v > 1000:
            raise ValueError("Size must be between 1 and 1000")
        return v
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        terms_config = {
            "field": self.field,
            "size": self.size
        }
        
        if self.order is not None:
            terms_config["order"] = self.order
        
        if self.include is not None:
            terms_config["include"] = self.include
        
        if self.exclude is not None:
            terms_config["exclude"] = self.exclude
        
        return {"terms": terms_config}


class ESMetricAggregation(BaseModel):
    """Agrégation métrique Elasticsearch (sum, avg, min, max, etc.)"""
    type: ESAggregationType = Field(..., description="Type d'agrégation")
    field: str = Field(..., description="Champ numérique")
    missing: Optional[Union[int, float]] = Field(default=None, description="Valeur par défaut")
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        metric_config = {"field": self.field}
        
        if self.missing is not None:
            metric_config["missing"] = self.missing
        
        return {self.type.value: metric_config}


class ESDateHistogramAggregation(BaseModel):
    """Agrégation date_histogram Elasticsearch"""
    field: str = Field(..., description="Champ date")
    calendar_interval: Optional[str] = Field(default=None, description="Intervalle calendaire")
    fixed_interval: Optional[str] = Field(default=None, description="Intervalle fixe")
    format: Optional[str] = Field(default=None, description="Format de date")
    min_doc_count: int = Field(default=1, description="Count minimum par bucket")
    
    @model_validator(mode='after')
    def validate_interval(self):
        """Valide qu'un intervalle est fourni"""
        if not self.calendar_interval and not self.fixed_interval:
            raise ValueError("Either calendar_interval or fixed_interval must be provided")
        return self
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        date_histogram_config = {
            "field": self.field,
            "min_doc_count": self.min_doc_count
        }
        
        if self.calendar_interval:
            date_histogram_config["calendar_interval"] = self.calendar_interval
        
        if self.fixed_interval:
            date_histogram_config["fixed_interval"] = self.fixed_interval
        
        if self.format:
            date_histogram_config["format"] = self.format
        
        return {"date_histogram": date_histogram_config}


class ESAggregationContainer(BaseModel):
    """Conteneur pour agrégations avec sous-agrégations"""
    name: str = Field(..., description="Nom de l'agrégation")
    aggregation: Union[ESTermsAggregation, ESMetricAggregation, ESDateHistogramAggregation] = Field(..., description="Agrégation principale")
    sub_aggregations: Dict[str, 'ESAggregationContainer'] = Field(default_factory=dict, description="Sous-agrégations")
    
    def add_sub_aggregation(self, name: str, sub_agg: 'ESAggregationContainer'):
        """Ajoute une sous-agrégation"""
        self.sub_aggregations[name] = sub_agg
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch"""
        agg_dict = self.aggregation.to_es_dict()
        
        # Ajouter les sous-agrégations
        if self.sub_aggregations:
            if len(agg_dict) == 1:
                # Trouver la clé principale de l'agrégation
                main_key = list(agg_dict.keys())[0]
                if "aggs" not in agg_dict:
                    agg_dict["aggs"] = {}
                
                for sub_name, sub_agg in self.sub_aggregations.items():
                    agg_dict["aggs"][sub_name] = sub_agg.to_es_dict()
        
        return agg_dict


# === REQUÊTE COMPLÈTE ===

class ESSearchQuery(BaseModel):
    """Requête de recherche Elasticsearch complète"""
    query: Optional[Dict[str, Any]] = Field(default=None, description="Clause query")
    size: int = Field(default=20, description="Nombre de résultats")
    from_: int = Field(default=0, alias="from", description="Offset pagination")
    sort: List[ESSort] = Field(default_factory=list, description="Critères de tri")
    # CORRECTION: Remplacer _source par source_fields avec alias
    source_fields: Optional[Union[bool, List[str]]] = Field(default=None, alias="_source", description="Champs à retourner")
    highlight: Optional[Dict[str, Any]] = Field(default=None, description="Configuration highlighting")
    aggs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Agrégations")
    timeout: Optional[str] = Field(default=None, description="Timeout de la requête")
    track_total_hits: Optional[Union[bool, int]] = Field(default=None, description="Tracking du total")
    min_score: Optional[float] = Field(default=None, description="Score minimum")
    
    @field_validator("size")
    @classmethod
    def validate_size(cls, v):
        """Valide la taille des résultats"""
        if v < 0 or v > settings.max_results_per_query:
            raise ValueError(f"Size must be between 0 and {settings.max_results_per_query}")
        return v
    
    @field_validator("from_")
    @classmethod
    def validate_from(cls, v):
        """Valide l'offset de pagination"""
        if v < 0 or v > settings.max_pagination_offset:
            raise ValueError(f"From must be between 0 and {settings.max_pagination_offset}")
        return v
    
    def add_sort(self, field: str, order: ESSortOrder = ESSortOrder.DESC, **kwargs):
        """Ajoute un critère de tri"""
        sort_obj = ESSort(field=field, order=order, **kwargs)
        self.sort.append(sort_obj)
    
    def add_aggregation(self, name: str, aggregation: ESAggregationContainer):
        """Ajoute une agrégation"""
        self.aggs[name] = aggregation.to_es_dict()
    
    def set_highlighting(self, fields: List[str], fragment_size: int = 150, num_fragments: int = 3):
        """Configure le highlighting"""
        highlight_config = {
            "fields": {},
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"]
        }
        
        for field in fields:
            highlight_config["fields"][field] = {
                "fragment_size": fragment_size,
                "number_of_fragments": num_fragments
            }
        
        self.highlight = highlight_config
    
    def to_es_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch final"""
        es_query = {
            "size": self.size,
            "from": self.from_
        }
        
        if self.query is not None:
            es_query["query"] = self.query
        
        if self.sort:
            es_query["sort"] = [sort_obj.to_es_dict() for sort_obj in self.sort]
        
        if self.source_fields is not None:
            es_query["_source"] = self.source_fields
        
        if self.highlight is not None:
            es_query["highlight"] = self.highlight
        
        if self.aggs:
            es_query["aggs"] = self.aggs
        
        if self.timeout is not None:
            es_query["timeout"] = self.timeout
        
        if self.track_total_hits is not None:
            es_query["track_total_hits"] = self.track_total_hits
        
        if self.min_score is not None:
            es_query["min_score"] = self.min_score
        
        return es_query
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convertit en JSON Elasticsearch"""
        return json.dumps(self.to_es_dict(), indent=indent, ensure_ascii=False)


# === BUILDERS SPÉCIALISÉS ===

class FinancialTransactionQueryBuilder:
    """Builder spécialisé pour requêtes de transactions financières"""
    
    def __init__(self):
        self.query = ESSearchQuery()
        self.bool_query = ESBoolQuery()
    
    def add_user_filter(self, user_id: int) -> 'FinancialTransactionQueryBuilder':
        """Ajoute le filtre utilisateur obligatoire"""
        user_term = ESTermQuery(field="user_id", value=user_id)
        self.bool_query.add_filter(user_term.to_es_dict())
        return self
    
    def add_category_filter(self, category: str) -> 'FinancialTransactionQueryBuilder':
        """Ajoute un filtre par catégorie"""
        category_term = ESTermQuery(field="category_name.keyword", value=category)
        self.bool_query.add_filter(category_term.to_es_dict())
        return self
    
    def add_merchant_filter(self, merchant: str) -> 'FinancialTransactionQueryBuilder':
        """Ajoute un filtre par marchand"""
        merchant_term = ESTermQuery(field="merchant_name.keyword", value=merchant)
        self.bool_query.add_filter(merchant_term.to_es_dict())
        return self
    
    def add_amount_range(self, min_amount: Optional[float] = None, max_amount: Optional[float] = None) -> 'FinancialTransactionQueryBuilder':
        """Ajoute un filtre de plage de montants"""
        if min_amount is not None or max_amount is not None:
            range_query = ESRangeQuery(
                field="amount_abs",
                gte=min_amount,
                lte=max_amount
            )
            self.bool_query.add_filter(range_query.to_es_dict())
        return self
    
    def add_date_range(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> 'FinancialTransactionQueryBuilder':
        """Ajoute un filtre de plage de dates"""
        if start_date is not None or end_date is not None:
            range_query = ESRangeQuery(
                field="date",
                gte=start_date,
                lte=end_date
            )
            self.bool_query.add_filter(range_query.to_es_dict())
        return self
    
    def add_text_search(self, search_text: str, fields: Optional[List[str]] = None) -> 'FinancialTransactionQueryBuilder':
        """Ajoute une recherche textuelle"""
        if not fields:
            fields = ["searchable_text", "primary_description", "merchant_name", "category_name"]
        
        # Convertir en ESField avec boost par défaut
        es_fields = []
        field_boosts = {
            "searchable_text": 2.0,
            "primary_description": 1.5,
            "merchant_name": 1.8,
            "category_name": 1.2
        }
        
        for field in fields:
            boost = field_boosts.get(field, 1.0)
            es_fields.append(ESField(name=field, boost=boost))
        
        multi_match = ESMultiMatchQuery(
            query=search_text,
            fields=es_fields,
            type=ESMultiMatchType.BEST_FIELDS,
            fuzziness="AUTO"
        )
        
        self.bool_query.add_must(multi_match.to_es_dict())
        return self
    
    def add_category_aggregation(self, size: int = 10) -> 'FinancialTransactionQueryBuilder':
        """Ajoute une agrégation par catégorie"""
        terms_agg = ESTermsAggregation(
            field="category_name.keyword",
            size=size,
            order={"total_amount": "desc"}
        )
        
        # Sous-agrégations pour métriques
        agg_container = ESAggregationContainer(name="by_category", aggregation=terms_agg)
        
        # Ajouter sous-agrégations
        sum_agg = ESAggregationContainer(
            name="total_amount",
            aggregation=ESMetricAggregation(type=ESAggregationType.SUM, field="amount_abs")
        )
        avg_agg = ESAggregationContainer(
            name="avg_amount", 
            aggregation=ESMetricAggregation(type=ESAggregationType.AVG, field="amount_abs")
        )
        count_agg = ESAggregationContainer(
            name="transaction_count",
            aggregation=ESMetricAggregation(type=ESAggregationType.COUNT, field="transaction_id")
        )
        
        agg_container.add_sub_aggregation("total_amount", sum_agg)
        agg_container.add_sub_aggregation("avg_amount", avg_agg)
        agg_container.add_sub_aggregation("transaction_count", count_agg)
        
        self.query.add_aggregation("by_category", agg_container)
        return self
    
    def set_pagination(self, limit: int = 20, offset: int = 0) -> 'FinancialTransactionQueryBuilder':
        """Configure la pagination"""
        self.query.size = limit
        self.query.from_ = offset
        return self
    
    def set_sort_by_relevance_and_date(self) -> 'FinancialTransactionQueryBuilder':
        """Configure le tri par pertinence puis date"""
        self.query.add_sort("_score", ESSortOrder.DESC)
        self.query.add_sort("date", ESSortOrder.DESC, unmapped_type="date")
        return self
    
    def set_highlighting(self, enabled: bool = True) -> 'FinancialTransactionQueryBuilder':
        """Configure le highlighting"""
        if enabled:
            highlight_fields = ["searchable_text", "primary_description", "merchant_name"]
            self.query.set_highlighting(highlight_fields, fragment_size=150, num_fragments=2)
        return self
    
    def set_timeout(self, timeout_ms: int) -> 'FinancialTransactionQueryBuilder':
        """Configure le timeout"""
        self.query.timeout = f"{timeout_ms}ms"
        return self
    
    def build(self) -> ESSearchQuery:
        """Construit la requête finale"""
        # Affecter la bool query si elle contient des clauses
        if (self.bool_query.must or self.bool_query.filter or 
            self.bool_query.should or self.bool_query.must_not):
            self.query.query = self.bool_query.to_es_dict()
        
        return self.query


# === TEMPLATES DE REQUÊTES PRÉDÉFINIES ===

class ESQueryTemplates:
    """Templates de requêtes Elasticsearch prédéfinies pour cas d'usage fréquents"""
    
    @staticmethod
    def simple_user_transactions(user_id: int, limit: int = 20) -> ESSearchQuery:
        """Template: toutes les transactions d'un utilisateur"""
        return (FinancialTransactionQueryBuilder()
                .add_user_filter(user_id)
                .set_pagination(limit, 0)
                .set_sort_by_relevance_and_date()
                .build())
    
    @staticmethod
    def category_search(user_id: int, category: str, limit: int = 20) -> ESSearchQuery:
        """Template: recherche par catégorie"""
        return (FinancialTransactionQueryBuilder()
                .add_user_filter(user_id)
                .add_category_filter(category)
                .set_pagination(limit, 0)
                .set_sort_by_relevance_and_date()
                .build())
    
    @staticmethod
    def merchant_search(user_id: int, merchant: str, limit: int = 20) -> ESSearchQuery:
        """Template: recherche par marchand"""
        return (FinancialTransactionQueryBuilder()
                .add_user_filter(user_id)
                .add_merchant_filter(merchant)
                .set_pagination(limit, 0)
                .set_sort_by_relevance_and_date()
                .build())
    
    @staticmethod
    def text_search(user_id: int, search_text: str, limit: int = 20) -> ESSearchQuery:
        """Template: recherche textuelle"""
        return (FinancialTransactionQueryBuilder()
                .add_user_filter(user_id)
                .add_text_search(search_text)
                .set_pagination(limit, 0)
                .set_sort_by_relevance_and_date()
                .set_highlighting(True)
                .build())
    
    @staticmethod
    def amount_range_search(
        user_id: int, 
        min_amount: Optional[float] = None, 
        max_amount: Optional[float] = None,
        limit: int = 20
    ) -> ESSearchQuery:
        """Template: recherche par plage de montants"""
        return (FinancialTransactionQueryBuilder()
                .add_user_filter(user_id)
                .add_amount_range(min_amount, max_amount)
                .set_pagination(limit, 0)
                .set_sort_by_relevance_and_date()
                .build())
    
    @staticmethod
    def date_range_search(
        user_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20
    ) -> ESSearchQuery:
        """Template: recherche par plage de dates"""
        return (FinancialTransactionQueryBuilder()
                .add_user_filter(user_id)
                .add_date_range(start_date, end_date)
                .set_pagination(limit, 0)
                .set_sort_by_relevance_and_date()
                .build())
    
    @staticmethod
    def combined_search(
        user_id: int,
        search_text: Optional[str] = None,
        category: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20
    ) -> ESSearchQuery:
        """Template: recherche combinée (texte + filtres)"""
        builder = FinancialTransactionQueryBuilder().add_user_filter(user_id)
        
        if search_text:
            builder.add_text_search(search_text)
            builder.set_highlighting(True)
        
        if category:
            builder.add_category_filter(category)
        
        if min_amount is not None or max_amount is not None:
            builder.add_amount_range(min_amount, max_amount)
        
        if start_date or end_date:
            builder.add_date_range(start_date, end_date)
        
        return (builder
                .set_pagination(limit, 0)
                .set_sort_by_relevance_and_date()
                .build())
    
    @staticmethod
    def category_aggregation(user_id: int, date_filter: Optional[Dict[str, str]] = None) -> ESSearchQuery:
        """Template: agrégation par catégorie"""
        builder = FinancialTransactionQueryBuilder().add_user_filter(user_id)
        
        if date_filter:
            builder.add_date_range(
                date_filter.get("start_date"),
                date_filter.get("end_date")
            )
        
        return (builder
                .add_category_aggregation(size=15)
                .set_pagination(0, 0)  # Pas de hits, seulement agrégations
                .build())


# === VALIDATEURS DE REQUÊTES ===

class ESQueryValidator:
    """Validateur pour requêtes Elasticsearch"""
    
    @staticmethod
    def validate_query_structure(query_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide la structure d'une requête Elasticsearch"""
        errors = []
        
        # Vérification des champs obligatoires
        if "query" not in query_dict and "aggs" not in query_dict:
            errors.append("Query must contain either 'query' or 'aggs' field")
        
        # Validation size
        if "size" in query_dict:
            size = query_dict["size"]
            if not isinstance(size, int) or size < 0:
                errors.append("Size must be a non-negative integer")
            elif size > settings.max_results_per_query:
                errors.append(f"Size exceeds maximum allowed ({settings.max_results_per_query})")
        
        # Validation from
        if "from" in query_dict:
            from_val = query_dict["from"]
            if not isinstance(from_val, int) or from_val < 0:
                errors.append("From must be a non-negative integer")
            elif from_val > settings.max_pagination_offset:
                errors.append(f"From exceeds maximum allowed ({settings.max_pagination_offset})")
        
        # Validation du timeout
        if "timeout" in query_dict:
            timeout = query_dict["timeout"]
            if not isinstance(timeout, str) or not timeout.endswith("ms"):
                errors.append("Timeout must be a string ending with 'ms'")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_user_security(query_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide qu'un filtre user_id est présent pour la sécurité"""
        errors = []
        
        # Vérifier la présence d'un filtre user_id
        has_user_filter = ESQueryValidator._find_user_filter(query_dict.get("query", {}))
        
        if not has_user_filter:
            errors.append("Query must contain a user_id filter for security")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _find_user_filter(query_part: Dict[str, Any]) -> bool:
        """Recherche récursivement un filtre user_id dans la requête"""
        if not isinstance(query_part, dict):
            return False
        
        # Vérifier term user_id
        if "term" in query_part and isinstance(query_part["term"], dict):
            if "user_id" in query_part["term"]:
                return True
        
        # Vérifier dans bool query
        if "bool" in query_part and isinstance(query_part["bool"], dict):
            bool_query = query_part["bool"]
            
            for clause in ["must", "filter", "should"]:
                if clause in bool_query and isinstance(bool_query[clause], list):
                    for sub_query in bool_query[clause]:
                        if ESQueryValidator._find_user_filter(sub_query):
                            return True
        
        # Recherche récursive dans les autres types de requêtes
        for key, value in query_part.items():
            if isinstance(value, dict):
                if ESQueryValidator._find_user_filter(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and ESQueryValidator._find_user_filter(item):
                        return True
        
        return False


# === UTILITAIRES ===

def optimize_es_query(query_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Optimise une requête Elasticsearch pour la performance"""
    optimized = query_dict.copy()
    
    # Activer le cache de requête pour les requêtes avec filtres
    if "query" in optimized and "bool" in optimized.get("query", {}):
        bool_query = optimized["query"]["bool"]
        if "filter" in bool_query and bool_query["filter"]:
            optimized["request_cache"] = True
    
    # Optimiser track_total_hits selon la taille
    size = optimized.get("size", 10)
    if size <= 10000:
        optimized["track_total_hits"] = True
    else:
        optimized["track_total_hits"] = 10000
    
    # Ajouter preference pour la cohérence
    optimized["preference"] = "_local"
    
    return optimized


def extract_query_metadata(query_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extrait les métadonnées d'une requête pour analyse"""
    metadata = {
        "has_text_search": False,
        "has_filters": False,
        "has_aggregations": False,
        "estimated_complexity": "simple",
        "fields_used": set(),
        "user_id": None
    }
    
    # Analyser la query
    if "query" in query_dict:
        query_part = query_dict["query"]
        metadata["has_text_search"] = _has_text_search(query_part)
        metadata["has_filters"] = _has_filters(query_part)
        metadata["fields_used"] = _extract_fields(query_part)
        metadata["user_id"] = _extract_user_id(query_part)
    
    # Analyser les agrégations
    if "aggs" in query_dict and query_dict["aggs"]:
        metadata["has_aggregations"] = True
    
    # Estimer la complexité
    complexity_score = 0
    if metadata["has_text_search"]:
        complexity_score += 2
    if metadata["has_filters"]:
        complexity_score += 1
    if metadata["has_aggregations"]:
        complexity_score += len(query_dict.get("aggs", {}))
    
    if complexity_score <= 2:
        metadata["estimated_complexity"] = "simple"
    elif complexity_score <= 5:
        metadata["estimated_complexity"] = "medium"
    else:
        metadata["estimated_complexity"] = "complex"
    
    metadata["fields_used"] = list(metadata["fields_used"])
    
    return metadata


def _has_text_search(query_part: Dict[str, Any]) -> bool:
    """Détecte la présence de recherche textuelle"""
    text_query_types = ["match", "multi_match", "match_phrase", "simple_query_string"]
    
    if not isinstance(query_part, dict):
        return False
    
    for query_type in text_query_types:
        if query_type in query_part:
            return True
    
    # Recherche récursive
    for value in query_part.values():
        if isinstance(value, dict):
            if _has_text_search(value):
                return True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and _has_text_search(item):
                    return True
    
    return False


def _has_filters(query_part: Dict[str, Any]) -> bool:
    """Détecte la présence de filtres"""
    filter_query_types = ["term", "terms", "range", "exists"]
    
    if not isinstance(query_part, dict):
        return False
    
    for query_type in filter_query_types:
        if query_type in query_part:
            return True
    
    # Vérifier bool/filter clause
    if "bool" in query_part and isinstance(query_part["bool"], dict):
        if "filter" in query_part["bool"]:
            return True
    
    return False


def _extract_fields(query_part: Dict[str, Any], fields: Optional[set] = None) -> set:
    """Extrait tous les champs utilisés dans une requête"""
    if fields is None:
        fields = set()
    
    if not isinstance(query_part, dict):
        return fields
    
    # Chercher les champs dans différents types de requêtes
    for key, value in query_part.items():
        if key == "field" and isinstance(value, str):
            fields.add(value)
        elif key == "fields" and isinstance(value, list):
            for field in value:
                if isinstance(field, str):
                    # Extraire le nom du champ (avant le ^boost)
                    field_name = field.split("^")[0]
                    fields.add(field_name)
        elif isinstance(value, dict):
            # Recherche récursive
            _extract_fields(value, fields)
            
            # Cas spéciaux : term, range, etc.
            if key in ["term", "terms", "range", "exists"]:
                for field_name in value.keys():
                    fields.add(field_name)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _extract_fields(item, fields)
    
    return fields


def _extract_user_id(query_part: Dict[str, Any]) -> Optional[int]:
    """Extrait l'user_id de la requête"""
    if not isinstance(query_part, dict):
        return None
    
    # Chercher term user_id
    if "term" in query_part and isinstance(query_part["term"], dict):
        if "user_id" in query_part["term"]:
            user_id_value = query_part["term"]["user_id"]
            if isinstance(user_id_value, int):
                return user_id_value
            elif isinstance(user_id_value, dict) and "value" in user_id_value:
                return user_id_value["value"]
    
    # Recherche récursive
    for value in query_part.values():
        if isinstance(value, dict):
            result = _extract_user_id(value)
            if result is not None:
                return result
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    result = _extract_user_id(item)
                    if result is not None:
                        return result
    
    return None


# === EXPORTS ===

__all__ = [
    # Enums
    "ESQueryType",
    "ESBoolClause",
    "ESSortOrder",
    "ESMultiMatchType",
    "ESAggregationType",
    
    # Modèles de base
    "ESField",
    "ESSort",
    
    # Requêtes simples
    "ESTermQuery",
    "ESTermsQuery",
    "ESRangeQuery",
    "ESMatchQuery",
    "ESMultiMatchQuery",
    
    # Requêtes composées
    "ESBoolQuery",
    
    # Agrégations
    "ESTermsAggregation",
    "ESMetricAggregation",
    "ESDateHistogramAggregation",
    "ESAggregationContainer",
    
    # Requête complète
    "ESSearchQuery",
    
    # Builders
    "FinancialTransactionQueryBuilder",
    
    # Templates
    "ESQueryTemplates",
    
    # Validateurs
    "ESQueryValidator",
    
    # Utilitaires
    "optimize_es_query",
    "extract_query_metadata"
]