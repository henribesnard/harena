"""
Builders et templates Elasticsearch pour le Search Service - Partie 2.

Ce fichier contient les builders, templates et utilitaires pour construire
des requêtes Elasticsearch complexes de manière fluide et optimisée.

ARCHITECTURE:
- QueryBuilder pour construction fluide de requêtes
- AggregationBuilder pour agrégations financières
- Templates prédéfinis pour intentions financières
- Mapping d'index optimisé
- Utilitaires d'optimisation et validation

CONFIGURATION CENTRALISÉE:
- Templates configurables via paramètres
- Mapping financier optimisé
- Validation et optimisation automatique
"""

from typing import Dict, List, Any, Optional, Union
import copy

from pydantic import BaseModel, Field, validator

# Configuration centralisée
from config_service.config import settings

# Import des modèles de base
from .elasticsearch_queries import (
    ElasticsearchQuery, ElasticsearchFilter, ElasticsearchAggregation,
    BoolQuery, MatchQuery, MultiMatchQuery, TermQuery, TermsQuery, RangeQuery,
    TermFilter, TermsFilter, RangeFilter,
    TermsAggregation, DateHistogramAggregation, SumAggregation, AvgAggregation,
    MaxAggregation, MinAggregation, StatsAggregation,
    SortOrder
)

# ==================== BUILDERS ====================

class QueryBuilder:
    """Builder pour construire des requêtes Elasticsearch complexes."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Remet à zéro le builder."""
        self._query = None
        self._filters = []
        self._aggregations = []
        self._sort = []
        self._source = None
        self._size = None
        self._from = None
        self._highlight = None
        self._min_score = None
        return self
    
    def query(self, query: ElasticsearchQuery):
        """Ajoute une requête principale."""
        self._query = query
        return self
    
    def filter(self, filter_obj: ElasticsearchFilter):
        """Ajoute un filtre."""
        self._filters.append(filter_obj)
        return self
    
    def aggregation(self, agg: ElasticsearchAggregation):
        """Ajoute une agrégation."""
        self._aggregations.append(agg)
        return self
    
    def sort(self, field: str, order: SortOrder = SortOrder.DESC):
        """Ajoute un tri."""
        self._sort.append({field: {"order": order}})
        return self
    
    def source(self, fields: List[str]):
        """Spécifie les champs à retourner."""
        self._source = fields
        return self
    
    def size(self, size: int):
        """Spécifie la taille des résultats."""
        self._size = size
        return self
    
    def from_offset(self, offset: int):
        """Spécifie l'offset pour la pagination."""
        self._from = offset
        return self
    
    def highlight(self, fields: List[str], fragment_size: int = 150):
        """Ajoute le highlighting."""
        self._highlight = {
            "fields": {field: {"fragment_size": fragment_size} for field in fields}
        }
        return self
    
    def min_score(self, score: float):
        """Spécifie le score minimum."""
        self._min_score = score
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construit la requête Elasticsearch finale."""
        query_dict = {}
        
        # Construction de la requête principale avec filtres
        if self._query or self._filters:
            if self._filters and self._query:
                # Bool query avec requête et filtres
                bool_query = BoolQuery(
                    must=[self._query] if self._query else [],
                    filter=self._filters
                )
                query_dict["query"] = bool_query.to_dict()
            elif self._filters:
                # Seulement des filtres
                bool_query = BoolQuery(filter=self._filters)
                query_dict["query"] = bool_query.to_dict()
            elif self._query:
                # Seulement une requête
                query_dict["query"] = self._query.to_dict()
        
        # Agrégations
        if self._aggregations:
            aggs_dict = {}
            for agg in self._aggregations:
                aggs_dict.update(agg.to_dict())
            query_dict["aggs"] = aggs_dict
        
        # Paramètres de réponse
        if self._sort:
            query_dict["sort"] = self._sort
        if self._source is not None:
            query_dict["_source"] = self._source
        if self._size is not None:
            query_dict["size"] = self._size
        if self._from is not None:
            query_dict["from"] = self._from
        if self._highlight:
            query_dict["highlight"] = self._highlight
        if self._min_score is not None:
            query_dict["min_score"] = self._min_score
        
        return query_dict

class AggregationBuilder:
    """Builder spécialisé pour les agrégations financières."""
    
    @staticmethod
    def category_breakdown(size: int = 10) -> TermsAggregation:
        """Agrégation par catégorie avec sous-métriques."""
        return TermsAggregation(
            name="category_breakdown",
            field="category_name.keyword",
            size=size,
            order={"total_amount": "desc"}
        )
    
    @staticmethod
    def merchant_analysis(size: int = 10) -> TermsAggregation:
        """Agrégation par marchand."""
        return TermsAggregation(
            name="merchant_analysis",
            field="merchant_name.keyword",
            size=size,
            order={"_count": "desc"}
        )
    
    @staticmethod
    def monthly_evolution() -> DateHistogramAggregation:
        """Évolution mensuelle des dépenses."""
        return DateHistogramAggregation(
            name="monthly_evolution",
            field="date",
            calendar_interval="month",
            format="yyyy-MM",
            min_doc_count=0
        )
    
    @staticmethod
    def weekly_pattern() -> DateHistogramAggregation:
        """Pattern hebdomadaire des dépenses."""
        return DateHistogramAggregation(
            name="weekly_pattern",
            field="date",
            calendar_interval="day",
            format="E",
            min_doc_count=0
        )
    
    @staticmethod
    def amount_statistics() -> StatsAggregation:
        """Statistiques sur les montants."""
        return StatsAggregation(
            name="amount_statistics",
            field="amount_abs"
        )
    
    @staticmethod
    def total_spending() -> SumAggregation:
        """Total des dépenses."""
        return SumAggregation(
            name="total_spending",
            field="amount_abs"
        )

class FilterBuilder:
    """Builder spécialisé pour les filtres financiers."""
    
    @staticmethod
    def user_filter(user_id: int) -> TermFilter:
        """Filtre utilisateur obligatoire."""
        return TermFilter(field="user_id", value=user_id)
    
    @staticmethod
    def category_filter(category: str) -> TermFilter:
        """Filtre par catégorie."""
        return TermFilter(field="category_name.keyword", value=category)
    
    @staticmethod
    def merchant_filter(merchant: str) -> TermFilter:
        """Filtre par marchand."""
        return TermFilter(field="merchant_name.keyword", value=merchant)
    
    @staticmethod
    def amount_range_filter(min_amount: Optional[float] = None, max_amount: Optional[float] = None) -> RangeFilter:
        """Filtre par plage de montant."""
        return RangeFilter(
            field="amount_abs",
            gte=min_amount,
            lte=max_amount
        )
    
    @staticmethod
    def date_range_filter(start_date: str, end_date: str) -> RangeFilter:
        """Filtre par plage de dates."""
        return RangeFilter(
            field="date",
            gte=start_date,
            lte=end_date,
            format="yyyy-MM-dd"
        )
    
    @staticmethod
    def transaction_type_filter(transaction_type: str) -> TermFilter:
        """Filtre par type de transaction."""
        return TermFilter(field="transaction_type", value=transaction_type)

# ==================== TEMPLATES ====================

class QueryTemplate(BaseModel):
    """Template de requête réutilisable."""
    name: str = Field(..., description="Nom du template")
    description: str = Field(..., description="Description du template")
    intent_types: List[str] = Field(..., description="Types d'intention supportés")
    template: Dict[str, Any] = Field(..., description="Template Elasticsearch")
    parameters: List[str] = Field(default=[], description="Paramètres dynamiques")
    
    def render(self, **params) -> Dict[str, Any]:
        """Rend le template avec les paramètres."""
        rendered = copy.deepcopy(self.template)
        
        # Remplacement simple des paramètres
        def replace_params(obj, params):
            if isinstance(obj, dict):
                return {k: replace_params(v, params) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_params(item, params) for item in obj]
            elif isinstance(obj, str) and obj.startswith("{{") and obj.endswith("}}"):
                param_name = obj[2:-2].strip()
                return params.get(param_name, obj)
            else:
                return obj
        
        return replace_params(rendered, params)

class FieldMapping(BaseModel):
    """Mapping d'un champ Elasticsearch."""
    type: str = Field(..., description="Type de champ")
    index: Optional[bool] = Field(None, description="Indexé ou non")
    store: Optional[bool] = Field(None, description="Stocké ou non")
    doc_values: Optional[bool] = Field(None, description="Doc values")
    analyzer: Optional[str] = Field(None, description="Analyseur")
    search_analyzer: Optional[str] = Field(None, description="Analyseur de recherche")
    fields: Optional[Dict[str, "FieldMapping"]] = Field(None, description="Sous-champs")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        mapping = {"type": self.type}
        
        if self.index is not None:
            mapping["index"] = self.index
        if self.store is not None:
            mapping["store"] = self.store
        if self.doc_values is not None:
            mapping["doc_values"] = self.doc_values
        if self.analyzer:
            mapping["analyzer"] = self.analyzer
        if self.search_analyzer:
            mapping["search_analyzer"] = self.search_analyzer
        if self.fields:
            mapping["fields"] = {k: v.to_dict() for k, v in self.fields.items()}
        
        return mapping

class IndexMapping(BaseModel):
    """Mapping complet d'un index Elasticsearch."""
    properties: Dict[str, FieldMapping] = Field(..., description="Propriétés des champs")
    dynamic: Optional[Union[bool, str]] = Field(None, description="Mapping dynamique")
    date_detection: Optional[bool] = Field(None, description="Détection automatique de dates")
    numeric_detection: Optional[bool] = Field(None, description="Détection automatique de nombres")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire Elasticsearch."""
        mapping = {
            "properties": {k: v.to_dict() for k, v in self.properties.items()}
        }
        
        if self.dynamic is not None:
            mapping["dynamic"] = self.dynamic
        if self.date_detection is not None:
            mapping["date_detection"] = self.date_detection
        if self.numeric_detection is not None:
            mapping["numeric_detection"] = self.numeric_detection
        
        return mapping

# ==================== TEMPLATES FINANCIERS PRÉDÉFINIS ====================

class FinancialQueryTemplates:
    """Templates de requêtes financières prédéfinies."""
    
    @staticmethod
    def search_by_category() -> QueryTemplate:
        """Template de recherche par catégorie."""
        return QueryTemplate(
            name="search_by_category",
            description="Recherche par catégorie financière",
            intent_types=["SEARCH_BY_CATEGORY"],
            template={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"user_id": "{{user_id}}"}},
                            {"term": {"category_name.keyword": "{{category}}"}}
                        ]
                    }
                },
                "sort": [{"date": {"order": "desc"}}],
                "size": "{{limit}}"
            },
            parameters=["user_id", "category", "limit"]
        )
    
    @staticmethod
    def search_by_merchant() -> QueryTemplate:
        """Template de recherche par marchand."""
        return QueryTemplate(
            name="search_by_merchant",
            description="Recherche par marchand",
            intent_types=["SEARCH_BY_MERCHANT"],
            template={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"user_id": "{{user_id}}"}},
                            {"term": {"merchant_name.keyword": "{{merchant}}"}}
                        ]
                    }
                },
                "sort": [{"date": {"order": "desc"}}],
                "size": "{{limit}}"
            },
            parameters=["user_id", "merchant", "limit"]
        )
    
    @staticmethod
    def text_search_with_category() -> QueryTemplate:
        """Template de recherche textuelle avec catégorie."""
        return QueryTemplate(
            name="text_search_with_category",
            description="Recherche textuelle filtrée par catégorie",
            intent_types=["TEXT_SEARCH_WITH_CATEGORY"],
            template={
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": "{{query_text}}",
                                    "fields": ["searchable_text^2", "primary_description^1.5", "merchant_name^1.8"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"user_id": "{{user_id}}"}},
                            {"term": {"category_name.keyword": "{{category}}"}}
                        ]
                    }
                },
                "highlight": {
                    "fields": {
                        "searchable_text": {"fragment_size": 150},
                        "primary_description": {"fragment_size": 150}
                    }
                },
                "sort": [{"_score": {"order": "desc"}}, {"date": {"order": "desc"}}],
                "size": "{{limit}}"
            },
            parameters=["user_id", "query_text", "category", "limit"]
        )
    
    @staticmethod
    def temporal_aggregation() -> QueryTemplate:
        """Template d'agrégation temporelle."""
        return QueryTemplate(
            name="temporal_aggregation",
            description="Analyse temporelle des dépenses",
            intent_types=["TEMPORAL_ANALYSIS", "SPENDING_EVOLUTION"],
            template={
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
                    "monthly_evolution": {
                        "date_histogram": {
                            "field": "date",
                            "calendar_interval": "month",
                            "format": "yyyy-MM",
                            "min_doc_count": 0
                        },
                        "aggs": {
                            "total_amount": {"sum": {"field": "amount_abs"}},
                            "transaction_count": {"value_count": {"field": "transaction_id"}},
                            "average_amount": {"avg": {"field": "amount_abs"}}
                        }
                    },
                    "category_breakdown": {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": 10,
                            "order": {"total_amount": "desc"}
                        },
                        "aggs": {
                            "total_amount": {"sum": {"field": "amount_abs"}}
                        }
                    }
                }
            },
            parameters=["user_id", "start_date", "end_date"]
        )
    
    @staticmethod
    def category_breakdown() -> QueryTemplate:
        """Template d'analyse par catégorie."""
        return QueryTemplate(
            name="category_breakdown",
            description="Répartition des dépenses par catégorie",
            intent_types=["CATEGORY_BREAKDOWN"],
            template={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"user_id": "{{user_id}}"}}
                        ]
                    }
                },
                "size": 0,
                "aggs": {
                    "category_breakdown": {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": "{{size}}",
                            "order": {"total_amount": "desc"}
                        },
                        "aggs": {
                            "total_amount": {"sum": {"field": "amount_abs"}},
                            "transaction_count": {"value_count": {"field": "transaction_id"}},
                            "average_amount": {"avg": {"field": "amount_abs"}},
                            "monthly_trend": {
                                "date_histogram": {
                                    "field": "date",
                                    "calendar_interval": "month",
                                    "format": "yyyy-MM"
                                },
                                "aggs": {
                                    "monthly_total": {"sum": {"field": "amount_abs"}}
                                }
                            }
                        }
                    },
                    "overall_stats": {
                        "stats": {"field": "amount_abs"}
                    }
                }
            },
            parameters=["user_id", "size"]
        )

# ==================== MAPPING FINANCIER ====================

class FinancialIndexMapping:
    """Mapping optimisé pour l'index des transactions financières."""
    
    @staticmethod
    def get_transactions_mapping() -> IndexMapping:
        """Retourne le mapping pour l'index des transactions."""
        return IndexMapping(
            dynamic="strict",  # Mapping strict pour éviter les erreurs
            date_detection=False,  # Pas de détection automatique
            numeric_detection=False,  # Pas de détection automatique
            properties={
                # Identifiants
                "user_id": FieldMapping(type="long", doc_values=True),
                "account_id": FieldMapping(type="long", doc_values=True),
                "transaction_id": FieldMapping(
                    type="keyword",
                    doc_values=True,
                    store=True
                ),
                
                # Montants
                "amount": FieldMapping(type="double", doc_values=True),
                "amount_abs": FieldMapping(type="double", doc_values=True),
                "currency_code": FieldMapping(type="keyword", doc_values=True),
                
                # Types et opérations
                "transaction_type": FieldMapping(type="keyword", doc_values=True),
                "operation_type": FieldMapping(
                    type="text",
                    analyzer="standard",
                    fields={
                        "keyword": FieldMapping(type="keyword", doc_values=True)
                    }
                ),
                
                # Dates
                "date": FieldMapping(type="date", format="yyyy-MM-dd", doc_values=True),
                "month_year": FieldMapping(type="keyword", doc_values=True),
                "weekday": FieldMapping(type="keyword", doc_values=True),
                
                # Descriptions et texte
                "primary_description": FieldMapping(
                    type="text",
                    analyzer="french",
                    search_analyzer="french",
                    fields={
                        "keyword": FieldMapping(type="keyword", doc_values=True)
                    }
                ),
                "searchable_text": FieldMapping(
                    type="text",
                    analyzer="french",
                    search_analyzer="french"
                ),
                
                # Marchand et catégorie
                "merchant_name": FieldMapping(
                    type="text",
                    analyzer="standard",
                    fields={
                        "keyword": FieldMapping(type="keyword", doc_values=True),
                        "suggest": FieldMapping(type="completion")
                    }
                ),
                "category_name": FieldMapping(
                    type="text",
                    analyzer="standard",
                    fields={
                        "keyword": FieldMapping(type="keyword", doc_values=True)
                    }
                ),
                
                # Métadonnées
                "created_at": FieldMapping(type="date", doc_values=True),
                "updated_at": FieldMapping(type="date", doc_values=True)
            }
        )

# ==================== CONSTANTES ET HELPERS ====================

# Champs optimisés pour les recherches textuelles
TEXT_SEARCH_FIELDS = [
    "searchable_text^2.0",
    "primary_description^1.5",
    "merchant_name^1.8",
    "category_name^1.2"
]

# Champs pour les suggestions d'autocomplétion
AUTOCOMPLETE_FIELDS = [
    "merchant_name.suggest",
    "category_name.keyword"
]

# Mapping des opérateurs vers les requêtes Elasticsearch
OPERATOR_MAPPING = {
    "eq": lambda field, value: TermQuery(field=field, value=value),
    "ne": lambda field, value: BoolQuery(must_not=[TermQuery(field=field, value=value)]),
    "in": lambda field, values: TermsQuery(field=field, values=values),
    "not_in": lambda field, values: BoolQuery(must_not=[TermsQuery(field=field, values=values)]),
    "gt": lambda field, value: RangeQuery(field=field, gt=value),
    "gte": lambda field, value: RangeQuery(field=field, gte=value),
    "lt": lambda field, value: RangeQuery(field=field, lt=value),
    "lte": lambda field, value: RangeQuery(field=field, lte=value),
    "between": lambda field, values: RangeQuery(field=field, gte=values[0], lte=values[1])
}

# Templates par intention
INTENT_TEMPLATE_MAPPING = {
    "SEARCH_BY_CATEGORY": FinancialQueryTemplates.search_by_category(),
    "SEARCH_BY_MERCHANT": FinancialQueryTemplates.search_by_merchant(),
    "TEXT_SEARCH_WITH_CATEGORY": FinancialQueryTemplates.text_search_with_category(),
    "TEMPORAL_ANALYSIS": FinancialQueryTemplates.temporal_aggregation(),
    "CATEGORY_BREAKDOWN": FinancialQueryTemplates.category_breakdown()
}

# Configuration des agrégations par défaut
DEFAULT_AGGREGATIONS = {
    "category_breakdown": AggregationBuilder.category_breakdown(),
    "merchant_analysis": AggregationBuilder.merchant_analysis(),
    "monthly_evolution": AggregationBuilder.monthly_evolution(),
    "amount_statistics": AggregationBuilder.amount_statistics(),
    "total_spending": AggregationBuilder.total_spending()
}

# ==================== UTILITAIRES ====================

def build_financial_query(
    user_id: int,
    filters: List[Dict[str, Any]] = None,
    text_query: str = None,
    aggregations: List[str] = None,
    sort_by: str = "date",
    sort_order: SortOrder = SortOrder.DESC,
    size: int = 20,
    from_offset: int = 0
) -> Dict[str, Any]:
    """
    Construit une requête Elasticsearch optimisée pour les données financières.
    
    Args:
        user_id: ID de l'utilisateur (obligatoire)
        filters: Liste de filtres à appliquer
        text_query: Requête textuelle optionnelle
        aggregations: Liste d'agrégations à inclure
        sort_by: Champ de tri
        sort_order: Ordre de tri
        size: Nombre de résultats
        from_offset: Offset pour pagination
        
    Returns:
        Dictionnaire de requête Elasticsearch
    """
    builder = QueryBuilder()
    
    # Filtre utilisateur obligatoire
    builder.filter(FilterBuilder.user_filter(user_id))
    
    # Ajout des filtres personnalisés
    if filters:
        for filter_config in filters:
            field = filter_config.get("field")
            operator = filter_config.get("operator", "eq")
            value = filter_config.get("value")
            
            if field and value is not None and operator in OPERATOR_MAPPING:
                query_obj = OPERATOR_MAPPING[operator](field, value)
                builder.filter(query_obj)
    
    # Requête textuelle si spécifiée
    if text_query:
        multi_match = MultiMatchQuery(
            query=text_query,
            fields=TEXT_SEARCH_FIELDS,
            type="best_fields",
            fuzziness="AUTO",
            tie_breaker=0.3
        )
        builder.query(multi_match)
    
    # Agrégations si demandées
    if aggregations:
        for agg_name in aggregations:
            if agg_name in DEFAULT_AGGREGATIONS:
                builder.aggregation(DEFAULT_AGGREGATIONS[agg_name])
    
    # Configuration de la réponse
    builder.sort(sort_by, sort_order)
    builder.size(size)
    builder.from_offset(from_offset)
    
    # Highlighting pour les recherches textuelles
    if text_query:
        builder.highlight(["searchable_text", "primary_description"])
    
    return builder.build()

def validate_elasticsearch_query(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide une requête Elasticsearch et retourne des informations de validation.
    
    Args:
        query: Requête Elasticsearch à valider
        
    Returns:
        Dictionnaire avec les résultats de validation
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    # Vérifications de base
    if not isinstance(query, dict):
        validation_result["is_valid"] = False
        validation_result["errors"].append("La requête doit être un dictionnaire")
        return validation_result
    
    # Vérification de la sécurité: filtre user_id
    has_user_filter = False
    if "query" in query:
        query_content = query["query"]
        if isinstance(query_content, dict) and "bool" in query_content:
            bool_query = query_content["bool"]
            if "filter" in bool_query:
                for filter_item in bool_query["filter"]:
                    if isinstance(filter_item, dict) and "term" in filter_item:
                        if "user_id" in filter_item["term"]:
                            has_user_filter = True
                            break
    
    if not has_user_filter:
        validation_result["errors"].append("Filtre user_id obligatoire pour la sécurité")
        validation_result["is_valid"] = False
    
    # Vérification de la taille
    size = query.get("size", 10)
    if size > settings.MAX_SEARCH_RESULTS:
        validation_result["warnings"].append(f"Taille élevée: {size} > {settings.MAX_SEARCH_RESULTS}")
    
    # Vérification des agrégations
    if "aggs" in query:
        agg_count = len(query["aggs"])
        if agg_count > 10:
            validation_result["warnings"].append(f"Nombreuses agrégations: {agg_count}")
    
    # Suggestions d'optimisation
    if "query" in query and "sort" not in query:
        validation_result["suggestions"].append("Ajouter un tri pour des résultats cohérents")
    
    if "highlight" not in query and "query" in query:
        validation_result["suggestions"].append("Considérer l'ajout de highlighting")
    
    return validation_result

def optimize_elasticsearch_query(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimise une requête Elasticsearch pour de meilleures performances.
    
    Args:
        query: Requête Elasticsearch à optimiser
        
    Returns:
        Requête optimisée
    """
    optimized = copy.deepcopy(query)
    
    # Optimisation 1: Déplacer les filtres exacts vers la clause filter
    if "query" in optimized:
        query_content = optimized["query"]
        if isinstance(query_content, dict) and "bool" in query_content:
            bool_query = query_content["bool"]
            
            # Déplacer les term queries vers filter
            if "must" in bool_query:
                must_clauses = bool_query["must"]
                filter_clauses = bool_query.get("filter", [])
                new_must = []
                
                for clause in must_clauses:
                    if isinstance(clause, dict) and "term" in clause:
                        filter_clauses.append(clause)
                    else:
                        new_must.append(clause)
                
                bool_query["must"] = new_must
                bool_query["filter"] = filter_clauses
    
    # Optimisation 2: Ajouter _source si pas spécifié
    if "_source" not in optimized:
        optimized["_source"] = [
            "user_id", "transaction_id", "amount", "amount_abs",
            "date", "category_name", "merchant_name", "primary_description"
        ]
    
    # Optimisation 3: Limiter la taille par défaut
    if "size" not in optimized:
        optimized["size"] = min(20, settings.DEFAULT_LIMIT)
    
    return optimized

# ==================== EXPORTS ====================

__all__ = [
    # Builders
    "QueryBuilder",
    "AggregationBuilder",
    "FilterBuilder",
    
    # Templates et mapping
    "QueryTemplate",
    "FieldMapping",
    "IndexMapping",
    "FinancialQueryTemplates",
    "FinancialIndexMapping",
    
    # Utilitaires
    "build_financial_query",
    "validate_elasticsearch_query",
    "optimize_elasticsearch_query",
    
    # Constantes
    "TEXT_SEARCH_FIELDS",
    "AUTOCOMPLETE_FIELDS",
    "OPERATOR_MAPPING",
    "INTENT_TEMPLATE_MAPPING",
    "DEFAULT_AGGREGATIONS"
]