"""
Templates de requêtes Elasticsearch pour le Search Service.

Bibliothèque de templates optimisés par intention et type de requête
pour améliorer les performances et la cohérence.
"""

import logging
from typing import Any, Dict, List, Optional
from enum import Enum

from ..models.service_contracts import SearchServiceQuery, IntentType, QueryType
from ..config.settings import SearchServiceSettings, get_settings


logger = logging.getLogger(__name__)


class TemplateType(str, Enum):
    """Types de templates disponibles."""
    BASIC_FILTER = "basic_filter"
    TEXT_SEARCH = "text_search"
    AGGREGATION = "aggregation"
    COMPLEX_QUERY = "complex_query"
    OPTIMIZED_INTENT = "optimized_intent"


class QueryTemplateManager:
    """
    Gestionnaire de templates de requêtes Elasticsearch.
    
    Fournit des templates optimisés pour chaque type d'intention
    et gère la construction dynamique des requêtes.
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        self.settings = settings or get_settings()
        
        # Cache des templates
        self.template_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialisation des templates
        self._initialize_templates()
        
        logger.info("Query template manager initialized with optimized templates")
    
    def _initialize_templates(self) -> None:
        """Initialise la bibliothèque de templates."""
        
        # Templates par intention
        self.intent_templates = {
            IntentType.SEARCH_BY_CATEGORY: self._create_category_search_template(),
            IntentType.SEARCH_BY_MERCHANT: self._create_merchant_search_template(),
            IntentType.SEARCH_BY_AMOUNT: self._create_amount_search_template(),
            IntentType.SEARCH_BY_DATE: self._create_date_search_template(),
            IntentType.TEXT_SEARCH: self._create_text_search_template(),
            IntentType.COUNT_OPERATIONS: self._create_count_operations_template(),
            IntentType.TEMPORAL_ANALYSIS: self._create_temporal_analysis_template(),
            IntentType.COUNT_OPERATIONS_BY_AMOUNT: self._create_count_by_amount_template(),
            IntentType.TEXT_SEARCH_WITH_CATEGORY: self._create_text_with_category_template(),
            IntentType.TEMPORAL_SPENDING_ANALYSIS: self._create_temporal_spending_template()
        }
        
        # Templates par type de requête
        self.query_type_templates = {
            QueryType.SIMPLE_SEARCH: self._create_simple_search_template(),
            QueryType.FILTERED_SEARCH: self._create_filtered_search_template(),
            QueryType.TEXT_SEARCH: self._create_pure_text_search_template(),
            QueryType.AGGREGATION: self._create_aggregation_only_template(),
            QueryType.FILTERED_AGGREGATION: self._create_filtered_aggregation_template(),
            QueryType.TEXT_SEARCH_WITH_FILTER: self._create_text_with_filter_template(),
            QueryType.TEMPORAL_AGGREGATION: self._create_temporal_aggregation_template()
        }
        
        logger.info(f"Templates initialized: {len(self.intent_templates)} intent templates, {len(self.query_type_templates)} query type templates")
    
    async def get_template_for_intent(self, intent: IntentType) -> Optional[Dict[str, Any]]:
        """
        Récupère le template optimisé pour une intention.
        
        Args:
            intent: Type d'intention
            
        Returns:
            Optional[Dict]: Template Elasticsearch ou None
        """
        template = self.intent_templates.get(intent)
        if template:
            logger.debug(f"Template trouvé pour intention {intent}")
            return template.copy()
        
        logger.debug(f"Aucun template pour intention {intent}")
        return None
    
    async def get_template_for_query_type(self, query_type: QueryType) -> Optional[Dict[str, Any]]:
        """
        Récupère le template pour un type de requête.
        
        Args:
            query_type: Type de requête
            
        Returns:
            Optional[Dict]: Template Elasticsearch ou None
        """
        template = self.query_type_templates.get(query_type)
        if template:
            logger.debug(f"Template trouvé pour type {query_type}")
            return template.copy()
        
        logger.debug(f"Aucun template pour type {query_type}")
        return None
    
    async def build_query_from_template(
        self,
        template: Dict[str, Any],
        query: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch à partir d'un template.
        
        Args:
            template: Template de base
            query: Requête selon le contrat
            
        Returns:
            Dict: Requête Elasticsearch construite
        """
        # Copie du template
        es_query = template.copy()
        
        # Injection des paramètres dynamiques
        es_query = self._inject_user_filter(es_query, query.query_metadata.user_id)
        es_query = self._inject_filters(es_query, query.filters)
        es_query = self._inject_text_search(es_query, query.filters.text_search)
        es_query = self._inject_aggregations(es_query, query.aggregations)
        es_query = self._inject_pagination(es_query, query.search_parameters)
        es_query = self._inject_options(es_query, query.options)
        
        return es_query
    
    # === TEMPLATES PAR INTENTION ===
    
    def _create_category_search_template(self) -> Dict[str, Any]:
        """Template optimisé pour recherche par catégorie."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Category filter sera injecté
                    ]
                }
            },
            "sort": [
                {"date": {"order": "desc"}},
                {"amount_abs": {"order": "desc"}}
            ],
            "_source": [
                "transaction_id", "user_id", "account_id",
                "amount", "amount_abs", "transaction_type", "currency_code",
                "date", "month_year", "primary_description", "merchant_name", "category_name"
            ],
            "track_scores": False  # Pas besoin de score pour filtrage exact
        }
    
    def _create_merchant_search_template(self) -> Dict[str, Any]:
        """Template optimisé pour recherche par marchand."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "should": [
                        # Recherche exacte sur merchant_name.keyword (boost élevé)
                        {"term": {"merchant_name.keyword": {"boost": 3.0}}},
                        # Recherche textuelle sur merchant_name (boost moyen)
                        {"match": {"merchant_name": {"boost": 2.0}}},
                        # Recherche dans la description (boost faible)
                        {"match": {"primary_description": {"boost": 1.0}}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"date": {"order": "desc"}}
            ],
            "_source": [
                "transaction_id", "user_id", "amount", "amount_abs",
                "date", "primary_description", "merchant_name", "category_name"
            ]
        }
    
    def _create_amount_search_template(self) -> Dict[str, Any]:
        """Template optimisé pour recherche par montant."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Amount range filter sera injecté
                    ]
                }
            },
            "sort": [
                {"amount_abs": {"order": "desc"}},
                {"date": {"order": "desc"}}
            ],
            "_source": [
                "transaction_id", "user_id", "amount", "amount_abs", "currency_code",
                "date", "primary_description", "merchant_name", "category_name"
            ],
            "track_scores": False
        }
    
    def _create_date_search_template(self) -> Dict[str, Any]:
        """Template optimisé pour recherche par date."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Date range filter sera injecté
                    ]
                }
            },
            "sort": [
                {"date": {"order": "desc"}},
                {"amount_abs": {"order": "desc"}}
            ],
            "_source": [
                "transaction_id", "user_id", "amount", "amount_abs",
                "date", "month_year", "weekday", "primary_description", 
                "merchant_name", "category_name"
            ],
            "track_scores": False
        }
    
    def _create_text_search_template(self) -> Dict[str, Any]:
        """Template optimisé pour recherche textuelle pure."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                        # Text search sera injecté
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "searchable_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    },
                    "primary_description": {
                        "fragment_size": 100,
                        "number_of_fragments": 2
                    },
                    "merchant_name": {
                        "fragment_size": 50,
                        "number_of_fragments": 1
                    }
                },
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"]
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"date": {"order": "desc"}}
            ],
            "_source": [
                "transaction_id", "user_id", "amount", "amount_abs",
                "date", "primary_description", "merchant_name", "category_name",
                "searchable_text"
            ]
        }
    
    def _create_count_operations_template(self) -> Dict[str, Any]:
        """Template optimisé pour comptage d'opérations."""
        return {
            "size": 0,  # Pas besoin de documents, juste le count
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Additional filters seront injectés
                    ]
                }
            },
            "aggs": {
                "total_count": {
                    "value_count": {
                        "field": "transaction_id"
                    }
                }
            },
            "track_total_hits": True
        }
    
    def _create_temporal_analysis_template(self) -> Dict[str, Any]:
        """Template optimisé pour analyse temporelle."""
        return {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Date filters seront injectés
                    ]
                }
            },
            "aggs": {
                "by_month": {
                    "terms": {
                        "field": "month_year",
                        "size": 24,
                        "order": {"_key": "desc"}
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount"}},
                        "total_amount_abs": {"sum": {"field": "amount_abs"}},
                        "avg_amount": {"avg": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                },
                "by_weekday": {
                    "terms": {
                        "field": "weekday",
                        "size": 7
                    },
                    "aggs": {
                        "avg_amount": {"avg": {"field": "amount_abs"}}
                    }
                }
            }
        }
    
    def _create_count_by_amount_template(self) -> Dict[str, Any]:
        """Template pour comptage par plage de montants."""
        return {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Amount range filter sera injecté
                    ]
                }
            },
            "aggs": {
                "amount_ranges": {
                    "range": {
                        "field": "amount_abs",
                        "ranges": [
                            {"to": 10},
                            {"from": 10, "to": 50},
                            {"from": 50, "to": 100},
                            {"from": 100, "to": 200},
                            {"from": 200}
                        ]
                    }
                },
                "total_count": {
                    "value_count": {"field": "transaction_id"}
                },
                "total_amount": {
                    "sum": {"field": "amount"}
                }
            }
        }
    
    def _create_text_with_category_template(self) -> Dict[str, Any]:
        """Template pour recherche textuelle avec filtre catégorie."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                        # Text search sera injecté
                    ],
                    "filter": [
                        # Category filter sera injecté
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "primary_description": {"fragment_size": 150},
                    "merchant_name": {"fragment_size": 100}
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"date": {"order": "desc"}}
            ],
            "_source": [
                "transaction_id", "user_id", "amount", "amount_abs",
                "date", "primary_description", "merchant_name", "category_name"
            ]
        }
    
    def _create_temporal_spending_template(self) -> Dict[str, Any]:
        """Template pour analyse des dépenses temporelles."""
        return {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        {"range": {"amount": {"lt": 0}}}  # Seulement les dépenses
                    ]
                }
            },
            "aggs": {
                "spending_by_month": {
                    "terms": {
                        "field": "month_year",
                        "size": 12,
                        "order": {"total_spending": "desc"}
                    },
                    "aggs": {
                        "total_spending": {
                            "sum": {
                                "field": "amount_abs"
                            }
                        },
                        "avg_spending": {
                            "avg": {
                                "field": "amount_abs"
                            }
                        },
                        "spending_by_category": {
                            "terms": {
                                "field": "category_name.keyword",
                                "size": 10
                            },
                            "aggs": {
                                "category_spending": {
                                    "sum": {"field": "amount_abs"}
                                }
                            }
                        }
                    }
                },
                "total_spending": {
                    "sum": {"field": "amount_abs"}
                },
                "spending_stats": {
                    "stats": {"field": "amount_abs"}
                }
            }
        }
    
    # === TEMPLATES PAR TYPE DE REQUÊTE ===
    
    def _create_simple_search_template(self) -> Dict[str, Any]:
        """Template pour recherche simple."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ]
                }
            },
            "sort": [
                {"date": {"order": "desc"}},
                {"amount_abs": {"order": "desc"}}
            ],
            "_source": [
                "transaction_id", "user_id", "amount", "amount_abs",
                "date", "primary_description", "merchant_name", "category_name"
            ],
            "track_scores": False
        }
    
    def _create_filtered_search_template(self) -> Dict[str, Any]:
        """Template pour recherche avec filtres."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Filters seront injectés
                    ]
                }
            },
            "sort": [
                {"date": {"order": "desc"}},
                {"amount_abs": {"order": "desc"}}
            ],
            "_source": [
                "transaction_id", "user_id", "amount", "amount_abs",
                "date", "primary_description", "merchant_name", "category_name"
            ],
            "track_scores": False
        }
    
    def _create_pure_text_search_template(self) -> Dict[str, Any]:
        """Template pour recherche textuelle pure."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                        # Multi-match query sera injecté
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "searchable_text": {},
                    "primary_description": {},
                    "merchant_name": {}
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"date": {"order": "desc"}}
            ]
        }
    
    def _create_aggregation_only_template(self) -> Dict[str, Any]:
        """Template pour agrégations uniquement."""
        return {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ]
                }
            },
            "aggs": {
                # Aggregations seront injectées
            }
        }
    
    def _create_filtered_aggregation_template(self) -> Dict[str, Any]:
        """Template pour agrégations avec filtres."""
        return {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Filters seront injectés
                    ]
                }
            },
            "aggs": {
                # Aggregations seront injectées
            }
        }
    
    def _create_text_with_filter_template(self) -> Dict[str, Any]:
        """Template pour recherche textuelle avec filtres."""
        return {
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                        # Text search sera injecté
                    ],
                    "filter": [
                        # Additional filters seront injectés
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "searchable_text": {"fragment_size": 150},
                    "primary_description": {"fragment_size": 100}
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"date": {"order": "desc"}}
            ]
        }
    
    def _create_temporal_aggregation_template(self) -> Dict[str, Any]:
        """Template pour agrégations temporelles."""
        return {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        # User filter sera injecté
                    ],
                    "filter": [
                        # Date filters seront injectés
                    ]
                }
            },
            "aggs": {
                "temporal_analysis": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "month",
                        "format": "yyyy-MM"
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount"}},
                        "total_amount_abs": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                }
            }
        }
    
    # === MÉTHODES D'INJECTION ===
    
    def _inject_user_filter(self, es_query: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """Injecte le filtre user_id obligatoire."""
        user_filter = {"term": {"user_id": user_id}}
        
        if "query" not in es_query:
            es_query["query"] = {"bool": {"must": []}}
        
        if "bool" not in es_query["query"]:
            es_query["query"] = {"bool": {"must": [es_query["query"]]}}
        
        if "must" not in es_query["query"]["bool"]:
            es_query["query"]["bool"]["must"] = []
        
        # Ajouter en premier pour optimisation
        es_query["query"]["bool"]["must"].insert(0, user_filter)
        
        return es_query
    
    def _inject_filters(self, es_query: Dict[str, Any], filters) -> Dict[str, Any]:
        """Injecte les filtres de la requête."""
        if not filters:
            return es_query
        
        # Filtres obligatoires (skip user_id déjà injecté)
        for filter_item in filters.required:
            if filter_item.field != "user_id":
                filter_clause = self._build_filter_clause(filter_item)
                if filter_clause:
                    es_query = self._add_to_bool_query(es_query, "must", filter_clause)
        
        # Filtres optionnels
        for filter_item in filters.optional:
            filter_clause = self._build_filter_clause(filter_item)
            if filter_clause:
                es_query = self._add_to_bool_query(es_query, "should", filter_clause)
        
        # Filtres de plage
        for filter_item in filters.ranges:
            range_clause = self._build_range_clause(filter_item)
            if range_clause:
                es_query = self._add_to_bool_query(es_query, "filter", range_clause)
        
        return es_query
    
    def _inject_text_search(self, es_query: Dict[str, Any], text_search) -> Dict[str, Any]:
        """Injecte la recherche textuelle."""
        if not text_search:
            return es_query
        
        # Construction de la clause multi_match
        multi_match = {
            "multi_match": {
                "query": text_search.query,
                "fields": self._apply_field_boosts(text_search.fields, text_search.boost),
                "type": "best_fields",
                "fuzziness": "AUTO",
                "operator": "and"
            }
        }
        
        es_query = self._add_to_bool_query(es_query, "must", multi_match)
        
        return es_query
    
    def _inject_aggregations(self, es_query: Dict[str, Any], aggregations) -> Dict[str, Any]:
        """Injecte les agrégations."""
        if not aggregations or not aggregations.enabled:
            return es_query
        
        aggs = {}
        
        # Métriques globales
        if "sum" in aggregations.types:
            for metric in aggregations.metrics:
                aggs[f"total_{metric}"] = {"sum": {"field": metric}}
        
        if "avg" in aggregations.types:
            for metric in aggregations.metrics:
                aggs[f"avg_{metric}"] = {"avg": {"field": metric}}
        
        if "count" in aggregations.types:
            aggs["doc_count"] = {"value_count": {"field": "transaction_id"}}
        
        # Agrégations par groupement
        for group_field in aggregations.group_by:
            field_key = f"{group_field}.keyword" if self._is_text_field(group_field) else group_field
            
            agg_def = {
                "terms": {
                    "field": field_key,
                    "size": 50,
                    "order": {"_count": "desc"}
                }
            }
            
            # Sous-agrégations
            if aggregations.metrics:
                agg_def["aggs"] = {}
                for metric in aggregations.metrics:
                    if "sum" in aggregations.types:
                        agg_def["aggs"][f"total_{metric}"] = {"sum": {"field": metric}}
                    if "avg" in aggregations.types:
                        agg_def["aggs"][f"avg_{metric}"] = {"avg": {"field": metric}}
            
            aggs[f"by_{group_field}"] = agg_def
        
        if aggs:
            es_query["aggs"] = aggs
        
        return es_query
    
    def _inject_pagination(self, es_query: Dict[str, Any], parameters) -> Dict[str, Any]:
        """Injecte les paramètres de pagination."""
        es_query["size"] = parameters.limit
        es_query["from"] = parameters.offset
        
        return es_query
    
    def _inject_options(self, es_query: Dict[str, Any], options) -> Dict[str, Any]:
        """Injecte les options de recherche."""
        if options.include_highlights and "highlight" not in es_query:
            es_query["highlight"] = {
                "fields": {
                    "searchable_text": {"fragment_size": 150, "number_of_fragments": 3},
                    "primary_description": {"fragment_size": 100, "number_of_fragments": 2}
                }
            }
        
        if options.cache_enabled:
            es_query["request_cache"] = True
        
        return es_query
    
    # === MÉTHODES UTILITAIRES ===
    
    def _build_filter_clause(self, filter_item) -> Optional[Dict[str, Any]]:
        """Construit une clause de filtre."""
        field = filter_item.field
        operator = filter_item.operator
        value = filter_item.value
        
        if operator.value == "eq":
            return {"term": {f"{field}.keyword" if self._is_text_field(field) else field: value}}
        elif operator.value == "in":
            return {"terms": {f"{field}.keyword" if self._is_text_field(field) else field: value}}
        elif operator.value == "match":
            return {"match": {field: {"query": value, "operator": "and"}}}
        
        return None
    
    def _build_range_clause(self, filter_item) -> Optional[Dict[str, Any]]:
        """Construit une clause de plage."""
        field = filter_item.field
        operator = filter_item.operator
        value = filter_item.value
        
        if operator.value == "between" and isinstance(value, list) and len(value) == 2:
            return {"range": {field: {"gte": value[0], "lte": value[1]}}}
        elif operator.value == "gt":
            return {"range": {field: {"gt": value}}}
        elif operator.value == "gte":
            return {"range": {field: {"gte": value}}}
        elif operator.value == "lt":
            return {"range": {field: {"lt": value}}}
        elif operator.value == "lte":
            return {"range": {field: {"lte": value}}}
        
        return None
    
    def _add_to_bool_query(self, es_query: Dict[str, Any], clause_type: str, clause: Dict[str, Any]) -> Dict[str, Any]:
        """Ajoute une clause à une requête bool."""
        if "query" not in es_query:
            es_query["query"] = {"bool": {}}
        
        if "bool" not in es_query["query"]:
            es_query["query"] = {"bool": {"must": [es_query["query"]]}}
        
        if clause_type not in es_query["query"]["bool"]:
            es_query["query"]["bool"][clause_type] = []
        
        es_query["query"]["bool"][clause_type].append(clause)
        
        return es_query
    
    def _apply_field_boosts(self, fields: List[str], boosts: Optional[Dict[str, float]]) -> List[str]:
        """Applique les boosts aux champs."""
        if not boosts:
            boosts = {
                "searchable_text": 2.0,
                "primary_description": 1.5,
                "merchant_name": 1.8,
                "category_name": 1.2
            }
        
        boosted_fields = []
        for field in fields:
            boost = boosts.get(field, 1.0)
            if boost != 1.0:
                boosted_fields.append(f"{field}^{boost}")
            else:
                boosted_fields.append(field)
        
        return boosted_fields
    
    def _is_text_field(self, field: str) -> bool:
        """Détermine si un champ est de type texte."""
        text_fields = {
            "primary_description", "merchant_name", "category_name",
            "operation_type", "searchable_text"
        }
        return field in text_fields
    
    async def shutdown(self) -> None:
        """Arrête le gestionnaire de templates."""
        self.template_cache.clear()
        logger.info("Query template manager shutdown")


# === HELPER FUNCTIONS ===

def create_query_template_manager(
    settings: Optional[SearchServiceSettings] = None
) -> QueryTemplateManager:
    """
    Factory pour créer un gestionnaire de templates.
    
    Args:
        settings: Configuration
        
    Returns:
        QueryTemplateManager: Gestionnaire configuré
    """
    return QueryTemplateManager(settings=settings or get_settings())