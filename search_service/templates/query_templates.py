"""
Query Templates pour Search Service

Templates de requêtes Elasticsearch optimisés pour chaque type d'intention
financière identifiée par le Conversation Service AutoGen.

Architecture:
- Templates par intention (12 catégories primaires + 104 sous-catégories)
- Paramètres dynamiques selon entités extraites
- Validation templates avant utilisation
- Versioning templates pour évolution
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

class TemplateValidationError(Exception):
    """Exception levée lors de la validation des templates"""
    pass

class QueryType(Enum):
    """Types de requêtes supportées"""
    FILTERED_SEARCH = "filtered_search"
    TEXT_SEARCH = "text_search"
    FILTERED_AGGREGATION = "filtered_aggregation"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    COMBINED_SEARCH = "combined_search"

@dataclass
class QueryTemplateConfig:
    """Configuration d'un template de requête"""
    name: str
    intent_type: str
    query_type: QueryType
    description: str
    required_params: List[str]
    optional_params: List[str]
    performance_tier: str  # "fast", "medium", "complex"
    cache_duration: int  # en secondes

class QueryTemplates:
    """Gestionnaire des templates de requêtes Elasticsearch"""
    
    def __init__(self):
        self.templates = FINANCIAL_QUERY_TEMPLATES
        self.intent_mapping = INTENT_TEMPLATE_MAPPING
        
    def get_template(self, intent_type: str, **params) -> Dict[str, Any]:
        """
        Récupère et paramètre un template selon l'intention
        
        Args:
            intent_type: Type d'intention (ex: "SEARCH_BY_CATEGORY")
            **params: Paramètres pour le template
            
        Returns:
            Template Elasticsearch paramétré
            
        Raises:
            TemplateValidationError: Si template invalide ou paramètres manquants
        """
        if intent_type not in self.intent_mapping:
            raise TemplateValidationError(f"Intent type non supporté: {intent_type}")
            
        template_name = self.intent_mapping[intent_type]
        template_config = self.templates[template_name]
        
        # Validation paramètres requis
        missing_params = []
        for required_param in template_config["required_params"]:
            if required_param not in params:
                missing_params.append(required_param)
                
        if missing_params:
            raise TemplateValidationError(
                f"Paramètres requis manquants: {missing_params}"
            )
            
        # Construction de la requête
        query_builder = QueryTemplateBuilder(template_config)
        return query_builder.build(**params)
    
    def validate_template(self, template: Dict[str, Any]) -> bool:
        """Valide la structure d'un template Elasticsearch"""
        required_keys = ["query", "size"]
        return all(key in template for key in required_keys)
    
    def get_available_intents(self) -> List[str]:
        """Retourne la liste des intentions supportées"""
        return list(self.intent_mapping.keys())

class QueryTemplateBuilder:
    """Constructeur de templates de requêtes"""
    
    def __init__(self, template_config: Dict[str, Any]):
        self.config = template_config
        self.template = template_config["template"]
        
    def build(self, **params) -> Dict[str, Any]:
        """Construit la requête Elasticsearch finale"""
        # Copie profonde du template
        query = json.loads(json.dumps(self.template))
        
        # Application des paramètres
        self._apply_user_filter(query, params.get("user_id"))
        self._apply_text_search(query, params)
        self._apply_filters(query, params)
        self._apply_ranges(query, params)
        self._apply_sorting(query, params)
        self._apply_limits(query, params)
        
        return query
    
    def _apply_user_filter(self, query: Dict[str, Any], user_id: int):
        """Applique le filtre utilisateur obligatoire"""
        if not user_id:
            raise TemplateValidationError("user_id est obligatoire")
            
        # Assure que le filtre user_id est présent
        bool_query = query.setdefault("query", {}).setdefault("bool", {})
        must_filters = bool_query.setdefault("must", [])
        
        # Ajoute le filtre user_id s'il n'existe pas
        user_filter_exists = any(
            f.get("term", {}).get("user_id") == user_id for f in must_filters
        )
        
        if not user_filter_exists:
            must_filters.append({"term": {"user_id": user_id}})
    
    def _apply_text_search(self, query: Dict[str, Any], params: Dict[str, Any]):
        """Applique la recherche textuelle"""
        search_query = params.get("search_query")
        search_fields = params.get("search_fields", [
            "searchable_text^2.0",
            "primary_description^1.5", 
            "merchant_name^1.8"
        ])
        
        if search_query:
            bool_query = query.setdefault("query", {}).setdefault("bool", {})
            must_queries = bool_query.setdefault("must", [])
            
            # Multi-match query avec boost
            multi_match = {
                "multi_match": {
                    "query": search_query,
                    "fields": search_fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
            must_queries.append(multi_match)
    
    def _apply_filters(self, query: Dict[str, Any], params: Dict[str, Any]):
        """Applique les filtres catégoriels"""
        filters = params.get("filters", {})
        
        bool_query = query.setdefault("query", {}).setdefault("bool", {})
        filter_queries = bool_query.setdefault("filter", [])
        
        # Filtres exacts
        for field, value in filters.items():
            if field in ["category_name", "merchant_name", "transaction_type", 
                        "currency_code", "operation_type", "month_year", "weekday"]:
                # Utilise .keyword pour les champs textuels
                filter_field = f"{field}.keyword" if field in [
                    "category_name", "merchant_name", "operation_type"
                ] else field
                
                filter_queries.append({
                    "term": {filter_field: value}
                })
    
    def _apply_ranges(self, query: Dict[str, Any], params: Dict[str, Any]):
        """Applique les filtres de plage"""
        ranges = params.get("ranges", {})
        
        bool_query = query.setdefault("query", {}).setdefault("bool", {})
        filter_queries = bool_query.setdefault("filter", [])
        
        for field, range_spec in ranges.items():
            if field in ["amount", "amount_abs", "date"]:
                range_query = {"range": {field: {}}}
                
                if "gte" in range_spec:
                    range_query["range"][field]["gte"] = range_spec["gte"]
                if "lte" in range_spec:
                    range_query["range"][field]["lte"] = range_spec["lte"]
                if "gt" in range_spec:
                    range_query["range"][field]["gt"] = range_spec["gt"]
                if "lt" in range_spec:
                    range_query["range"][field]["lt"] = range_spec["lt"]
                    
                filter_queries.append(range_query)
    
    def _apply_sorting(self, query: Dict[str, Any], params: Dict[str, Any]):
        """Applique le tri"""
        sort_spec = params.get("sort", [{"date": {"order": "desc"}}])
        query["sort"] = sort_spec
    
    def _apply_limits(self, query: Dict[str, Any], params: Dict[str, Any]):
        """Applique les limites de pagination"""
        query["size"] = params.get("limit", 20)
        query["from"] = params.get("offset", 0)
        
        # Timeout de sécurité
        query["timeout"] = params.get("timeout", "30s")

# Templates de requêtes par intention financière
FINANCIAL_QUERY_TEMPLATES = {
    "category_search": {
        "name": "category_search",
        "intent_type": "SEARCH_BY_CATEGORY",
        "query_type": QueryType.FILTERED_SEARCH,
        "description": "Recherche par catégorie financière",
        "required_params": ["user_id", "category_name"],
        "optional_params": ["limit", "offset", "date_range"],
        "performance_tier": "fast",
        "cache_duration": 300,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 20,
            "sort": [{"date": {"order": "desc"}}],
            "track_total_hits": True
        }
    },
    
    "merchant_search": {
        "name": "merchant_search", 
        "intent_type": "SEARCH_BY_MERCHANT",
        "query_type": QueryType.FILTERED_SEARCH,
        "description": "Recherche par nom de marchand",
        "required_params": ["user_id", "merchant_name"],
        "optional_params": ["limit", "offset", "fuzzy_matching"],
        "performance_tier": "fast",
        "cache_duration": 300,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 20,
            "sort": [{"date": {"order": "desc"}}],
            "track_total_hits": True
        }
    },
    
    "amount_range_search": {
        "name": "amount_range_search",
        "intent_type": "SEARCH_BY_AMOUNT",
        "query_type": QueryType.FILTERED_SEARCH,
        "description": "Recherche par montant ou plage de montants",
        "required_params": ["user_id", "amount_range"],
        "optional_params": ["limit", "offset", "currency_code"],
        "performance_tier": "fast",
        "cache_duration": 180,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 20,
            "sort": [{"amount_abs": {"order": "desc"}}],
            "track_total_hits": True
        }
    },
    
    "date_range_search": {
        "name": "date_range_search",
        "intent_type": "SEARCH_BY_DATE",
        "query_type": QueryType.FILTERED_SEARCH,
        "description": "Recherche par période temporelle",
        "required_params": ["user_id", "date_range"],
        "optional_params": ["limit", "offset", "grouping"],
        "performance_tier": "fast",
        "cache_duration": 600,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 20,
            "sort": [{"date": {"order": "desc"}}],
            "track_total_hits": True
        }
    },
    
    "text_search": {
        "name": "text_search",
        "intent_type": "TEXT_SEARCH",
        "query_type": QueryType.TEXT_SEARCH,
        "description": "Recherche textuelle libre",
        "required_params": ["user_id", "search_query"],
        "optional_params": ["limit", "offset", "search_fields", "fuzziness"],
        "performance_tier": "medium",
        "cache_duration": 120,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 20,
            "sort": ["_score", {"date": {"order": "desc"}}],
            "track_total_hits": True,
            "highlight": {
                "fields": {
                    "searchable_text": {},
                    "primary_description": {},
                    "merchant_name": {}
                }
            }
        }
    },
    
    "category_text_search": {
        "name": "category_text_search",
        "intent_type": "TEXT_SEARCH_WITH_CATEGORY",
        "query_type": QueryType.COMBINED_SEARCH,
        "description": "Recherche textuelle avec filtre catégorie",
        "required_params": ["user_id", "search_query", "category_name"],
        "optional_params": ["limit", "offset", "search_fields"],
        "performance_tier": "medium",
        "cache_duration": 120,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 20,
            "sort": ["_score", {"date": {"order": "desc"}}],
            "track_total_hits": True,
            "highlight": {
                "fields": {
                    "searchable_text": {},
                    "primary_description": {},
                    "merchant_name": {}
                }
            }
        }
    },
    
    "count_operations": {
        "name": "count_operations",
        "intent_type": "COUNT_OPERATIONS",
        "query_type": QueryType.FILTERED_AGGREGATION,
        "description": "Comptage d'opérations avec filtres",
        "required_params": ["user_id"],
        "optional_params": ["filters", "ranges", "group_by"],
        "performance_tier": "fast",
        "cache_duration": 300,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 0,
            "track_total_hits": True,
            "aggs": {
                "transaction_count": {
                    "value_count": {
                        "field": "transaction_id"
                    }
                }
            }
        }
    },
    
    "spending_analysis": {
        "name": "spending_analysis",
        "intent_type": "SPENDING_ANALYSIS",
        "query_type": QueryType.FILTERED_AGGREGATION,
        "description": "Analyse des dépenses avec agrégations",
        "required_params": ["user_id"],
        "optional_params": ["filters", "ranges", "group_by"],
        "performance_tier": "medium",
        "cache_duration": 300,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 0,
            "track_total_hits": True,
            "aggs": {
                "total_spending": {
                    "sum": {
                        "field": "amount_abs"
                    }
                },
                "average_spending": {
                    "avg": {
                        "field": "amount_abs"
                    }
                },
                "spending_stats": {
                    "stats": {
                        "field": "amount_abs"
                    }
                }
            }
        }
    },
    
    "temporal_analysis": {
        "name": "temporal_analysis",
        "intent_type": "TEMPORAL_ANALYSIS",
        "query_type": QueryType.TEMPORAL_ANALYSIS,
        "description": "Analyse temporelle des transactions",
        "required_params": ["user_id", "time_grouping"],
        "optional_params": ["filters", "ranges", "metrics"],
        "performance_tier": "medium",
        "cache_duration": 600,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 0,
            "track_total_hits": True,
            "aggs": {
                "temporal_buckets": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "month"
                    },
                    "aggs": {
                        "total_amount": {
                            "sum": {
                                "field": "amount_abs"
                            }
                        },
                        "transaction_count": {
                            "value_count": {
                                "field": "transaction_id"
                            }
                        }
                    }
                }
            }
        }
    },
    
    "merchant_analysis": {
        "name": "merchant_analysis",
        "intent_type": "MERCHANT_ANALYSIS",
        "query_type": QueryType.FILTERED_AGGREGATION,
        "description": "Analyse par marchand",
        "required_params": ["user_id"],
        "optional_params": ["filters", "ranges", "top_n"],
        "performance_tier": "medium",
        "cache_duration": 300,
        "template": {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 0,
            "track_total_hits": True,
            "aggs": {
                "top_merchants": {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": 10,
                        "order": {
                            "total_spent": "desc"
                        }
                    },
                    "aggs": {
                        "total_spent": {
                            "sum": {
                                "field": "amount_abs"
                            }
                        },
                        "transaction_count": {
                            "value_count": {
                                "field": "transaction_id"
                            }
                        }
                    }
                }
            }
        }
    }
}

# Mapping des intentions vers les templates
INTENT_TEMPLATE_MAPPING = {
    "SEARCH_BY_CATEGORY": "category_search",
    "SEARCH_BY_MERCHANT": "merchant_search",
    "SEARCH_BY_AMOUNT": "amount_range_search",
    "SEARCH_BY_DATE": "date_range_search",
    "TEXT_SEARCH": "text_search",
    "TEXT_SEARCH_WITH_CATEGORY": "category_text_search",
    "COUNT_OPERATIONS": "count_operations",
    "COUNT_OPERATIONS_BY_AMOUNT": "count_operations",
    "COUNT_OPERATIONS_BY_CATEGORY": "count_operations",
    "SPENDING_ANALYSIS": "spending_analysis",
    "TEMPORAL_ANALYSIS": "temporal_analysis",
    "TEMPORAL_SPENDING_ANALYSIS": "temporal_analysis",
    "MERCHANT_ANALYSIS": "merchant_analysis",
    "CATEGORY_ANALYSIS": "spending_analysis",
    "BALANCE_ANALYSIS": "spending_analysis",
    "TRANSACTION_DETAILS": "text_search",
    "SEARCH_SIMILAR": "text_search",
    "COMPARE_PERIODS": "temporal_analysis",
    "TOP_MERCHANTS": "merchant_analysis",
    "TOP_CATEGORIES": "spending_analysis"
}

# Helpers pour la construction dynamique de templates
class TemplateHelpers:
    """Utilitaires pour la construction de templates"""
    
    @staticmethod
    def build_user_filter(user_id: int) -> Dict[str, Any]:
        """Construit le filtre utilisateur obligatoire"""
        return {"term": {"user_id": user_id}}
    
    @staticmethod
    def build_category_filter(category: str) -> Dict[str, Any]:
        """Construit un filtre par catégorie"""
        return {"term": {"category_name.keyword": category}}
    
    @staticmethod
    def build_merchant_filter(merchant: str) -> Dict[str, Any]:
        """Construit un filtre par marchand"""
        return {"term": {"merchant_name.keyword": merchant}}
    
    @staticmethod
    def build_amount_range(min_amount: float = None, max_amount: float = None) -> Dict[str, Any]:
        """Construit un filtre de plage de montants"""
        range_spec = {}
        if min_amount is not None:
            range_spec["gte"] = min_amount
        if max_amount is not None:
            range_spec["lte"] = max_amount
        return {"range": {"amount_abs": range_spec}}
    
    @staticmethod
    def build_date_range(start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Construit un filtre de plage de dates"""
        range_spec = {}
        if start_date:
            range_spec["gte"] = start_date
        if end_date:
            range_spec["lte"] = end_date
        return {"range": {"date": range_spec}}
    
    @staticmethod
    def build_text_search(query: str, fields: List[str] = None) -> Dict[str, Any]:
        """Construit une requête de recherche textuelle"""
        if fields is None:
            fields = [
                "searchable_text^2.0",
                "primary_description^1.5",
                "merchant_name^1.8"
            ]
        
        return {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }