"""
Aggregation Templates pour Search Service

Templates d'agrégations Elasticsearch optimisés pour les analyses financières.
Support des analyses temporelles, catégorielles et statistiques avancées.

Architecture:
- Templates agrégations par type d'analyse
- Configurations temporelles (daily, weekly, monthly)
- Métriques financières (sommes, moyennes, statistiques)
- Groupements multiples et bucket aggregations
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

class AggregationValidationError(Exception):
    """Exception levée lors de la validation des agrégations"""
    pass

class AggregationType(Enum):
    """Types d'agrégations supportées"""
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    STATISTICAL = "statistical"
    MULTI_LEVEL = "multi_level"
    COMPOSITE = "composite"

class TimeInterval(Enum):
    """Intervalles temporels supportés"""
    DAILY = "day"
    WEEKLY = "week"
    MONTHLY = "month"
    QUARTERLY = "quarter"
    YEARLY = "year"

@dataclass
class AggregationConfig:
    """Configuration d'une agrégation"""
    name: str
    type: AggregationType
    description: str
    required_params: List[str]
    optional_params: List[str]
    performance_impact: str  # "low", "medium", "high"
    cache_duration: int

class AggregationTemplates:
    """Gestionnaire des templates d'agrégations Elasticsearch"""
    
    def __init__(self):
        self.templates = {
            **FINANCIAL_AGGREGATION_TEMPLATES,
            **TEMPORAL_AGGREGATION_TEMPLATES,
            **CATEGORICAL_AGGREGATION_TEMPLATES
        }
    
    def get_aggregation(self, agg_type: str, **params) -> Dict[str, Any]:
        """
        Récupère et paramètre une agrégation selon le type
        
        Args:
            agg_type: Type d'agrégation (ex: "monthly_spending")
            **params: Paramètres pour l'agrégation
            
        Returns:
            Template d'agrégation Elasticsearch paramétré
        """
        if agg_type not in self.templates:
            raise AggregationValidationError(f"Type d'agrégation non supporté: {agg_type}")
        
        template_config = self.templates[agg_type]
        
        # Validation paramètres requis
        missing_params = []
        for required_param in template_config["required_params"]:
            if required_param not in params:
                missing_params.append(required_param)
        
        if missing_params:
            raise AggregationValidationError(
                f"Paramètres requis manquants: {missing_params}"
            )
        
        # Construction de l'agrégation
        builder = AggregationTemplateBuilder(template_config)
        return builder.build(**params)
    
    def get_available_aggregations(self) -> List[str]:
        """Retourne la liste des agrégations disponibles"""
        return list(self.templates.keys())
    
    def validate_aggregation(self, agg: Dict[str, Any]) -> bool:
        """Valide la structure d'une agrégation"""
        # Vérification structure basique
        if not isinstance(agg, dict):
            return False
        
        # Au moins une agrégation doit être définie
        valid_agg_types = [
            "terms", "date_histogram", "histogram", "range", "stats",
            "sum", "avg", "min", "max", "value_count", "cardinality",
            "percentiles", "significant_terms", "composite"
        ]
        
        return any(agg_type in agg for agg_type in valid_agg_types)

class AggregationTemplateBuilder:
    """Constructeur de templates d'agrégations"""
    
    def __init__(self, template_config: Dict[str, Any]):
        self.config = template_config
        self.template = template_config["template"]
    
    def build(self, **params) -> Dict[str, Any]:
        """Construit l'agrégation Elasticsearch finale"""
        # Copie profonde du template
        agg = json.loads(json.dumps(self.template))
        
        # Application des paramètres
        self._apply_temporal_params(agg, params)
        self._apply_categorical_params(agg, params)
        self._apply_statistical_params(agg, params)
        self._apply_filters(agg, params)
        
        return agg
    
    def _apply_temporal_params(self, agg: Dict[str, Any], params: Dict[str, Any]):
        """Applique les paramètres temporels"""
        interval = params.get("interval", "month")
        field = params.get("date_field", "date")
        
        # Recherche des date_histogram dans l'agrégation
        self._update_date_histograms(agg, interval, field)
    
    def _apply_categorical_params(self, agg: Dict[str, Any], params: Dict[str, Any]):
        """Applique les paramètres catégoriels"""
        size = params.get("size", 10)
        order = params.get("order", {"total_amount": "desc"})
        
        # Recherche des terms aggregations
        self._update_terms_aggregations(agg, size, order)
    
    def _apply_statistical_params(self, agg: Dict[str, Any], params: Dict[str, Any]):
        """Applique les paramètres statistiques"""
        field = params.get("field", "amount_abs")
        
        # Mise à jour des champs pour les stats
        self._update_statistical_fields(agg, field)
    
    def _apply_filters(self, agg: Dict[str, Any], params: Dict[str, Any]):
        """Applique les filtres aux agrégations"""
        filters = params.get("filters", {})
        
        # Si des filtres sont spécifiés, on les applique via filter aggregation
        if filters:
            self._wrap_with_filter(agg, filters)
    
    def _update_date_histograms(self, agg: Dict[str, Any], interval: str, field: str):
        """Met à jour les date_histogram dans l'agrégation"""
        for key, value in agg.items():
            if isinstance(value, dict):
                if "date_histogram" in value:
                    value["date_histogram"]["field"] = field
                    value["date_histogram"]["calendar_interval"] = interval
                else:
                    self._update_date_histograms(value, interval, field)
    
    def _update_terms_aggregations(self, agg: Dict[str, Any], size: int, order: Dict[str, str]):
        """Met à jour les terms aggregations"""
        for key, value in agg.items():
            if isinstance(value, dict):
                if "terms" in value:
                    value["terms"]["size"] = size
                    if order:
                        value["terms"]["order"] = order
                else:
                    self._update_terms_aggregations(value, size, order)
    
    def _update_statistical_fields(self, agg: Dict[str, Any], field: str):
        """Met à jour les champs pour les agrégations statistiques"""
        statistical_types = ["sum", "avg", "min", "max", "stats", "extended_stats", "percentiles"]
        
        for key, value in agg.items():
            if isinstance(value, dict):
                for stat_type in statistical_types:
                    if stat_type in value:
                        value[stat_type]["field"] = field
                        break
                else:
                    self._update_statistical_fields(value, field)
    
    def _wrap_with_filter(self, agg: Dict[str, Any], filters: Dict[str, Any]):
        """Encapsule l'agrégation dans un filtre"""
        # Conversion des filtres en query Elasticsearch
        filter_query = self._build_filter_query(filters)
        
        # Encapsulation de l'agrégation existante
        original_agg = agg.copy()
        agg.clear()
        agg["filtered_data"] = {
            "filter": filter_query,
            "aggs": original_agg
        }
    
    def _build_filter_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Construit une query de filtre à partir des paramètres"""
        bool_query = {"bool": {"must": []}}
        
        for field, value in filters.items():
            if isinstance(value, (str, int, float)):
                bool_query["bool"]["must"].append({
                    "term": {field: value}
                })
            elif isinstance(value, dict) and "range" in value:
                bool_query["bool"]["must"].append({
                    "range": {field: value["range"]}
                })
        
        return bool_query

# Templates d'agrégations financières de base
FINANCIAL_AGGREGATION_TEMPLATES = {
    "spending_summary": {
        "name": "spending_summary",
        "type": AggregationType.STATISTICAL,
        "description": "Résumé statistique des dépenses",
        "required_params": [],
        "optional_params": ["field", "filters"],
        "performance_impact": "low",
        "cache_duration": 300,
        "template": {
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
            "transaction_count": {
                "value_count": {
                    "field": "transaction_id"
                }
            },
            "spending_stats": {
                "stats": {
                    "field": "amount_abs"
                }
            },
            "spending_percentiles": {
                "percentiles": {
                    "field": "amount_abs",
                    "percents": [25, 50, 75, 90, 95, 99]
                }
            }
        }
    },
    
    "category_breakdown": {
        "name": "category_breakdown",
        "type": AggregationType.CATEGORICAL,
        "description": "Répartition par catégorie",
        "required_params": [],
        "optional_params": ["size", "order", "filters"],
        "performance_impact": "medium",
        "cache_duration": 300,
        "template": {
            "categories": {
                "terms": {
                    "field": "category_name.keyword",
                    "size": 20,
                    "order": {
                        "total_amount": "desc"
                    }
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
                    },
                    "average_amount": {
                        "avg": {
                            "field": "amount_abs"
                        }
                    },
                    "amount_stats": {
                        "stats": {
                            "field": "amount_abs"
                        }
                    }
                }
            }
        }
    },
    
    "merchant_breakdown": {
        "name": "merchant_breakdown",
        "type": AggregationType.CATEGORICAL,
        "description": "Répartition par marchand",
        "required_params": [],
        "optional_params": ["size", "order", "filters"],
        "performance_impact": "medium",
        "cache_duration": 300,
        "template": {
            "merchants": {
                "terms": {
                    "field": "merchant_name.keyword",
                    "size": 20,
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
                    },
                    "average_transaction": {
                        "avg": {
                            "field": "amount_abs"
                        }
                    },
                    "first_transaction": {
                        "min": {
                            "field": "date"
                        }
                    },
                    "last_transaction": {
                        "max": {
                            "field": "date"
                        }
                    }
                }
            }
        }
    },
    
    "amount_distribution": {
        "name": "amount_distribution",
        "type": AggregationType.STATISTICAL,
        "description": "Distribution des montants par tranches",
        "required_params": [],
        "optional_params": ["ranges", "field"],
        "performance_impact": "medium",
        "cache_duration": 600,
        "template": {
            "amount_ranges": {
                "range": {
                    "field": "amount_abs",
                    "ranges": [
                        {"to": 10},
                        {"from": 10, "to": 50},
                        {"from": 50, "to": 100},
                        {"from": 100, "to": 200},
                        {"from": 200, "to": 500},
                        {"from": 500}
                    ]
                },
                "aggs": {
                    "transaction_count": {
                        "value_count": {
                            "field": "transaction_id"
                        }
                    },
                    "total_amount": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    }
                }
            }
        }
    },
    
    "transaction_types": {
        "name": "transaction_types",
        "type": AggregationType.CATEGORICAL,
        "description": "Répartition par type de transaction",
        "required_params": [],
        "optional_params": ["filters"],
        "performance_impact": "low",
        "cache_duration": 600,
        "template": {
            "by_type": {
                "terms": {
                    "field": "transaction_type",
                    "size": 10
                },
                "aggs": {
                    "total_amount": {
                        "sum": {
                            "field": "amount"
                        }
                    },
                    "total_absolute": {
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

# Templates d'agrégations temporelles
TEMPORAL_AGGREGATION_TEMPLATES = {
    "monthly_trends": {
        "name": "monthly_trends",
        "type": AggregationType.TEMPORAL,
        "description": "Tendances mensuelles",
        "required_params": [],
        "optional_params": ["interval", "date_field", "filters"],
        "performance_impact": "medium",
        "cache_duration": 600,
        "template": {
            "monthly_data": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "month",
                    "format": "yyyy-MM",
                    "order": {
                        "_key": "desc"
                    }
                },
                "aggs": {
                    "total_spending": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    },
                    "total_income": {
                        "sum": {
                            "script": {
                                "source": "doc['amount'].value > 0 ? doc['amount'].value : 0"
                            }
                        }
                    },
                    "total_expenses": {
                        "sum": {
                            "script": {
                                "source": "doc['amount'].value < 0 ? Math.abs(doc['amount'].value) : 0"
                            }
                        }
                    },
                    "net_balance": {
                        "sum": {
                            "field": "amount"
                        }
                    },
                    "transaction_count": {
                        "value_count": {
                            "field": "transaction_id"
                        }
                    },
                    "average_transaction": {
                        "avg": {
                            "field": "amount_abs"
                        }
                    },
                    "unique_merchants": {
                        "cardinality": {
                            "field": "merchant_name.keyword"
                        }
                    },
                    "unique_categories": {
                        "cardinality": {
                            "field": "category_name.keyword"
                        }
                    }
                }
            }
        }
    },
    
    "weekly_patterns": {
        "name": "weekly_patterns",
        "type": AggregationType.TEMPORAL,
        "description": "Patterns hebdomadaires",
        "required_params": [],
        "optional_params": ["interval", "filters"],
        "performance_impact": "medium",
        "cache_duration": 600,
        "template": {
            "weekly_data": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "week",
                    "format": "yyyy-'W'ww",
                    "order": {
                        "_key": "desc"
                    }
                },
                "aggs": {
                    "total_spending": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    },
                    "transaction_count": {
                        "value_count": {
                            "field": "transaction_id"
                        }
                    },
                    "daily_breakdown": {
                        "terms": {
                            "field": "weekday",
                            "order": {
                                "_key": "asc"
                            }
                        },
                        "aggs": {
                            "daily_spending": {
                                "sum": {
                                    "field": "amount_abs"
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    
    "daily_activity": {
        "name": "daily_activity",
        "type": AggregationType.TEMPORAL,
        "description": "Activité quotidienne",
        "required_params": [],
        "optional_params": ["interval", "filters"],
        "performance_impact": "high",
        "cache_duration": 300,
        "template": {
            "daily_data": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "day",
                    "format": "yyyy-MM-dd",
                    "order": {
                        "_key": "desc"
                    }
                },
                "aggs": {
                    "total_spending": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    },
                    "transaction_count": {
                        "value_count": {
                            "field": "transaction_id"
                        }
                    },
                    "largest_transaction": {
                        "max": {
                            "field": "amount_abs"
                        }
                    },
                    "categories_used": {
                        "cardinality": {
                            "field": "category_name.keyword"
                        }
                    }
                }
            }
        }
    },
    
    "yearly_overview": {
        "name": "yearly_overview",
        "type": AggregationType.TEMPORAL,
        "description": "Vue d'ensemble annuelle",
        "required_params": [],
        "optional_params": ["interval", "filters"],
        "performance_impact": "low",
        "cache_duration": 3600,
        "template": {
            "yearly_data": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "year",
                    "format": "yyyy",
                    "order": {
                        "_key": "desc"
                    }
                },
                "aggs": {
                    "total_spending": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    },
                    "total_income": {
                        "sum": {
                            "script": {
                                "source": "doc['amount'].value > 0 ? doc['amount'].value : 0"
                            }
                        }
                    },
                    "net_balance": {
                        "sum": {
                            "field": "amount"
                        }
                    },
                    "transaction_count": {
                        "value_count": {
                            "field": "transaction_id"
                        }
                    },
                    "monthly_breakdown": {
                        "date_histogram": {
                            "field": "date",
                            "calendar_interval": "month",
                            "format": "MM"
                        },
                        "aggs": {
                            "monthly_spending": {
                                "sum": {
                                    "field": "amount_abs"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

# Templates d'agrégations catégorielles avancées
CATEGORICAL_AGGREGATION_TEMPLATES = {
    "category_merchant_matrix": {
        "name": "category_merchant_matrix",
        "type": AggregationType.MULTI_LEVEL,
        "description": "Matrice catégorie x marchand",
        "required_params": [],
        "optional_params": ["category_size", "merchant_size", "filters"],
        "performance_impact": "high",
        "cache_duration": 600,
        "template": {
            "categories": {
                "terms": {
                    "field": "category_name.keyword",
                    "size": 10,
                    "order": {
                        "total_amount": "desc"
                    }
                },
                "aggs": {
                    "total_amount": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    },
                    "top_merchants": {
                        "terms": {
                            "field": "merchant_name.keyword",
                            "size": 5,
                            "order": {
                                "merchant_spending": "desc"
                            }
                        },
                        "aggs": {
                            "merchant_spending": {
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
    },
    
    "temporal_category_trends": {
        "name": "temporal_category_trends",
        "type": AggregationType.MULTI_LEVEL,
        "description": "Tendances temporelles par catégorie",
        "required_params": [],
        "optional_params": ["interval", "category_size", "filters"],
        "performance_impact": "high",
        "cache_duration": 600,
        "template": {
            "monthly_trends": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "month",
                    "format": "yyyy-MM"
                },
                "aggs": {
                    "categories": {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": 10
                        },
                        "aggs": {
                            "category_spending": {
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
                    },
                    "total_monthly": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    }
                }
            }
        }
    },
    
    "spending_concentration": {
        "name": "spending_concentration",
        "type": AggregationType.STATISTICAL,
        "description": "Concentration des dépenses (analyse Pareto)",
        "required_params": [],
        "optional_params": ["field", "percentile"],
        "performance_impact": "medium",
        "cache_duration": 600,
        "template": {
            "merchant_concentration": {
                "terms": {
                    "field": "merchant_name.keyword",
                    "size": 100,
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
                    "cumulative_percentage": {
                        "cumulative_sum": {
                            "buckets_path": "total_spent"
                        }
                    }
                }
            },
            "category_concentration": {
                "terms": {
                    "field": "category_name.keyword",
                    "size": 50,
                    "order": {
                        "total_spent": "desc"
                    }
                },
                "aggs": {
                    "total_spent": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    }
                }
            }
        }
    },
    
    "comparative_analysis": {
        "name": "comparative_analysis",
        "type": AggregationType.COMPOSITE,
        "description": "Analyse comparative multi-dimensionnelle",
        "required_params": [],
        "optional_params": ["compare_field", "baseline_filters", "comparison_filters"],
        "performance_impact": "high",
        "cache_duration": 300,
        "template": {
            "baseline_metrics": {
                "filter": {
                    "bool": {
                        "must": []
                    }
                },
                "aggs": {
                    "total_spending": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    },
                    "transaction_count": {
                        "value_count": {
                            "field": "transaction_id"
                        }
                    },
                    "category_breakdown": {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": 10
                        },
                        "aggs": {
                            "category_total": {
                                "sum": {
                                    "field": "amount_abs"
                                }
                            }
                        }
                    }
                }
            },
            "comparison_metrics": {
                "filter": {
                    "bool": {
                        "must": []
                    }
                },
                "aggs": {
                    "total_spending": {
                        "sum": {
                            "field": "amount_abs"
                        }
                    },
                    "transaction_count": {
                        "value_count": {
                            "field": "transaction_id"
                        }
                    },
                    "category_breakdown": {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": 10
                        },
                        "aggs": {
                            "category_total": {
                                "sum": {
                                    "field": "amount_abs"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

# Helpers pour les agrégations
class AggregationHelpers:
    """Utilitaires pour la construction d'agrégations"""
    
    @staticmethod
    def build_date_histogram(field: str = "date", interval: str = "month", format: str = None) -> Dict[str, Any]:
        """Construit une agrégation date_histogram"""
        date_hist = {
            "field": field,
            "calendar_interval": interval
        }
        
        if format:
            date_hist["format"] = format
            
        return {"date_histogram": date_hist}
    
    @staticmethod
    def build_terms_aggregation(field: str, size: int = 10, order: Dict[str, str] = None) -> Dict[str, Any]:
        """Construit une agrégation terms"""
        terms_agg = {
            "field": field,
            "size": size
        }
        
        if order:
            terms_agg["order"] = order
            
        return {"terms": terms_agg}
    
    @staticmethod
    def build_stats_aggregation(field: str = "amount_abs") -> Dict[str, Any]:
        """Construit une agrégation stats complète"""
        return {
            "stats": {
                "field": field
            }
        }
    
    @staticmethod
    def build_sum_aggregation(field: str = "amount_abs") -> Dict[str, Any]:
        """Construit une agrégation sum"""
        return {
            "sum": {
                "field": field
            }
        }
    
    @staticmethod
    def build_count_aggregation(field: str = "transaction_id") -> Dict[str, Any]:
        """Construit une agrégation value_count"""
        return {
            "value_count": {
                "field": field
            }
        }
    
    @staticmethod
    def build_range_aggregation(field: str, ranges: List[Dict[str, float]]) -> Dict[str, Any]:
        """Construit une agrégation range"""
        return {
            "range": {
                "field": field,
                "ranges": ranges
            }
        }
    
    @staticmethod
    def build_filter_aggregation(filter_query: Dict[str, Any], sub_aggs: Dict[str, Any]) -> Dict[str, Any]:
        """Construit une agrégation avec filtre"""
        return {
            "filter": filter_query,
            "aggs": sub_aggs
        }

# Configuration des agrégations par défaut
DEFAULT_AGGREGATION_CONFIG = {
    "max_buckets": 10000,
    "timeout": "30s",
    "execution_hint": "map",
    "collect_mode": "depth_first"
}

# Mapping des intentions vers les agrégations
INTENT_AGGREGATION_MAPPING = {
    "SPENDING_ANALYSIS": "spending_summary",
    "TEMPORAL_ANALYSIS": "monthly_trends",
    "CATEGORY_ANALYSIS": "category_breakdown",
    "MERCHANT_ANALYSIS": "merchant_breakdown",
    "BALANCE_ANALYSIS": "spending_summary",
    "COUNT_OPERATIONS": "spending_summary",
    "COMPARE_PERIODS": "comparative_analysis",
    "TOP_MERCHANTS": "merchant_breakdown",
    "TOP_CATEGORIES": "category_breakdown"
}