"""
Aggregation Templates pour Search Service

Gestion des templates d'agrégations Elasticsearch spécialisés pour les analyses financières.
Supporte les agrégations complexes pour visualisations et rapports analytiques.

Classes principales:
- AggregationTemplateManager: Gestionnaire principal des agrégations
- FinancialAggregationTemplates: Agrégations financières spécialisées
- DateAggregationTemplates: Agrégations temporelles
- StatisticalAggregationTemplates: Agrégations statistiques

Types d'agrégations supportés:
- Répartitions (marchands, catégories, montants)
- Évolutions temporelles (histogrammes de dates)
- Statistiques (somme, moyenne, min, max, percentiles)
- Agrégations composites (bucket + metrics)
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from copy import deepcopy

from ..models.service_contracts import IntentType, AggregationType
from ..models.requests import AggregationRequest

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

AGGREGATION_CONFIG = {
    "default_bucket_size": 10,
    "max_bucket_size": 1000,
    "default_min_doc_count": 1,
    "enable_metadata": True,
    "precision_threshold": 3000,  # Pour cardinality
    "execution_hint": "map",  # Pour terms aggregations
    "collect_mode": "depth_first"
}

# Configuration des buckets pour montants
AMOUNT_BUCKETS = [
    {"to": 10, "key": "micro"},
    {"from": 10, "to": 50, "key": "small"},
    {"from": 50, "to": 200, "key": "medium"},
    {"from": 200, "to": 1000, "key": "large"},
    {"from": 1000, "key": "very_large"}
]

# Intervalles temporels supportés
TIME_INTERVALS = {
    "daily": "1d",
    "weekly": "1w", 
    "monthly": "1M",
    "quarterly": "1q",
    "yearly": "1y",
    "hourly": "1h"
}

# ==================== TEMPLATES D'AGRÉGATIONS FINANCIÈRES ====================

class FinancialAggregationTemplates:
    """Templates pour agrégations financières."""
    
    @staticmethod
    def merchant_spending_template(
        size: int = 10,
        min_doc_count: int = 1,
        include_metrics: bool = True,
        order_by: str = "total_amount"
    ) -> Dict[str, Any]:
        """Template pour répartition des dépenses par marchand."""
        agg = {
            "merchants": {
                "terms": {
                    "field": "merchant_name.keyword",
                    "size": size,
                    "min_doc_count": min_doc_count,
                    "order": {}
                }
            }
        }
        
        if include_metrics:
            agg["merchants"]["aggs"] = {
                "total_amount": {"sum": {"field": "amount_abs"}},
                "avg_amount": {"avg": {"field": "amount_abs"}},
                "transaction_count": {"value_count": {"field": "transaction_id"}},
                "max_amount": {"max": {"field": "amount_abs"}},
                "min_amount": {"min": {"field": "amount_abs"}}
            }
            
            # Définir l'ordre selon la métrique choisie
            if order_by == "total_amount":
                agg["merchants"]["terms"]["order"] = {"total_amount": "desc"}
            elif order_by == "transaction_count":
                agg["merchants"]["terms"]["order"] = {"_count": "desc"}
            elif order_by == "avg_amount":
                agg["merchants"]["terms"]["order"] = {"avg_amount": "desc"}
            else:
                agg["merchants"]["terms"]["order"] = {"_key": "asc"}
        
        return agg
    
    @staticmethod
    def category_distribution_template(
        size: int = 20,
        min_doc_count: int = 1,
        include_subcategories: bool = False
    ) -> Dict[str, Any]:
        """Template pour distribution par catégorie."""
        agg = {
            "categories": {
                "terms": {
                    "field": "category_id",
                    "size": size,
                    "min_doc_count": min_doc_count,
                    "order": {"total_spent": "desc"}
                },
                "aggs": {
                    "total_spent": {"sum": {"field": "amount_abs"}},
                    "avg_transaction": {"avg": {"field": "amount_abs"}},
                    "transaction_count": {"value_count": {"field": "transaction_id"}},
                    "category_name": {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": 1
                        }
                    }
                }
            }
        }
        
        # Ajouter sous-catégories si demandé
        if include_subcategories:
            agg["categories"]["aggs"]["subcategories"] = {
                "terms": {
                    "field": "subcategory_name.keyword",
                    "size": 5,
                    "min_doc_count": 1
                },
                "aggs": {
                    "subcategory_total": {"sum": {"field": "amount_abs"}}
                }
            }
        
        return agg
    
    @staticmethod
    def amount_distribution_template(
        buckets: List[Dict[str, Any]] = None,
        include_stats: bool = True
    ) -> Dict[str, Any]:
        """Template pour distribution des montants."""
        if not buckets:
            buckets = AMOUNT_BUCKETS
        
        agg = {
            "amount_ranges": {
                "range": {
                    "field": "amount_abs",
                    "ranges": buckets
                },
                "aggs": {
                    "avg_amount": {"avg": {"field": "amount_abs"}},
                    "total_amount": {"sum": {"field": "amount_abs"}}
                }
            }
        }
        
        if include_stats:
            agg["amount_statistics"] = {
                "stats": {"field": "amount_abs"}
            }
            agg["amount_percentiles"] = {
                "percentiles": {
                    "field": "amount_abs",
                    "percents": [25, 50, 75, 90, 95, 99]
                }
            }
        
        return agg
    
    @staticmethod
    def spending_evolution_template(
        interval: str = "monthly",
        include_cumulative: bool = True,
        extended_bounds: bool = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Template pour évolution des dépenses dans le temps."""
        calendar_interval = TIME_INTERVALS.get(interval, "1M")
        
        date_histogram = {
            "field": "transaction_date",
            "calendar_interval": calendar_interval,
            "min_doc_count": 0
        }
        
        # Ajouter les bornes étendues si spécifiées
        if extended_bounds and start_date and end_date:
            date_histogram["extended_bounds"] = {
                "min": start_date.isoformat(),
                "max": end_date.isoformat()
            }
        
        agg = {
            "spending_over_time": {
                "date_histogram": date_histogram,
                "aggs": {
                    "total_spent": {"sum": {"field": "amount_abs"}},
                    "transaction_count": {"value_count": {"field": "transaction_id"}},
                    "avg_transaction": {"avg": {"field": "amount_abs"}},
                    "unique_merchants": {
                        "cardinality": {
                            "field": "merchant_name.keyword",
                            "precision_threshold": AGGREGATION_CONFIG["precision_threshold"]
                        }
                    },
                    "unique_categories": {
                        "cardinality": {
                            "field": "category_id",
                            "precision_threshold": AGGREGATION_CONFIG["precision_threshold"]
                        }
                    }
                }
            }
        }
        
        # Ajouter cumul si demandé
        if include_cumulative:
            agg["spending_over_time"]["aggs"]["cumulative_total"] = {
                "cumulative_sum": {
                    "buckets_path": "total_spent"
                }
            }
        
        return agg
    
    @staticmethod
    def comparative_spending_template(
        compare_field: str = "category_id",
        time_periods: List[Dict[str, datetime]] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """Template pour analyse comparative des dépenses."""
        agg = {
            "comparison": {
                "terms": {
                    "field": f"{compare_field}.keyword" if compare_field != "category_id" else compare_field,
                    "size": top_n,
                    "order": {"current_period.total": "desc"}
                },
                "aggs": {
                    "current_period": {
                        "filter": {
                            "range": {
                                "transaction_date": {
                                    "gte": time_periods[0]["start"].isoformat() if time_periods else "now-1M",
                                    "lte": time_periods[0]["end"].isoformat() if time_periods else "now"
                                }
                            }
                        },
                        "aggs": {
                            "total": {"sum": {"field": "amount_abs"}},
                            "count": {"value_count": {"field": "transaction_id"}}
                        }
                    }
                }
            }
        }
        
        # Ajouter période de comparaison si fournie
        if time_periods and len(time_periods) > 1:
            agg["comparison"]["aggs"]["previous_period"] = {
                "filter": {
                    "range": {
                        "transaction_date": {
                            "gte": time_periods[1]["start"].isoformat(),
                            "lte": time_periods[1]["end"].isoformat()
                        }
                    }
                },
                "aggs": {
                    "total": {"sum": {"field": "amount_abs"}},
                    "count": {"value_count": {"field": "transaction_id"}}
                }
            }
            
            # Ajouter calcul de variation
            agg["comparison"]["aggs"]["variation"] = {
                "bucket_script": {
                    "buckets_path": {
                        "current": "current_period.total",
                        "previous": "previous_period.total"
                    },
                    "script": "params.previous != 0 ? ((params.current - params.previous) / params.previous) * 100 : 0"
                }
            }
        
        return agg

class DateAggregationTemplates:
    """Templates pour agrégations temporelles."""
    
    @staticmethod
    def daily_spending_pattern_template(
        timezone: str = "Europe/Paris"
    ) -> Dict[str, Any]:
        """Template pour patterns de dépenses quotidiennes."""
        return {
            "daily_patterns": {
                "date_histogram": {
                    "field": "transaction_date",
                    "calendar_interval": "1d",
                    "time_zone": timezone,
                    "min_doc_count": 0
                },
                "aggs": {
                    "total_spent": {"sum": {"field": "amount_abs"}},
                    "transaction_count": {"value_count": {"field": "transaction_id"}},
                    "hour_distribution": {
                        "date_histogram": {
                            "field": "transaction_date",
                            "calendar_interval": "1h",
                            "time_zone": timezone
                        },
                        "aggs": {
                            "hourly_total": {"sum": {"field": "amount_abs"}}
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def weekday_analysis_template(
        timezone: str = "Europe/Paris"
    ) -> Dict[str, Any]:
        """Template pour analyse par jour de la semaine."""
        return {
            "weekday_spending": {
                "terms": {
                    "script": {
                        "source": "doc['transaction_date'].value.dayOfWeek",
                        "lang": "painless"
                    },
                    "size": 7,
                    "order": {"_key": "asc"}
                },
                "aggs": {
                    "total_spent": {"sum": {"field": "amount_abs"}},
                    "avg_spent": {"avg": {"field": "amount_abs"}},
                    "transaction_count": {"value_count": {"field": "transaction_id"}},
                    "weekday_name": {
                        "bucket_script": {
                            "buckets_path": {},
                            "script": {
                                "source": """
                                String[] days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'];
                                return days[(int)params._key - 1];
                                """
                            }
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def monthly_trends_template(
        num_months: int = 12,
        include_forecast: bool = False
    ) -> Dict[str, Any]:
        """Template pour tendances mensuelles."""
        agg = {
            "monthly_trends": {
                "date_histogram": {
                    "field": "transaction_date",
                    "calendar_interval": "1M",
                    "min_doc_count": 0,
                    "extended_bounds": {
                        "min": f"now-{num_months}M",
                        "max": "now"
                    }
                },
                "aggs": {
                    "total_spent": {"sum": {"field": "amount_abs"}},
                    "transaction_count": {"value_count": {"field": "transaction_id"}},
                    "avg_transaction": {"avg": {"field": "amount_abs"}},
                    "moving_avg": {
                        "moving_avg": {
                            "buckets_path": "total_spent",
                            "window": 3,
                            "model": "linear"
                        }
                    }
                }
            }
        }
        
        if include_forecast:
            agg["monthly_trends"]["aggs"]["forecast"] = {
                "moving_avg": {
                    "buckets_path": "total_spent",
                    "window": 6,
                    "model": "holt",
                    "predict": 3,
                    "settings": {
                        "alpha": 0.3,
                        "beta": 0.1
                    }
                }
            }
        
        return agg

class StatisticalAggregationTemplates:
    """Templates pour agrégations statistiques."""
    
    @staticmethod
    def financial_statistics_template(
        field: str = "amount_abs",
        include_percentiles: bool = True,
        include_histogram: bool = False
    ) -> Dict[str, Any]:
        """Template pour statistiques financières complètes."""
        agg = {
            "financial_stats": {
                "stats": {"field": field}
            },
            "extended_stats": {
                "extended_stats": {"field": field}
            }
        }
        
        if include_percentiles:
            agg["percentiles"] = {
                "percentiles": {
                    "field": field,
                    "percents": [1, 5, 25, 50, 75, 95, 99],
                    "keyed": True
                }
            }
            agg["percentile_ranks"] = {
                "percentile_ranks": {
                    "field": field,
                    "values": [10, 50, 100, 500, 1000]
                }
            }
        
        if include_histogram:
            agg["histogram"] = {
                "histogram": {
                    "field": field,
                    "interval": 50,
                    "min_doc_count": 1
                }
            }
        
        return agg
    
    @staticmethod
    def correlation_analysis_template(
        field_x: str,
        field_y: str,
        bucket_size: int = 20
    ) -> Dict[str, Any]:
        """Template pour analyse de corrélation."""
        return {
            "correlation": {
                "terms": {
                    "field": field_x,
                    "size": bucket_size
                },
                "aggs": {
                    "avg_y": {"avg": {"field": field_y}},
                    "sum_x": {"sum": {"field": field_x}},
                    "sum_y": {"sum": {"field": field_y}},
                    "doc_count": {"value_count": {"field": field_x}}
                }
            }
        }
    
    @staticmethod
    def outlier_detection_template(
        field: str = "amount_abs",
        iqr_multiplier: float = 1.5
    ) -> Dict[str, Any]:
        """Template pour détection d'anomalies."""
        return {
            "outlier_analysis": {
                "percentiles": {
                    "field": field,
                    "percents": [25, 75]
                }
            },
            "potential_outliers": {
                "bucket_script": {
                    "buckets_path": {
                        "q1": "outlier_analysis.25.0",
                        "q3": "outlier_analysis.75.0"
                    },
                    "script": {
                        "source": f"""
                        double iqr = params.q3 - params.q1;
                        double lower = params.q1 - {iqr_multiplier} * iqr;
                        double upper = params.q3 + {iqr_multiplier} * iqr;
                        return ['lower_bound': lower, 'upper_bound': upper];
                        """
                    }
                }
            }
        }

# ==================== GESTIONNAIRE PRINCIPAL ====================

class AggregationTemplateManager:
    """Gestionnaire principal des templates d'agrégation."""
    
    def __init__(self):
        """Initialise le gestionnaire."""
        self._templates = {}
        self._cache = {}
        self._load_default_templates()
        logger.info("✅ AggregationTemplateManager initialisé")
    
    def _load_default_templates(self):
        """Charge les templates d'agrégation par défaut."""
        # Templates financiers
        self._templates[AggregationType.MERCHANTS] = {
            "template": FinancialAggregationTemplates.merchant_spending_template,
            "description": "Répartition des dépenses par marchand",
            "category": "financial"
        }
        
        self._templates[AggregationType.CATEGORIES] = {
            "template": FinancialAggregationTemplates.category_distribution_template,
            "description": "Distribution par catégorie",
            "category": "financial"
        }
        
        self._templates[AggregationType.AMOUNTS] = {
            "template": FinancialAggregationTemplates.amount_distribution_template,
            "description": "Distribution des montants",
            "category": "financial"
        }
        
        self._templates[AggregationType.SPENDING_EVOLUTION] = {
            "template": FinancialAggregationTemplates.spending_evolution_template,
            "description": "Évolution des dépenses dans le temps",
            "category": "temporal"
        }
        
        # Templates temporels
        self._templates[AggregationType.DATE_HISTOGRAM] = {
            "template": DateAggregationTemplates.monthly_trends_template,
            "description": "Tendances mensuelles",
            "category": "temporal"
        }
        
        # Templates statistiques
        self._templates["financial_stats"] = {
            "template": StatisticalAggregationTemplates.financial_statistics_template,
            "description": "Statistiques financières complètes",
            "category": "statistical"
        }
    
    def get_template(
        self,
        aggregation_type: Union[AggregationType, str],
        **params
    ) -> Dict[str, Any]:
        """Récupère et rend un template d'agrégation."""
        if aggregation_type not in self._templates:
            raise ValueError(f"Template d'agrégation non trouvé: {aggregation_type}")
        
        template_info = self._templates[aggregation_type]
        template_func = template_info["template"]
        
        try:
            aggregation = template_func(**params)
            logger.debug(f"Template d'agrégation généré pour {aggregation_type}")
            return aggregation
        except Exception as e:
            logger.error(f"Erreur génération template agrégation {aggregation_type}: {e}")
            raise
    
    def get_merchant_spending_aggregation(self, **params) -> Dict[str, Any]:
        """Raccourci pour agrégation des dépenses par marchand."""
        return self.get_template(AggregationType.MERCHANTS, **params)
    
    def get_category_distribution_aggregation(self, **params) -> Dict[str, Any]:
        """Raccourci pour agrégation par catégorie."""
        return self.get_template(AggregationType.CATEGORIES, **params)
    
    def get_spending_evolution_aggregation(self, **params) -> Dict[str, Any]:
        """Raccourci pour évolution des dépenses."""
        return self.get_template(AggregationType.SPENDING_EVOLUTION, **params)
    
    def get_daily_patterns_aggregation(self, **params) -> Dict[str, Any]:
        """Agrégation pour patterns quotidiens."""
        return DateAggregationTemplates.daily_spending_pattern_template(**params)
    
    def get_weekday_analysis_aggregation(self, **params) -> Dict[str, Any]:
        """Agrégation pour analyse par jour de semaine."""
        return DateAggregationTemplates.weekday_analysis_template(**params)
    
    def get_financial_statistics_aggregation(self, **params) -> Dict[str, Any]:
        """Agrégation pour statistiques financières."""
        return StatisticalAggregationTemplates.financial_statistics_template(**params)
    
    def create_composite_aggregation(
        self,
        aggregation_types: List[Union[AggregationType, str]],
        **common_params
    ) -> Dict[str, Any]:
        """Crée une agrégation composite avec plusieurs types."""
        composite_agg = {}
        
        for agg_type in aggregation_types:
            try:
                # Générer un nom unique pour chaque agrégation
                agg_name = f"{agg_type.value if hasattr(agg_type, 'value') else agg_type}_agg"
                
                # Récupérer le template et l'appliquer
                template_agg = self.get_template(agg_type, **common_params)
                
                # Fusionner dans l'agrégation composite
                composite_agg.update(template_agg)
                
            except Exception as e:
                logger.warning(f"Impossible d'ajouter l'agrégation {agg_type}: {e}")
                continue
        
        return composite_agg
    
    def get_available_templates(self) -> Dict[str, Any]:
        """Retourne la liste des templates disponibles."""
        return {
            template_key: {
                "description": template_info["description"],
                "category": template_info["category"]
            }
            for template_key, template_info in self._templates.items()
        }
    
    def clear_cache(self):
        """Vide le cache."""
        self._cache.clear()
        logger.info("🗑️ Cache des templates d'agrégation vidé")

# ==================== FONCTIONS DE VALIDATION ====================

def validate_aggregation_template(template: Dict[str, Any]) -> bool:
    """Valide un template d'agrégation Elasticsearch."""
    try:
        # Vérification de la structure de base
        if not isinstance(template, dict):
            raise ValueError("Le template doit être un dictionnaire")
        
        # Types d'agrégation supportés
        supported_agg_types = {
            "terms", "range", "date_histogram", "histogram", "sum", "avg", "min", "max",
            "stats", "extended_stats", "percentiles", "percentile_ranks", "cardinality",
            "value_count", "bucket_script", "cumulative_sum", "moving_avg", "derivative",
            "serial_diff", "bucket_sort", "top_hits", "significant_terms", "sampler",
            "diversified_sampler", "composite", "rare_terms", "geo_distance", "ip_range",
            "date_range", "missing", "nested", "reverse_nested", "children", "parent",
            "adjacency_matrix", "auto_date_histogram", "variable_width_histogram"
        }
        
        def validate_agg_node(node: Dict[str, Any], path: str = ""):
            """Valide récursivement un nœud d'agrégation."""
            if not isinstance(node, dict):
                return True
            
            for key, value in node.items():
                current_path = f"{path}.{key}" if path else key
                
                # Vérifier les types d'agrégation
                if key in supported_agg_types:
                    if not isinstance(value, dict):
                        raise ValueError(f"Valeur invalide pour {current_path}: doit être un dict")
                    
                    # Validation spécifique par type
                    if key == "terms":
                        required_fields = ["field"]
                        if not any(req_field in value for req_field in required_fields):
                            if "script" not in value:
                                raise ValueError(f"Terms aggregation manque 'field' ou 'script' à {current_path}")
                    
                    elif key == "range":
                        if "field" not in value or "ranges" not in value:
                            raise ValueError(f"Range aggregation manque 'field' ou 'ranges' à {current_path}")
                    
                    elif key == "date_histogram":
                        if "field" not in value:
                            raise ValueError(f"Date histogram manque 'field' à {current_path}")
                        if not any(interval in value for interval in ["calendar_interval", "fixed_interval", "interval"]):
                            raise ValueError(f"Date histogram manque interval à {current_path}")
                
                # Validation récursive pour sous-agrégations
                elif key == "aggs" or key == "aggregations":
                    if not isinstance(value, dict):
                        raise ValueError(f"Sous-agrégations invalides à {current_path}")
                    
                    for sub_agg_name, sub_agg_def in value.items():
                        validate_agg_node(sub_agg_def, f"{current_path}.{sub_agg_name}")
                
                # Validation récursive pour autres structures
                elif isinstance(value, dict):
                    validate_agg_node(value, current_path)
        
        validate_agg_node(template)
        return True
        
    except Exception as e:
        logger.error(f"Validation template agrégation échouée: {e}")
        return False

def render_aggregation_template(
    template: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Rend un template d'agrégation avec des paramètres."""
    try:
        # Deep copy pour éviter la mutation
        rendered = deepcopy(template)
        
        def replace_placeholders(obj: Any, parameters: Dict[str, Any]) -> Any:
            """Remplace les placeholders dans l'objet."""
            if isinstance(obj, dict):
                return {k: replace_placeholders(v, parameters) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(item, parameters) for item in obj]
            elif isinstance(obj, str):
                # Remplacer les placeholders de type {{param}}
                if obj.startswith("{{") and obj.endswith("}}"):
                    param_name = obj[2:-2].strip()
                    if param_name in parameters:
                        return parameters[param_name]
                    else:
                        raise ValueError(f"Paramètre manquant: {param_name}")
                # Remplacer les placeholders intégrés
                import re
                pattern = r'\{\{([^}]+)\}\}'
                
                def replacer(match):
                    param_name = match.group(1).strip()
                    if param_name in parameters:
                        return str(parameters[param_name])
                    else:
                        raise ValueError(f"Paramètre manquant: {param_name}")
                
                return re.sub(pattern, replacer, obj)
            else:
                return obj
        
        return replace_placeholders(rendered, params)
        
    except Exception as e:
        logger.error(f"Erreur rendu template agrégation: {e}")
        raise

# ==================== TEMPLATES PRÉDÉFINIS ====================

PREDEFINED_AGGREGATION_TEMPLATES = {
    "merchant_top_10": {
        "merchants": {
            "terms": {
                "field": "merchant_name.keyword",
                "size": 10,
                "order": {"total_amount": "desc"}
            },
            "aggs": {
                "total_amount": {"sum": {"field": "amount_abs"}},
                "transaction_count": {"value_count": {"field": "transaction_id"}}
            }
        }
    },
    
    "monthly_spending": {
        "monthly_trends": {
            "date_histogram": {
                "field": "transaction_date",
                "calendar_interval": "1M",
                "min_doc_count": 0
            },
            "aggs": {
                "total_spent": {"sum": {"field": "amount_abs"}},
                "avg_transaction": {"avg": {"field": "amount_abs"}},
                "transaction_count": {"value_count": {"field": "transaction_id"}}
            }
        }
    },
    
    "category_breakdown": {
        "categories": {
            "terms": {
                "field": "category_id",
                "size": 20,
                "order": {"total_spent": "desc"}
            },
            "aggs": {
                "total_spent": {"sum": {"field": "amount_abs"}},
                "avg_spent": {"avg": {"field": "amount_abs"}},
                "category_name": {
                    "terms": {
                        "field": "category_name.keyword",
                        "size": 1
                    }
                }
            }
        }
    },
    
    "spending_stats": {
        "amount_stats": {"stats": {"field": "amount_abs"}},
        "amount_percentiles": {
            "percentiles": {
                "field": "amount_abs",
                "percents": [25, 50, 75, 90, 95, 99]
            }
        }
    },
    
    "weekly_patterns": {
        "weekday_spending": {
            "terms": {
                "script": {
                    "source": "doc['transaction_date'].value.dayOfWeek",
                    "lang": "painless"
                },
                "size": 7,
                "order": {"_key": "asc"}
            },
            "aggs": {
                "daily_total": {"sum": {"field": "amount_abs"}},
                "daily_avg": {"avg": {"field": "amount_abs"}},
                "transaction_count": {"value_count": {"field": "transaction_id"}}
            }
        }
    },
    
    "amount_ranges": {
        "amount_distribution": {
            "range": {
                "field": "amount_abs",
                "ranges": AMOUNT_BUCKETS
            },
            "aggs": {
                "range_total": {"sum": {"field": "amount_abs"}},
                "range_avg": {"avg": {"field": "amount_abs"}}
            }
        }
    }
}

# ==================== FONCTIONS UTILITAIRES ====================

def get_aggregation_template_by_name(template_name: str) -> Dict[str, Any]:
    """Récupère un template d'agrégation prédéfini par nom."""
    if template_name not in PREDEFINED_AGGREGATION_TEMPLATES:
        raise ValueError(f"Template d'agrégation '{template_name}' non trouvé")
    
    return deepcopy(PREDEFINED_AGGREGATION_TEMPLATES[template_name])

def list_available_aggregation_templates() -> List[str]:
    """Liste tous les templates d'agrégation disponibles."""
    return list(PREDEFINED_AGGREGATION_TEMPLATES.keys())

def create_multi_level_aggregation(
    primary_field: str,
    secondary_field: str,
    metric_field: str = "amount_abs",
    primary_size: int = 10,
    secondary_size: int = 5
) -> Dict[str, Any]:
    """Crée une agrégation à plusieurs niveaux."""
    return {
        "multi_level": {
            "terms": {
                "field": primary_field,
                "size": primary_size,
                "order": {"total_amount": "desc"}
            },
            "aggs": {
                "total_amount": {"sum": {"field": metric_field}},
                "avg_amount": {"avg": {"field": metric_field}},
                "sub_breakdown": {
                    "terms": {
                        "field": secondary_field,
                        "size": secondary_size,
                        "order": {"sub_total": "desc"}
                    },
                    "aggs": {
                        "sub_total": {"sum": {"field": metric_field}},
                        "sub_avg": {"avg": {"field": metric_field}}
                    }
                }
            }
        }
    }

def create_time_series_aggregation(
    interval: str = "1M",
    metric_fields: List[str] = None,
    include_moving_avg: bool = True,
    window_size: int = 3
) -> Dict[str, Any]:
    """Crée une agrégation de série temporelle."""
    if not metric_fields:
        metric_fields = ["amount_abs"]
    
    aggs = {
        "time_series": {
            "date_histogram": {
                "field": "transaction_date",
                "calendar_interval": interval,
                "min_doc_count": 0
            },
            "aggs": {}
        }
    }
    
    # Ajouter les métriques pour chaque champ
    for field in metric_fields:
        field_name = field.replace(".", "_")
        aggs["time_series"]["aggs"][f"{field_name}_sum"] = {"sum": {"field": field}}
        aggs["time_series"]["aggs"][f"{field_name}_avg"] = {"avg": {"field": field}}
        
        if include_moving_avg:
            aggs["time_series"]["aggs"][f"{field_name}_moving_avg"] = {
                "moving_avg": {
                    "buckets_path": f"{field_name}_sum",
                    "window": window_size,
                    "model": "linear"
                }
            }
    
    return aggs

def optimize_aggregation_for_performance(
    aggregation: Dict[str, Any],
    max_buckets: int = 1000,
    enable_execution_hints: bool = True
) -> Dict[str, Any]:
    """Optimise une agrégation pour les performances."""
    optimized = deepcopy(aggregation)
    
    def optimize_node(node: Dict[str, Any]):
        """Optimise récursivement un nœud d'agrégation."""
        if not isinstance(node, dict):
            return
        
        for key, value in node.items():
            if isinstance(value, dict):
                # Optimisations pour terms aggregations
                if key == "terms":
                    if "size" not in value or value["size"] > max_buckets:
                        value["size"] = min(value.get("size", 10), max_buckets)
                    
                    if enable_execution_hints:
                        value["execution_hint"] = AGGREGATION_CONFIG["execution_hint"]
                        value["collect_mode"] = AGGREGATION_CONFIG["collect_mode"]
                
                # Optimisations pour cardinality
                elif key == "cardinality":
                    if "precision_threshold" not in value:
                        value["precision_threshold"] = AGGREGATION_CONFIG["precision_threshold"]
                
                # Optimisation récursive
                optimize_node(value)
    
    optimize_node(optimized)
    return optimized

# ==================== EXPORTS ====================

__all__ = [
    # Classes principales
    "AggregationTemplateManager",
    "FinancialAggregationTemplates",
    "DateAggregationTemplates", 
    "StatisticalAggregationTemplates",
    
    # Fonctions de validation et rendu
    "validate_aggregation_template",
    "render_aggregation_template",
    
    # Utilitaires
    "get_aggregation_template_by_name",
    "list_available_aggregation_templates",
    "create_multi_level_aggregation",
    "create_time_series_aggregation",
    "optimize_aggregation_for_performance",
    
    # Configuration
    "AGGREGATION_CONFIG",
    "AMOUNT_BUCKETS",
    "TIME_INTERVALS",
    "PREDEFINED_AGGREGATION_TEMPLATES"
]