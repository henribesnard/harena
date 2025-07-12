"""
Templates d'agrégation Elasticsearch pour analyses financières
Bibliothèque spécialisée pour agrégations temporelles, catégorielles et métriques
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from models.elasticsearch_queries import (
    ESSearchQuery, ESAggregationType, ESTermsAggregation, ESMetricAggregation,
    ESDateHistogramAggregation, ESAggregationContainer, FinancialTransactionQueryBuilder
)
from models.service_contracts import AggregationRequest, AggregationType
from config import settings


logger = logging.getLogger(__name__)


class AggregationIntent(str, Enum):
    """Types d'intentions d'agrégation financière"""
    # === AGRÉGATIONS TEMPORELLES ===
    SPENDING_OVER_TIME = "spending_over_time"
    MONTHLY_BREAKDOWN = "monthly_breakdown"
    WEEKLY_PATTERN = "weekly_pattern"
    DAILY_TREND = "daily_trend"
    QUARTERLY_ANALYSIS = "quarterly_analysis"
    YEARLY_COMPARISON = "yearly_comparison"
    SEASONAL_ANALYSIS = "seasonal_analysis"
    
    # === AGRÉGATIONS CATÉGORIELLES ===
    SPENDING_BY_CATEGORY = "spending_by_category"
    TOP_CATEGORIES = "top_categories"
    CATEGORY_DISTRIBUTION = "category_distribution"
    CATEGORY_TRENDS = "category_trends"
    CATEGORY_COMPARISON = "category_comparison"
    
    # === AGRÉGATIONS MARCHANDS ===
    TOP_MERCHANTS = "top_merchants"
    MERCHANT_SPENDING = "merchant_spending"
    MERCHANT_FREQUENCY = "merchant_frequency"
    MERCHANT_TRENDS = "merchant_trends"
    NEW_MERCHANTS = "new_merchants"
    
    # === AGRÉGATIONS MÉTRIQUES ===
    SPENDING_STATISTICS = "spending_statistics"
    TRANSACTION_PATTERNS = "transaction_patterns"
    AMOUNT_DISTRIBUTION = "amount_distribution"
    PAYMENT_TYPE_ANALYSIS = "payment_type_analysis"
    CURRENCY_BREAKDOWN = "currency_breakdown"
    
    # === AGRÉGATIONS COMPARATIVES ===
    PERIOD_COMPARISON = "period_comparison"
    BUDGET_ANALYSIS = "budget_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    GROWTH_ANALYSIS = "growth_analysis"
    
    # === AGRÉGATIONS COMPLEXES ===
    COHORT_ANALYSIS = "cohort_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    PREDICTIVE_METRICS = "predictive_metrics"


class AggregationComplexity(str, Enum):
    """Niveaux de complexité des agrégations"""
    SIMPLE = "simple"          # 1 niveau, ~10-20ms
    MODERATE = "moderate"      # 2 niveaux, ~30-50ms  
    COMPLEX = "complex"        # 3+ niveaux, ~50-100ms
    ADVANCED = "advanced"      # Nested/pipeline, ~100-200ms


class AggregationDimension(str, Enum):
    """Dimensions d'agrégation disponibles"""
    # Dimensions temporelles
    DATE = "date"
    MONTH_YEAR = "month_year"
    WEEKDAY = "weekday"
    QUARTER = "quarter"
    YEAR = "year"
    
    # Dimensions catégorielles
    CATEGORY = "category_name.keyword"
    MERCHANT = "merchant_name.keyword"
    TRANSACTION_TYPE = "transaction_type"
    OPERATION_TYPE = "operation_type.keyword"
    CURRENCY = "currency_code"
    
    # Dimensions métriques
    AMOUNT_RANGE = "amount_range"
    FREQUENCY_TIER = "frequency_tier"


class AggregationMetric(str, Enum):
    """Métriques calculables"""
    # Métriques financières
    TOTAL_AMOUNT = "amount_abs"
    NET_AMOUNT = "amount"
    AVERAGE_AMOUNT = "amount_abs"
    MIN_AMOUNT = "amount_abs"
    MAX_AMOUNT = "amount_abs"
    
    # Métriques de comptage
    TRANSACTION_COUNT = "transaction_id"
    UNIQUE_MERCHANTS = "merchant_name.keyword"
    UNIQUE_CATEGORIES = "category_name.keyword"
    
    # Métriques temporelles
    DAYS_ACTIVE = "date"
    MONTHS_ACTIVE = "month_year"


@dataclass
class AggregationTemplate:
    """Template d'agrégation avec métadonnées"""
    intent: AggregationIntent
    name: str
    description: str
    complexity: AggregationComplexity
    dimensions: List[AggregationDimension]
    metrics: List[AggregationMetric]
    estimated_time_ms: int
    cache_duration_minutes: int
    elasticsearch_aggs: Dict[str, Any]
    sub_aggregations: Optional[Dict[str, Any]] = None
    filters_required: List[str] = None
    min_doc_count: int = 1
    size_limit: int = 100


class FinancialAggregationEngine:
    """Moteur d'agrégations financières avec templates spécialisés"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.dimension_configs = self._initialize_dimension_configs()
        self.metric_configs = self._initialize_metric_configs()
    
    def _initialize_templates(self) -> Dict[AggregationIntent, AggregationTemplate]:
        """Initialise tous les templates d'agrégation"""
        templates = {}
        
        # === TEMPLATES TEMPORELS ===
        templates[AggregationIntent.MONTHLY_BREAKDOWN] = AggregationTemplate(
            intent=AggregationIntent.MONTHLY_BREAKDOWN,
            name="monthly_spending_breakdown",
            description="Répartition des dépenses par mois",
            complexity=AggregationComplexity.SIMPLE,
            dimensions=[AggregationDimension.MONTH_YEAR],
            metrics=[AggregationMetric.TOTAL_AMOUNT, AggregationMetric.TRANSACTION_COUNT],
            estimated_time_ms=25,
            cache_duration_minutes=60,
            elasticsearch_aggs={
                "monthly_spending": {
                    "terms": {
                        "field": "month_year",
                        "size": 24,
                        "order": {"_key": "desc"}
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}},
                        "avg_amount": {"avg": {"field": "amount_abs"}}
                    }
                }
            }
        )
        
        templates[AggregationIntent.SPENDING_OVER_TIME] = AggregationTemplate(
            intent=AggregationIntent.SPENDING_OVER_TIME,
            name="spending_time_series",
            description="Évolution des dépenses dans le temps",
            complexity=AggregationComplexity.MODERATE,
            dimensions=[AggregationDimension.DATE],
            metrics=[AggregationMetric.TOTAL_AMOUNT, AggregationMetric.TRANSACTION_COUNT],
            estimated_time_ms=45,
            cache_duration_minutes=30,
            elasticsearch_aggs={
                "spending_over_time": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "month",
                        "format": "yyyy-MM",
                        "min_doc_count": 1
                    },
                    "aggs": {
                        "total_spending": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}},
                        "daily_avg": {
                            "avg": {"field": "amount_abs"}
                        },
                        "spending_trend": {
                            "derivative": {
                                "buckets_path": "total_spending"
                            }
                        }
                    }
                }
            }
        )
        
        templates[AggregationIntent.WEEKLY_PATTERN] = AggregationTemplate(
            intent=AggregationIntent.WEEKLY_PATTERN,
            name="weekly_spending_pattern",
            description="Patterns de dépenses par jour de la semaine",
            complexity=AggregationComplexity.SIMPLE,
            dimensions=[AggregationDimension.WEEKDAY],
            metrics=[AggregationMetric.TOTAL_AMOUNT, AggregationMetric.TRANSACTION_COUNT],
            estimated_time_ms=20,
            cache_duration_minutes=120,
            elasticsearch_aggs={
                "weekly_pattern": {
                    "terms": {
                        "field": "weekday",
                        "size": 7,
                        "order": {"avg_amount": "desc"}
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount_abs"}},
                        "avg_amount": {"avg": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                }
            }
        )
        
        # === TEMPLATES CATÉGORIELS ===
        templates[AggregationIntent.SPENDING_BY_CATEGORY] = AggregationTemplate(
            intent=AggregationIntent.SPENDING_BY_CATEGORY,
            name="category_spending_analysis",
            description="Analyse des dépenses par catégorie",
            complexity=AggregationComplexity.MODERATE,
            dimensions=[AggregationDimension.CATEGORY],
            metrics=[AggregationMetric.TOTAL_AMOUNT, AggregationMetric.TRANSACTION_COUNT],
            estimated_time_ms=35,
            cache_duration_minutes=45,
            elasticsearch_aggs={
                "spending_by_category": {
                    "terms": {
                        "field": "category_name.keyword",
                        "size": 20,
                        "order": {"total_spending": "desc"}
                    },
                    "aggs": {
                        "total_spending": {"sum": {"field": "amount_abs"}},
                        "avg_spending": {"avg": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}},
                        "spending_percentage": {
                            "bucket_script": {
                                "buckets_path": {
                                    "my_spending": "total_spending"
                                },
                                "script": "params.my_spending / params.total_spending * 100"
                            }
                        }
                    }
                }
            }
        )
        
        templates[AggregationIntent.TOP_CATEGORIES] = AggregationTemplate(
            intent=AggregationIntent.TOP_CATEGORIES,
            name="top_spending_categories",
            description="Top catégories par volume de dépenses",
            complexity=AggregationComplexity.SIMPLE,
            dimensions=[AggregationDimension.CATEGORY],
            metrics=[AggregationMetric.TOTAL_AMOUNT],
            estimated_time_ms=18,
            cache_duration_minutes=60,
            elasticsearch_aggs={
                "top_categories": {
                    "terms": {
                        "field": "category_name.keyword",
                        "size": 10,
                        "order": {"total_amount": "desc"}
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                }
            }
        )
        
        # === TEMPLATES MARCHANDS ===
        templates[AggregationIntent.TOP_MERCHANTS] = AggregationTemplate(
            intent=AggregationIntent.TOP_MERCHANTS,
            name="top_merchants_analysis",
            description="Analyse des top marchands par dépenses",
            complexity=AggregationComplexity.MODERATE,
            dimensions=[AggregationDimension.MERCHANT],
            metrics=[AggregationMetric.TOTAL_AMOUNT, AggregationMetric.TRANSACTION_COUNT],
            estimated_time_ms=40,
            cache_duration_minutes=30,
            elasticsearch_aggs={
                "top_merchants": {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": 15,
                        "order": {"total_spending": "desc"}
                    },
                    "aggs": {
                        "total_spending": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}},
                        "avg_transaction": {"avg": {"field": "amount_abs"}},
                        "first_transaction": {"min": {"field": "date"}},
                        "last_transaction": {"max": {"field": "date"}},
                        "spending_frequency": {
                            "bucket_script": {
                                "buckets_path": {
                                    "transactions": "transaction_count"
                                },
                                "script": "params.transactions / 30.0"  # Par mois approximatif
                            }
                        }
                    }
                }
            }
        )
        
        templates[AggregationIntent.MERCHANT_FREQUENCY] = AggregationTemplate(
            intent=AggregationIntent.MERCHANT_FREQUENCY,
            name="merchant_transaction_frequency",
            description="Fréquence de transactions par marchand",
            complexity=AggregationComplexity.SIMPLE,
            dimensions=[AggregationDimension.MERCHANT],
            metrics=[AggregationMetric.TRANSACTION_COUNT],
            estimated_time_ms=22,
            cache_duration_minutes=90,
            elasticsearch_aggs={
                "merchant_frequency": {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": 20,
                        "order": {"transaction_count": "desc"}
                    },
                    "aggs": {
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                }
            }
        )
        
        # === TEMPLATES MÉTRIQUES ===
        templates[AggregationIntent.SPENDING_STATISTICS] = AggregationTemplate(
            intent=AggregationIntent.SPENDING_STATISTICS,
            name="comprehensive_spending_stats",
            description="Statistiques complètes des dépenses",
            complexity=AggregationComplexity.SIMPLE,
            dimensions=[],
            metrics=[AggregationMetric.TOTAL_AMOUNT, AggregationMetric.AVERAGE_AMOUNT],
            estimated_time_ms=15,
            cache_duration_minutes=120,
            elasticsearch_aggs={
                "spending_stats": {
                    "stats": {"field": "amount_abs"}
                },
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                },
                "extended_stats": {
                    "extended_stats": {"field": "amount_abs"}
                },
                "percentiles": {
                    "percentiles": {
                        "field": "amount_abs",
                        "percents": [25, 50, 75, 90, 95, 99]
                    }
                }
            }
        )
        
        templates[AggregationIntent.AMOUNT_DISTRIBUTION] = AggregationTemplate(
            intent=AggregationIntent.AMOUNT_DISTRIBUTION,
            name="transaction_amount_distribution",
            description="Distribution des montants de transactions",
            complexity=AggregationComplexity.MODERATE,
            dimensions=[AggregationDimension.AMOUNT_RANGE],
            metrics=[AggregationMetric.TRANSACTION_COUNT],
            estimated_time_ms=30,
            cache_duration_minutes=60,
            elasticsearch_aggs={
                "amount_ranges": {
                    "range": {
                        "field": "amount_abs",
                        "ranges": [
                            {"to": 10, "key": "0-10"},
                            {"from": 10, "to": 25, "key": "10-25"},
                            {"from": 25, "to": 50, "key": "25-50"},
                            {"from": 50, "to": 100, "key": "50-100"},
                            {"from": 100, "to": 250, "key": "100-250"},
                            {"from": 250, "to": 500, "key": "250-500"},
                            {"from": 500, "key": "500+"}
                        ]
                    },
                    "aggs": {
                        "transaction_count": {"value_count": {"field": "transaction_id"}},
                        "total_amount": {"sum": {"field": "amount_abs"}}
                    }
                }
            }
        )
        
        # === TEMPLATES COMPARATIFS ===
        templates[AggregationIntent.PERIOD_COMPARISON] = AggregationTemplate(
            intent=AggregationIntent.PERIOD_COMPARISON,
            name="period_over_period_comparison",
            description="Comparaison entre périodes",
            complexity=AggregationComplexity.COMPLEX,
            dimensions=[AggregationDimension.MONTH_YEAR],
            metrics=[AggregationMetric.TOTAL_AMOUNT, AggregationMetric.TRANSACTION_COUNT],
            estimated_time_ms=65,
            cache_duration_minutes=30,
            elasticsearch_aggs={
                "current_period": {
                    "filter": {
                        "range": {
                            "date": {
                                "gte": "now-1M/M",
                                "lte": "now/M"
                            }
                        }
                    },
                    "aggs": {
                        "total_spending": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                },
                "previous_period": {
                    "filter": {
                        "range": {
                            "date": {
                                "gte": "now-2M/M",
                                "lte": "now-1M/M"
                            }
                        }
                    },
                    "aggs": {
                        "total_spending": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                },
                "growth_rate": {
                    "bucket_script": {
                        "buckets_path": {
                            "current": "current_period>total_spending",
                            "previous": "previous_period>total_spending"
                        },
                        "script": "(params.current - params.previous) / params.previous * 100"
                    }
                }
            }
        )
        
        # === TEMPLATES AVANCÉS ===
        templates[AggregationIntent.CATEGORY_TRENDS] = AggregationTemplate(
            intent=AggregationIntent.CATEGORY_TRENDS,
            name="category_spending_trends",
            description="Tendances de dépenses par catégorie dans le temps",
            complexity=AggregationComplexity.COMPLEX,
            dimensions=[AggregationDimension.CATEGORY, AggregationDimension.MONTH_YEAR],
            metrics=[AggregationMetric.TOTAL_AMOUNT],
            estimated_time_ms=85,
            cache_duration_minutes=45,
            elasticsearch_aggs={
                "categories": {
                    "terms": {
                        "field": "category_name.keyword",
                        "size": 10
                    },
                    "aggs": {
                        "monthly_trend": {
                            "date_histogram": {
                                "field": "date",
                                "calendar_interval": "month",
                                "format": "yyyy-MM"
                            },
                            "aggs": {
                                "monthly_spending": {"sum": {"field": "amount_abs"}},
                                "trend": {
                                    "derivative": {
                                        "buckets_path": "monthly_spending"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        )
        
        return templates
    
    def _initialize_dimension_configs(self) -> Dict[AggregationDimension, Dict[str, Any]]:
        """Configuration des dimensions d'agrégation"""
        return {
            AggregationDimension.MONTH_YEAR: {
                "field": "month_year",
                "type": "terms",
                "default_size": 24,
                "sort_order": "_key"
            },
            AggregationDimension.CATEGORY: {
                "field": "category_name.keyword",
                "type": "terms",
                "default_size": 20,
                "sort_order": "total_amount"
            },
            AggregationDimension.MERCHANT: {
                "field": "merchant_name.keyword",
                "type": "terms",
                "default_size": 15,
                "sort_order": "total_amount"
            },
            AggregationDimension.WEEKDAY: {
                "field": "weekday",
                "type": "terms",
                "default_size": 7,
                "sort_order": "_key"
            },
            AggregationDimension.DATE: {
                "field": "date",
                "type": "date_histogram",
                "default_interval": "month",
                "format": "yyyy-MM"
            }
        }
    
    def _initialize_metric_configs(self) -> Dict[AggregationMetric, Dict[str, Any]]:
        """Configuration des métriques"""
        return {
            AggregationMetric.TOTAL_AMOUNT: {
                "agg_type": "sum",
                "field": "amount_abs",
                "name": "total_amount"
            },
            AggregationMetric.AVERAGE_AMOUNT: {
                "agg_type": "avg",
                "field": "amount_abs",
                "name": "avg_amount"
            },
            AggregationMetric.TRANSACTION_COUNT: {
                "agg_type": "value_count",
                "field": "transaction_id",
                "name": "transaction_count"
            },
            AggregationMetric.MIN_AMOUNT: {
                "agg_type": "min",
                "field": "amount_abs",
                "name": "min_amount"
            },
            AggregationMetric.MAX_AMOUNT: {
                "agg_type": "max",
                "field": "amount_abs",
                "name": "max_amount"
            }
        }
    
    def get_template(self, intent: AggregationIntent) -> Optional[AggregationTemplate]:
        """Récupère un template par intention"""
        return self.templates.get(intent)
    
    def list_templates(self) -> List[AggregationTemplate]:
        """Liste tous les templates disponibles"""
        return list(self.templates.values())
    
    def get_templates_by_complexity(self, complexity: AggregationComplexity) -> List[AggregationTemplate]:
        """Récupère les templates par niveau de complexité"""
        return [t for t in self.templates.values() if t.complexity == complexity]


class AggregationQueryBuilder:
    """Builder pour construire des requêtes d'agrégation complexes"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.base_query = FinancialTransactionQueryBuilder().add_user_filter(user_id)
        self.aggregations = {}
        self.filters_applied = []
        
    def apply_date_filter(self, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> 'AggregationQueryBuilder':
        """Applique un filtre de date"""
        if start_date or end_date:
            self.base_query.add_date_range_filter(start_date, end_date)
            self.filters_applied.append(f"date_range_{start_date}_{end_date}")
        return self
    
    def apply_category_filter(self, categories: List[str]) -> 'AggregationQueryBuilder':
        """Applique un filtre de catégories"""
        for category in categories:
            self.base_query.add_category_filter(category)
            self.filters_applied.append(f"category_{category}")
        return self
    
    def apply_amount_filter(self, min_amount: Optional[float] = None,
                           max_amount: Optional[float] = None) -> 'AggregationQueryBuilder':
        """Applique un filtre de montant"""
        if min_amount is not None or max_amount is not None:
            self.base_query.add_amount_range_filter(min_amount, max_amount)
            self.filters_applied.append(f"amount_{min_amount}_{max_amount}")
        return self
    
    def add_template_aggregation(self, template: AggregationTemplate) -> 'AggregationQueryBuilder':
        """Ajoute une agrégation basée sur un template"""
        for agg_name, agg_config in template.elasticsearch_aggs.items():
            self.aggregations[agg_name] = agg_config
        return self
    
    def add_custom_aggregation(self, name: str, config: Dict[str, Any]) -> 'AggregationQueryBuilder':
        """Ajoute une agrégation personnalisée"""
        self.aggregations[name] = config
        return self
    
    def build(self) -> ESSearchQuery:
        """Construit la requête finale"""
        query = self.base_query.build()
        query.size = 0  # Pas de documents, seulement agrégations
        
        if self.aggregations:
            query.aggs = self.aggregations
        
        return query
    
    def get_cache_key(self) -> str:
        """Génère une clé de cache basée sur les filtres appliqués"""
        filter_signature = "_".join(sorted(self.filters_applied))
        agg_signature = "_".join(sorted(self.aggregations.keys()))
        return f"agg_{self.user_id}_{filter_signature}_{agg_signature}"


class AggregationResultProcessor:
    """Processeur pour les résultats d'agrégation Elasticsearch"""
    
    @staticmethod
    def process_monthly_breakdown(es_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les résultats d'une répartition mensuelle"""
        if "monthly_spending" not in es_result.get("aggregations", {}):
            return {"months": [], "total": 0, "error": "No monthly data found"}
        
        monthly_agg = es_result["aggregations"]["monthly_spending"]
        months = []
        
        for bucket in monthly_agg["buckets"]:
            months.append({
                "month": bucket["key"],
                "total_amount": bucket["total_amount"]["value"],
                "transaction_count": bucket["transaction_count"]["value"],
                "average_amount": bucket["avg_amount"]["value"]
            })
        
        total_amount = sum(month["total_amount"] for month in months)
        total_transactions = sum(month["transaction_count"] for month in months)
        
        return {
            "months": months,
            "summary": {
                "total_amount": total_amount,
                "total_transactions": total_transactions,
                "average_monthly": total_amount / len(months) if months else 0,
                "months_analyzed": len(months)
            }
        }
    
    @staticmethod
    def process_category_spending(es_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les résultats d'analyse par catégorie"""
        if "spending_by_category" not in es_result.get("aggregations", {}):
            return {"categories": [], "error": "No category data found"}
        
        category_agg = es_result["aggregations"]["spending_by_category"]
        categories = []
        
        for bucket in category_agg["buckets"]:
            categories.append({
                "category": bucket["key"],
                "total_spending": bucket["total_spending"]["value"],
                "average_spending": bucket["avg_spending"]["value"],
                "transaction_count": bucket["transaction_count"]["value"],
                "percentage": (bucket["total_spending"]["value"] / 
                             sum(b["total_spending"]["value"] for b in category_agg["buckets"]) * 100)
            })
        
        # Tri par montant total décroissant
        categories.sort(key=lambda x: x["total_spending"], reverse=True)
        
        total_spending = sum(cat["total_spending"] for cat in categories)
        
        return {
            "categories": categories,
            "summary": {
                "total_spending": total_spending,
                "categories_count": len(categories),
                "top_category": categories[0] if categories else None,
                "spending_concentration": categories[0]["percentage"] if categories else 0
            }
        }
    
    @staticmethod
    def process_merchant_analysis(es_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les résultats d'analyse des marchands"""
        if "top_merchants" not in es_result.get("aggregations", {}):
            return {"merchants": [], "error": "No merchant data found"}
        
        merchant_agg = es_result["aggregations"]["top_merchants"]
        merchants = []
        
        for bucket in merchant_agg["buckets"]:
            merchants.append({
                "merchant": bucket["key"],
                "total_spending": bucket["total_spending"]["value"],
                "transaction_count": bucket["transaction_count"]["value"],
                "average_transaction": bucket["avg_transaction"]["value"],
                "first_transaction": bucket["first_transaction"]["value_as_string"],
                "last_transaction": bucket["last_transaction"]["value_as_string"],
                "monthly_frequency": bucket.get("spending_frequency", {}).get("value", 0)
            })
        
        return {
            "merchants": merchants,
            "summary": {
                "total_merchants": len(merchants),
                "total_spending": sum(m["total_spending"] for m in merchants),
                "most_frequent": max(merchants, key=lambda x: x["transaction_count"]) if merchants else None,
                "highest_spending": max(merchants, key=lambda x: x["total_spending"]) if merchants else None
            }
        }
    
    @staticmethod
    def process_spending_statistics(es_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les résultats de statistiques de dépenses"""
        aggregations = es_result.get("aggregations", {})
        
        if "spending_stats" not in aggregations:
            return {"error": "No spending statistics found"}
        
        stats = aggregations["spending_stats"]
        extended_stats = aggregations.get("extended_stats", {})
        percentiles = aggregations.get("percentiles", {})
        transaction_count = aggregations.get("transaction_count", {}).get("value", 0)
        
        return {
            "basic_statistics": {
                "total_amount": stats.get("sum", 0),
                "average_amount": stats.get("avg", 0),
                "min_amount": stats.get("min", 0),
                "max_amount": stats.get("max", 0),
                "transaction_count": transaction_count
            },
            "extended_statistics": {
                "variance": extended_stats.get("variance", 0),
                "std_deviation": extended_stats.get("std_deviation", 0),
                "sum_of_squares": extended_stats.get("sum_of_squares", 0)
            },
            "percentiles": {
                f"p{int(k)}": v for k, v in percentiles.get("values", {}).items()
            },
            "insights": {
                "spending_volatility": "high" if extended_stats.get("std_deviation", 0) > stats.get("avg", 0) else "low",
                "outlier_threshold": percentiles.get("values", {}).get("95.0", 0),
                "typical_range": {
                    "low": percentiles.get("values", {}).get("25.0", 0),
                    "high": percentiles.get("values", {}).get("75.0", 0)
                }
            }
        }
    
    @staticmethod
    def process_period_comparison(es_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les résultats de comparaison entre périodes"""
        aggregations = es_result.get("aggregations", {})
        
        current = aggregations.get("current_period", {})
        previous = aggregations.get("previous_period", {})
        growth_rate = aggregations.get("growth_rate", {}).get("value")
        
        current_spending = current.get("total_spending", {}).get("value", 0)
        current_transactions = current.get("transaction_count", {}).get("value", 0)
        
        previous_spending = previous.get("total_spending", {}).get("value", 0)
        previous_transactions = previous.get("transaction_count", {}).get("value", 0)
        
        return {
            "current_period": {
                "total_spending": current_spending,
                "transaction_count": current_transactions,
                "average_transaction": current_spending / current_transactions if current_transactions > 0 else 0
            },
            "previous_period": {
                "total_spending": previous_spending,
                "transaction_count": previous_transactions,
                "average_transaction": previous_spending / previous_transactions if previous_transactions > 0 else 0
            },
            "comparison": {
                "spending_change": current_spending - previous_spending,
                "spending_change_percent": growth_rate if growth_rate is not None else 0,
                "transaction_change": current_transactions - previous_transactions,
                "transaction_change_percent": (
                    (current_transactions - previous_transactions) / previous_transactions * 100
                    if previous_transactions > 0 else 0
                ),
                "trend": "increasing" if current_spending > previous_spending else "decreasing"
            }
        }
    
    @staticmethod
    def process_amount_distribution(es_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les résultats de distribution des montants"""
        if "amount_ranges" not in es_result.get("aggregations", {}):
            return {"ranges": [], "error": "No amount distribution data found"}
        
        range_agg = es_result["aggregations"]["amount_ranges"]
        ranges = []
        total_transactions = 0
        total_amount = 0
        
        for bucket in range_agg["buckets"]:
            range_transactions = bucket["transaction_count"]["value"]
            range_amount = bucket["total_amount"]["value"]
            
            ranges.append({
                "range": bucket["key"],
                "from_amount": bucket.get("from", 0),
                "to_amount": bucket.get("to", float('inf')),
                "transaction_count": range_transactions,
                "total_amount": range_amount,
                "average_amount": range_amount / range_transactions if range_transactions > 0 else 0
            })
            
            total_transactions += range_transactions
            total_amount += range_amount
        
        # Calculer les pourcentages
        for range_data in ranges:
            range_data["transaction_percentage"] = (
                range_data["transaction_count"] / total_transactions * 100 
                if total_transactions > 0 else 0
            )
            range_data["amount_percentage"] = (
                range_data["total_amount"] / total_amount * 100 
                if total_amount > 0 else 0
            )
        
        return {
            "ranges": ranges,
            "summary": {
                "total_transactions": total_transactions,
                "total_amount": total_amount,
                "most_common_range": max(ranges, key=lambda x: x["transaction_count"]) if ranges else None,
                "highest_value_range": max(ranges, key=lambda x: x["total_amount"]) if ranges else None
            }
        }


class AggregationComposer:
    """Compositeur pour créer des agrégations multi-dimensionnelles complexes"""
    
    def __init__(self, engine: FinancialAggregationEngine):
        self.engine = engine
        
    def compose_temporal_category_analysis(self, user_id: int, 
                                         date_range: Optional[Tuple[str, str]] = None) -> ESSearchQuery:
        """Compose une analyse temporelle par catégorie"""
        builder = AggregationQueryBuilder(user_id)
        
        if date_range:
            builder.apply_date_filter(date_range[0], date_range[1])
        
        # Agrégation composite: catégories -> temps
        composite_agg = {
            "category_temporal_analysis": {
                "terms": {
                    "field": "category_name.keyword",
                    "size": 15
                },
                "aggs": {
                    "monthly_trend": {
                        "date_histogram": {
                            "field": "date",
                            "calendar_interval": "month",
                            "format": "yyyy-MM"
                        },
                        "aggs": {
                            "monthly_spending": {"sum": {"field": "amount_abs"}},
                            "transaction_count": {"value_count": {"field": "transaction_id"}}
                        }
                    },
                    "total_category_spending": {"sum": {"field": "amount_abs"}},
                    "avg_monthly": {
                        "avg_bucket": {
                            "buckets_path": "monthly_trend>monthly_spending"
                        }
                    }
                }
            }
        }
        
        builder.add_custom_aggregation("category_temporal_analysis", composite_agg["category_temporal_analysis"])
        return builder.build()
    
    def compose_merchant_category_cross_analysis(self, user_id: int, 
                                                top_merchants: int = 10,
                                                top_categories: int = 10) -> ESSearchQuery:
        """Compose une analyse croisée marchands-catégories"""
        builder = AggregationQueryBuilder(user_id)
        
        # Multi-agrégation pour voir les relations marchands-catégories
        cross_analysis = {
            "merchants": {
                "terms": {
                    "field": "merchant_name.keyword",
                    "size": top_merchants
                },
                "aggs": {
                    "categories": {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": 5  # Top 5 catégories par marchand
                        },
                        "aggs": {
                            "spending": {"sum": {"field": "amount_abs"}},
                            "transactions": {"value_count": {"field": "transaction_id"}}
                        }
                    },
                    "total_merchant_spending": {"sum": {"field": "amount_abs"}}
                }
            },
            "categories": {
                "terms": {
                    "field": "category_name.keyword",
                    "size": top_categories
                },
                "aggs": {
                    "top_merchants": {
                        "terms": {
                            "field": "merchant_name.keyword",
                            "size": 5  # Top 5 marchands par catégorie
                        },
                        "aggs": {
                            "spending": {"sum": {"field": "amount_abs"}},
                            "transactions": {"value_count": {"field": "transaction_id"}}
                        }
                    },
                    "total_category_spending": {"sum": {"field": "amount_abs"}}
                }
            }
        }
        
        for agg_name, agg_config in cross_analysis.items():
            builder.add_custom_aggregation(agg_name, agg_config)
        
        return builder.build()
    
    def compose_spending_velocity_analysis(self, user_id: int) -> ESSearchQuery:
        """Compose une analyse de vélocité des dépenses"""
        builder = AggregationQueryBuilder(user_id)
        
        velocity_agg = {
            "spending_velocity": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "week",
                    "format": "yyyy-'W'ww"
                },
                "aggs": {
                    "weekly_spending": {"sum": {"field": "amount_abs"}},
                    "transaction_count": {"value_count": {"field": "transaction_id"}},
                    "unique_merchants": {
                        "cardinality": {"field": "merchant_name.keyword"}
                    },
                    "unique_categories": {
                        "cardinality": {"field": "category_name.keyword"}
                    },
                    "spending_acceleration": {
                        "derivative": {
                            "buckets_path": "weekly_spending"
                        }
                    },
                    "transaction_velocity": {
                        "bucket_script": {
                            "buckets_path": {
                                "transactions": "transaction_count"
                            },
                            "script": "params.transactions / 7.0"  # Transactions par jour
                        }
                    }
                }
            }
        }
        
        builder.add_custom_aggregation("spending_velocity", velocity_agg["spending_velocity"])
        return builder.build()


class AggregationCache:
    """Gestionnaire de cache pour les résultats d'agrégation"""
    
    def __init__(self):
        self.cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        self.max_cache_size = 1000
    
    def get_cache_key(self, user_id: int, template_name: str, 
                     filters: Dict[str, Any] = None) -> str:
        """Génère une clé de cache unique"""
        filter_str = ""
        if filters:
            filter_items = sorted(filters.items())
            filter_str = "_".join(f"{k}:{v}" for k, v in filter_items)
        
        return f"agg_{user_id}_{template_name}_{filter_str}"
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupère un résultat du cache"""
        if cache_key in self.cache:
            result, timestamp, ttl = self.cache[cache_key]
            if datetime.now().timestamp() - timestamp < ttl:
                self.cache_stats["hits"] += 1
                return result
            else:
                # Expiré, supprimer
                del self.cache[cache_key]
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, cache_key: str, result: Dict[str, Any], ttl_minutes: int = 30):
        """Met en cache un résultat"""
        if len(self.cache) >= self.max_cache_size:
            # Éviction LRU simple: supprimer le plus ancien
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            self.cache_stats["evictions"] += 1
        
        ttl_seconds = ttl_minutes * 60
        self.cache[cache_key] = (result, datetime.now().timestamp(), ttl_seconds)
    
    def invalidate_user_cache(self, user_id: int):
        """Invalide tout le cache pour un utilisateur"""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"agg_{user_id}_")]
        for key in keys_to_remove:
            del self.cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de cache"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "total_hits": self.cache_stats["hits"],
            "total_misses": self.cache_stats["misses"],
            "total_evictions": self.cache_stats["evictions"],
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size
        }


class AggregationPerformanceMonitor:
    """Moniteur de performance pour les agrégations"""
    
    def __init__(self):
        self.performance_history = {}
        self.slow_query_threshold_ms = 200
        
    def record_execution(self, template_name: str, execution_time_ms: int, 
                        result_count: int, cache_hit: bool = False):
        """Enregistre une exécution d'agrégation"""
        if template_name not in self.performance_history:
            self.performance_history[template_name] = []
        
        self.performance_history[template_name].append({
            "execution_time_ms": execution_time_ms,
            "result_count": result_count,
            "cache_hit": cache_hit,
            "timestamp": datetime.now().isoformat(),
            "is_slow": execution_time_ms > self.slow_query_threshold_ms
        })
        
        # Garder seulement les 100 dernières exécutions
        if len(self.performance_history[template_name]) > 100:
            self.performance_history[template_name] = self.performance_history[template_name][-100:]
    
    def get_performance_summary(self, template_name: str) -> Dict[str, Any]:
        """Retourne un résumé de performance pour un template"""
        if template_name not in self.performance_history:
            return {"error": "No performance data available"}
        
        history = self.performance_history[template_name]
        execution_times = [h["execution_time_ms"] for h in history]
        cache_hits = [h for h in history if h["cache_hit"]]
        slow_queries = [h for h in history if h["is_slow"]]
        
        return {
            "total_executions": len(history),
            "average_time_ms": sum(execution_times) / len(execution_times),
            "min_time_ms": min(execution_times),
            "max_time_ms": max(execution_times),
            "cache_hit_rate": len(cache_hits) / len(history) if history else 0,
            "slow_query_rate": len(slow_queries) / len(history) if history else 0,
            "last_execution": history[-1] if history else None
        }
    
    def get_slow_queries(self) -> List[Dict[str, Any]]:
        """Retourne les requêtes lentes récentes"""
        slow_queries = []
        
        for template_name, history in self.performance_history.items():
            for execution in history:
                if execution["is_slow"]:
                    slow_queries.append({
                        "template_name": template_name,
                        **execution
                    })
        
        # Trier par temps d'exécution décroissant
        return sorted(slow_queries, key=lambda x: x["execution_time_ms"], reverse=True)[:20]


class AggregationOrchestrator:
    """Orchestrateur principal pour les agrégations financières"""
    
    def __init__(self):
        self.engine = FinancialAggregationEngine()
        self.composer = AggregationComposer(self.engine)
        self.processor = AggregationResultProcessor()
        self.cache = AggregationCache()
        self.monitor = AggregationPerformanceMonitor()
    
    def execute_aggregation(self, intent: AggregationIntent, user_id: int,
                           filters: Optional[Dict[str, Any]] = None,
                           use_cache: bool = True) -> Dict[str, Any]:
        """Exécute une agrégation avec gestion de cache et monitoring"""
        template = self.engine.get_template(intent)
        if not template:
            return {"error": f"Template not found for intent: {intent}"}
        
        # Vérifier le cache d'abord
        cache_key = self.cache.get_cache_key(user_id, template.name, filters)
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.monitor.record_execution(
                    template.name, 0, len(cached_result.get("results", [])), cache_hit=True
                )
                return cached_result
        
        start_time = datetime.now()
        
        try:
            # Construire la requête
            builder = AggregationQueryBuilder(user_id)
            
            # Appliquer les filtres
            if filters:
                if "date_range" in filters:
                    start_date, end_date = filters["date_range"]
                    builder.apply_date_filter(start_date, end_date)
                if "categories" in filters:
                    builder.apply_category_filter(filters["categories"])
                if "amount_range" in filters:
                    min_amount, max_amount = filters["amount_range"]
                    builder.apply_amount_filter(min_amount, max_amount)
            
            # Ajouter l'agrégation du template
            builder.add_template_aggregation(template)
            query = builder.build()
            
            # Exécuter la requête (ici on simule, dans la vraie vie on appellerait Elasticsearch)
            # es_result = elasticsearch_client.search(query)
            es_result = self._simulate_elasticsearch_result(template, user_id)
            
            # Traiter les résultats selon le type d'intention
            processed_result = self._process_result_by_intent(intent, es_result)
            
            # Calculer le temps d'exécution
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Enregistrer les performances
            self.monitor.record_execution(
                template.name, execution_time, 
                len(processed_result.get("results", [])), cache_hit=False
            )
            
            # Mettre en cache
            if use_cache:
                self.cache.set(cache_key, processed_result, template.cache_duration_minutes)
            
            # Ajouter les métadonnées
            processed_result["metadata"] = {
                "template_name": template.name,
                "execution_time_ms": execution_time,
                "cache_hit": False,
                "complexity": template.complexity.value,
                "estimated_time_ms": template.estimated_time_ms
            }
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error executing aggregation {intent}: {str(e)}")
            return {"error": f"Aggregation execution failed: {str(e)}"}
    
    def _process_result_by_intent(self, intent: AggregationIntent, 
                                 es_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les résultats selon le type d'intention"""
        if intent == AggregationIntent.MONTHLY_BREAKDOWN:
            return self.processor.process_monthly_breakdown(es_result)
        elif intent == AggregationIntent.SPENDING_BY_CATEGORY:
            return self.processor.process_category_spending(es_result)
        elif intent == AggregationIntent.TOP_MERCHANTS:
            return self.processor.process_merchant_analysis(es_result)
        elif intent == AggregationIntent.SPENDING_STATISTICS:
            return self.processor.process_spending_statistics(es_result)
        elif intent == AggregationIntent.PERIOD_COMPARISON:
            return self.processor.process_period_comparison(es_result)
        elif intent == AggregationIntent.AMOUNT_DISTRIBUTION:
            return self.processor.process_amount_distribution(es_result)
        else:
            # Traitement générique pour les autres intentions
            return {"raw_result": es_result, "processed": False}
    
    def _simulate_elasticsearch_result(self, template: AggregationTemplate, 
                                     user_id: int) -> Dict[str, Any]:
        """Simule un résultat Elasticsearch pour les tests"""
        # Cette méthode serait remplacée par un vrai appel ES en production
        return {
            "took": template.estimated_time_ms,
            "hits": {"total": {"value": 150}},
            "aggregations": {
                "monthly_spending": {
                    "buckets": [
                        {
                            "key": "2024-01",
                            "doc_count": 45,
                            "total_amount": {"value": 567.89},
                            "transaction_count": {"value": 45},
                            "avg_amount": {"value": 12.62}
                        },
                        {
                            "key": "2023-12",
                            "doc_count": 38,
                            "total_amount": {"value": 423.45},
                            "transaction_count": {"value": 38},
                            "avg_amount": {"value": 11.14}
                        }
                    ]
                }
            }
        }
    
    def get_available_aggregations(self) -> List[Dict[str, Any]]:
        """Retourne la liste des agrégations disponibles"""
        templates = self.engine.list_templates()
        return [
            {
                "intent": template.intent.value,
                "name": template.name,
                "description": template.description,
                "complexity": template.complexity.value,
                "estimated_time_ms": template.estimated_time_ms,
                "dimensions": [dim.value for dim in template.dimensions],
                "metrics": [metric.value for metric in template.metrics]
            }
            for template in templates
        ]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance global"""
        cache_stats = self.cache.get_cache_stats()
        slow_queries = self.monitor.get_slow_queries()
        
        template_performance = {}
        for template in self.engine.list_templates():
            template_performance[template.name] = self.monitor.get_performance_summary(template.name)
        
        return {
            "cache_statistics": cache_stats,
            "slow_queries": slow_queries,
            "template_performance": template_performance,
            "total_templates": len(self.engine.templates),
            "report_generated_at": datetime.now().isoformat()
        }


# === INSTANCES GLOBALES ===
aggregation_engine = FinancialAggregationEngine()
aggregation_orchestrator = AggregationOrchestrator()


# === FONCTIONS D'UTILITÉ PRINCIPALES ===

def execute_financial_aggregation(intent: AggregationIntent, user_id: int,
                                 filters: Optional[Dict[str, Any]] = None,
                                 use_cache: bool = True) -> Dict[str, Any]:
    """
    Fonction principale pour exécuter une agrégation financière
    """
    return aggregation_orchestrator.execute_aggregation(intent, user_id, filters, use_cache)


def get_monthly_spending_breakdown(user_id: int, months_back: int = 12) -> Dict[str, Any]:
    """Récupère la répartition mensuelle des dépenses"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
    
    filters = {"date_range": (start_date, end_date)}
    return execute_financial_aggregation(AggregationIntent.MONTHLY_BREAKDOWN, user_id, filters)


def get_category_analysis(user_id: int, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """Analyse des dépenses par catégorie"""
    filters = {"date_range": date_range} if date_range else None
    return execute_financial_aggregation(AggregationIntent.SPENDING_BY_CATEGORY, user_id, filters)


def get_top_merchants(user_id: int, limit: int = 10) -> Dict[str, Any]:
    """Analyse des top marchands"""
    return execute_financial_aggregation(AggregationIntent.TOP_MERCHANTS, user_id)


def get_spending_statistics(user_id: int) -> Dict[str, Any]:
    """Statistiques complètes des dépenses"""
    return execute_financial_aggregation(AggregationIntent.SPENDING_STATISTICS, user_id)


def compare_periods(user_id: int, current_start: str, current_end: str,
                   previous_start: str, previous_end: str) -> Dict[str, Any]:
    """Compare les dépenses entre deux périodes"""
    filters = {
        "current_period": (current_start, current_end),
        "previous_period": (previous_start, previous_end)
    }
    return execute_financial_aggregation(AggregationIntent.PERIOD_COMPARISON, user_id, filters)


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # Classes principales
    "FinancialAggregationEngine",
    "AggregationQueryBuilder", 
    "AggregationResultProcessor",
    "AggregationComposer",
    "AggregationOrchestrator",
    "AggregationCache",
    "AggregationPerformanceMonitor",
    
    # Enums
    "AggregationIntent",
    "AggregationComplexity",
    "AggregationDimension", 
    "AggregationMetric",
    
    # Modèles
    "AggregationTemplate",
    
    # Fonctions utilitaires
    "execute_financial_aggregation",
    "get_monthly_spending_breakdown",
    "get_category_analysis",
    "get_top_merchants", 
    "get_spending_statistics",
    "compare_periods",
    
    # Instances globales
    "aggregation_engine",
    "aggregation_orchestrator"
]