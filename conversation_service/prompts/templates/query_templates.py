"""
Templates de requêtes par type d'intention
Structures de base pour génération requêtes search_service optimisées
"""
from typing import Dict, Any, List, Optional


class QueryTemplates:
    """Templates de requêtes search_service par type d'intention"""
    
    @staticmethod
    def get_template(intent_type: str) -> Dict[str, Any]:
        """Retourne le template pour un type d'intention donné"""
        templates = {
            "SEARCH_BY_MERCHANT": QueryTemplates._merchant_search_template(),
            "SEARCH_BY_AMOUNT": QueryTemplates._amount_search_template(),
            "SPENDING_ANALYSIS": QueryTemplates._spending_analysis_template(),
            "BALANCE_INQUIRY": QueryTemplates._balance_inquiry_template(),
            "SEARCH_BY_DATE": QueryTemplates._date_search_template(),
            "SEARCH_BY_OPERATION_TYPE": QueryTemplates._operation_type_search_template(),
            "SEARCH_BY_CATEGORY": QueryTemplates._category_search_template(),
            "TRANSACTION_HISTORY": QueryTemplates._transaction_history_template(),
            "COUNT_TRANSACTIONS": QueryTemplates._count_transactions_template(),
        }
        
        return templates.get(intent_type, QueryTemplates._default_template())
    
    @staticmethod
    def _merchant_search_template() -> Dict[str, Any]:
        """Template pour recherche par marchand"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}",
                "merchant_name": {"match": "{merchant_name}"},
                "transaction_type": "{transaction_type}"  # debit par défaut
            },
            "aggregations": {
                "merchant_stats": {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": 10
                    },
                    "aggs": {
                        "total_spent": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}},
                        "average_amount": {"avg": {"field": "amount_abs"}}
                    }
                },
                "daily_breakdown": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "day"
                    },
                    "aggs": {
                        "daily_total": {"sum": {"field": "amount_abs"}}
                    }
                }
            },
            "sort": [{"date": {"order": "desc"}}],
            "page_size": 200,
            "include_fields": [
                "transaction_id", "amount", "amount_abs", "merchant_name",
                "date", "primary_description", "category_name"
            ]
        }
    
    @staticmethod
    def _amount_search_template() -> Dict[str, Any]:
        """Template pour recherche par montant"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}",
                "amount_abs": "{amount_filter}"  # gte, lte, range
            },
            "aggregations": {
                "amount_distribution": {
                    "terms": {
                        "field": "amount_abs",
                        "size": 15
                    },
                    "aggs": {
                        "merchant_breakdown": {
                            "terms": {
                                "field": "merchant_name.keyword",
                                "size": 5
                            }
                        }
                    }
                },
                "total_matching": {
                    "sum": {"field": "amount_abs"}
                }
            },
            "sort": [{"amount_abs": {"order": "desc"}}],
            "page_size": 50,
            "include_fields": [
                "transaction_id", "amount", "amount_abs", "merchant_name",
                "date", "primary_description"
            ]
        }
    
    @staticmethod
    def _spending_analysis_template() -> Dict[str, Any]:
        """Template pour analyse des dépenses"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}",
                "transaction_type": "debit"
            },
            "aggregations": {
                "category_breakdown": {
                    "terms": {
                        "field": "category_name.keyword",
                        "size": 15
                    },
                    "aggs": {
                        "category_total": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}},
                        "average_amount": {"avg": {"field": "amount_abs"}}
                    }
                },
                "monthly_spending": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "month"
                    },
                    "aggs": {
                        "monthly_total": {"sum": {"field": "amount_abs"}}
                    }
                },
                "total_spending": {
                    "sum": {"field": "amount_abs"}
                }
            },
            "aggregation_only": True,
            "page_size": 0
        }
    
    @staticmethod
    def _balance_inquiry_template() -> Dict[str, Any]:
        """Template pour consultation solde"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}"
            },
            "aggregations": {
                "balance_by_account": {
                    "terms": {
                        "field": "account_id.keyword",
                        "size": 10
                    },
                    "aggs": {
                        "current_balance": {"sum": {"field": "amount"}},
                        "last_transaction": {"max": {"field": "date"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                },
                "total_balance": {
                    "sum": {"field": "amount"}
                }
            },
            "aggregation_only": True,
            "page_size": 0
        }
    
    @staticmethod
    def _date_search_template() -> Dict[str, Any]:
        """Template pour recherche par date"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}",
                "date": "{date_range}"  # gte, lte selon période
            },
            "aggregations": {
                "daily_transactions": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "day"
                    },
                    "aggs": {
                        "daily_count": {"value_count": {"field": "transaction_id"}},
                        "daily_total_in": {
                            "sum": {
                                "field": "amount",
                                "script": "params._source.amount > 0 ? params._source.amount : 0"
                            }
                        },
                        "daily_total_out": {
                            "sum": {
                                "field": "amount_abs",
                                "script": "params._source.amount < 0 ? params._source.amount_abs : 0"
                            }
                        }
                    }
                }
            },
            "sort": [{"date": {"order": "desc"}}],
            "page_size": 100,
            "include_fields": [
                "transaction_id", "amount", "merchant_name", "date",
                "primary_description", "operation_type"
            ]
        }
    
    @staticmethod
    def _operation_type_search_template() -> Dict[str, Any]:
        """Template pour recherche par type d'opération"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}",
                "operation_type": {"match": "{operation_type}"}
            },
            "aggregations": {
                "operation_stats": {
                    "terms": {
                        "field": "operation_type.keyword",
                        "size": 10
                    },
                    "aggs": {
                        "operation_total": {"sum": {"field": "amount_abs"}},
                        "operation_count": {"value_count": {"field": "transaction_id"}},
                        "avg_amount": {"avg": {"field": "amount_abs"}}
                    }
                },
                "monthly_operations": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "month"
                    },
                    "aggs": {
                        "monthly_count": {"value_count": {"field": "transaction_id"}},
                        "monthly_total": {"sum": {"field": "amount_abs"}}
                    }
                }
            },
            "sort": [{"date": {"order": "desc"}}],
            "page_size": 50,
            "include_fields": [
                "transaction_id", "amount", "merchant_name", "date",
                "operation_type", "primary_description"
            ]
        }
    
    @staticmethod
    def _category_search_template() -> Dict[str, Any]:
        """Template pour recherche par catégorie"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}",
                "category_name": {"match": "{category_name}"}
            },
            "aggregations": {
                "category_analysis": {
                    "terms": {
                        "field": "category_name.keyword",
                        "size": 10
                    },
                    "aggs": {
                        "category_total": {"sum": {"field": "amount_abs"}},
                        "merchant_breakdown": {
                            "terms": {
                                "field": "merchant_name.keyword",
                                "size": 5
                            },
                            "aggs": {
                                "merchant_total": {"sum": {"field": "amount_abs"}}
                            }
                        }
                    }
                }
            },
            "sort": [{"date": {"order": "desc"}}],
            "page_size": 30,
            "include_fields": [
                "transaction_id", "amount", "merchant_name", "date",
                "category_name", "primary_description"
            ]
        }
    
    @staticmethod
    def _transaction_history_template() -> Dict[str, Any]:
        """Template pour historique transactions général"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}"
            },
            "sort": [{"date": {"order": "desc"}}],
            "page_size": 50,
            "include_fields": [
                "transaction_id", "amount", "amount_abs", "merchant_name",
                "date", "primary_description", "category_name", "operation_type",
                "transaction_type"
            ]
        }
    
    @staticmethod
    def _count_transactions_template() -> Dict[str, Any]:
        """Template pour comptage de transactions"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}"
            },
            "aggregations": {
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                },
                "count_by_type": {
                    "terms": {
                        "field": "transaction_type.keyword",
                        "size": 5
                    }
                },
                "count_by_merchant": {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": 10
                    }
                },
                "count_by_category": {
                    "terms": {
                        "field": "category_name.keyword",
                        "size": 10
                    }
                }
            },
            "aggregation_only": True,
            "page_size": 0
        }
    
    @staticmethod
    def _default_template() -> Dict[str, Any]:
        """Template par défaut pour intentions inconnues"""
        return {
            "user_id": "{user_id}",
            "filters": {
                "user_id": "{user_id}"
            },
            "sort": [{"date": {"order": "desc"}}],
            "page_size": 200,
            "include_fields": [
                "transaction_id", "amount", "merchant_name", "date",
                "primary_description", "category_name"
            ]
        }


class AggregationTemplates:
    """Templates d'agrégations spécialisées réutilisables"""
    
    @staticmethod
    def merchant_analysis(size: int = 10) -> Dict[str, Any]:
        """Analyse par marchand"""
        return {
            "terms": {
                "field": "merchant_name.keyword",
                "size": size
            },
            "aggs": {
                "total_spent": {"sum": {"field": "amount_abs"}},
                "transaction_count": {"value_count": {"field": "transaction_id"}},
                "average_amount": {"avg": {"field": "amount_abs"}},
                "last_transaction": {"max": {"field": "date"}}
            }
        }
    
    @staticmethod
    def category_analysis(size: int = 15) -> Dict[str, Any]:
        """Analyse par catégorie"""
        return {
            "terms": {
                "field": "category_name.keyword", 
                "size": size
            },
            "aggs": {
                "category_total": {"sum": {"field": "amount_abs"}},
                "transaction_count": {"value_count": {"field": "transaction_id"}},
                "average_amount": {"avg": {"field": "amount_abs"}}
            }
        }
    
    @staticmethod
    def temporal_analysis(interval: str = "month") -> Dict[str, Any]:
        """Analyse temporelle"""
        return {
            "date_histogram": {
                "field": "date",
                "calendar_interval": interval
            },
            "aggs": {
                "period_total": {"sum": {"field": "amount_abs"}},
                "period_count": {"value_count": {"field": "transaction_id"}},
                "avg_transaction": {"avg": {"field": "amount_abs"}}
            }
        }
    
    @staticmethod
    def operation_type_analysis(size: int = 10) -> Dict[str, Any]:
        """Analyse par type d'opération"""
        return {
            "terms": {
                "field": "operation_type.keyword",
                "size": size
            },
            "aggs": {
                "operation_total": {"sum": {"field": "amount_abs"}},
                "operation_count": {"value_count": {"field": "transaction_id"}},
                "avg_amount": {"avg": {"field": "amount_abs"}}
            }
        }
    
    @staticmethod
    def balance_by_account() -> Dict[str, Any]:
        """Solde par compte"""
        return {
            "terms": {
                "field": "account_id.keyword",
                "size": 20
            },
            "aggs": {
                "current_balance": {"sum": {"field": "amount"}},
                "last_transaction": {"max": {"field": "date"}},
                "transaction_count": {"value_count": {"field": "transaction_id"}},
                "last_activity": {
                    "top_hits": {
                        "size": 1,
                        "sort": [{"date": {"order": "desc"}}],
                        "_source": ["date", "amount", "merchant_name"]
                    }
                }
            }
        }


class FilterTemplates:
    """Templates de filtres réutilisables"""
    
    @staticmethod
    def date_range(start_date: str, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Filtre plage de dates"""
        filter_config = {"gte": start_date}
        if end_date:
            filter_config["lte"] = end_date
        return filter_config
    
    @staticmethod
    def amount_range(min_amount: Optional[float] = None, 
                    max_amount: Optional[float] = None) -> Dict[str, Any]:
        """Filtre plage de montants"""
        filter_config = {}
        if min_amount is not None:
            filter_config["gte"] = min_amount
        if max_amount is not None:
            filter_config["lte"] = max_amount
        return filter_config
    
    @staticmethod
    def merchant_filter(merchant_name: str, exact: bool = False) -> Dict[str, Any]:
        """Filtre marchand"""
        if exact:
            return {"term": merchant_name}
        else:
            return {"match": merchant_name}
    
    @staticmethod
    def category_filter(category_name: str, exact: bool = False) -> Dict[str, Any]:
        """Filtre catégorie"""
        if exact:
            return {"term": category_name}
        else:
            return {"match": category_name}
    
    @staticmethod
    def operation_type_filter(operation_types: List[str]) -> Dict[str, Any]:
        """Filtre types d'opération"""
        if len(operation_types) == 1:
            return {"match": operation_types[0]}
        else:
            return {"terms": operation_types}
    
    @staticmethod
    def transaction_type_filter(transaction_type: str) -> Dict[str, Any]:
        """Filtre type de transaction (credit/debit)"""
        return {"term": transaction_type}


class SortTemplates:
    """Templates de tri réutilisables"""
    
    @staticmethod
    def date_desc() -> List[Dict[str, Any]]:
        """Tri par date décroissante"""
        return [{"date": {"order": "desc"}}]
    
    @staticmethod
    def amount_desc() -> List[Dict[str, Any]]:
        """Tri par montant décroissant"""
        return [{"amount_abs": {"order": "desc"}}]
    
    @staticmethod
    def relevance_then_date() -> List[Dict[str, Any]]:
        """Tri par pertinence puis date"""
        return [
            {"_score": {"order": "desc"}},
            {"date": {"order": "desc"}}
        ]
    
    @staticmethod
    def date_amount() -> List[Dict[str, Any]]:
        """Tri par date puis montant"""
        return [
            {"date": {"order": "desc"}},
            {"amount_abs": {"order": "desc"}}
        ]