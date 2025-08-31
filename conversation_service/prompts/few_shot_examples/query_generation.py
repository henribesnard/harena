"""
Exemples few-shot pour génération de requêtes search_service
40+ exemples couvrant tous les cas d'usage critiques
"""
from typing import Dict, Any, List


class QueryGenerationExamples:
    """Collection d'exemples few-shot pour génération requêtes"""
    
    @staticmethod
    def get_all_examples() -> List[Dict[str, Any]]:
        """Retourne tous les exemples few-shot organisés par intention"""
        examples = []
        
        # SEARCH_BY_MERCHANT
        examples.extend(QueryGenerationExamples._merchant_examples())
        
        # SEARCH_BY_AMOUNT  
        examples.extend(QueryGenerationExamples._amount_examples())
        
        # SPENDING_ANALYSIS
        examples.extend(QueryGenerationExamples._spending_examples())
        
        # BALANCE_INQUIRY
        examples.extend(QueryGenerationExamples._balance_examples())
        
        # SEARCH_BY_DATE
        examples.extend(QueryGenerationExamples._date_examples())
        
        # SEARCH_BY_OPERATION_TYPE
        examples.extend(QueryGenerationExamples._operation_type_examples())
        
        # SEARCH_BY_CATEGORY
        examples.extend(QueryGenerationExamples._category_examples())
        
        # TRANSACTION_HISTORY
        examples.extend(QueryGenerationExamples._history_examples())
        
        # COUNT_TRANSACTIONS
        examples.extend(QueryGenerationExamples._count_examples())
        
        # Cas complexes et edge cases
        examples.extend(QueryGenerationExamples._complex_examples())
        
        return examples
    
    @staticmethod
    def _merchant_examples() -> List[Dict[str, Any]]:
        """Exemples SEARCH_BY_MERCHANT"""
        return [
            {
                "description": "Recherche simple Amazon",
                "input": {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "intent_confidence": 0.94,
                    "user_id": 123,
                    "entities": {
                        "merchants": ["Amazon"],
                        "dates": [],
                        "amounts": [],
                        "transaction_types": ["debit"]
                    },
                    "user_message": "Mes achats Amazon"
                },
                "output": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "merchant_name": {"match": "Amazon"},
                        "transaction_type": "debit"
                    },
                    "aggregations": {
                        "merchant_stats": {
                            "terms": {"field": "merchant_name.keyword", "size": 10},
                            "aggs": {
                                "total_spent": {"sum": {"field": "amount_abs"}},
                                "transaction_count": {"value_count": {"field": "transaction_id"}},
                                "average_amount": {"avg": {"field": "amount_abs"}}
                            }
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 20,
                    "include_fields": ["transaction_id", "amount", "amount_abs", "merchant_name", "date", "primary_description"]
                }
            },
            {
                "description": "Amazon avec période spécifique",
                "input": {
                    "intent_type": "SEARCH_BY_MERCHANT", 
                    "intent_confidence": 0.92,
                    "user_id": 456,
                    "entities": {
                        "merchants": ["Amazon"],
                        "dates": [{"type": "period", "value": "2024-08", "text": "août"}],
                        "transaction_types": []
                    },
                    "user_message": "Mes achats Amazon en août"
                },
                "output": {
                    "user_id": 456,
                    "filters": {
                        "user_id": 456,
                        "merchant_name": {"match": "Amazon"},
                        "date": {"gte": "2024-08-01", "lte": "2024-08-31"}
                    },
                    "aggregations": {
                        "merchant_stats": {
                            "terms": {"field": "merchant_name.keyword", "size": 10},
                            "aggs": {
                                "total_spent": {"sum": {"field": "amount_abs"}},
                                "transaction_count": {"value_count": {"field": "transaction_id"}}
                            }
                        },
                        "daily_breakdown": {
                            "date_histogram": {"field": "date", "calendar_interval": "day"},
                            "aggs": {"daily_total": {"sum": {"field": "amount_abs"}}}
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 20,
                    "include_fields": ["transaction_id", "amount", "merchant_name", "date", "primary_description"]
                }
            },
            {
                "description": "Multiples marchands",
                "input": {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "intent_confidence": 0.89,
                    "user_id": 789,
                    "entities": {
                        "merchants": ["Carrefour", "Leclerc"],
                        "dates": [],
                        "amounts": []
                    },
                    "user_message": "Mes courses Carrefour et Leclerc"
                },
                "output": {
                    "user_id": 789,
                    "filters": {
                        "user_id": 789,
                        "merchant_name": {"terms": ["Carrefour", "Leclerc"]},
                        "transaction_type": "debit"
                    },
                    "aggregations": {
                        "merchant_comparison": {
                            "terms": {"field": "merchant_name.keyword", "size": 10},
                            "aggs": {
                                "total_spent": {"sum": {"field": "amount_abs"}},
                                "avg_basket": {"avg": {"field": "amount_abs"}}
                            }
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 30,
                    "include_fields": ["transaction_id", "amount", "merchant_name", "date", "category_name"]
                }
            }
        ]
    
    @staticmethod
    def _amount_examples() -> List[Dict[str, Any]]:
        """Exemples SEARCH_BY_AMOUNT"""
        return [
            {
                "description": "Montant supérieur à 100€",
                "input": {
                    "intent_type": "SEARCH_BY_AMOUNT",
                    "intent_confidence": 0.91,
                    "user_id": 123,
                    "entities": {
                        "amounts": [{"value": 100, "operator": "gte", "currency": "EUR"}],
                        "dates": []
                    },
                    "user_message": "Transactions supérieures à 100€"
                },
                "output": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "amount_abs": {"gte": 100}
                    },
                    "aggregations": {
                        "amount_distribution": {
                            "terms": {"field": "amount_abs", "size": 15},
                            "aggs": {
                                "merchant_breakdown": {
                                    "terms": {"field": "merchant_name.keyword", "size": 5}
                                }
                            }
                        },
                        "total_matching": {"sum": {"field": "amount_abs"}}
                    },
                    "sort": [{"amount_abs": {"order": "desc"}}],
                    "page_size": 50,
                    "include_fields": ["transaction_id", "amount", "amount_abs", "merchant_name", "date"]
                }
            },
            {
                "description": "Plage de montants",
                "input": {
                    "intent_type": "SEARCH_BY_AMOUNT",
                    "intent_confidence": 0.88,
                    "user_id": 456,
                    "entities": {
                        "amounts": [
                            {"value": 50, "operator": "gte", "currency": "EUR"},
                            {"value": 200, "operator": "lte", "currency": "EUR"}
                        ]
                    },
                    "user_message": "Transactions entre 50€ et 200€"
                },
                "output": {
                    "user_id": 456,
                    "filters": {
                        "user_id": 456,
                        "amount_abs": {"gte": 50, "lte": 200}
                    },
                    "aggregations": {
                        "amount_ranges": {
                            "range": {
                                "field": "amount_abs",
                                "ranges": [
                                    {"from": 50, "to": 100},
                                    {"from": 100, "to": 150},
                                    {"from": 150, "to": 200}
                                ]
                            }
                        }
                    },
                    "sort": [{"amount_abs": {"order": "desc"}}],
                    "page_size": 30,
                    "include_fields": ["transaction_id", "amount", "merchant_name", "date", "category_name"]
                }
            }
        ]
    
    @staticmethod
    def _spending_examples() -> List[Dict[str, Any]]:
        """Exemples SPENDING_ANALYSIS"""
        return [
            {
                "description": "Analyse globale des dépenses",
                "input": {
                    "intent_type": "SPENDING_ANALYSIS",
                    "intent_confidence": 0.93,
                    "user_id": 123,
                    "entities": {
                        "dates": [{"type": "period", "value": "2024-08", "text": "août"}],
                        "transaction_types": ["debit"]
                    },
                    "user_message": "Combien j'ai dépensé en août ?"
                },
                "output": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "transaction_type": "debit",
                        "date": {"gte": "2024-08-01", "lte": "2024-08-31"}
                    },
                    "aggregations": {
                        "category_breakdown": {
                            "terms": {"field": "category_name.keyword", "size": 15},
                            "aggs": {
                                "category_total": {"sum": {"field": "amount_abs"}},
                                "transaction_count": {"value_count": {"field": "transaction_id"}}
                            }
                        },
                        "daily_spending": {
                            "date_histogram": {"field": "date", "calendar_interval": "day"},
                            "aggs": {"daily_total": {"sum": {"field": "amount_abs"}}}
                        },
                        "total_spending": {"sum": {"field": "amount_abs"}}
                    },
                    "aggregation_only": True,
                    "page_size": 0
                }
            },
            {
                "description": "Analyse par catégorie spécifique",
                "input": {
                    "intent_type": "SPENDING_ANALYSIS",
                    "intent_confidence": 0.90,
                    "user_id": 456,
                    "entities": {
                        "categories": ["restaurant"],
                        "dates": [{"type": "relative", "value": "this_month", "text": "ce mois"}]
                    },
                    "user_message": "Mes dépenses restaurants ce mois"
                },
                "output": {
                    "user_id": 456,
                    "filters": {
                        "user_id": 456,
                        "category_name": {"match": "restaurant"},
                        "transaction_type": "debit",
                        "date": {"gte": "2024-08-01"}
                    },
                    "aggregations": {
                        "restaurant_details": {
                            "terms": {"field": "merchant_name.keyword", "size": 20},
                            "aggs": {
                                "merchant_total": {"sum": {"field": "amount_abs"}},
                                "visit_count": {"value_count": {"field": "transaction_id"}}
                            }
                        },
                        "weekly_trend": {
                            "date_histogram": {"field": "date", "calendar_interval": "week"},
                            "aggs": {"weekly_total": {"sum": {"field": "amount_abs"}}}
                        }
                    },
                    "aggregation_only": True
                }
            }
        ]
    
    @staticmethod
    def _balance_examples() -> List[Dict[str, Any]]:
        """Exemples BALANCE_INQUIRY"""
        return [
            {
                "description": "Solde global",
                "input": {
                    "intent_type": "BALANCE_INQUIRY",
                    "intent_confidence": 0.98,
                    "user_id": 123,
                    "entities": {},
                    "user_message": "Mon solde"
                },
                "output": {
                    "user_id": 123,
                    "filters": {"user_id": 123},
                    "aggregations": {
                        "balance_by_account": {
                            "terms": {"field": "account_id.keyword", "size": 10},
                            "aggs": {
                                "current_balance": {"sum": {"field": "amount"}},
                                "last_transaction": {"max": {"field": "date"}},
                                "transaction_count": {"value_count": {"field": "transaction_id"}}
                            }
                        },
                        "total_balance": {"sum": {"field": "amount"}}
                    },
                    "aggregation_only": True,
                    "page_size": 0
                }
            },
            {
                "description": "Solde compte spécifique",
                "input": {
                    "intent_type": "BALANCE_INQUIRY",
                    "intent_confidence": 0.95,
                    "user_id": 456,
                    "entities": {
                        "accounts": ["compte courant"]
                    },
                    "user_message": "Solde de mon compte courant"
                },
                "output": {
                    "user_id": 456,
                    "filters": {
                        "user_id": 456,
                        "account_type": {"match": "checking"}
                    },
                    "aggregations": {
                        "account_balance": {
                            "sum": {"field": "amount"}
                        },
                        "last_movements": {
                            "terms": {"field": "transaction_type.keyword"},
                            "aggs": {
                                "type_total": {"sum": {"field": "amount"}}
                            }
                        }
                    },
                    "aggregation_only": True
                }
            }
        ]
    
    @staticmethod
    def _date_examples() -> List[Dict[str, Any]]:
        """Exemples SEARCH_BY_DATE"""
        return [
            {
                "description": "Transactions d'hier",
                "input": {
                    "intent_type": "SEARCH_BY_DATE",
                    "intent_confidence": 0.94,
                    "user_id": 123,
                    "entities": {
                        "dates": [{"type": "specific", "value": "2024-08-25", "text": "hier"}]
                    },
                    "user_message": "Mes transactions d'hier"
                },
                "output": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "date": {"gte": "2024-08-25", "lte": "2024-08-25"}
                    },
                    "aggregations": {
                        "daily_summary": {
                            "terms": {"field": "transaction_type.keyword"},
                            "aggs": {
                                "type_total": {"sum": {"field": "amount_abs"}},
                                "type_count": {"value_count": {"field": "transaction_id"}}
                            }
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 100,
                    "include_fields": ["transaction_id", "amount", "merchant_name", "date", "primary_description", "operation_type"]
                }
            },
            {
                "description": "Période étendue - semaine dernière",
                "input": {
                    "intent_type": "SEARCH_BY_DATE",
                    "intent_confidence": 0.91,
                    "user_id": 456,
                    "entities": {
                        "dates": [{"type": "period", "value": "2024-W33", "text": "semaine dernière"}]
                    },
                    "user_message": "Transactions de la semaine dernière"
                },
                "output": {
                    "user_id": 456,
                    "filters": {
                        "user_id": 456,
                        "date": {"gte": "2024-08-19", "lte": "2024-08-25"}
                    },
                    "aggregations": {
                        "daily_breakdown": {
                            "date_histogram": {"field": "date", "calendar_interval": "day"},
                            "aggs": {
                                "daily_count": {"value_count": {"field": "transaction_id"}},
                                "daily_in": {"sum": {"field": "amount", "script": "params._source.amount > 0 ? params._source.amount : 0"}},
                                "daily_out": {"sum": {"field": "amount_abs", "script": "params._source.amount < 0 ? params._source.amount_abs : 0"}}
                            }
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 50
                }
            }
        ]
    
    @staticmethod
    def _operation_type_examples() -> List[Dict[str, Any]]:
        """Exemples SEARCH_BY_OPERATION_TYPE"""
        return [
            {
                "description": "Virements reçus",
                "input": {
                    "intent_type": "SEARCH_BY_OPERATION_TYPE",
                    "intent_confidence": 0.92,
                    "user_id": 123,
                    "entities": {
                        "operation_types": ["virement"],
                        "transaction_types": ["credit"]
                    },
                    "user_message": "Mes virements reçus en mai"
                },
                "output": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "operation_type": {"match": "virement"},
                        "transaction_type": "credit",
                        "date": {"gte": "2024-05-01", "lte": "2024-05-31"}
                    },
                    "aggregations": {
                        "transfer_stats": {
                            "sum": {"field": "amount_abs"}
                        },
                        "transfer_count": {
                            "value_count": {"field": "transaction_id"}
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 30,
                    "include_fields": ["transaction_id", "amount", "date", "primary_description", "operation_type"]
                }
            },
            {
                "description": "Paiements par carte",
                "input": {
                    "intent_type": "SEARCH_BY_OPERATION_TYPE",
                    "intent_confidence": 0.89,
                    "user_id": 456,
                    "entities": {
                        "operation_types": ["carte", "CB"]
                    },
                    "user_message": "Mes paiements par carte"
                },
                "output": {
                    "user_id": 456,
                    "filters": {
                        "user_id": 456,
                        "operation_type": {"terms": ["carte", "CB"]},
                        "transaction_type": "debit"
                    },
                    "aggregations": {
                        "card_usage": {
                            "terms": {"field": "merchant_name.keyword", "size": 20},
                            "aggs": {
                                "merchant_total": {"sum": {"field": "amount_abs"}},
                                "visit_count": {"value_count": {"field": "transaction_id"}}
                            }
                        },
                        "monthly_card_usage": {
                            "date_histogram": {"field": "date", "calendar_interval": "month"},
                            "aggs": {"monthly_total": {"sum": {"field": "amount_abs"}}}
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 50
                }
            }
        ]
    
    @staticmethod
    def _category_examples() -> List[Dict[str, Any]]:
        """Exemples SEARCH_BY_CATEGORY"""
        return [
            {
                "description": "Dépenses transport",
                "input": {
                    "intent_type": "SEARCH_BY_CATEGORY",
                    "intent_confidence": 0.90,
                    "user_id": 123,
                    "entities": {
                        "categories": ["transport"]
                    },
                    "user_message": "Mes dépenses transport"
                },
                "output": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "category_name": {"match": "transport"}
                    },
                    "aggregations": {
                        "transport_breakdown": {
                            "terms": {"field": "merchant_name.keyword", "size": 10},
                            "aggs": {
                                "merchant_total": {"sum": {"field": "amount_abs"}},
                                "frequency": {"value_count": {"field": "transaction_id"}}
                            }
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 30,
                    "include_fields": ["transaction_id", "amount", "merchant_name", "date", "category_name"]
                }
            }
        ]
    
    @staticmethod
    def _history_examples() -> List[Dict[str, Any]]:
        """Exemples TRANSACTION_HISTORY"""
        return [
            {
                "description": "Historique général",
                "input": {
                    "intent_type": "TRANSACTION_HISTORY",
                    "intent_confidence": 0.95,
                    "user_id": 123,
                    "entities": {},
                    "user_message": "Mon historique de transactions"
                },
                "output": {
                    "user_id": 123,
                    "filters": {"user_id": 123},
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 50,
                    "include_fields": [
                        "transaction_id", "amount", "amount_abs", "merchant_name",
                        "date", "primary_description", "category_name", "operation_type", "transaction_type"
                    ]
                }
            }
        ]
    
    @staticmethod
    def _count_examples() -> List[Dict[str, Any]]:
        """Exemples COUNT_TRANSACTIONS"""
        return [
            {
                "description": "Nombre de transactions Amazon",
                "input": {
                    "intent_type": "COUNT_TRANSACTIONS",
                    "intent_confidence": 0.93,
                    "user_id": 123,
                    "entities": {
                        "merchants": ["Amazon"]
                    },
                    "user_message": "Combien de fois j'ai acheté sur Amazon ?"
                },
                "output": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "merchant_name": {"match": "Amazon"}
                    },
                    "aggregations": {
                        "amazon_count": {"value_count": {"field": "transaction_id"}},
                        "amazon_total": {"sum": {"field": "amount_abs"}},
                        "monthly_amazon": {
                            "date_histogram": {"field": "date", "calendar_interval": "month"},
                            "aggs": {"monthly_count": {"value_count": {"field": "transaction_id"}}}
                        }
                    },
                    "aggregation_only": True,
                    "page_size": 0
                }
            }
        ]
    
    @staticmethod
    def _complex_examples() -> List[Dict[str, Any]]:
        """Exemples complexes et edge cases"""
        return [
            {
                "description": "Requête multi-critères complexe",
                "input": {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "intent_confidence": 0.87,
                    "user_id": 123,
                    "entities": {
                        "merchants": ["Amazon"],
                        "amounts": [{"value": 50, "operator": "gte"}],
                        "dates": [{"type": "period", "value": "2024-08", "text": "août"}],
                        "categories": ["shopping"]
                    },
                    "user_message": "Achats Amazon supérieurs à 50€ en août catégorie shopping"
                },
                "output": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "merchant_name": {"match": "Amazon"},
                        "amount_abs": {"gte": 50},
                        "date": {"gte": "2024-08-01", "lte": "2024-08-31"},
                        "category_name": {"match": "shopping"}
                    },
                    "aggregations": {
                        "complex_analysis": {
                            "terms": {"field": "amount_abs", "size": 10},
                            "aggs": {
                                "avg_by_amount": {"avg": {"field": "amount_abs"}}
                            }
                        }
                    },
                    "sort": [{"amount_abs": {"order": "desc"}}],
                    "page_size": 25,
                    "include_fields": ["transaction_id", "amount", "merchant_name", "date", "category_name"]
                }
            },
            {
                "description": "Cas sans entités spécifiques - intention générale",
                "input": {
                    "intent_type": "SPENDING_ANALYSIS",
                    "intent_confidence": 0.75,
                    "user_id": 789,
                    "entities": {},
                    "user_message": "Analyse de mes dépenses"
                },
                "output": {
                    "user_id": 789,
                    "filters": {
                        "user_id": 789,
                        "transaction_type": "debit"
                    },
                    "aggregations": {
                        "general_spending": {
                            "terms": {"field": "category_name.keyword", "size": 15},
                            "aggs": {"category_total": {"sum": {"field": "amount_abs"}}}
                        },
                        "monthly_trend": {
                            "date_histogram": {"field": "date", "calendar_interval": "month"},
                            "aggs": {"monthly_total": {"sum": {"field": "amount_abs"}}}
                        },
                        "total_spending": {"sum": {"field": "amount_abs"}}
                    },
                    "aggregation_only": True,
                    "page_size": 0
                }
            },
            {
                "description": "Recherche avec text_search spécifique",
                "input": {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "intent_confidence": 0.85,
                    "user_id": 123,
                    "entities": {
                        "text_search": ["remboursement mutuelle"],
                        "transaction_types": ["credit"]
                    },
                    "user_message": "Remboursements mutuelle"
                },
                "output": {
                    "user_id": 123,
                    "query": "remboursement mutuelle",
                    "filters": {
                        "user_id": 123,
                        "transaction_type": "credit"
                    },
                    "sort": [{"_score": {"order": "desc"}}, {"date": {"order": "desc"}}],
                    "page_size": 20,
                    "include_fields": ["transaction_id", "amount", "primary_description", "date", "merchant_name"]
                }
            }
        ]
    
    @staticmethod
    def get_examples_by_intent(intent_type: str) -> List[Dict[str, Any]]:
        """Retourne les exemples pour un type d'intention spécifique"""
        all_examples = QueryGenerationExamples.get_all_examples()
        return [
            example for example in all_examples 
            if example["input"]["intent_type"] == intent_type
        ]
    
    @staticmethod
    def get_prompt_examples(intent_type: str, max_examples: int = 3) -> str:
        """Génère les exemples few-shot formatés pour le prompt"""
        examples = QueryGenerationExamples.get_examples_by_intent(intent_type)[:max_examples]
        
        if not examples:
            # Fallback avec exemples génériques
            examples = QueryGenerationExamples.get_all_examples()[:max_examples]
        
        prompt_examples = []
        for i, example in enumerate(examples, 1):
            prompt_examples.append(f"""
EXEMPLE {i}:
INPUT: {example['input']}
OUTPUT: {example['output']}
DESCRIPTION: {example['description']}
""")
        
        return "\n".join(prompt_examples)


class QueryValidationExamples:
    """Exemples pour validation de requêtes générées"""
    
    @staticmethod
    def get_validation_examples() -> List[Dict[str, Any]]:
        """Exemples de validation avec erreurs corrigées"""
        return [
            {
                "description": "Requête sans user_id - correction automatique",
                "invalid_query": {
                    "filters": {"merchant_name": {"match": "Amazon"}},
                    "aggregations": {}
                },
                "corrected_query": {
                    "user_id": 123,
                    "filters": {
                        "user_id": 123,
                        "merchant_name": {"match": "Amazon"}
                    },
                    "page_size": 20,
                    "include_fields": ["transaction_id", "amount", "merchant_name", "date"]
                },
                "applied_fixes": [
                    "Ajout user_id manquant",
                    "Configuration page_size par défaut", 
                    "Ajout champs essentiels"
                ]
            },
            {
                "description": "Agrégations trop nombreuses - limitation automatique",
                "invalid_query": {
                    "user_id": 123,
                    "aggregations": {
                        "agg1": {"terms": {"field": "field1", "size": 100}},
                        "agg2": {"terms": {"field": "field2", "size": 150}}
                    }
                },
                "corrected_query": {
                    "user_id": 123,
                    "aggregations": {
                        "agg1": {"terms": {"field": "field1", "size": 20}},
                        "agg2": {"terms": {"field": "field2", "size": 20}}
                    }
                },
                "applied_fixes": [
                    "Limitation buckets agrégation 'agg1' à 20",
                    "Limitation buckets agrégation 'agg2' à 20"
                ]
            }
        ]