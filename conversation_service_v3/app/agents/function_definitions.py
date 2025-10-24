"""
Définitions des fonctions disponibles pour le LLM via function calling
Ces fonctions couvrent 95% des questions financières possibles
"""

from typing import Dict, Any, List

# ============================================================================
# FONCTION PRINCIPALE : search_transactions
# ============================================================================

SEARCH_TRANSACTIONS_FUNCTION = {
    "name": "search_transactions",
    "description": """Recherche flexible de transactions financières avec filtres et agrégations.

    Cette fonction unique peut répondre à presque toutes les questions sur les transactions:
    - Recherches simples: "Mes achats Amazon", "Transactions > 100€"
    - Analyses: "Combien dépensé en restaurants?", "Répartition par catégorie"
    - Tendances: "Évolution mensuelle", "Comparaisons temporelles"
    - Statistiques: "Moyenne", "Total", "Max/Min"

    Pour les comparaisons de périodes, appelle cette fonction 2 fois avec des filtres de dates différents.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "integer",
                "description": "ID de l'utilisateur (OBLIGATOIRE - toujours fourni par le système)"
            },
            "query": {
                "type": "string",
                "description": "Recherche textuelle libre pour merchant_name, primary_description (optionnel)"
            },
            "filters": {
                "type": "object",
                "description": "Filtres structurés sur les champs",
                "properties": {
                    "date": {
                        "type": "object",
                        "description": "Filtre de dates au format ISO 8601",
                        "properties": {
                            "gte": {
                                "type": "string",
                                "description": "Date début (incluse) format: 2025-01-01T00:00:00Z"
                            },
                            "lte": {
                                "type": "string",
                                "description": "Date fin (incluse) format: 2025-01-31T23:59:59Z"
                            },
                            "gt": {
                                "type": "string",
                                "description": "Date après (exclusive)"
                            },
                            "lt": {
                                "type": "string",
                                "description": "Date avant (exclusive)"
                            }
                        }
                    },
                    "amount_abs": {
                        "type": "object",
                        "description": "Filtre sur montant ABSOLU (toujours positif). IMPORTANT: 'plus de X' = gt (exclut X), 'au moins X' = gte (inclut X)",
                        "properties": {
                            "gte": {
                                "type": "number",
                                "description": "Montant minimum (inclus) - utiliser pour 'au moins', 'minimum'"
                            },
                            "lte": {
                                "type": "number",
                                "description": "Montant maximum (inclus) - utiliser pour 'au maximum', 'jusqu'à'"
                            },
                            "gt": {
                                "type": "number",
                                "description": "Montant supérieur (exclusif) - utiliser pour 'plus de', 'supérieur à'"
                            },
                            "lt": {
                                "type": "number",
                                "description": "Montant inférieur (exclusif) - utiliser pour 'moins de', 'inférieur à'"
                            }
                        }
                    },
                    "merchant_name": {
                        "type": "object",
                        "description": "Filtre sur nom du marchand",
                        "properties": {
                            "match": {
                                "type": "string",
                                "description": "Recherche floue (ex: 'amazon' trouvera 'Amazon.fr', 'Amazon Prime')"
                            },
                            "term": {
                                "type": "string",
                                "description": "Terme exact (sensible à la casse)"
                            },
                            "terms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Liste de termes exacts (OU logique)"
                            }
                        }
                    },
                    "category_name": {
                        "type": "object",
                        "description": "Filtre sur catégorie de transaction",
                        "properties": {
                            "match": {
                                "type": "string",
                                "description": "Recherche floue sur catégorie (ex: 'restaurant' trouvera 'Restaurants', 'Restaurant rapide')"
                            },
                            "term": {
                                "type": "string",
                                "description": "Catégorie exacte"
                            },
                            "terms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Liste de catégories (OU logique)"
                            }
                        }
                    },
                    "transaction_type": {
                        "type": "string",
                        "enum": ["debit", "credit"],
                        "description": "Type de transaction: 'debit' pour dépenses/sorties d'argent, 'credit' pour revenus/entrées d'argent"
                    },
                    "operation_type": {
                        "type": "object",
                        "description": "Type d'opération bancaire",
                        "properties": {
                            "term": {
                                "type": "string",
                                "description": "Type exact: 'Carte', 'Prélèvement', 'Virement', 'Chèque'"
                            }
                        }
                    },
                    "account_id": {
                        "type": "integer",
                        "description": "Filtre sur un compte bancaire spécifique"
                    }
                }
            },
            "aggregations": {
                "type": "object",
                "description": "Agrégations à calculer. Exemples courants fournis ci-dessous.",
                "additionalProperties": True
            },
            "sort": {
                "type": "array",
                "description": "Tri des résultats. Par défaut: [{'date': {'order': 'desc'}}]",
                "items": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "order": {
                                "type": "string",
                                "enum": ["asc", "desc"]
                            }
                        }
                    }
                },
                "default": [{"date": {"order": "desc"}}]
            },
            "page_size": {
                "type": "integer",
                "description": "Nombre de transactions à retourner (max 200, défaut 50)",
                "minimum": 1,
                "maximum": 200,
                "default": 50
            },
            "offset": {
                "type": "integer",
                "description": "Offset pour pagination",
                "minimum": 0,
                "default": 0
            }
        },
        "required": ["user_id", "filters", "sort", "page_size"]
    }
}

# ============================================================================
# TEMPLATES D'AGRÉGATIONS COURANTES
# ============================================================================

AGGREGATION_TEMPLATES = {
    "total_by_category": {
        "description": "Total des dépenses par catégorie avec comptage",
        "use_cases": [
            "Combien par catégorie?",
            "Répartition des dépenses",
            "Quelle catégorie coûte le plus?"
        ],
        "template": {
            "by_category": {
                "terms": {
                    "field": "category_name.keyword",
                    "size": 20,
                    "order": {"total_amount": "desc"}
                },
                "aggs": {
                    "total_amount": {
                        "sum": {"field": "amount_abs"}
                    },
                    "transaction_count": {
                        "value_count": {"field": "transaction_id"}
                    },
                    "avg_transaction": {
                        "avg": {"field": "amount_abs"}
                    }
                }
            }
        }
    },

    "monthly_trend": {
        "description": "Évolution mensuelle des dépenses",
        "use_cases": [
            "Tendance mensuelle",
            "Évolution dans le temps",
            "Comparaison mois par mois"
        ],
        "template": {
            "monthly_breakdown": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "month",
                    "format": "yyyy-MM"
                },
                "aggs": {
                    "total_spent": {
                        "sum": {"field": "amount_abs"}
                    },
                    "transaction_count": {
                        "value_count": {"field": "transaction_id"}
                    },
                    "avg_transaction": {
                        "avg": {"field": "amount_abs"}
                    }
                }
            }
        }
    },

    "weekly_trend": {
        "description": "Évolution hebdomadaire",
        "use_cases": [
            "Tendance hebdomadaire",
            "Dépenses par semaine"
        ],
        "template": {
            "weekly_breakdown": {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "week",
                    "format": "yyyy-'W'ww"
                },
                "aggs": {
                    "total_spent": {
                        "sum": {"field": "amount_abs"}
                    }
                }
            }
        }
    },

    "top_merchants": {
        "description": "Marchands les plus fréquents et coûteux",
        "use_cases": [
            "Où je dépense le plus?",
            "Marchands fréquents",
            "Top 10 marchands"
        ],
        "template": {
            "merchants_ranking": {
                "terms": {
                    "field": "merchant_name.keyword",
                    "size": 10,
                    "order": {"total_spent": "desc"}
                },
                "aggs": {
                    "total_spent": {
                        "sum": {"field": "amount_abs"}
                    },
                    "frequency": {
                        "value_count": {"field": "transaction_id"}
                    },
                    "avg_basket": {
                        "avg": {"field": "amount_abs"}
                    }
                }
            }
        }
    },

    "transaction_statistics": {
        "description": "Statistiques globales des transactions (débits et crédits, total, moyenne, min, max)",
        "use_cases": [
            "Résumé des transactions",
            "Statistiques globales",
            "Vue d'ensemble",
            "Analyse des revenus et dépenses"
        ],
        "template": {
            "global_stats": {
                "stats": {"field": "amount_abs"}
            },
            "total_transactions": {
                "value_count": {"field": "transaction_id"}
            },
            "total_debit": {
                "filter": {"term": {"transaction_type": "debit"}},
                "aggs": {
                    "sum_debit": {"sum": {"field": "amount_abs"}},
                    "count_debit": {"value_count": {"field": "transaction_id"}}
                }
            },
            "total_credit": {
                "filter": {"term": {"transaction_type": "credit"}},
                "aggs": {
                    "sum_credit": {"sum": {"field": "amount"}},
                    "count_credit": {"value_count": {"field": "transaction_id"}}
                }
            }
        }
    },

    "day_of_week_pattern": {
        "description": "Répartition par jour de la semaine",
        "use_cases": [
            "Quel jour je dépense le plus?",
            "Pattern hebdomadaire",
            "Dépenses weekend vs semaine"
        ],
        "template": {
            "by_weekday": {
                "terms": {
                    "field": "weekday",
                    "size": 7,
                    "order": {"_key": "asc"}
                },
                "aggs": {
                    "total_spent": {
                        "sum": {"field": "amount_abs"}
                    },
                    "avg_spent": {
                        "avg": {"field": "amount_abs"}
                    }
                }
            }
        }
    }
}

# ============================================================================
# FONCTION COMPLÉMENTAIRE : get_account_summary
# ============================================================================

GET_ACCOUNT_SUMMARY_FUNCTION = {
    "name": "get_account_summary",
    "description": """Récupère le résumé des comptes bancaires de l'utilisateur.

    Utiliser pour:
    - "Mes comptes"
    - "Solde de mes comptes"
    - "Quels comptes j'ai?"
    - "Balance de mon compte courant"
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "integer",
                "description": "ID de l'utilisateur"
            },
            "include_balance": {
                "type": "boolean",
                "description": "Inclure les soldes actuels",
                "default": True
            },
            "account_id": {
                "type": "integer",
                "description": "ID d'un compte spécifique (optionnel)"
            }
        },
        "required": ["user_id"]
    }
}

# ============================================================================
# FONCTION COMPLÉMENTAIRE : detect_recurring_transactions
# ============================================================================

DETECT_RECURRING_FUNCTION = {
    "name": "detect_recurring_transactions",
    "description": """Détecte les transactions récurrentes (abonnements, factures régulières).

    Utiliser pour:
    - "Mes abonnements"
    - "Paiements récurrents"
    - "Factures mensuelles"
    - "Dépenses régulières"
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "integer",
                "description": "ID de l'utilisateur"
            },
            "min_occurrences": {
                "type": "integer",
                "description": "Nombre minimum d'occurrences pour être considéré récurrent",
                "default": 3,
                "minimum": 2
            },
            "lookback_months": {
                "type": "integer",
                "description": "Nombre de mois à analyser",
                "default": 6,
                "minimum": 3,
                "maximum": 24
            },
            "merchant_name": {
                "type": "string",
                "description": "Filtrer sur un marchand spécifique (optionnel)"
            }
        },
        "required": ["user_id"]
    }
}

# ============================================================================
# TOUTES LES FONCTIONS DISPONIBLES
# ============================================================================

ALL_FUNCTIONS = [
    SEARCH_TRANSACTIONS_FUNCTION,
    GET_ACCOUNT_SUMMARY_FUNCTION,
    DETECT_RECURRING_FUNCTION
]


# ============================================================================
# EXEMPLES D'UTILISATION POUR LE LLM
# ============================================================================

FUNCTION_USAGE_EXAMPLES = """
EXEMPLES D'UTILISATION DES FONCTIONS:

1. Question simple: "Mes 10 dernières transactions chez Amazon"
   → search_transactions(
       user_id=123,
       query="Amazon",
       sort=[{"date": {"order": "desc"}}],
       page_size=10
     )

2. Question avec filtre montant: "Dépenses de plus de 100€"
   → search_transactions(
       user_id=123,
       filters={
         "transaction_type": "debit",
         "amount_abs": {"gt": 100}  // gt car "plus de" EXCLUT 100
       }
     )

3. Question analytique: "Combien j'ai dépensé en restaurants ce mois?"
   → search_transactions(
       user_id=123,
       filters={
         "category_name": {"match": "restaurant"},
         "transaction_type": "debit",
         "date": {
           "gte": "2025-10-01T00:00:00Z",
           "lte": "2025-10-31T23:59:59Z"
         }
       },
       aggregations={
         "total_spent": {"sum": {"field": "amount_abs"}},
         "transaction_count": {"value_count": {"field": "transaction_id"}}
       }
     )

4. Question de tendance: "Évolution de mes courses sur 6 mois"
   → search_transactions(
       user_id=123,
       filters={
         "category_name": {"match": "alimentation"},
         "transaction_type": "debit",
         "date": {
           "gte": "2025-04-01T00:00:00Z",
           "lte": "2025-10-31T23:59:59Z"
         }
       },
       aggregations={
         "monthly_breakdown": {
           "date_histogram": {
             "field": "date",
             "calendar_interval": "month"
           },
           "aggs": {
             "total_spent": {"sum": {"field": "amount_abs"}}
           }
         }
       },
       page_size=0  // Pas besoin de transactions détaillées
     )

5. Comparaison de périodes: "Compare janvier vs février"
   → APPELER 2 FOIS:

   Appel 1 - Janvier:
   search_transactions(
       user_id=123,
       filters={
         "date": {
           "gte": "2025-01-01T00:00:00Z",
           "lte": "2025-01-31T23:59:59Z"
         },
         "transaction_type": "debit"
       },
       aggregations={
         "total_spent": {"sum": {"field": "amount_abs"}},
         "by_category": {
           "terms": {"field": "category_name.keyword", "size": 20},
           "aggs": {"total": {"sum": {"field": "amount_abs"}}}
         }
       }
     )

   Appel 2 - Février:
   search_transactions(
       user_id=123,
       filters={
         "date": {
           "gte": "2025-02-01T00:00:00Z",
           "lte": "2025-02-28T23:59:59Z"
         },
         "transaction_type": "debit"
       },
       aggregations={
         "total_spent": {"sum": {"field": "amount_abs"}},
         "by_category": {
           "terms": {"field": "category_name.keyword", "size": 20},
           "aggs": {"total": {"sum": {"field": "amount_abs"}}}
         }
       }
     )

   Puis comparer les résultats dans la réponse

6. Détection d'abonnements: "Mes abonnements mensuels"
   → detect_recurring_transactions(
       user_id=123,
       min_occurrences=3,
       lookback_months=6
     )

7. Soldes des comptes: "Quel est le solde de mon compte courant?"
   → get_account_summary(
       user_id=123,
       include_balance=True
     )

8. Top marchands: "Où je dépense le plus?"
   → search_transactions(
       user_id=123,
       filters={"transaction_type": "debit"},
       aggregations={
         "merchants_ranking": {
           "terms": {
             "field": "merchant_name.keyword",
             "size": 10,
             "order": {"total_spent": "desc"}
           },
           "aggs": {
             "total_spent": {"sum": {"field": "amount_abs"}},
             "frequency": {"value_count": {"field": "transaction_id"}}
           }
         }
       },
       page_size=0
     )
"""


def get_aggregation_template(template_name: str) -> Dict[str, Any]:
    """Retourne un template d'agrégation"""
    return AGGREGATION_TEMPLATES.get(template_name, {}).get("template", {})


def get_all_templates_description() -> str:
    """Retourne la description de tous les templates disponibles"""
    descriptions = []
    for name, data in AGGREGATION_TEMPLATES.items():
        descriptions.append(f"{name}: {data['description']}")
        descriptions.append(f"  Cas d'usage: {', '.join(data['use_cases'])}")
    return "\n".join(descriptions)
