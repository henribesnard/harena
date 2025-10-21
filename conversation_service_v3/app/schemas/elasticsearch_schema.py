"""
Elasticsearch Schema Definition
Permet aux agents LangChain de comprendre la structure des données
"""

ELASTICSEARCH_SCHEMA = {
    "index": "transactions",
    "fields": {
        "id": {
            "type": "long",
            "description": "Identifiant unique de la transaction"
        },
        "user_id": {
            "type": "long",
            "description": "Identifiant de l'utilisateur propriétaire de la transaction",
            "required": True
        },
        "amount": {
            "type": "float",
            "description": "Montant de la transaction en euros (positif ou négatif)",
            "aggregatable": True
        },
        "amount_abs": {
            "type": "float",
            "description": "Valeur absolue du montant de la transaction (toujours positif)",
            "aggregatable": True,
            "note": "À utiliser pour filtrer par montant sans se soucier du signe"
        },
        "date": {
            "type": "date",
            "format": "yyyy-MM-dd",
            "description": "Date de la transaction",
            "aggregatable": True
        },
        "merchant_name": {
            "type": "keyword",
            "description": "Nom du marchand/commerçant",
            "aggregatable": True
        },
        "category_name": {
            "type": "keyword",
            "description": "Catégorie de la transaction (ex: Alimentation, Transport, Loisirs)",
            "aggregatable": True
        },
        "transaction_type": {
            "type": "keyword",
            "description": "Type de transaction: 'debit' (dépense) ou 'credit' (revenu)",
            "values": ["debit", "credit"],
            "aggregatable": True
        },
        "operation_type": {
            "type": "keyword",
            "description": "Type d'opération bancaire (ex: CARD, TRANSFER, CHECK)",
            "aggregatable": True
        },
        "primary_description": {
            "type": "text",
            "description": "Description textuelle de la transaction",
            "analyzable": True
        },
        "bank_name": {
            "type": "keyword",
            "description": "Nom de la banque",
            "aggregatable": True
        },
        "account_name": {
            "type": "keyword",
            "description": "Nom du compte bancaire",
            "aggregatable": True
        },
        "account_id": {
            "type": "long",
            "description": "Identifiant du compte bancaire",
            "aggregatable": True
        },
        "transaction_id": {
            "type": "keyword",
            "description": "Identifiant unique de la transaction (utilisé pour compter les transactions)",
            "aggregatable": True
        },
        "currency_code": {
            "type": "keyword",
            "description": "Code de la devise (ex: EUR, USD)",
            "aggregatable": True
        },
        "month_year": {
            "type": "keyword",
            "description": "Mois et année au format YYYY-MM",
            "aggregatable": True
        },
        "weekday": {
            "type": "keyword",
            "description": "Jour de la semaine (ex: monday, tuesday)",
            "aggregatable": True
        },
        "searchable_text": {
            "type": "text",
            "description": "Champ textuel combiné pour la recherche full-text",
            "analyzable": True
        }
    },
    "common_aggregations": {
        "total_amount": {
            "type": "sum",
            "field": "amount",
            "description": "Somme totale des montants"
        },
        "by_category": {
            "type": "terms",
            "field": "category_name",
            "description": "Regroupement par catégorie avec sous-totaux"
        },
        "by_merchant": {
            "type": "terms",
            "field": "merchant_name",
            "description": "Regroupement par marchand avec sous-totaux"
        },
        "by_date": {
            "type": "date_histogram",
            "field": "date",
            "interval": "month",
            "description": "Évolution temporelle par mois"
        },
        "statistics": {
            "type": "stats",
            "field": "amount",
            "description": "Statistiques complètes (min, max, avg, sum, count)"
        }
    },
    "search_service_format_examples": {
        "filter_by_category": {
            "description": "Filtrer par catégorie - Format search_service",
            "request": {
                "user_id": 3,
                "filters": {
                    "category_name": ["Alimentation"]
                },
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 50
            }
        },
        "filter_by_amount_range": {
            "description": "Filtrer par plage de montants - Format search_service",
            "request": {
                "user_id": 3,
                "filters": {
                    "amount_abs": {"gte": 50, "lte": 200}
                },
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 50
            }
        },
        "filter_by_date_range": {
            "description": "Filtrer par période - Format search_service",
            "request": {
                "user_id": 3,
                "filters": {
                    "date": {"gte": "2025-01-01", "lte": "2025-01-31"}
                },
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 50
            }
        },
        "aggregate_by_category": {
            "description": "Agrégation par catégorie avec somme - Format search_service",
            "request": {
                "user_id": 3,
                "filters": {},
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 50,
                "aggregations": {
                    "by_category": {
                        "terms": {"field": "category_name", "size": 20},
                        "aggs": {
                            "total_amount": {"sum": {"field": "amount"}}
                        }
                    }
                }
            }
        },
        "expenses_above_amount": {
            "description": "Dépenses supérieures à un montant - Format search_service",
            "request": {
                "user_id": 3,
                "filters": {
                    "transaction_type": "debit",
                    "amount_abs": {"gt": 100}
                },
                "sort": [{"date": {"order": "desc"}}],
                "page_size": 50,
                "aggregations": {
                    "transaction_count": {
                        "value_count": {"field": "transaction_id"}
                    },
                    "total_debit": {
                        "filter": {"term": {"transaction_type": "debit"}},
                        "aggs": {
                            "sum_amount": {"sum": {"field": "amount_abs"}}
                        }
                    }
                }
            }
        }
    }
}

def get_schema_description() -> str:
    """Retourne une description textuelle du schéma pour les LLMs"""
    desc = "# Elasticsearch Transaction Schema\n\n"
    desc += "## Available Fields:\n"

    for field_name, field_info in ELASTICSEARCH_SCHEMA["fields"].items():
        desc += f"- **{field_name}** ({field_info['type']}): {field_info['description']}\n"
        if "values" in field_info:
            desc += f"  Possible values: {', '.join(field_info['values'])}\n"
        if field_info.get("aggregatable"):
            desc += f"  Can be used in aggregations\n"

    desc += "\n## Common Aggregation Patterns:\n"
    for agg_name, agg_info in ELASTICSEARCH_SCHEMA["common_aggregations"].items():
        desc += f"- **{agg_name}**: {agg_info['description']}\n"

    desc += "\n## IMPORTANT: Search Service Format Examples\n"
    desc += "You MUST generate queries in the search_service format (NOT Elasticsearch DSL):\n\n"
    for example_name, example_info in ELASTICSEARCH_SCHEMA["search_service_format_examples"].items():
        desc += f"### {example_name}\n"
        desc += f"{example_info['description']}\n"
        import json
        desc += f"```json\n{json.dumps(example_info['request'], indent=2, ensure_ascii=False)}\n```\n\n"

    return desc

def get_query_template(template_name: str) -> dict:
    """Retourne un template de query au format search_service"""
    example = ELASTICSEARCH_SCHEMA["search_service_format_examples"].get(template_name, {})
    return example.get("request", {})
