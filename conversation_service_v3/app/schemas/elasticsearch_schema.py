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
    "query_examples": {
        "filter_by_category": {
            "description": "Filtrer par catégorie",
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": "<USER_ID>"}},
                        {"term": {"category_name": "<CATEGORY>"}}
                    ]
                }
            }
        },
        "filter_by_amount_range": {
            "description": "Filtrer par plage de montants",
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": "<USER_ID>"}},
                        {"range": {"amount": {"gte": "<MIN>", "lte": "<MAX>"}}}
                    ]
                }
            }
        },
        "filter_by_date_range": {
            "description": "Filtrer par période",
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": "<USER_ID>"}},
                        {"range": {"date": {"gte": "<START_DATE>", "lte": "<END_DATE>"}}}
                    ]
                }
            }
        },
        "aggregate_by_category": {
            "description": "Agrégation par catégorie avec somme",
            "query": {
                "bool": {
                    "must": [{"term": {"user_id": "<USER_ID>"}}]
                }
            },
            "aggs": {
                "by_category": {
                    "terms": {"field": "category_name", "size": 20},
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount"}}
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

    return desc

def get_query_template(template_name: str) -> dict:
    """Retourne un template de query prêt à être modifié"""
    return ELASTICSEARCH_SCHEMA["query_examples"].get(template_name, {})
