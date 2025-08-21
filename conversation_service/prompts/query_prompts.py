"""
Query Generation Prompts for Harena Conversation Service.

This module contains specialized prompts for the Query Generator Agent,
including system prompts, few-shot examples, and intent-specific templates
for optimal Elasticsearch query generation.

Key Features:
- Intent-specific query generation templates
- Few-shot examples for complex financial queries
- Elasticsearch field mapping and optimization rules
- Query validation and error handling prompts

Author: Harena Conversation Team
Created: 2025-01-31
Version: 1.0.0 - Elasticsearch Query Generation
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

# ================================
# SYSTEM PROMPT
# ================================

QUERY_GENERATION_SYSTEM_PROMPT = """Tu es un expert en génération de requêtes Elasticsearch pour le domaine financier.

**Ton rôle :**
Tu transformes les intentions utilisateur et entités extraites en requêtes Elasticsearch optimisées pour le service de recherche Harena.

**Types de requêtes supportées :**
- `simple_search` : Recherche de base avec filtres user_id
- `filtered_search` : Recherche avec filtres multiples (catégorie, marchand, montant)
- `aggregated_search` : Recherche avec agrégations (sommes, moyennes, comptages)
- `text_search` : Recherche textuelle sur descriptions et noms de marchands

**Champs Elasticsearch disponibles :**
- `user_id` (integer) : Identifiant utilisateur [OBLIGATOIRE]
- `amount` (float) : Montant transaction (peut être négatif)
- `amount_abs` (float) : Montant absolu transaction
- `merchant_name` (text/keyword) : Nom du marchand
- `category_name` (text/keyword) : Catégorie transaction
- `primary_description` (text) : Description principale
- `searchable_text` (text) : Texte searchable combiné
- `date` (date) : Date transaction (format YYYY-MM-DD)
- `month_year` (keyword) : Mois-année (format YYYY-MM)
- `currency_code` (keyword) : Code devise (EUR, USD, etc.)

**Opérateurs de filtrage :**
- `eq` : égal à
- `ne` : différent de
- `gt` : supérieur à
- `gte` : supérieur ou égal à
- `lt` : inférieur à
- `lte` : inférieur ou égal à
- `in` : dans la liste
- `between` : entre deux valeurs
- `match` : correspondance textuelle

**Règles importantes :**
1. TOUJOURS inclure `{"field": "user_id", "operator": "eq", "value": USER_ID}` dans les filtres required
2. Utilise `amount_abs` pour les comparaisons de montants
3. Utilise `category_name.keyword` pour les filtres exacts de catégorie
4. Utilise `merchant_name.keyword` pour les filtres exacts de marchand
5. Pour les recherches textuelles, utilise les champs `searchable_text`, `primary_description`, `merchant_name`
6. Les agrégations sont pertinentes pour les analyses (category_analysis, temporal_analysis, spending_analysis)

**Format de sortie :**
Réponds UNIQUEMENT en JSON valide avec la structure SearchServiceQuery exacte.
Pas de texte avant ou après le JSON.

**Optimisations :**
- Limite à 20 résultats par défaut (sauf pour les agrégations)
- Timeout de 5000ms maximum
- Utilise les champs pertinents uniquement
- Applique les tris appropriés selon l'intention"""

# ================================
# FEW-SHOT EXAMPLES
# ================================

QUERY_FEW_SHOT_EXAMPLES = [
    {
        "input": """Message utilisateur : Mes dépenses restaurant ce mois
Intention détectée : CATEGORY_ANALYSIS
User ID : 34
Entités extraites :
- CATEGORY: restaurant
- DATE_RANGE: this_month

Génère une requête Elasticsearch optimisée au format SearchServiceQuery.""",
        
        "output": """{
  "query_metadata": {
    "intent_type": "CATEGORY_ANALYSIS",
    "agent_name": "query_generator",
    "timestamp": "2024-01-31T10:30:00Z"
  },
  "search_parameters": {
    "query_type": "aggregated_search",
    "fields": ["user_id", "category_name", "amount", "merchant_name", "date"],
    "limit": 50,
    "timeout_ms": 5000
  },
  "filters": {
    "required": [
      {"field": "user_id", "operator": "eq", "value": 34},
      {"field": "category_name.keyword", "operator": "eq", "value": "restaurant"}
    ],
    "ranges": [
      {"field": "date", "operator": "between", "value": ["2024-01-01", "2024-01-31"]}
    ]
  },
  "aggregations": {
    "enabled": true,
    "types": ["sum", "count"],
    "metrics": ["amount"],
    "group_by": ["merchant_name"]
  }
}"""
    },
    
    {
        "input": """Message utilisateur : Combien j'ai dépensé chez Amazon cette année ?
Intention détectée : MERCHANT_ANALYSIS
User ID : 42
Entités extraites :
- MERCHANT: Amazon
- DATE_RANGE: this_year

Génère une requête Elasticsearch optimisée au format SearchServiceQuery.""",
        
        "output": """{
  "query_metadata": {
    "intent_type": "MERCHANT_ANALYSIS",
    "agent_name": "query_generator",
    "timestamp": "2024-01-31T10:30:00Z"
  },
  "search_parameters": {
    "query_type": "text_search",
    "fields": ["user_id", "merchant_name", "amount", "date", "category_name"],
    "limit": 30,
    "timeout_ms": 4000
  },
  "filters": {
    "required": [
      {"field": "user_id", "operator": "eq", "value": 42}
    ],
    "ranges": [
      {"field": "date", "operator": "between", "value": ["2024-01-01", "2024-12-31"]}
    ]
  },
  "text_search": {
    "query": "Amazon",
    "fields": ["merchant_name", "primary_description", "searchable_text"],
    "operator": "match"
  },
  "aggregations": {
    "enabled": true,
    "types": ["sum", "count"],
    "metrics": ["amount"]
  }
}"""
    },
    
    {
        "input": """Message utilisateur : Transactions supérieures à 100 euros hier
Intention détectée : TRANSACTION_SEARCH
User ID : 67
Entités extraites :
- AMOUNT: 100
- DATE_RANGE: yesterday

Génère une requête Elasticsearch optimisée au format SearchServiceQuery.""",
        
        "output": """{
  "query_metadata": {
    "intent_type": "TRANSACTION_SEARCH",
    "agent_name": "query_generator",
    "timestamp": "2024-01-31T10:30:00Z"
  },
  "search_parameters": {
    "query_type": "filtered_search",
    "fields": ["user_id", "amount", "merchant_name", "primary_description", "date", "category_name"],
    "limit": 20,
    "timeout_ms": 3000,
    "sort": [{"amount": "desc"}]
  },
  "filters": {
    "required": [
      {"field": "user_id", "operator": "eq", "value": 67}
    ],
    "ranges": [
      {"field": "amount_abs", "operator": "gt", "value": 100},
      {"field": "date", "operator": "eq", "value": "2024-01-30"}
    ]
  }
}"""
    },
    
    {
        "input": """Message utilisateur : Évolution de mes dépenses par mois cette année
Intention détectée : TEMPORAL_ANALYSIS
User ID : 89
Entités extraites :
- DATE_RANGE: this_year

Génère une requête Elasticsearch optimisée au format SearchServiceQuery.""",
        
        "output": """{
  "query_metadata": {
    "intent_type": "TEMPORAL_ANALYSIS",
    "agent_name": "query_generator",
    "timestamp": "2024-01-31T10:30:00Z"
  },
  "search_parameters": {
    "query_type": "aggregated_search",
    "fields": ["user_id", "amount", "date", "month_year"],
    "limit": 1000,
    "timeout_ms": 6000
  },
  "filters": {
    "required": [
      {"field": "user_id", "operator": "eq", "value": 89}
    ],
    "ranges": [
      {"field": "date", "operator": "between", "value": ["2024-01-01", "2024-12-31"]}
    ]
  },
  "aggregations": {
    "enabled": true,
    "types": ["sum", "count", "avg"],
    "metrics": ["amount"],
    "group_by": ["month_year"],
    "bucket_size": "month"
  }
}"""
    },
    
    {
        "input": """Message utilisateur : Mes achats en magasin bio
Intention détectée : TEXT_SEARCH_WITH_CATEGORY
User ID : 123
Entités extraites :
- SEARCH_TERM: bio

Génère une requête Elasticsearch optimisée au format SearchServiceQuery.""",
        
        "output": """{
  "query_metadata": {
    "intent_type": "TEXT_SEARCH_WITH_CATEGORY",
    "agent_name": "query_generator",
    "timestamp": "2024-01-31T10:30:00Z"
  },
  "search_parameters": {
    "query_type": "text_search",
    "fields": ["user_id", "merchant_name", "primary_description", "amount", "date", "category_name"],
    "limit": 25,
    "timeout_ms": 4000,
    "sort": [{"date": "desc"}]
  },
  "filters": {
    "required": [
      {"field": "user_id", "operator": "eq", "value": 123}
    ]
  },
  "text_search": {
    "query": "bio",
    "fields": ["merchant_name", "primary_description", "searchable_text"],
    "operator": "match"
  }
}"""
    }
]

_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(path: Optional[str] = None, *, cache: Optional[Dict[str, str]] = None, cache_key: str = "system") -> str:
    """Charger le prompt système depuis un fichier ou un cache."""
    cache = _PROMPT_CACHE if cache is None else cache
    if cache_key in cache:
        return cache[cache_key]
    if path:
        prompt = Path(path).read_text(encoding="utf-8")
        cache[cache_key] = prompt
        return prompt
    return QUERY_GENERATION_SYSTEM_PROMPT



def get_examples() -> List[Dict[str, str]]:
    """Récupérer les exemples few-shot pour la génération de requêtes."""
    return list(QUERY_FEW_SHOT_EXAMPLES)



def update_examples(examples: List[Dict[str, str]]) -> None:
    """Mettre à jour les exemples few-shot utilisés pour la génération de requêtes."""
    QUERY_FEW_SHOT_EXAMPLES.clear()
    QUERY_FEW_SHOT_EXAMPLES.extend(examples)

# ================================
# INTENT-SPECIFIC TEMPLATES
# ================================

INTENT_QUERY_TEMPLATES = {
    "BALANCE_INQUIRY": {
        "query_type": "simple_search",
        "fields": ["user_id", "amount", "date", "merchant_name"],
        "limit": 10,
        "sort": [{"date": "desc"}],
        "aggregations": {
            "enabled": True,
            "types": ["sum"],
            "metrics": ["amount"]
        }
    },
    
    "TRANSACTION_SEARCH": {
        "query_type": "filtered_search",
        "fields": ["user_id", "amount", "merchant_name", "primary_description", "date", "category_name"],
        "limit": 20,
        "sort": [{"date": "desc"}]
    },
    
    "CATEGORY_ANALYSIS": {
        "query_type": "aggregated_search",
        "fields": ["user_id", "category_name", "amount", "merchant_name", "date"],
        "limit": 100,
        "aggregations": {
            "enabled": True,
            "types": ["sum", "count", "avg"],
            "metrics": ["amount"],
            "group_by": ["category_name"]
        }
    },
    
    "MERCHANT_ANALYSIS": {
        "query_type": "text_search",
        "fields": ["user_id", "merchant_name", "amount", "date", "category_name"],
        "limit": 30,
        "aggregations": {
            "enabled": True,
            "types": ["sum", "count"],
            "metrics": ["amount"]
        }
    },
    
    "SPENDING_ANALYSIS": {
        "query_type": "aggregated_search",
        "fields": ["user_id", "amount", "category_name", "date", "month_year"],
        "limit": 200,
        "aggregations": {
            "enabled": True,
            "types": ["sum", "count", "avg", "min", "max"],
            "metrics": ["amount"],
            "group_by": ["category_name", "month_year"]
        }
    },
    
    "TEMPORAL_ANALYSIS": {
        "query_type": "aggregated_search",
        "fields": ["user_id", "amount", "date", "month_year", "category_name"],
        "limit": 500,
        "aggregations": {
            "enabled": True,
            "types": ["sum", "count", "avg"],
            "metrics": ["amount"],
            "group_by": ["month_year"],
            "bucket_size": "month"
        }
    },
    
    "BUDGET_ANALYSIS": {
        "query_type": "aggregated_search",
        "fields": ["user_id", "amount", "category_name", "date"],
        "limit": 300,
        "aggregations": {
            "enabled": True,
            "types": ["sum", "count"],
            "metrics": ["amount"],
            "group_by": ["category_name"]
        }
    }
}

# ================================
# ELASTICSEARCH FIELD MAPPING
# ================================

ELASTICSEARCH_FIELD_MAPPING = {
    "user_identifiers": {
        "user_id": {"type": "integer", "required": True}
    },
    
    "transaction_amounts": {
        "amount": {"type": "float", "description": "Montant transaction (peut être négatif)"},
        "amount_abs": {"type": "float", "description": "Montant absolu pour comparaisons"},
        "currency_code": {"type": "keyword", "description": "Code devise (EUR, USD, etc.)"}
    },
    
    "merchant_information": {
        "merchant_name": {"type": "text/keyword", "description": "Nom du marchand"},
        "merchant_category": {"type": "keyword", "description": "Catégorie du marchand"}
    },
    
    "transaction_categorization": {
        "category_name": {"type": "text/keyword", "description": "Catégorie transaction"},
        "subcategory_name": {"type": "keyword", "description": "Sous-catégorie transaction"}
    },
    
    "transaction_descriptions": {
        "primary_description": {"type": "text", "description": "Description principale"},
        "secondary_description": {"type": "text", "description": "Description secondaire"},
        "searchable_text": {"type": "text", "description": "Texte searchable combiné"}
    },
    
    "temporal_fields": {
        "date": {"type": "date", "format": "yyyy-MM-dd", "description": "Date transaction"},
        "month_year": {"type": "keyword", "format": "yyyy-MM", "description": "Mois-année"},
        "day_of_week": {"type": "keyword", "description": "Jour de la semaine"},
        "hour": {"type": "integer", "description": "Heure de la transaction"}
    },
    
    "location_fields": {
        "city": {"type": "keyword", "description": "Ville de la transaction"},
        "country": {"type": "keyword", "description": "Pays de la transaction"}
    }
}

# ================================
# QUERY OPTIMIZATION RULES
# ================================

QUERY_OPTIMIZATION_RULES = {
    "performance_optimizations": {
        "high_volume_users": {
            "reduce_limit": True,
            "add_timeout": True,
            "optimize_fields": True
        },
        "slow_queries": {
            "reduce_aggregations": True,
            "simplify_text_search": True,
            "add_caching": True
        }
    },
    
    "intent_specific_optimizations": {
        "CATEGORY_ANALYSIS": {
            "preferred_fields": ["category_name", "amount", "date"],
            "sort_by": "amount",
            "limit_override": 100
        },
        "TEMPORAL_ANALYSIS": {
            "preferred_fields": ["date", "month_year", "amount"],
            "sort_by": "date",
            "aggregation_bucket": "month"
        },
        "MERCHANT_ANALYSIS": {
            "preferred_fields": ["merchant_name", "amount", "date"],
            "text_search_boost": True,
            "limit_override": 50
        }
    },
    
    "error_prevention": {
        "always_include_user_id": True,
        "validate_date_formats": True,
        "sanitize_text_queries": True,
        "check_field_existence": True
    }
}

# ================================
# VALIDATION PROMPTS
# ================================

QUERY_VALIDATION_PROMPTS = {
    "structure_validation": """
Vérifie que la requête générée respecte la structure SearchServiceQuery :
- query_metadata avec intent_type, agent_name, timestamp
- search_parameters avec query_type, fields, limit
- filters avec au minimum user_id dans required
- aggregations si pertinent pour l'intention
""",
    
    "field_validation": """
Vérifie que tous les champs utilisés existent dans le mapping Elasticsearch :
- Champs obligatoires : user_id
- Champs numériques : amount, amount_abs
- Champs texte : merchant_name, primary_description, searchable_text
- Champs de catégorie : category_name (utilise .keyword pour filtres exacts)
- Champs temporels : date, month_year
""",
    
    "operator_validation": """
Vérifie que les opérateurs utilisés sont valides :
- Comparaisons : eq, ne, gt, gte, lt, lte
- Plages : between (avec array de 2 valeurs)
- Listes : in (avec array de valeurs)
- Texte : match (pour text_search uniquement)
"""
}

# ================================
# ERROR HANDLING TEMPLATES
# ================================

QUERY_ERROR_TEMPLATES = {
    "missing_user_id": {
        "error": "user_id is required in all queries",
        "fix": "Add {\"field\": \"user_id\", \"operator\": \"eq\", \"value\": USER_ID} to filters.required"
    },
    
    "invalid_field": {
        "error": "Field does not exist in Elasticsearch mapping",
        "fix": "Use valid fields from ELASTICSEARCH_FIELD_MAPPING"
    },
    
    "invalid_operator": {
        "error": "Operator not supported",
        "fix": "Use valid operators: eq, ne, gt, gte, lt, lte, in, between, match"
    },
    
    "malformed_aggregation": {
        "error": "Aggregation structure is invalid",
        "fix": "Use enabled:bool, types:array, metrics:array, group_by:array structure"
    },
    
    "timeout_too_high": {
        "error": "Timeout exceeds maximum allowed",
        "fix": "Set timeout_ms to maximum 10000ms"
    }
}
