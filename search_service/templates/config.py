"""
Configuration des Templates - Search Service

Configuration centralisée pour tous les templates de requêtes Elasticsearch.
Définit les mappings de champs, groupes de champs, et paramètres d'optimisation.
"""

from typing import Dict, Any, List
from enum import Enum

# ==================== CONFIGURATION GLOBALE ====================

TEMPLATE_CONFIG = {
    "version": "1.0.0",
    "default_size": 20,
    "max_size": 1000,
    "default_from": 0,
    "enable_highlighting": True,
    "enable_explain": False,
    "default_fuzziness": "AUTO",
    "min_score": 0.1,
    "timeout": "30s",
    "track_total_hits": True,
    "max_inner_hits": 100,
    "cache_enabled": True,
    "cache_size": 1000,
    "cache_ttl_seconds": 3600
}

# ==================== MAPPING DES CHAMPS ====================

FIELD_MAPPINGS = {
    # Champs textuels principaux
    "searchable_text": {
        "boost": 2.0,
        "analyzer": "financial_text_analyzer",
        "type": "text"
    },
    "primary_description": {
        "boost": 1.8,
        "analyzer": "standard",
        "type": "text"
    },
    "secondary_description": {
        "boost": 1.2,
        "analyzer": "standard", 
        "type": "text"
    },
    "notes": {
        "boost": 1.0,
        "analyzer": "simple",
        "type": "text"
    },
    
    # Champs marchands
    "merchant_name": {
        "boost": 2.5,
        "analyzer": "keyword_analyzer",
        "type": "text"
    },
    "merchant_name.keyword": {
        "boost": 3.0,
        "type": "keyword"
    },
    "merchant_alias": {
        "boost": 1.8,
        "analyzer": "standard",
        "type": "text"
    },
    
    # Champs catégories
    "category_name": {
        "boost": 2.0,
        "analyzer": "category_analyzer",
        "type": "text"
    },
    "category_name.keyword": {
        "boost": 2.8,
        "type": "keyword"
    },
    "subcategory_name": {
        "boost": 1.5,
        "analyzer": "standard",
        "type": "text"
    },
    
    # Champs numériques
    "amount": {
        "type": "double",
        "boost": 1.0
    },
    "amount_abs": {
        "type": "double", 
        "boost": 1.0
    },
    
    # Champs de dates
    "transaction_date": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
    },
    "created_at": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
    },
    
    # Champs identifiants
    "transaction_id": {
        "type": "keyword"
    },
    "user_id": {
        "type": "long"
    },
    "category_id": {
        "type": "long"
    },
    "merchant_id": {
        "type": "long"
    }
}

# ==================== GROUPES DE CHAMPS ====================

FIELD_GROUPS = {
    "all_text": [
        "searchable_text^2.0",
        "primary_description^1.8", 
        "secondary_description^1.2",
        "notes^1.0",
        "merchant_name^2.5",
        "category_name^2.0"
    ],
    
    "primary_text": [
        "searchable_text^2.5",
        "primary_description^2.0"
    ],
    
    "merchant_fields": [
        "merchant_name.keyword^3.0",
        "merchant_name^2.5",
        "merchant_alias^1.8"
    ],
    
    "category_fields": [
        "category_name.keyword^2.8",
        "category_name^2.0",
        "subcategory_name^1.5"
    ],
    
    "description_fields": [
        "primary_description^2.0",
        "secondary_description^1.5",
        "notes^1.0"
    ]
}

# ==================== CONFIGURATION FUZZINESS ====================

FUZZINESS_CONFIG = {
    "text_fields": "AUTO:4,7",
    "merchant_fields": "AUTO:3,6",
    "category_fields": "1",
    "strict_fields": "0"
}

# ==================== CONFIGURATION BM25 ====================

BM25_CONFIG = {
    "k1": 1.2,
    "b": 0.75,
    "discount_overlaps": True
}

# ==================== CONFIGURATION HIGHLIGHTING ====================

HIGHLIGHT_CONFIG = {
    "pre_tags": ["<mark>"],
    "post_tags": ["</mark>"],
    "fragment_size": 150,
    "number_of_fragments": 3,
    "default_fields": [
        "searchable_text",
        "primary_description", 
        "merchant_name"
    ]
}

# ==================== CONFIGURATION PERFORMANCE ====================

PERFORMANCE_CONFIG = {
    "max_clauses": 1024,
    "max_expansions": 50,
    "prefix_length": 1,
    "tie_breaker": 0.3,
    "minimum_should_match": "75%",
    "boost_factor_limit": 10.0,
    "decay_functions_limit": 5
}

# ==================== TEMPLATES PRÉDÉFINIS ====================

PREDEFINED_TEMPLATES = {
    "text_search_simple": {
        "multi_match": {
            "query": "{{query_text}}",
            "fields": FIELD_GROUPS["primary_text"],
            "type": "best_fields",
            "fuzziness": "AUTO",
            "tie_breaker": 0.3
        }
    },
    
    "merchant_exact": {
        "bool": {
            "must": [
                {"term": {"merchant_name.keyword": "{{merchant_name}}"}}
            ],
            "filter": [
                {"term": {"user_id": "{{user_id}}"}}
            ]
        }
    },
    
    "amount_range": {
        "bool": {
            "must": [
                {"range": {"amount_abs": {
                    "gte": "{{min_amount}}",
                    "lte": "{{max_amount}}"
                }}}
            ],
            "filter": [
                {"term": {"user_id": "{{user_id}}"}}
            ]
        }
    },
    
    "date_range": {
        "bool": {
            "must": [
                {"range": {"transaction_date": {
                    "gte": "{{start_date}}",
                    "lte": "{{end_date}}",
                    "time_zone": "{{timezone}}"
                }}}
            ],
            "filter": [
                {"term": {"user_id": "{{user_id}}"}}
            ]
        }
    },
    
    "category_filter": {
        "bool": {
            "filter": [
                {"terms": {"category_id": "{{category_ids}}"}},
                {"term": {"user_id": "{{user_id}}"}}
            ]
        }
    }
}

# ==================== FONCTIONS UTILITAIRES ====================

def get_field_boost(field_name: str) -> float:
    """Retourne le boost d'un champ."""
    return FIELD_MAPPINGS.get(field_name, {}).get("boost", 1.0)

def get_field_group(group_name: str) -> List[str]:
    """Retourne un groupe de champs."""
    return FIELD_GROUPS.get(group_name, [])

def get_fuzziness_for_field_type(field_type: str) -> str:
    """Retourne la configuration de fuzziness pour un type de champ."""
    return FUZZINESS_CONFIG.get(field_type, "AUTO")

def get_predefined_template(template_name: str) -> Dict[str, Any]:
    """Retourne un template prédéfini."""
    if template_name not in PREDEFINED_TEMPLATES:
        raise ValueError(f"Template prédéfini '{template_name}' non trouvé")
    return PREDEFINED_TEMPLATES[template_name].copy()

def validate_template_config() -> bool:
    """Valide la configuration des templates."""
    try:
        # Vérifier que tous les champs dans les groupes existent
        for group_name, fields in FIELD_GROUPS.items():
            for field in fields:
                field_name = field.split('^')[0]  # Enlever le boost
                if field_name not in FIELD_MAPPINGS:
                    raise ValueError(f"Champ {field_name} dans le groupe {group_name} non défini dans FIELD_MAPPINGS")
        
        # Vérifier les valeurs de configuration
        if TEMPLATE_CONFIG["default_size"] > TEMPLATE_CONFIG["max_size"]:
            raise ValueError("default_size ne peut pas être supérieur à max_size")
        
        return True
    except Exception as e:
        print(f"Erreur validation configuration: {e}")
        return False

# Valider la configuration au chargement
if not validate_template_config():
    raise RuntimeError("Configuration des templates invalide")