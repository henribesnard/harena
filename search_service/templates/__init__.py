"""
Search Service Templates Module

Ce module contient les templates de requêtes Elasticsearch optimisés
pour chaque type d'intention financière identifiée par le Conversation Service.

Architecture:
- QueryTemplates: Templates requêtes par intention (12+104 catégories)
- AggregationTemplates: Templates agrégations financières
- TemplateRegistry: Registre centralisé templates avec validation
"""

from .query_templates import (
    QueryTemplates,
    FINANCIAL_QUERY_TEMPLATES,
    INTENT_TEMPLATE_MAPPING,
    QueryTemplateBuilder,
    TemplateValidationError
)

from .aggregation_templates import (
    AggregationTemplates,
    FINANCIAL_AGGREGATION_TEMPLATES,
    TEMPORAL_AGGREGATION_TEMPLATES,
    CATEGORICAL_AGGREGATION_TEMPLATES,
    AggregationTemplateBuilder,
    AggregationValidationError
)

__all__ = [
    # Query Templates
    "QueryTemplates",
    "FINANCIAL_QUERY_TEMPLATES", 
    "INTENT_TEMPLATE_MAPPING",
    "QueryTemplateBuilder",
    "TemplateValidationError",
    
    # Aggregation Templates
    "AggregationTemplates",
    "FINANCIAL_AGGREGATION_TEMPLATES",
    "TEMPORAL_AGGREGATION_TEMPLATES", 
    "CATEGORICAL_AGGREGATION_TEMPLATES",
    "AggregationTemplateBuilder",
    "AggregationValidationError"
]

# Version des templates pour versionning
TEMPLATE_VERSION = "1.0.0"

# Configuration templates par défaut
DEFAULT_TEMPLATE_CONFIG = {
    "timeout": "30s",
    "size": 20,
    "track_total_hits": True,
    "explain": False,
    "highlight": {
        "enabled": False,
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"]
    }
}