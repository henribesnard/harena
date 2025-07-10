"""
Templates Module pour Search Service

Module responsable de la gestion des templates de requêtes Elasticsearch 
optimisés pour les intentions financières et les agrégations.

Ce module fournit:
- Templates de requêtes par type d'intention (12+104 catégories)  
- Templates d'agrégations financières
- Validation et rendu des templates
- Cache des templates compilés
- Versioning des templates

Architecture:
- query_templates.py: Templates requêtes par intention
- aggregation_templates.py: Templates agrégations spécialisées

Utilisation:
    from search_service.templates import QueryTemplateManager, AggregationTemplateManager
    
    # Templates de requêtes
    query_mgr = QueryTemplateManager()
    template = query_mgr.get_template_by_intent(IntentType.SPENDING_ANALYSIS)
    
    # Templates d'agrégations  
    agg_mgr = AggregationTemplateManager()
    agg_template = agg_mgr.get_spending_evolution_template()
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

# ==================== TYPES ET ENUMS ====================

class TemplateType(str, Enum):
    """Types de templates disponibles."""
    
    # Templates de base
    TEXT_SEARCH = "text_search"
    EXACT_MATCH = "exact_match"
    FUZZY_SEARCH = "fuzzy_search"
    
    # Templates financiers
    SPENDING_ANALYSIS = "spending_analysis"
    MERCHANT_SEARCH = "merchant_search"
    CATEGORY_SEARCH = "category_search"
    AMOUNT_RANGE = "amount_range"
    DATE_RANGE = "date_range"
    
    # Templates d'agrégation
    MERCHANT_AGGREGATION = "merchant_aggregation"
    CATEGORY_AGGREGATION = "category_aggregation"
    DATE_HISTOGRAM = "date_histogram"
    AMOUNT_DISTRIBUTION = "amount_distribution"
    
    # Templates avancés
    SPENDING_EVOLUTION = "spending_evolution"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    ANOMALY_DETECTION = "anomaly_detection"

class TemplateStatus(str, Enum):
    """Statut des templates."""
    ACTIVE = "active"
    DEPRECATED = "deprecated" 
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"

# ==================== EXCEPTIONS ====================

class TemplateError(Exception):
    """Erreur de base pour les templates."""
    pass

class TemplateNotFoundError(TemplateError):
    """Template introuvable."""
    pass

class TemplateValidationError(TemplateError):
    """Erreur de validation de template."""
    pass

class TemplateRenderError(TemplateError):
    """Erreur de rendu de template."""
    pass

# ==================== CONFIGURATION ====================

TEMPLATE_CONFIG = {
    "cache_enabled": True,
    "cache_ttl_seconds": 3600,
    "validation_enabled": True,
    "auto_reload": False,
    "version": "1.0.0",
    "supported_elasticsearch_versions": ["7.x", "8.x"]
}

# Champs par défaut pour les différents types de recherche
DEFAULT_SEARCH_FIELDS = {
    "text": [
        "searchable_text^2.0",
        "primary_description^1.5", 
        "merchant_name^1.8",
        "category_name^1.2"
    ],
    "merchant": [
        "merchant_name.keyword^3.0",
        "merchant_name^2.0",
        "merchant_alias^1.5"
    ],
    "category": [
        "category_name.keyword^3.0",
        "category_name^2.0",
        "subcategory_name^1.5"
    ],
    "description": [
        "primary_description^2.0",
        "secondary_description^1.5",
        "notes^1.0"
    ]
}

# Configuration des agrégations par défaut
DEFAULT_AGGREGATION_CONFIG = {
    "merchants": {
        "size": 10,
        "min_doc_count": 1,
        "order": {"total_amount": "desc"}
    },
    "categories": {
        "size": 20,
        "min_doc_count": 1,
        "order": {"_count": "desc"}
    },
    "date_histogram": {
        "calendar_interval": "month",
        "min_doc_count": 0,
        "extended_bounds": True
    },
    "amount_ranges": {
        "ranges": [
            {"to": 10, "key": "micro"},
            {"from": 10, "to": 50, "key": "small"},
            {"from": 50, "to": 200, "key": "medium"},
            {"from": 200, "to": 1000, "key": "large"},
            {"from": 1000, "key": "very_large"}
        ]
    }
}

# ==================== IMPORTS DE MODULES ====================

# Import des gestionnaires principaux
from .query_templates import (
    QueryTemplateManager,
    FinancialQueryTemplates,
    TextSearchTemplates,
    validate_query_template,
    render_query_template
)

from .aggregation_templates import (
    AggregationTemplateManager,
    FinancialAggregationTemplates,
    DateAggregationTemplates,
    validate_aggregation_template,
    render_aggregation_template
)

# ==================== GESTIONNAIRE PRINCIPAL ====================

class TemplateManager:
    """Gestionnaire principal des templates."""
    
    def __init__(self):
        """Initialise le gestionnaire."""
        self.query_manager = QueryTemplateManager()
        self.aggregation_manager = AggregationTemplateManager()
        self._cache = {}
        
        logger.info("✅ TemplateManager initialisé")
    
    def get_query_template(self, template_type: TemplateType, **kwargs) -> Dict[str, Any]:
        """Récupère un template de requête."""
        return self.query_manager.get_template(template_type, **kwargs)
    
    def get_aggregation_template(self, template_type: TemplateType, **kwargs) -> Dict[str, Any]:
        """Récupère un template d'agrégation."""
        return self.aggregation_manager.get_template(template_type, **kwargs)
    
    def validate_template(self, template: Dict[str, Any], template_type: TemplateType) -> bool:
        """Valide un template."""
        if template_type.value.endswith("_aggregation") or template_type in [
            TemplateType.DATE_HISTOGRAM,
            TemplateType.AMOUNT_DISTRIBUTION
        ]:
            return validate_aggregation_template(template)
        else:
            return validate_query_template(template)
    
    def clear_cache(self):
        """Vide le cache des templates."""
        self._cache.clear()
        self.query_manager.clear_cache()
        self.aggregation_manager.clear_cache()
        logger.info("🗑️ Cache des templates vidé")

# ==================== HELPERS ET UTILITAIRES ====================

def get_template_info() -> Dict[str, Any]:
    """Retourne les informations sur les templates disponibles."""
    return {
        "version": TEMPLATE_CONFIG["version"],
        "query_templates": [t.value for t in TemplateType if not t.value.endswith("_aggregation")],
        "aggregation_templates": [t.value for t in TemplateType if t.value.endswith("_aggregation")],
        "config": TEMPLATE_CONFIG,
        "search_fields": DEFAULT_SEARCH_FIELDS,
        "aggregation_config": DEFAULT_AGGREGATION_CONFIG
    }

def validate_template_params(params: Dict[str, Any], required_params: List[str]) -> bool:
    """Valide les paramètres d'un template."""
    missing = set(required_params) - set(params.keys())
    if missing:
        raise TemplateValidationError(f"Paramètres manquants: {missing}")
    return True

# ==================== EXPORTS ====================

__all__ = [
    # Classes principales
    "TemplateManager",
    "QueryTemplateManager", 
    "AggregationTemplateManager",
    
    # Templates spécialisés
    "FinancialQueryTemplates",
    "TextSearchTemplates",
    "FinancialAggregationTemplates",
    "DateAggregationTemplates",
    
    # Enums
    "TemplateType",
    "TemplateStatus",
    
    # Exceptions
    "TemplateError",
    "TemplateNotFoundError",
    "TemplateValidationError", 
    "TemplateRenderError",
    
    # Fonctions de validation
    "validate_query_template",
    "validate_aggregation_template",
    "validate_template_params",
    
    # Fonctions de rendu
    "render_query_template",
    "render_aggregation_template",
    
    # Utilitaires
    "get_template_info",
    
    # Configuration
    "TEMPLATE_CONFIG",
    "DEFAULT_SEARCH_FIELDS",
    "DEFAULT_AGGREGATION_CONFIG"
]