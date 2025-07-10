"""
Query Templates - Search Service

Module principal des templates de requêtes Elasticsearch optimisés pour les intentions financières.
Point d'entrée unifié qui importe et expose toutes les fonctionnalités des sous-modules spécialisés.

Architecture modulaire:
- config.py: Configuration et constantes
- exceptions.py: Gestion d'erreurs spécialisée  
- text_search.py: Templates recherche textuelle
- financial_templates.py: Templates financiers spécialisés
- query_builder.py: Builder pattern pour requêtes complexes
- template_manager.py: Gestionnaire central avec cache

Usage:
    from search_service.templates import QueryTemplateManager, QueryTemplateBuilder
    
    # Gestionnaire principal
    manager = QueryTemplateManager()
    query = manager.get_template(IntentType.MERCHANT_SEARCH, merchant_name="McDonald's", user_id=123)
    
    # Builder pour requêtes complexes
    builder = QueryTemplateBuilder(user_id=123)
    complex_query = (builder
        .add_text_search("restaurant")
        .add_amount_range(min_amount=10.0)
        .add_date_range(start_date=datetime.now() - timedelta(days=30))
        .optimize_for_intent(IntentType.SPENDING_ANALYSIS)
        .build())
"""

from typing import Dict, Any, List, Optional, Union
import logging

# Import de la configuration
from .config import (
    TEMPLATE_CONFIG,
    FIELD_MAPPINGS,
    FIELD_GROUPS,
    FUZZINESS_CONFIG,
    BM25_CONFIG,
    HIGHLIGHT_CONFIG,
    PERFORMANCE_CONFIG,
    PREDEFINED_TEMPLATES,
    get_field_boost,
    get_field_group,
    get_fuzziness_for_field_type,
    get_predefined_template
)

# Import des exceptions
from .exceptions import (
    QueryTemplateError,
    TemplateNotFoundError,
    TemplateValidationError,
    TemplateRenderError,
    InvalidParametersError,
    QueryBuilderError,
    TemplateConfigurationError,
    CacheError,
    PerformanceError,
    handle_template_error,
    create_validation_error,
    create_parameter_error
)

# Import des templates spécialisés
from .text_search import (
    TextSearchTemplates,
    validate_text_query_params,
    optimize_text_search_for_performance,
    create_adaptive_text_search
)

from .financial_templates import (
    FinancialQueryTemplates,
    validate_financial_params,
    create_financial_filter_combination,
    optimize_financial_query_performance
)

# Import du builder
from .query_builder import (
    QueryTemplateBuilder,
    create_quick_query,
    validate_builder_query
)

# Import du gestionnaire principal
from .template_manager import (
    QueryTemplateManager,
    QueryTemplateMetadata,
    TemplateCache,
    create_template_from_intent,
    validate_query_template,
    optimize_query_for_performance
)

# Import des modèles pour les types
from ..models.service_contracts import IntentType, QueryType

logger = logging.getLogger(__name__)

# ==================== FAÇADE PRINCIPALE ====================

class QueryTemplates:
    """
    Façade principale pour l'accès aux templates de requêtes.
    Simplifie l'utilisation en exposant les fonctionnalités les plus communes.
    """
    
    def __init__(self, cache_enabled: bool = True):
        """Initialise la façade avec un gestionnaire de templates."""
        self.manager = QueryTemplateManager(cache_enabled=cache_enabled)
        self.text_templates = TextSearchTemplates()
        self.financial_templates = FinancialQueryTemplates()
    
    def get_template(self, intent_type: IntentType, **params) -> Dict[str, Any]:
        """Récupère un template selon l'intention."""
        return self.manager.get_template(intent_type, **params)
    
    def create_builder(self, user_id: Optional[int] = None) -> QueryTemplateBuilder:
        """Crée un builder de requête."""
        return self.manager.create_builder(user_id)
    
    def get_predefined(self, template_name: str, **params) -> Dict[str, Any]:
        """Récupère un template prédéfini."""
        return self.manager.get_predefined_template(template_name, params)
    
    def quick_search(self, user_id: int, query_text: str, **filters) -> Dict[str, Any]:
        """Crée rapidement une requête de recherche."""
        return create_quick_query(user_id=user_id, query_text=query_text, **filters)
    
    def clear_cache(self):
        """Vide le cache."""
        self.manager.clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques."""
        return {
            "cache_stats": self.manager.get_cache_stats(),
            "available_templates": len(self.manager.get_available_templates()),
            "predefined_templates": len(PREDEFINED_TEMPLATES)
        }

# ==================== FONCTIONS UTILITAIRES GLOBALES ====================

def create_financial_search(
    user_id: int,
    intent_type: IntentType,
    **params
) -> Dict[str, Any]:
    """
    Crée une recherche financière optimisée.
    """
    if intent_type == IntentType.MERCHANT_SEARCH:
        if "merchant_name" not in params:
            raise InvalidParametersError(missing_params=["merchant_name"])
        
        if params.get("exact_match", False):
            return FinancialQueryTemplates.merchant_exact_search(
                merchant_name=params["merchant_name"],
                user_id=user_id,
                boost=params.get("boost", 1.0)
            )
        else:
            return FinancialQueryTemplates.merchant_fuzzy_search(
                merchant_name=params["merchant_name"],
                user_id=user_id,
                fuzziness=params.get("fuzziness", "AUTO:3,6"),
                boost=params.get("boost", 1.0)
            )
    
    elif intent_type == IntentType.CATEGORY_SEARCH:
        if "category_id" in params:
            return FinancialQueryTemplates.category_search_by_id(
                category_id=params["category_id"],
                user_id=user_id,
                include_subcategories=params.get("include_subcategories", False),
                boost=params.get("boost", 1.0)
            )
        elif "category_name" in params:
            return FinancialQueryTemplates.category_search_by_name(
                category_name=params["category_name"],
                user_id=user_id,
                include_subcategories=params.get("include_subcategories", True),
                exact_match=params.get("exact_match", False),
                boost=params.get("boost", 1.0)
            )
        else:
            raise InvalidParametersError(missing_params=["category_id ou category_name"])
    
    elif intent_type == IntentType.SPENDING_ANALYSIS:
        required_params = ["period_start", "period_end"]
        missing = [p for p in required_params if p not in params]
        if missing:
            raise InvalidParametersError(missing_params=missing)
        
        return FinancialQueryTemplates.spending_analysis_template(
            user_id=user_id,
            **params
        )
    
    else:
        raise InvalidParametersError(invalid_params=["intent_type"])


def create_text_search(
    query_text: str,
    search_type: str = "best_fields",
    fields: Optional[List[str]] = None,
    **params
) -> Dict[str, Any]:
    """
    Crée une recherche textuelle optimisée.
    """
    if search_type == "best_fields":
        return TextSearchTemplates.multi_match_best_fields(
            query_text=query_text,
            fields=fields,
            **params
        )
    elif search_type == "cross_fields":
        return TextSearchTemplates.multi_match_cross_fields(
            query_text=query_text,
            fields=fields,
            **params
        )
    elif search_type == "phrase":
        return TextSearchTemplates.phrase_search(
            query_text=query_text,
            fields=fields,
            **params
        )
    elif search_type == "phrase_prefix":
        return TextSearchTemplates.phrase_prefix_search(
            query_text=query_text,
            fields=fields,
            **params
        )
    elif search_type == "fuzzy":
        field = params.get("field", "searchable_text")
        return TextSearchTemplates.fuzzy_search(
            query_text=query_text,
            field=field,
            **{k: v for k, v in params.items() if k != "field"}
        )
    elif search_type == "adaptive":
        return create_adaptive_text_search(query_text, **params)
    else:
        raise InvalidParametersError(invalid_params=["search_type"])


def create_complete_search(
    user_id: int,
    intent_type: IntentType,
    entities: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crée une recherche complète avec builder selon l'intention et les entités.
    """
    builder = QueryTemplateBuilder(user_id)
    
    # Optimiser selon l'intention
    builder.optimize_for_intent(intent_type)
    
    # Ajouter la recherche textuelle si présente
    if "query_text" in entities:
        builder.add_text_search(entities["query_text"])
    
    # Ajouter les filtres selon les entités
    if "merchant_name" in entities:
        builder.add_merchant_search(
            entities["merchant_name"],
            exact_match=entities.get("exact_match", False)
        )
    
    if "category_ids" in entities:
        builder.add_category_filter(category_ids=entities["category_ids"])
    
    if "category_names" in entities:
        builder.add_category_filter(category_names=entities["category_names"])
    
    if "min_amount" in entities or "max_amount" in entities:
        builder.add_amount_range(
            min_amount=entities.get("min_amount"),
            max_amount=entities.get("max_amount")
        )
    
    if "start_date" in entities or "end_date" in entities:
        builder.add_date_range(
            start_date=entities.get("start_date"),
            end_date=entities.get("end_date")
        )
    
    # Appliquer les options
    if options:
        if "size" in options or "from" in options:
            builder.set_pagination(
                size=options.get("size", 20),
                from_=options.get("from", 0)
            )
        
        if "highlight" in options and options["highlight"]:
            builder.enable_highlighting()
        
        if "sort" in options:
            for sort_field in options["sort"]:
                builder.add_sort(**sort_field)
    
    return builder.build()


def get_template_info() -> Dict[str, Any]:
    """
    Retourne les informations complètes sur les templates disponibles.
    """
    return {
        "version": TEMPLATE_CONFIG["version"],
        "config": TEMPLATE_CONFIG,
        "field_mappings": FIELD_MAPPINGS,
        "field_groups": FIELD_GROUPS,
        "predefined_templates": list(PREDEFINED_TEMPLATES.keys()),
        "supported_intents": [intent.value for intent in IntentType],
        "performance_limits": PERFORMANCE_CONFIG,
        "features": {
            "caching": True,
            "validation": True,
            "optimization": True,
            "builder_pattern": True,
            "predefined_templates": True,
            "custom_templates": True
        }
    }


# ==================== INSTANCE GLOBALE ====================

# Instance globale pour usage simple
_global_templates = None

def get_templates() -> QueryTemplates:
    """Retourne l'instance globale des templates."""
    global _global_templates
    if _global_templates is None:
        _global_templates = QueryTemplates()
    return _global_templates


# ==================== EXPORTS ====================

__all__ = [
    # Classes principales
    "QueryTemplates",
    "QueryTemplateManager",
    "QueryTemplateBuilder",
    "TextSearchTemplates",
    "FinancialQueryTemplates",
    "QueryTemplateMetadata",
    "TemplateCache",
    
    # Exceptions
    "QueryTemplateError",
    "TemplateNotFoundError",
    "TemplateValidationError", 
    "TemplateRenderError",
    "InvalidParametersError",
    "QueryBuilderError",
    "TemplateConfigurationError",
    "CacheError",
    "PerformanceError",
    
    # Configuration
    "TEMPLATE_CONFIG",
    "FIELD_MAPPINGS",
    "FIELD_GROUPS", 
    "FUZZINESS_CONFIG",
    "BM25_CONFIG",
    "HIGHLIGHT_CONFIG",
    "PERFORMANCE_CONFIG",
    "PREDEFINED_TEMPLATES",
    
    # Fonctions utilitaires principales
    "create_financial_search",
    "create_text_search", 
    "create_complete_search",
    "create_template_from_intent",
    "create_quick_query",
    "get_template_info",
    "get_templates",
    
    # Fonctions de validation
    "validate_query_template",
    "validate_text_query_params",
    "validate_financial_params",
    "validate_builder_query",
    
    # Fonctions d'optimisation
    "optimize_query_for_performance",
    "optimize_text_search_for_performance",
    "optimize_financial_query_performance",
    
    # Fonctions de configuration
    "get_field_boost",
    "get_field_group",
    "get_fuzziness_for_field_type",
    "get_predefined_template",
    
    # Fonctions avancées
    "create_adaptive_text_search",
    "create_financial_filter_combination",
    "handle_template_error",
    "create_validation_error",
    "create_parameter_error"
]