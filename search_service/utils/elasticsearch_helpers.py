"""
Helpers Elasticsearch pour le Search Service - Module Principal.

Ce module centralise tous les utilitaires Elasticsearch et expose une API unifiée
pour la construction, optimisation et traitement des requêtes dans le contexte financier.

ARCHITECTURE MODULAIRE:
- elasticsearch/builders.py : Construction de requêtes (QueryBuilder, ElasticsearchHelpers)
- elasticsearch/formatters.py : Formatage des résultats (ResultFormatter) 
- elasticsearch/scoring.py : Calcul de scores personnalisés (ScoreCalculator)
- elasticsearch/highlights.py : Traitement des highlights (HighlightProcessor)
- elasticsearch/filters.py : Gestion avancée des filtres (FilterManager)
- elasticsearch/templates.py : Templates de requêtes (TemplateManager)
- elasticsearch/config.py : Configuration et constantes

USAGE SIMPLIFIÉ:
    from search_service.utils.elasticsearch_helpers import (
        QueryBuilder, ElasticsearchHelpers, ResultFormatter
    )
    
    # Construction rapide
    query = ElasticsearchHelpers.build_financial_query(
        query="virement café", user_id=123
    )
    
    # Builder avancé
    builder = QueryBuilder()
    query = (builder
        .with_user_filter(123)
        .with_financial_search("virement café")
        .build())
"""

# Imports des composants spécialisés
from .elasticsearch.config import (
    QueryStrategy, SortStrategy, BoostType, AggregationType,
    FINANCIAL_SYNONYMS, FINANCIAL_SEARCH_FIELDS, HIGHLIGHT_FIELDS,
    DEFAULT_BOOST_VALUES, AMOUNT_AGGREGATION_BUCKETS
)

from .elasticsearch.builders import (
    ElasticsearchHelpers, QueryBuilder, QueryContext
)

from .elasticsearch.formatters import (
    ResultFormatter, FormattedHit, AggregationResult
)

from .elasticsearch.scoring import (
    ScoreCalculator
)

from .elasticsearch.highlights import (
    HighlightProcessor
)

from .elasticsearch.filters import (
    FilterManager
)

from .elasticsearch.templates import (
    TemplateManager
)

# Fonctions utilitaires de haut niveau
from .elasticsearch.utils import (
    format_search_results, extract_highlights, calculate_relevance_score,
    optimize_query_for_performance, build_suggestion_query,
    validate_query_structure
)

# Factory functions pour création rapide
def create_query_builder() -> QueryBuilder:
    """Crée un QueryBuilder configuré."""
    return QueryBuilder()

def create_result_formatter(**kwargs) -> ResultFormatter:
    """Crée un ResultFormatter configuré."""
    return ResultFormatter(**kwargs)

def create_score_calculator(**kwargs) -> ScoreCalculator:
    """Crée un ScoreCalculator configuré."""
    return ScoreCalculator(**kwargs)

def create_highlight_processor(**kwargs) -> HighlightProcessor:
    """Crée un HighlightProcessor configuré."""
    return HighlightProcessor(**kwargs)

def create_filter_manager() -> FilterManager:
    """Crée un FilterManager configuré."""
    return FilterManager()

def create_template_manager() -> TemplateManager:
    """Crée un TemplateManager configuré."""
    return TemplateManager()

# Exports principaux
__all__ = [
    # Classes principales
    'ElasticsearchHelpers',
    'QueryBuilder',
    'ResultFormatter', 
    'ScoreCalculator',
    'HighlightProcessor',
    'FilterManager',
    'TemplateManager',
    
    # Structures de données
    'QueryContext',
    'FormattedHit',
    'AggregationResult',
    'QueryStrategy',
    'SortStrategy',
    'BoostType',
    'AggregationType',
    
    # Fonctions utilitaires
    'format_search_results',
    'extract_highlights',
    'calculate_relevance_score',
    'optimize_query_for_performance',
    'build_suggestion_query',
    'validate_query_structure',
    
    # Factory functions
    'create_query_builder',
    'create_result_formatter',
    'create_score_calculator',
    'create_highlight_processor',
    'create_filter_manager',
    'create_template_manager',
    
    # Configuration
    'FINANCIAL_SYNONYMS',
    'FINANCIAL_SEARCH_FIELDS',
    'HIGHLIGHT_FIELDS',
    'DEFAULT_BOOST_VALUES',
    'AMOUNT_AGGREGATION_BUCKETS'
]