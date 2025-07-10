"""
Sous-module elasticsearch - Imports centralisés.

Ce module __init__.py permet d'importer facilement les helpers Elasticsearch
depuis le sous-module elasticsearch/.
"""

# Imports des classes principales (lazy loading pour éviter les imports circulaires)
def _import_elasticsearch_helpers():
    from .builders import ElasticsearchHelpers
    return ElasticsearchHelpers

def _import_query_builder():
    from .builders import QueryBuilder
    return QueryBuilder

def _import_result_formatter():
    from .formatters import ResultFormatter
    return ResultFormatter

def _import_score_calculator():
    from .scoring import ScoreCalculator
    return ScoreCalculator

def _import_highlight_processor():
    from .highlights import HighlightProcessor
    return HighlightProcessor

def _import_filter_manager():
    from .filters import FilterManager
    return FilterManager

def _import_template_manager():
    from .templates import TemplateManager
    return TemplateManager

# Lazy loading des classes
def __getattr__(name):
    if name == 'ElasticsearchHelpers':
        return _import_elasticsearch_helpers()
    elif name == 'QueryBuilder':
        return _import_query_builder()
    elif name == 'ResultFormatter':
        return _import_result_formatter()
    elif name == 'ScoreCalculator':
        return _import_score_calculator()
    elif name == 'HighlightProcessor':
        return _import_highlight_processor()
    elif name == 'FilterManager':
        return _import_filter_manager()
    elif name == 'TemplateManager':
        return _import_template_manager()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")