"""
Sous-module validators - Imports centralisés.

Ce module __init__.py permet d'importer facilement les validateurs
depuis le sous-module validators/.
"""

# Imports des classes principales (lazy loading pour éviter les imports circulaires)
def _import_query_validator():
    from .query import QueryValidator
    return QueryValidator

def _import_filter_validator():
    from .filters import FilterValidator
    return FilterValidator

def _import_result_validator():
    from .results import ResultValidator
    return ResultValidator

def _import_parameter_validator():
    from .parameters import ParameterValidator
    return ParameterValidator

def _import_security_validator():
    from .security import SecurityValidator
    return SecurityValidator

# Lazy loading des classes
def __getattr__(name):
    if name == 'QueryValidator':
        return _import_query_validator()
    elif name == 'FilterValidator':
        return _import_filter_validator()
    elif name == 'ResultValidator':
        return _import_result_validator()
    elif name == 'ParameterValidator':
        return _import_parameter_validator()
    elif name == 'SecurityValidator':
        return _import_security_validator()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")