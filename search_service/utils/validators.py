"""
Validateurs pour le Search Service - Module Principal.

Ce module centralise tous les validateurs spécialisés et expose une API unifiée
pour la validation des requêtes, filtres, et résultats du service de recherche.

ARCHITECTURE MODULAIRE:
- validators/base.py : Validateur de base et utilitaires communs
- validators/query.py : Validation requêtes Elasticsearch
- validators/filters.py : Validation filtres de recherche
- validators/results.py : Validation résultats
- validators/parameters.py : Validation paramètres API
- validators/security.py : Validations de sécurité

USAGE SIMPLIFIÉ:
    from search_service.utils.validators import (
        QueryValidator, FilterValidator, ResultValidator
    )
    
    # Validation complète
    query_validator = QueryValidator()
    result = query_validator.validate_search_query(query_body)
    
    # Validation rapide
    from search_service.utils.validators import validate_search_request
    validated = validate_search_request(query, user_id, filters)
"""

# Imports des composants spécialisés
from .validators.base import (
    BaseValidator, ValidationResult, QueryComplexity, FieldValidationRule,
    ValidationLevel, FieldType, DEFAULT_LIMITS
)

from .validators.query import QueryValidator
from .validators.filters import FilterValidator
from .validators.results import ResultValidator
from .validators.parameters import ParameterValidator
from .validators.security import SecurityValidator

# Exceptions centralisées
from .validators.base import (
    ValidationError, QueryValidationError, FilterValidationError,
    ResultValidationError, ParameterValidationError, SecurityValidationError
)

# Fonctions utilitaires de haut niveau
from .validators.utils import (
    validate_search_request, validate_user_id, validate_amount,
    validate_date, sanitize_query, is_safe_query, 
    escape_elasticsearch_query
)

# Configuration et constantes
from .validators.config import (
    ALLOWED_SEARCH_FIELDS, SENSITIVE_FIELDS, ALLOWED_QUERY_TYPES,
    DANGEROUS_PATTERNS, ES_SPECIAL_CHARS
)

# Factory functions pour création rapide
def create_query_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    **kwargs
) -> QueryValidator:
    """Crée un validateur de requêtes configuré."""
    return QueryValidator(validation_level=validation_level, **kwargs)

def create_filter_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> FilterValidator:
    """Crée un validateur de filtres configuré."""
    return FilterValidator(validation_level=validation_level)

def create_result_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> ResultValidator:
    """Crée un validateur de résultats configuré."""
    return ResultValidator(validation_level=validation_level)

def create_parameter_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> ParameterValidator:
    """Crée un validateur de paramètres configuré."""
    return ParameterValidator(validation_level=validation_level)

def create_security_validator(
    validation_level: ValidationLevel = ValidationLevel.STRICT
) -> SecurityValidator:
    """Crée un validateur de sécurité configuré."""
    return SecurityValidator(validation_level=validation_level)

# Exports principaux
__all__ = [
    # Classes principales
    'BaseValidator',
    'QueryValidator', 
    'FilterValidator',
    'ResultValidator',
    'ParameterValidator',
    'SecurityValidator',
    
    # Exceptions
    'ValidationError',
    'QueryValidationError',
    'FilterValidationError', 
    'ResultValidationError',
    'ParameterValidationError',
    'SecurityValidationError',
    
    # Structures de données
    'ValidationResult',
    'QueryComplexity',
    'FieldValidationRule',
    'ValidationLevel',
    'FieldType',
    
    # Fonctions utilitaires
    'validate_search_request',
    'validate_user_id',
    'validate_amount',
    'validate_date',
    'sanitize_query',
    'is_safe_query',
    'escape_elasticsearch_query',
    
    # Factory functions
    'create_query_validator',
    'create_filter_validator',
    'create_result_validator',
    'create_parameter_validator',
    'create_security_validator',
    
    # Configuration
    'ALLOWED_SEARCH_FIELDS',
    'SENSITIVE_FIELDS',
    'ALLOWED_QUERY_TYPES',
    'DEFAULT_LIMITS'
]