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
- validators/utils.py : Fonctions utilitaires
- validators/config.py : Configuration et constantes

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

import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

# ==================== IMPORTS DES SOUS-MODULES ====================

# Import des composants de base
try:
    from .validators.base import (
        BaseValidator, ValidationResult, QueryComplexity, FieldValidationRule,
        ValidationLevel, FieldType, DEFAULT_LIMITS
    )
    _BASE_AVAILABLE = True
except ImportError:
    logger.warning("validators.base module not available - using fallback implementations")
    _BASE_AVAILABLE = False
    
    # Implémentations de fallback
    from enum import Enum
    from dataclasses import dataclass, field
    
    class ValidationLevel(str, Enum):
        BASIC = "basic"
        STANDARD = "standard"
        STRICT = "strict"
        PARANOID = "paranoid"
    
    class FieldType(str, Enum):
        STRING = "string"
        INTEGER = "integer"
        FLOAT = "float"
        BOOLEAN = "boolean"
        DATE = "date"
        USER_ID = "user_id"
        AMOUNT = "amount"
    
    @dataclass
    class ValidationResult:
        is_valid: bool
        errors: List[str] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)
        sanitized_data: Optional[Any] = None
        validation_time_ms: float = 0.0
        security_flags: List[str] = field(default_factory=list)
        
        def add_error(self, error: str, field: str = ""):
            self.is_valid = False
            error_msg = f"{field}: {error}" if field else error
            self.errors.append(error_msg)
        
        def add_warning(self, warning: str, field: str = ""):
            warning_msg = f"{field}: {warning}" if field else warning
            self.warnings.append(warning_msg)
        
        def add_security_flag(self, flag: str):
            self.security_flags.append(flag)
    
    @dataclass
    class QueryComplexity:
        score: int = 0
        nested_depth: int = 0
        bool_clauses: int = 0
        wildcard_count: int = 0
        regexp_count: int = 0
        function_score_count: int = 0
        
        @property
        def is_complex(self) -> bool:
            return self.score > 100
    
    @dataclass 
    class FieldValidationRule:
        field_type: FieldType
        required: bool = False
        min_length: Optional[int] = None
        max_length: Optional[int] = None
    
    class BaseValidator:
        def __init__(self, validation_level=ValidationLevel.STANDARD):
            self.validation_level = validation_level
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        def validate(self, data, **kwargs):
            return ValidationResult(is_valid=True)
    
    DEFAULT_LIMITS = {"max_query_length": 1000, "max_results_limit": 1000}

# Import du validateur de requêtes
try:
    from .validators.query import QueryValidator
    _QUERY_AVAILABLE = True
except ImportError:
    logger.warning("validators.query module not available - using fallback implementation")
    _QUERY_AVAILABLE = False
    
    class QueryValidator(BaseValidator):
        def validate_search_query(self, query_body):
            result = ValidationResult(is_valid=True)
            if not isinstance(query_body, dict):
                result.is_valid = False
                result.add_error("Query body must be a dictionary")
            return result
        
        def validate(self, query, **kwargs):
            if isinstance(query, dict) and "query" in query:
                return self.validate_search_query(query["query"])
            return self.validate_search_query(query)

# Import du validateur de filtres
try:
    from .validators.filters import FilterValidator
    _FILTERS_AVAILABLE = True
except ImportError:
    logger.warning("validators.filters module not available - using fallback implementation")
    _FILTERS_AVAILABLE = False
    
    class FilterValidator(BaseValidator):
        def validate(self, filters):
            result = ValidationResult(is_valid=True)
            if not isinstance(filters, dict):
                result.add_error("Filters must be a dictionary")
            return result

# Import du validateur de résultats
try:
    from .validators.results import ResultValidator
    _RESULTS_AVAILABLE = True
except ImportError:
    logger.warning("validators.results module not available - using fallback implementation")
    _RESULTS_AVAILABLE = False
    
    class ResultValidator(BaseValidator):
        def validate(self, results):
            result = ValidationResult(is_valid=True)
            if not isinstance(results, dict):
                result.add_error("Results must be a dictionary")
            elif "hits" not in results:
                result.add_error("Results must contain 'hits' field")
            return result

# Import du validateur de paramètres
try:
    from .validators.parameters import ParameterValidator
    _PARAMETERS_AVAILABLE = True
except ImportError:
    logger.warning("validators.parameters module not available - using fallback implementation")
    _PARAMETERS_AVAILABLE = False
    
    class ParameterValidator(BaseValidator):
        def validate(self, parameters):
            result = ValidationResult(is_valid=True)
            if not isinstance(parameters, dict):
                result.add_error("Parameters must be a dictionary")
            return result
        
        def validate_pagination(self, size=None, from_=None, page=None):
            result = ValidationResult(is_valid=True)
            if size is not None and (not isinstance(size, int) or size < 0):
                result.add_error("Size must be a positive integer")
            if from_ is not None and (not isinstance(from_, int) or from_ < 0):
                result.add_error("From must be a positive integer")
            if page is not None and (not isinstance(page, int) or page < 1):
                result.add_error("Page must be >= 1")
            return result

# Import du validateur de sécurité
try:
    from .validators.security import SecurityValidator
    _SECURITY_AVAILABLE = True
except ImportError:
    logger.warning("validators.security module not available - using fallback implementation")
    _SECURITY_AVAILABLE = False
    
    class SecurityValidator(BaseValidator):
        def validate(self, data, context="general"):
            result = ValidationResult(is_valid=True)
            
            if isinstance(data, str):
                # Validation basique contre les patterns dangereux
                dangerous_patterns = ["<script", "javascript:", "eval(", "union select"]
                for pattern in dangerous_patterns:
                    if pattern.lower() in data.lower():
                        result.add_error("Potentially dangerous content detected")
                        result.add_security_flag(f"dangerous_pattern: {pattern}")
                        break
            
            return result

# Import des exceptions
try:
    from .validators.base import (
        ValidationError, QueryValidationError, FilterValidationError,
        ResultValidationError, ParameterValidationError, SecurityValidationError
    )
    _EXCEPTIONS_AVAILABLE = True
except ImportError:
    logger.warning("Exception classes not available - using fallback implementations")
    _EXCEPTIONS_AVAILABLE = False
    
    class ValidationError(Exception):
        def __init__(self, message, field="", details=None):
            super().__init__(message)
            self.field = field
            self.details = details or {}
    
    class QueryValidationError(ValidationError):
        pass
    
    class FilterValidationError(ValidationError):
        pass
    
    class ResultValidationError(ValidationError):
        pass
    
    class ParameterValidationError(ValidationError):
        pass
    
    class SecurityValidationError(ValidationError):
        pass

# Import des fonctions utilitaires
try:
    from .validators.utils import (
        validate_search_request, validate_user_id, validate_amount,
        validate_date, sanitize_query, is_safe_query, 
        escape_elasticsearch_query
    )
    _UTILS_AVAILABLE = True
except ImportError:
    logger.warning("validators.utils module not available - using fallback implementations")
    _UTILS_AVAILABLE = False
    
    # Implémentations de fallback
    def validate_search_request(query: str, user_id: int, filters: Dict = None) -> ValidationResult:
        """Validation basique d'une requête de recherche."""
        result = ValidationResult(is_valid=True)
        
        if not query or not isinstance(query, str):
            result.add_error("Query is required and must be a string")
        
        if not user_id or not isinstance(user_id, int) or user_id <= 0:
            result.add_error("Valid user_id is required")
        
        if filters and not isinstance(filters, dict):
            result.add_error("Filters must be a dictionary if provided")
        
        return result
    
    def validate_user_id(user_id: Any) -> bool:
        """Valide un ID utilisateur."""
        return isinstance(user_id, int) and user_id > 0
    
    def validate_amount(amount: Any) -> bool:
        """Valide un montant."""
        try:
            amount_float = float(amount)
            return 0 <= amount_float <= 999999999.99
        except (ValueError, TypeError):
            return False
    
    def validate_date(date_value: Any) -> bool:
        """Valide une date."""
        from datetime import datetime
        if isinstance(date_value, datetime):
            return True
        if isinstance(date_value, str):
            try:
                datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                return True
            except ValueError:
                return False
        return False
    
    def sanitize_query(query: str) -> str:
        """Nettoie une requête."""
        import re
        if not isinstance(query, str):
            return ""
        # Suppression des caractères dangereux
        sanitized = re.sub(r'[<>"\';\\]', '', query)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized[:1000]  # Limitation longueur
    
    def is_safe_query(query: str) -> bool:
        """Vérifie si une requête est sûre."""
        import re
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript\s*:',
            r'eval\s*\(',
            r'union\s+select'
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        return True
    
    def escape_elasticsearch_query(query: str) -> str:
        """Échappe une requête Elasticsearch."""
        if not isinstance(query, str):
            return query
        special_chars = r'+-=&|><!(){}[]^"~*?:\/\\'
        escaped = query
        for char in special_chars:
            escaped = escaped.replace(char, f"\\{char}")
        return escaped

# Import de la configuration
try:
    from .validators.config import (
        ALLOWED_SEARCH_FIELDS, SENSITIVE_FIELDS, ALLOWED_QUERY_TYPES,
        DANGEROUS_PATTERNS, ES_SPECIAL_CHARS, VALIDATION_LIMITS
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("validators.config module not available - using fallback constants")
    _CONFIG_AVAILABLE = False
    
    # Constantes de fallback
    ALLOWED_SEARCH_FIELDS = {
        'searchable_text', 'primary_description', 'clean_description',
        'provider_description', 'merchant_name', 'category_id',
        'transaction_date', 'amount', 'user_id', 'account_id',
        'transaction_type', 'currency_code', 'operation_type'
    }
    
    SENSITIVE_FIELDS = {'user_id', 'account_id', 'amount'}
    
    ALLOWED_QUERY_TYPES = {
        'match', 'match_all', 'match_phrase', 'multi_match',
        'term', 'terms', 'range', 'bool', 'wildcard', 'regexp'
    }
    
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript\s*:',
        r'eval\s*\(',
        r'union\s+select'
    ]
    
    ES_SPECIAL_CHARS = r'+-=&|><!(){}[]^"~*?:\/\\'
    
    VALIDATION_LIMITS = {
        "max_query_length": 1000,
        "max_results_limit": 1000,
        "max_filter_values": 100,
        "max_timeout_ms": 30000
    }

# ==================== FACTORY FUNCTIONS ====================

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

# ==================== FONCTIONS UTILITAIRES AVANCÉES ====================

def validate_elasticsearch_query(query_dict: Dict[str, Any]) -> ValidationResult:
    """
    Valide une requête Elasticsearch complète.
    
    Args:
        query_dict: Dictionnaire de requête Elasticsearch
        
    Returns:
        ValidationResult avec détails
    """
    validator = create_query_validator()
    if hasattr(validator, 'validate'):
        return validator.validate(query_dict)
    else:
        return validator.validate_search_query(query_dict.get("query", {}))

def validate_financial_search(query: str, user_id: int, amount_filter: Dict = None) -> ValidationResult:
    """
    Valide une recherche financière avec paramètres spécialisés.
    
    Args:
        query: Texte de recherche
        user_id: ID utilisateur
        amount_filter: Filtre de montant optionnel
        
    Returns:
        ValidationResult
    """
    result = validate_search_request(query, user_id)
    
    if amount_filter and isinstance(amount_filter, dict):
        for key, value in amount_filter.items():
            if key in ['min_amount', 'max_amount']:
                if not validate_amount(value):
                    result.is_valid = False
                    result.add_error(f"Invalid {key}: {value}")
    
    return result

def sanitize_elasticsearch_query(query_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitise une requête Elasticsearch complète.
    
    Args:
        query_dict: Requête à sanitiser
        
    Returns:
        Requête sanitisée
    """
    def sanitize_node(node):
        if isinstance(node, dict):
            return {k: sanitize_node(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [sanitize_node(item) for item in node]
        elif isinstance(node, str):
            return sanitize_query(node)
        else:
            return node
    
    return sanitize_node(query_dict)

def validate_complete_search_request(query: str, user_id: int, 
                                   filters: Dict[str, Any] = None,
                                   parameters: Dict[str, Any] = None) -> ValidationResult:
    """
    Valide une requête de recherche complète avec tous ses composants.
    
    Args:
        query: Texte de recherche
        user_id: ID utilisateur
        filters: Filtres de recherche
        parameters: Paramètres API
        
    Returns:
        ValidationResult global
    """
    global_result = ValidationResult(is_valid=True)
    
    # Validation de la requête de base
    query_result = validate_search_request(query, user_id, filters)
    global_result.errors.extend(query_result.errors)
    global_result.warnings.extend(query_result.warnings)
    
    if query_result.errors:
        global_result.is_valid = False
    
    # Validation des filtres si fournis
    if filters:
        filter_validator = create_filter_validator()
        filter_result = filter_validator.validate(filters)
        global_result.errors.extend(filter_result.errors)
        global_result.warnings.extend(filter_result.warnings)
        if hasattr(filter_result, 'security_flags'):
            global_result.security_flags.extend(filter_result.security_flags)
        
        if filter_result.errors:
            global_result.is_valid = False
    
    # Validation des paramètres si fournis
    if parameters:
        param_validator = create_parameter_validator()
        param_result = param_validator.validate(parameters)
        global_result.errors.extend(param_result.errors)
        global_result.warnings.extend(param_result.warnings)
        if hasattr(param_result, 'security_flags'):
            global_result.security_flags.extend(param_result.security_flags)
        
        if param_result.errors:
            global_result.is_valid = False
    
    # Validation de sécurité globale
    security_validator = create_security_validator()
    security_data = {
        "query": query,
        "user_id": user_id,
        "filters": filters or {},
        "parameters": parameters or {}
    }
    security_result = security_validator.validate(security_data, "search_request")
    global_result.errors.extend(security_result.errors)
    global_result.warnings.extend(security_result.warnings)
    if hasattr(security_result, 'security_flags'):
        global_result.security_flags.extend(security_result.security_flags)
    
    if security_result.errors:
        global_result.is_valid = False
    
    return global_result

def check_query_performance_impact(query_dict: Dict[str, Any]) -> List[str]:
    """
    Analyse l'impact performance potentiel d'une requête.
    
    Args:
        query_dict: Requête à analyser
        
    Returns:
        Liste des avertissements de performance
    """
    warnings = []
    
    def analyze_node(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                current_path = f"{path}.{key}" if path else key
                
                if key == "wildcard":
                    warnings.append(f"Wildcard query detected at {current_path} - may impact performance")
                elif key == "regexp":
                    warnings.append(f"Regexp query detected at {current_path} - may impact performance")
                elif key == "fuzzy":
                    warnings.append(f"Fuzzy query detected at {current_path} - may impact performance")
                elif key == "function_score":
                    warnings.append(f"Function score query detected at {current_path} - may impact performance")
                elif key == "should" and isinstance(value, list) and len(value) > 10:
                    warnings.append(f"Many should clauses at {current_path} - may impact performance")
                
                analyze_node(value, current_path)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                analyze_node(item, f"{path}[{i}]")
    
    if "query" in query_dict:
        analyze_node(query_dict["query"])
    
    # Vérification de la taille
    if query_dict.get("size", 0) > 100:
        warnings.append("Large result set requested - may impact performance")
    
    # Vérification de l'offset
    if query_dict.get("from", 0) > 1000:
        warnings.append("Large offset requested - may impact performance")
    
    return warnings

# ==================== GESTIONNAIRE DE VALIDATION ====================

class ValidationManager:
    """
    Gestionnaire centralisé pour toutes les validations.
    
    Coordonne l'exécution des différents validateurs et fournit
    une interface unifiée pour la validation.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validators = {
            'query': create_query_validator(validation_level),
            'filter': create_filter_validator(validation_level),
            'parameter': create_parameter_validator(validation_level),
            'result': create_result_validator(validation_level),
            'security': create_security_validator(ValidationLevel.STRICT)
        }
    
    def validate_search_request_complete(self, request_data: Dict[str, Any]) -> ValidationResult:
        """
        Valide une requête de recherche complète.
        
        Args:
            request_data: Données de la requête
            
        Returns:
            ValidationResult global
        """
        # Extraction des composants
        query = request_data.get('query', '')
        user_id = request_data.get('user_id')
        filters = request_data.get('filters', {})
        parameters = request_data.get('parameters', {})
        
        # Utilisation de la fonction de validation complète
        return validate_complete_search_request(query, user_id, filters, parameters)
    
    def validate_elasticsearch_response(self, response_data: Dict[str, Any]) -> ValidationResult:
        """
        Valide une réponse Elasticsearch.
        
        Args:
            response_data: Réponse Elasticsearch
            
        Returns:
            ValidationResult
        """
        return self.validators['result'].validate(response_data)
    
    def get_validation_status(self) -> Dict[str, bool]:
        """
        Retourne le statut des modules de validation disponibles.
        
        Returns:
            Statut des modules
        """
        return {
            'base': _BASE_AVAILABLE,
            'query': _QUERY_AVAILABLE,
            'filters': _FILTERS_AVAILABLE,
            'results': _RESULTS_AVAILABLE,
            'parameters': _PARAMETERS_AVAILABLE,
            'security': _SECURITY_AVAILABLE,
            'utils': _UTILS_AVAILABLE,
            'config': _CONFIG_AVAILABLE,
            'exceptions': _EXCEPTIONS_AVAILABLE
        }

# ==================== EXPORTS PRINCIPAUX ====================

__all__ = [
    # Classes principales
    'BaseValidator',
    'QueryValidator', 
    'FilterValidator',
    'ResultValidator',
    'ParameterValidator',
    'SecurityValidator',
    'ValidationManager',
    
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
    
    # Fonctions utilitaires de base
    'validate_search_request',
    'validate_user_id',
    'validate_amount',
    'validate_date',
    'sanitize_query',
    'is_safe_query',
    'escape_elasticsearch_query',
    
    # Fonctions utilitaires avancées
    'validate_elasticsearch_query',
    'validate_financial_search',
    'sanitize_elasticsearch_query',
    'validate_complete_search_request',
    'check_query_performance_impact',
    
    # Factory functions
    'create_query_validator',
    'create_filter_validator',
    'create_result_validator',
    'create_parameter_validator',
    'create_security_validator',
    
    # Configuration et constantes
    'ALLOWED_SEARCH_FIELDS',
    'SENSITIVE_FIELDS',
    'ALLOWED_QUERY_TYPES',
    'DEFAULT_LIMITS',
    'DANGEROUS_PATTERNS',
    'ES_SPECIAL_CHARS',
    'VALIDATION_LIMITS'
] 