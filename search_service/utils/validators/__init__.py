"""
Sous-module validators - Imports centralisés.

Ce module __init__.py permet d'importer facilement les validateurs
depuis le sous-module validators/ et expose ValidationError.
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ==================== EXCEPTIONS DE VALIDATION ====================

class ValidationError(Exception):
    """
    Exception de validation personnalisée.
    
    Utilisée pour toutes les erreurs de validation dans le système.
    """
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)
    
    def __str__(self):
        if self.field:
            return f"Validation error in field '{self.field}': {self.message}"
        return f"Validation error: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'erreur en dictionnaire."""
        return {
            "error": "validation_error",
            "message": self.message,
            "field": self.field,
            "value": self.value
        }

class QueryValidationError(ValidationError):
    """Exception spécifique à la validation des requêtes."""
    pass

class FilterValidationError(ValidationError):
    """Exception spécifique à la validation des filtres."""
    pass

class ParameterValidationError(ValidationError):
    """Exception spécifique à la validation des paramètres."""
    pass

class SecurityValidationError(ValidationError):
    """Exception spécifique à la validation de sécurité."""
    pass

class ResultValidationError(ValidationError):
    """Exception spécifique à la validation des résultats."""
    pass

# ==================== CLASSES DE BASE ====================

class BaseValidator:
    """
    Validateur de base avec fonctionnalités communes.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.errors: List[ValidationError] = []
    
    def add_error(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """Ajoute une erreur de validation."""
        error = ValidationError(message, field, value)
        self.errors.append(error)
        
        if self.strict_mode:
            raise error
    
    def has_errors(self) -> bool:
        """Vérifie s'il y a des erreurs."""
        return len(self.errors) > 0
    
    def get_errors(self) -> List[ValidationError]:
        """Retourne la liste des erreurs."""
        return self.errors.copy()
    
    def clear_errors(self):
        """Efface toutes les erreurs."""
        self.errors.clear()
    
    def validate_required(self, value: Any, field_name: str) -> bool:
        """Valide qu'un champ requis n'est pas vide."""
        if value is None or (isinstance(value, str) and not value.strip()):
            self.add_error(f"Field '{field_name}' is required", field_name, value)
            return False
        return True
    
    def validate_type(self, value: Any, expected_type: type, field_name: str) -> bool:
        """Valide le type d'une valeur."""
        if not isinstance(value, expected_type):
            self.add_error(
                f"Field '{field_name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
                field_name, value
            )
            return False
        return True
    
    def validate_range(self, value: Union[int, float], min_val: Optional[Union[int, float]], 
                      max_val: Optional[Union[int, float]], field_name: str) -> bool:
        """Valide qu'une valeur est dans une plage donnée."""
        if min_val is not None and value < min_val:
            self.add_error(f"Field '{field_name}' must be >= {min_val}", field_name, value)
            return False
        
        if max_val is not None and value > max_val:
            self.add_error(f"Field '{field_name}' must be <= {max_val}", field_name, value)
            return False
        
        return True
    
    def validate_length(self, value: str, min_length: Optional[int], 
                       max_length: Optional[int], field_name: str) -> bool:
        """Valide la longueur d'une chaîne."""
        length = len(value) if value else 0
        
        if min_length is not None and length < min_length:
            self.add_error(f"Field '{field_name}' must be at least {min_length} characters", field_name, value)
            return False
        
        if max_length is not None and length > max_length:
            self.add_error(f"Field '{field_name}' must be at most {max_length} characters", field_name, value)
            return False
        
        return True
    
    def validate_choices(self, value: Any, choices: List[Any], field_name: str) -> bool:
        """Valide qu'une valeur fait partie des choix autorisés."""
        if value not in choices:
            self.add_error(f"Field '{field_name}' must be one of {choices}", field_name, value)
            return False
        return True

# ==================== IMPORTS AVEC FALLBACK ====================

# Imports des classes principales (lazy loading pour éviter les imports circulaires)
def _import_query_validator():
    try:
        from .query import QueryValidator
        return QueryValidator
    except ImportError:
        logger.warning("QueryValidator not available, using fallback")
        return BaseValidator

def _import_filter_validator():
    try:
        from .filters import FilterValidator
        return FilterValidator
    except ImportError:
        logger.warning("FilterValidator not available, using fallback")
        return BaseValidator

def _import_result_validator():
    try:
        from .results import ResultValidator
        return ResultValidator
    except ImportError:
        logger.warning("ResultValidator not available, using fallback")
        return BaseValidator

def _import_parameter_validator():
    try:
        from .parameters import ParameterValidator
        return ParameterValidator
    except ImportError:
        logger.warning("ParameterValidator not available, using fallback")
        return BaseValidator

def _import_security_validator():
    try:
        from .security import SecurityValidator
        return SecurityValidator
    except ImportError:
        logger.warning("SecurityValidator not available, using fallback")
        return BaseValidator

# ==================== FONCTIONS UTILITAIRES ====================

def validate_user_id(user_id: Any) -> bool:
    """
    Valide un ID utilisateur.
    
    Args:
        user_id: ID utilisateur à valider
        
    Returns:
        True si valide, False sinon
        
    Raises:
        ValidationError: Si l'ID n'est pas valide
    """
    if user_id is None:
        raise ValidationError("User ID cannot be None", "user_id", user_id)
    
    if not isinstance(user_id, int):
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            raise ValidationError("User ID must be an integer", "user_id", user_id)
    
    if user_id <= 0:
        raise ValidationError("User ID must be positive", "user_id", user_id)
    
    return True

def validate_query_text(query: Any, min_length: int = 1, max_length: int = 1000) -> bool:
    """
    Valide un texte de requête.
    
    Args:
        query: Texte de requête à valider
        min_length: Longueur minimum
        max_length: Longueur maximum
        
    Returns:
        True si valide, False sinon
        
    Raises:
        ValidationError: Si le texte n'est pas valide
    """
    if query is None:
        raise ValidationError("Query text cannot be None", "query", query)
    
    if not isinstance(query, str):
        raise ValidationError("Query text must be a string", "query", query)
    
    query = query.strip()
    
    if len(query) < min_length:
        raise ValidationError(f"Query text must be at least {min_length} characters", "query", query)
    
    if len(query) > max_length:
        raise ValidationError(f"Query text must be at most {max_length} characters", "query", query)
    
    return True

def validate_pagination(size: Any, from_: Any) -> bool:
    """
    Valide les paramètres de pagination.
    
    Args:
        size: Taille de la page
        from_: Offset
        
    Returns:
        True si valide, False sinon
        
    Raises:
        ValidationError: Si les paramètres ne sont pas valides
    """
    # Validation de size
    if size is not None:
        if not isinstance(size, int):
            try:
                size = int(size)
            except (ValueError, TypeError):
                raise ValidationError("Size must be an integer", "size", size)
        
        if size < 0:
            raise ValidationError("Size must be non-negative", "size", size)
        
        if size > 1000:
            raise ValidationError("Size must be <= 1000", "size", size)
    
    # Validation de from_
    if from_ is not None:
        if not isinstance(from_, int):
            try:
                from_ = int(from_)
            except (ValueError, TypeError):
                raise ValidationError("From must be an integer", "from", from_)
        
        if from_ < 0:
            raise ValidationError("From must be non-negative", "from", from_)
        
        if from_ > 10000:
            raise ValidationError("From must be <= 10000", "from", from_)
    
    return True

def validate_filters(filters: Any) -> bool:
    """
    Valide un dictionnaire de filtres.
    
    Args:
        filters: Dictionnaire de filtres
        
    Returns:
        True si valide, False sinon
        
    Raises:
        ValidationError: Si les filtres ne sont pas valides
    """
    if filters is None:
        return True
    
    if not isinstance(filters, dict):
        raise ValidationError("Filters must be a dictionary", "filters", filters)
    
    # Validation des filtres de montant
    if "amount_min" in filters:
        amount_min = filters["amount_min"]
        if not isinstance(amount_min, (int, float)):
            try:
                amount_min = float(amount_min)
            except (ValueError, TypeError):
                raise ValidationError("amount_min must be a number", "amount_min", amount_min)
    
    if "amount_max" in filters:
        amount_max = filters["amount_max"]
        if not isinstance(amount_max, (int, float)):
            try:
                amount_max = float(amount_max)
            except (ValueError, TypeError):
                raise ValidationError("amount_max must be a number", "amount_max", amount_max)
    
    # Validation des filtres de date
    if "date_start" in filters:
        date_start = filters["date_start"]
        if not isinstance(date_start, str):
            raise ValidationError("date_start must be a string", "date_start", date_start)
    
    if "date_end" in filters:
        date_end = filters["date_end"]
        if not isinstance(date_end, str):
            raise ValidationError("date_end must be a string", "date_end", date_end)
    
    # Validation des filtres de liste
    for list_filter in ["categories", "merchants"]:
        if list_filter in filters:
            filter_value = filters[list_filter]
            if not isinstance(filter_value, list):
                raise ValidationError(f"{list_filter} must be a list", list_filter, filter_value)
    
    return True

def sanitize_query_text(query: str) -> str:
    """
    Nettoie et sécurise un texte de requête.
    
    Args:
        query: Texte de requête
        
    Returns:
        Texte nettoyé
    """
    if not isinstance(query, str):
        return ""
    
    # Suppression des espaces en début/fin
    query = query.strip()
    
    # Suppression des caractères de contrôle
    query = ''.join(char for char in query if ord(char) >= 32)
    
    # Limitation de la longueur
    if len(query) > 1000:
        query = query[:1000]
    
    return query

def validate_elasticsearch_response(response: Any) -> bool:
    """
    Valide une réponse Elasticsearch.
    
    Args:
        response: Réponse Elasticsearch
        
    Returns:
        True si valide, False sinon
        
    Raises:
        ValidationError: Si la réponse n'est pas valide
    """
    if response is None:
        raise ValidationError("Elasticsearch response cannot be None", "response", response)
    
    if not isinstance(response, dict):
        raise ValidationError("Elasticsearch response must be a dictionary", "response", response)
    
    # Vérification de la structure de base
    if "hits" not in response:
        raise ValidationError("Elasticsearch response missing 'hits' field", "response", response)
    
    hits = response["hits"]
    if not isinstance(hits, dict):
        raise ValidationError("Elasticsearch 'hits' must be a dictionary", "hits", hits)
    
    if "hits" not in hits:
        raise ValidationError("Elasticsearch hits missing 'hits' array", "hits", hits)
    
    if not isinstance(hits["hits"], list):
        raise ValidationError("Elasticsearch hits['hits'] must be a list", "hits", hits["hits"])
    
    return True

# ==================== LAZY LOADING ====================

def __getattr__(name):
    """Lazy loading des classes de validation."""
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

# ==================== FACTORY FUNCTIONS ====================

def create_base_validator(strict_mode: bool = False) -> BaseValidator:
    """
    Crée un validateur de base.
    
    Args:
        strict_mode: Mode strict (lance des exceptions immédiatement)
        
    Returns:
        Instance de BaseValidator
    """
    return BaseValidator(strict_mode=strict_mode)

def validate_search_request(query: str, user_id: int, 
                          filters: Optional[Dict[str, Any]] = None,
                          size: Optional[int] = None,
                          from_: Optional[int] = None) -> Dict[str, Any]:
    """
    Valide une requête de recherche complète.
    
    Args:
        query: Texte de requête
        user_id: ID utilisateur
        filters: Filtres optionnels
        size: Taille de page
        from_: Offset
        
    Returns:
        Dictionnaire des valeurs validées
        
    Raises:
        ValidationError: Si la validation échoue
    """
    # Validation des paramètres obligatoires
    validate_query_text(query)
    validate_user_id(user_id)
    
    # Validation des paramètres optionnels
    if filters is not None:
        validate_filters(filters)
    
    if size is not None or from_ is not None:
        validate_pagination(size, from_)
    
    # Nettoyage du texte de requête
    clean_query = sanitize_query_text(query)
    
    return {
        "query": clean_query,
        "user_id": user_id,
        "filters": filters or {},
        "size": size or 20,
        "from": from_ or 0
    }

def validate_and_sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide et nettoie un dictionnaire d'entrée.
    
    Args:
        data: Données d'entrée
        
    Returns:
        Données validées et nettoyées
        
    Raises:
        ValidationError: Si la validation échoue
    """
    if not isinstance(data, dict):
        raise ValidationError("Input data must be a dictionary", "data", data)
    
    validated_data = {}
    
    # Validation et nettoyage des champs communs
    for key, value in data.items():
        if key == "query" and value is not None:
            validate_query_text(value)
            validated_data[key] = sanitize_query_text(value)
        elif key == "user_id" and value is not None:
            validate_user_id(value)
            validated_data[key] = int(value)
        elif key == "filters" and value is not None:
            validate_filters(value)
            validated_data[key] = value
        elif key in ["size", "from"] and value is not None:
            if key == "size":
                validate_pagination(value, None)
            else:
                validate_pagination(None, value)
            validated_data[key] = int(value)
        else:
            # Autres champs : validation basique
            if isinstance(value, str):
                validated_data[key] = value.strip()
            else:
                validated_data[key] = value
    
    return validated_data

# ==================== VALIDATEURS SPÉCIALISÉS ====================

class SimpleQueryValidator(BaseValidator):
    """Validateur simple pour les requêtes de base."""
    
    def validate_query(self, query: str, user_id: int) -> bool:
        """Valide une requête simple."""
        try:
            validate_query_text(query)
            validate_user_id(user_id)
            return True
        except ValidationError as e:
            self.add_error(e.message, e.field, e.value)
            return False

class SimpleFilterValidator(BaseValidator):
    """Validateur simple pour les filtres."""
    
    def validate_filters(self, filters: Dict[str, Any]) -> bool:
        """Valide un dictionnaire de filtres."""
        try:
            validate_filters(filters)
            return True
        except ValidationError as e:
            self.add_error(e.message, e.field, e.value)
            return False

class SimpleParameterValidator(BaseValidator):
    """Validateur simple pour les paramètres."""
    
    def validate_pagination(self, size: int, from_: int) -> bool:
        """Valide les paramètres de pagination."""
        try:
            validate_pagination(size, from_)
            return True
        except ValidationError as e:
            self.add_error(e.message, e.field, e.value)
            return False

class SimpleSecurityValidator(BaseValidator):
    """Validateur simple pour la sécurité."""
    
    def validate_user_access(self, user_id: int, requested_user_id: int) -> bool:
        """Valide l'accès utilisateur."""
        if user_id != requested_user_id:
            self.add_error("User can only access their own data", "user_id", requested_user_id)
            return False
        return True
    
    def validate_query_safety(self, query: str) -> bool:
        """Valide la sécurité d'une requête."""
        # Détection de patterns dangereux
        dangerous_patterns = [
            "<script", "javascript:", "eval(", "function(",
            "import ", "require(", "process.", "global.",
            "__", "prototype."
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if pattern in query_lower:
                self.add_error(f"Query contains potentially dangerous pattern: {pattern}", "query", query)
                return False
        
        return True

class SimpleResultValidator(BaseValidator):
    """Validateur simple pour les résultats."""
    
    def validate_result_structure(self, result: Dict[str, Any]) -> bool:
        """Valide la structure d'un résultat."""
        try:
            validate_elasticsearch_response(result)
            return True
        except ValidationError as e:
            self.add_error(e.message, e.field, e.value)
            return False

# ==================== EXPORTS PRINCIPAUX ====================

__all__ = [
    # Exceptions
    'ValidationError',
    'QueryValidationError',
    'FilterValidationError',
    'ParameterValidationError',
    'SecurityValidationError',
    'ResultValidationError',
    
    # Classes de base
    'BaseValidator',
    
    # Validateurs simples
    'SimpleQueryValidator',
    'SimpleFilterValidator',
    'SimpleParameterValidator',
    'SimpleSecurityValidator',
    'SimpleResultValidator',
    
    # Fonctions de validation
    'validate_user_id',
    'validate_query_text',
    'validate_pagination',
    'validate_filters',
    'validate_elasticsearch_response',
    'validate_search_request',
    'validate_and_sanitize_input',
    
    # Fonctions utilitaires
    'sanitize_query_text',
    'create_base_validator',
    
    # Classes principales (lazy loaded)
    'QueryValidator',
    'FilterValidator',
    'ResultValidator',
    'ParameterValidator',
    'SecurityValidator'
]

# ==================== CONFIGURATION PAR DÉFAUT ====================

# Configuration globale des validateurs
DEFAULT_CONFIG = {
    "strict_mode": False,
    "max_query_length": 1000,
    "max_page_size": 1000,
    "max_offset": 10000,
    "enable_security_checks": True,
    "sanitize_input": True
}

def get_validator_config() -> Dict[str, Any]:
    """Retourne la configuration des validateurs."""
    return DEFAULT_CONFIG.copy()

def set_validator_config(config: Dict[str, Any]):
    """Met à jour la configuration des validateurs."""
    DEFAULT_CONFIG.update(config)

# ==================== INITIALISATION ====================

# Log de l'initialisation du module
logger.info("Validators module initialized with ValidationError support")

# Création d'instances par défaut pour usage rapide
default_query_validator = SimpleQueryValidator()
default_filter_validator = SimpleFilterValidator()
default_parameter_validator = SimpleParameterValidator()
default_security_validator = SimpleSecurityValidator()
default_result_validator = SimpleResultValidator()

# Fonctions de convenance pour validation rapide
def quick_validate_query(query: str, user_id: int) -> bool:
    """Validation rapide d'une requête."""
    return default_query_validator.validate_query(query, user_id)

def quick_validate_filters(filters: Dict[str, Any]) -> bool:
    """Validation rapide de filtres."""
    return default_filter_validator.validate_filters(filters)

def quick_validate_pagination(size: int, from_: int) -> bool:
    """Validation rapide de pagination."""
    return default_parameter_validator.validate_pagination(size, from_)

def quick_validate_security(query: str, user_id: int, requested_user_id: int) -> bool:
    """Validation rapide de sécurité."""
    return (default_security_validator.validate_user_access(user_id, requested_user_id) and
            default_security_validator.validate_query_safety(query))

def quick_validate_result(result: Dict[str, Any]) -> bool:
    """Validation rapide de résultat."""
    return default_result_validator.validate_result_structure(result)

# Ajout des fonctions de convenance aux exports
__all__.extend([
    'quick_validate_query',
    'quick_validate_filters', 
    'quick_validate_pagination',
    'quick_validate_security',
    'quick_validate_result',
    'get_validator_config',
    'set_validator_config'
])