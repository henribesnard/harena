"""
Validateur de base et structures communes pour le Search Service.

Ce module fournit les classes de base, enums, exceptions et structures
de données partagées par tous les validateurs spécialisés.
"""

import re
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Callable

logger = logging.getLogger(__name__)

# ==================== ENUMS ET CONSTANTES ====================

class ValidationLevel(str, Enum):
    """Niveaux de validation disponibles."""
    BASIC = "basic"          # Validation minimale
    STANDARD = "standard"    # Validation normale (défaut)
    STRICT = "strict"        # Validation stricte
    PARANOID = "paranoid"    # Validation maximale

class FieldType(str, Enum):
    """Types de champs supportés."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    USER_ID = "user_id"
    AMOUNT = "amount"

# Limites par défaut
DEFAULT_LIMITS = {
    "max_query_length": 1000,
    "max_filter_values": 100,
    "max_nested_depth": 10,
    "max_bool_clauses": 50,
    "max_wildcard_queries": 10,
    "min_query_length": 1,
    "default_timeout_ms": 5000,
    "max_timeout_ms": 30000,
}

# Patterns de sécurité dangereux
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'javascript\s*:',
    r'on\w+\s*=',
    r'\$\{.*?\}',
    r'<%.*?%>',
    r'eval\s*\(',
    r'exec\s*\(',
    r'__.*__',
    r'\.\./',
    r'null\s*;',
    r'union\s+select',
    r'drop\s+table',
    r'delete\s+from',
    r'insert\s+into',
    r'update\s+.*\s+set',
]

# Caractères spéciaux Elasticsearch à échapper
ES_SPECIAL_CHARS = r'+-=&|><!(){}[]^"~*?:\/\\'

# ==================== EXCEPTIONS ====================

class ValidationError(Exception):
    """Exception de base pour les erreurs de validation."""
    
    def __init__(self, message: str, field: str = "", details: Optional[Dict] = None):
        super().__init__(message)
        self.field = field
        self.details = details or {}
        self.timestamp = datetime.utcnow()

class QueryValidationError(ValidationError):
    """Erreur de validation de requête Elasticsearch."""
    pass

class FilterValidationError(ValidationError):
    """Erreur de validation de filtre."""
    pass

class ResultValidationError(ValidationError):
    """Erreur de validation de résultat."""
    pass

class ParameterValidationError(ValidationError):
    """Erreur de validation de paramètre API."""
    pass

class SecurityValidationError(ValidationError):
    """Erreur de sécurité lors de la validation."""
    pass

# ==================== STRUCTURES DE DONNÉES ====================

@dataclass
class ValidationResult:
    """Résultat d'une validation avec détails."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Any] = None
    validation_time_ms: float = 0.0
    security_flags: List[str] = field(default_factory=list)
    
    def add_error(self, error: str, field: str = ""):
        """Ajoute une erreur avec contexte."""
        self.is_valid = False
        error_msg = f"{field}: {error}" if field else error
        self.errors.append(error_msg)
    
    def add_warning(self, warning: str, field: str = ""):
        """Ajoute un avertissement avec contexte."""
        warning_msg = f"{field}: {warning}" if field else warning
        self.warnings.append(warning_msg)
    
    def add_security_flag(self, flag: str):
        """Ajoute un flag de sécurité."""
        self.security_flags.append(flag)

@dataclass
class QueryComplexity:
    """Mesure de complexité d'une requête Elasticsearch."""
    score: int = 0
    nested_depth: int = 0
    bool_clauses: int = 0
    wildcard_count: int = 0
    regexp_count: int = 0
    function_score_count: int = 0
    
    @property
    def is_complex(self) -> bool:
        """Détermine si la requête est complexe."""
        return (self.score > 100 or 
                self.nested_depth > 8 or 
                self.bool_clauses > 30 or
                self.wildcard_count > 15 or
                self.regexp_count > 5)

@dataclass
class FieldValidationRule:
    """Règle de validation pour un champ."""
    field_type: FieldType
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[Set[Any]] = None
    pattern: Optional[str] = None
    custom_validator: Optional[Callable] = None

# ==================== VALIDATEUR BASE ====================

class BaseValidator:
    """
    Validateur de base avec fonctionnalités communes.
    
    Fournit les utilitaires de base pour tous les validateurs spécialisés.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _create_result(self, is_valid: bool = True, sanitized_data: Any = None) -> ValidationResult:
        """Crée un résultat de validation vide."""
        return ValidationResult(
            is_valid=is_valid,
            sanitized_data=sanitized_data
        )
    
    def _validate_required_field(self, value: Any, field_name: str, result: ValidationResult) -> bool:
        """Valide qu'un champ requis est présent."""
        if value is None or value == "":
            result.add_error(f"Field '{field_name}' is required")
            return False
        return True
    
    def _validate_string_length(self, value: str, field_name: str, min_length: int = None,
                              max_length: int = None, result: ValidationResult = None) -> bool:
        """Valide la longueur d'une chaîne."""
        if result is None:
            result = self._create_result()
        
        if not isinstance(value, str):
            result.add_error(f"Field '{field_name}' must be a string")
            return False
        
        length = len(value)
        
        if min_length is not None and length < min_length:
            result.add_error(f"Field '{field_name}' must be at least {min_length} characters")
            return False
        
        if max_length is not None and length > max_length:
            result.add_error(f"Field '{field_name}' must be at most {max_length} characters")
            return False
        
        return True
    
    def _validate_numeric_range(self, value: Union[int, float], field_name: str,
                              min_value: Union[int, float] = None,
                              max_value: Union[int, float] = None,
                              result: ValidationResult = None) -> bool:
        """Valide la plage d'une valeur numérique."""
        if result is None:
            result = self._create_result()
        
        if not isinstance(value, (int, float)):
            result.add_error(f"Field '{field_name}' must be numeric")
            return False
        
        if min_value is not None and value < min_value:
            result.add_error(f"Field '{field_name}' must be at least {min_value}")
            return False
        
        if max_value is not None and value > max_value:
            result.add_error(f"Field '{field_name}' must be at most {max_value}")
            return False
        
        return True
    
    def _validate_pattern(self, value: str, pattern: str, field_name: str,
                         result: ValidationResult = None) -> bool:
        """Valide qu'une valeur correspond à un pattern regex."""
        if result is None:
            result = self._create_result()
        
        try:
            if not re.match(pattern, value):
                result.add_error(f"Field '{field_name}' format is invalid")
                return False
        except re.error as e:
            result.add_error(f"Invalid pattern for field '{field_name}': {e}")
            return False
        
        return True
    
    def _check_security_patterns(self, value: str, result: ValidationResult) -> bool:
        """Vérifie les patterns de sécurité dangereux."""
        if not isinstance(value, str):
            return True
        
        for pattern in DANGEROUS_PATTERNS:
            try:
                if re.search(pattern, value, re.IGNORECASE):
                    result.add_security_flag(f"Dangerous pattern detected: {pattern}")
                    if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                        result.add_error("Potentially dangerous content detected")
                        return False
            except re.error:
                continue
        
        return True
    
    def _escape_elasticsearch_special_chars(self, query: str) -> str:
        """Échappe les caractères spéciaux Elasticsearch."""
        if not isinstance(query, str):
            return query
        
        escaped = query
        for char in ES_SPECIAL_CHARS:
            escaped = escaped.replace(char, f"\\{char}")
        
        return escaped
    
    def _sanitize_string(self, value: str, allow_html: bool = False) -> str:
        """Nettoie une chaîne de caractères."""
        if not isinstance(value, str):
            return value
        
        # Suppression des caractères de contrôle
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
        
        # Suppression des tags HTML si non autorisés
        if not allow_html:
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Normalisation des espaces
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def _measure_validation_time(self, start_time: float) -> float:
        """Mesure le temps de validation en millisecondes."""
        return round((time.time() - start_time) * 1000, 2)
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Méthode de validation principale à implémenter."""
        pass

# ==================== FONCTIONS UTILITAIRES ====================

def validate_user_id(user_id: Any) -> bool:
    """Valide un ID utilisateur."""
    if user_id is None:
        return False
    
    if isinstance(user_id, str):
        try:
            user_id = int(user_id)
        except ValueError:
            return False
    
    return isinstance(user_id, int) and user_id > 0

def validate_amount(amount: Any) -> bool:
    """Valide un montant financier."""
    if amount is None:
        return False
    
    try:
        amount_float = float(amount)
        return amount_float >= 0 and amount_float <= 999999999.99
    except (ValueError, TypeError):
        return False

def validate_date(date_value: Any) -> bool:
    """Valide une date."""
    if date_value is None:
        return False
    
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
    """Nettoie une requête de recherche."""
    if not isinstance(query, str):
        return ""
    
    # Suppression des caractères dangereux
    sanitized = re.sub(r'[<>"\';\\]', '', query)
    
    # Limitation de la longueur
    sanitized = sanitized[:DEFAULT_LIMITS["max_query_length"]]
    
    # Normalisation des espaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    return sanitized

def is_safe_query(query: str) -> bool:
    """Vérifie si une requête est sûre."""
    if not isinstance(query, str):
        return False
    
    for pattern in DANGEROUS_PATTERNS:
        try:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        except re.error:
            continue
    
    return True

def escape_elasticsearch_query(query: str) -> str:
    """Échappe une requête pour Elasticsearch."""
    if not isinstance(query, str):
        return query
    
    escaped = query
    for char in ES_SPECIAL_CHARS:
        escaped = escaped.replace(char, f"\\{char}")
    
    return escaped