"""
Validateur de base et structures communes pour le Search Service.

Ce module fournit la classe BaseValidator avec toutes les fonctionnalités communes
ainsi que les structures de données partagées par tous les validateurs.
"""

import re
import logging
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import unquote

# Configuration centralisée
from config_service.config import settings

logger = logging.getLogger(__name__)

# ==================== ENUMS ET CONSTANTES ====================

class ValidationLevel(str, Enum):
    """Niveaux de validation."""
    BASIC = "basic"         # Validation de base uniquement
    STANDARD = "standard"   # Validation standard avec sécurité
    STRICT = "strict"       # Validation stricte avec toutes les vérifications

class FieldType(str, Enum):
    """Types de champs supportés."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"

# Limites par défaut
DEFAULT_LIMITS = {
    "max_query_length": getattr(settings, 'SEARCH_MAX_QUERY_LENGTH', 500),
    "max_results": getattr(settings, 'SEARCH_MAX_LIMIT', 100),
    "max_filter_values": getattr(settings, 'SEARCH_MAX_FILTER_VALUES', 50),
    "max_nested_depth": 10,
    "max_bool_clauses": 50,
    "max_wildcard_terms": 10
}

# Patterns de sécurité dangereux
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # Scripts
    r'javascript\s*:',             # Javascript URLs
    r'on\w+\s*=',                 # Event handlers
    r'\$\{.*?\}',                 # Template injection
    r'<%.*?%>',                   # Template tags
    r'eval\s*\(',                 # Eval functions
    r'exec\s*\(',                 # Exec functions
    r'__.*__',                    # Python magic methods
    r'\.\./',                     # Directory traversal
    r'null\s*;',                  # SQL injection attempts
    r'union\s+select',            # SQL injection
    r'drop\s+table',              # SQL injection
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
    custom_validator: Optional[callable] = None

# ==================== VALIDATEUR BASE ====================

class BaseValidator:
    """
    Validateur de base avec fonctionnalités communes.
    
    Fournit les utilitaires de base pour tous les validateurs spécialisés.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.limits = DEFAULT_LIMITS.copy()
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL) 
            for pattern in DANGEROUS_PATTERNS
        ]
    
    def sanitize_string(
        self, 
        value: str, 
        max_length: Optional[int] = None,
        escape_es_chars: bool = True
    ) -> str:
        """
        Sanitise une chaîne de caractères.
        
        Args:
            value: Valeur à sanitiser
            max_length: Longueur maximum
            escape_es_chars: Échapper les caractères spéciaux Elasticsearch
            
        Returns:
            Chaîne sanitisée
            
        Raises:
            SecurityValidationError: Si patterns dangereux détectés
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Décoder les caractères URL-encodés
        try:
            value = unquote(value)
        except Exception:
            pass  # Continuer avec la valeur originale si le décodage échoue
        
        # Nettoyer et limiter la longueur
        value = value.strip()
        if max_length and len(value) > max_length:
            value = value[:max_length]
            logger.warning(f"String truncated to {max_length} characters")
        
        # Détecter les patterns dangereux
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            for pattern in self._compiled_patterns:
                if pattern.search(value):
                    dangerous_content = pattern.search(value).group(0)[:50]
                    logger.warning(f"Dangerous pattern detected: {dangerous_content}...")
                    
                    if self.validation_level == ValidationLevel.STRICT:
                        raise SecurityValidationError(
                            f"Input contains dangerous pattern: {pattern.pattern}",
                            details={"detected_content": dangerous_content}
                        )
                    else:
                        # En mode standard, on supprime le contenu dangereux
                        value = pattern.sub('', value)
        
        # Échapper les caractères spéciaux Elasticsearch
        if escape_es_chars:
            value = self._escape_elasticsearch_chars(value)
        
        return value
    
    def _escape_elasticsearch_chars(self, value: str) -> str:
        """Échappe les caractères spéciaux Elasticsearch."""
        for char in ES_SPECIAL_CHARS:
            if char in value:
                value = value.replace(char, f'\\{char}')
        return value
    
    def validate_user_id(self, user_id: Union[int, str]) -> int:
        """
        Valide un ID utilisateur.
        
        Args:
            user_id: ID à valider
            
        Returns:
            ID utilisateur validé
            
        Raises:
            ParameterValidationError: Si l'ID n'est pas valide
        """
        try:
            user_id_int = int(user_id)
            if user_id_int <= 0:
                raise ParameterValidationError("User ID must be positive")
            if user_id_int > 2147483647:  # Max int32
                raise ParameterValidationError("User ID too large")
            return user_id_int
        except (ValueError, TypeError):
            raise ParameterValidationError(f"Invalid user ID format: {user_id}")
    
    def validate_amount(self, amount: Union[int, float, str, Decimal]) -> Decimal:
        """
        Valide un montant financier.
        
        Args:
            amount: Montant à valider
            
        Returns:
            Montant validé en Decimal
            
        Raises:
            ParameterValidationError: Si le montant n'est pas valide
        """
        try:
            if isinstance(amount, str):
                # Nettoyer la chaîne (supprimer espaces, symboles monétaires)
                amount = re.sub(r'[^\d.,\-]', '', amount)
                amount = amount.replace(',', '.')
            
            decimal_amount = Decimal(str(amount))
            
            # Vérifications business
            if decimal_amount < Decimal('-999999999.99'):
                raise ParameterValidationError("Amount too small")
            if decimal_amount > Decimal('999999999.99'):
                raise ParameterValidationError("Amount too large")
            
            # Vérifier le nombre de décimales (max 2 pour les devises)
            if abs(decimal_amount.as_tuple().exponent) > 2:
                raise ParameterValidationError("Too many decimal places (max 2)")
            
            return decimal_amount
            
        except (ValueError, InvalidOperation, TypeError):
            raise ParameterValidationError(f"Invalid amount format: {amount}")
    
    def validate_date(self, date_value: Union[str, datetime, date]) -> datetime:
        """
        Valide une date.
        
        Args:
            date_value: Date à valider
            
        Returns:
            Date validée
            
        Raises:
            ParameterValidationError: Si la date n'est pas valide
        """
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, date):
            return datetime.combine(date_value, datetime.min.time())
        
        if isinstance(date_value, str):
            # Essayer différents formats de date
            date_formats = [
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
            
            raise ParameterValidationError(f"Invalid date format: {date_value}")
        
        raise ParameterValidationError(f"Invalid date type: {type(date_value)}")
    
    def validate_field_by_rule(self, value: Any, rule: FieldValidationRule, field_name: str) -> Any:
        """
        Valide un champ selon une règle définie.
        
        Args:
            value: Valeur à valider
            rule: Règle de validation
            field_name: Nom du champ
            
        Returns:
            Valeur validée
            
        Raises:
            ParameterValidationError: Si la validation échoue
        """
        # Vérifier si requis
        if rule.required and (value is None or value == ""):
            raise ParameterValidationError(f"Field '{field_name}' is required")
        
        if value is None:
            return None
        
        # Validation selon le type
        if rule.field_type == FieldType.STRING:
            if not isinstance(value, str):
                value = str(value)
            
            # Longueur
            if rule.min_length and len(value) < rule.min_length:
                raise ParameterValidationError(
                    f"Field '{field_name}' too short (min {rule.min_length})"
                )
            if rule.max_length and len(value) > rule.max_length:
                raise ParameterValidationError(
                    f"Field '{field_name}' too long (max {rule.max_length})"
                )
            
            # Pattern
            if rule.pattern and not re.match(rule.pattern, value):
                raise ParameterValidationError(
                    f"Field '{field_name}' doesn't match required pattern"
                )
            
            # Sanitisation
            value = self.sanitize_string(value, rule.max_length)
        
        elif rule.field_type == FieldType.INTEGER:
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ParameterValidationError(f"Field '{field_name}' must be an integer")
            
            if rule.min_value is not None and value < rule.min_value:
                raise ParameterValidationError(
                    f"Field '{field_name}' too small (min {rule.min_value})"
                )
            if rule.max_value is not None and value > rule.max_value:
                raise ParameterValidationError(
                    f"Field '{field_name}' too large (max {rule.max_value})"
                )
        
        elif rule.field_type == FieldType.FLOAT:
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ParameterValidationError(f"Field '{field_name}' must be a number")
            
            if rule.min_value is not None and value < rule.min_value:
                raise ParameterValidationError(
                    f"Field '{field_name}' too small (min {rule.min_value})"
                )
            if rule.max_value is not None and value > rule.max_value:
                raise ParameterValidationError(
                    f"Field '{field_name}' too large (max {rule.max_value})"
                )
        
        elif rule.field_type == FieldType.DECIMAL:
            value = self.validate_amount(value)
        
        elif rule.field_type == FieldType.DATETIME:
            value = self.validate_date(value)
        
        elif rule.field_type == FieldType.BOOLEAN:
            if isinstance(value, str):
                value = value.lower() in ['true', '1', 'yes', 'on']
            else:
                value = bool(value)
        
        elif rule.field_type == FieldType.LIST:
            if not isinstance(value, list):
                raise ParameterValidationError(f"Field '{field_name}' must be a list")
            
            if rule.max_length and len(value) > rule.max_length:
                raise ParameterValidationError(
                    f"Field '{field_name}' has too many items (max {rule.max_length})"
                )
        
        # Valeurs autorisées
        if rule.allowed_values and value not in rule.allowed_values:
            raise ParameterValidationError(
                f"Field '{field_name}' has invalid value. Allowed: {rule.allowed_values}"
            )
        
        # Validateur personnalisé
        if rule.custom_validator:
            try:
                value = rule.custom_validator(value)
            except Exception as e:
                raise ParameterValidationError(
                    f"Custom validation failed for '{field_name}': {e}"
                )
        
        return value