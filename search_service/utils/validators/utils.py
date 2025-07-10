"""
Fonctions utilitaires pour les validateurs.

Ce module fournit des fonctions utilitaires communes utilisées
par tous les validateurs du service de recherche.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from decimal import Decimal, InvalidOperation

from .base import ValidationResult, validate_user_id, validate_amount, validate_date
from .config import ALLOWED_SEARCH_FIELDS, VALIDATION_PATTERNS, ES_SPECIAL_CHARS

logger = logging.getLogger(__name__)

# ==================== FONCTIONS DE VALIDATION PRINCIPALES ====================

def validate_search_request(query: str, user_id: int, filters: Dict[str, Any] = None) -> ValidationResult:
    """
    Valide une requête de recherche complète avec tous ses paramètres.
    
    Args:
        query: Texte de recherche
        user_id: ID utilisateur
        filters: Filtres optionnels
        
    Returns:
        ValidationResult avec détails de validation
    """
    result = ValidationResult(is_valid=True)
    
    # Validation du texte de recherche
    if not _validate_query_text(query, result):
        return result
    
    # Validation de l'utilisateur
    if not _validate_user_id_param(user_id, result):
        return result
    
    # Validation des filtres
    if filters:
        if not _validate_filters_dict(filters, result):
            return result
    
    # Construction de la requête sanitisée
    sanitized_query = sanitize_query(query)
    result.sanitized_data = {
        "query": sanitized_query,
        "user_id": user_id,
        "filters": filters or {}
    }
    
    return result

def sanitize_query(query: str) -> str:
    """
    Nettoie et sécurise une requête de recherche.
    
    Args:
        query: Texte de recherche à nettoyer
        
    Returns:
        Texte de recherche nettoyé
    """
    if not isinstance(query, str):
        return ""
    
    # Suppression des caractères de contrôle
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)
    
    # Suppression des tags HTML
    sanitized = re.sub(r'<[^>]+>', '', sanitized)
    
    # Suppression des caractères dangereux
    sanitized = re.sub(r'[<>"\';\\]', '', sanitized)
    
    # Normalisation des espaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Limitation de la longueur
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000]
    
    return sanitized

def is_safe_query(query: str) -> bool:
    """
    Vérifie si une requête est sûre du point de vue sécurité.
    
    Args:
        query: Texte de requête à vérifier
        
    Returns:
        True si la requête est sûre, False sinon
    """
    if not isinstance(query, str):
        return False
    
    # Patterns dangereux
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript\s*:',
        r'on\w+\s*=',
        r'\$\{.*?\}',
        r'<%.*?%>',
        r'eval\s*\(',
        r'exec\s*\(',
        r'__.*__',
        r'\.\./',
        r'union\s+select',
        r'drop\s+table'
    ]
    
    for pattern in dangerous_patterns:
        try:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        except re.error:
            continue
    
    return True

def escape_elasticsearch_query(query: str) -> str:
    """
    Échappe les caractères spéciaux Elasticsearch dans une requête.
    
    Args:
        query: Requête à échapper
        
    Returns:
        Requête échappée
    """
    if not isinstance(query, str):
        return query
    
    escaped = query
    for char in ES_SPECIAL_CHARS:
        escaped = escaped.replace(char, f"\\{char}")
    
    return escaped

# ==================== VALIDATION DES TYPES DE DONNÉES ====================

def validate_email(email: str) -> bool:
    """Valide une adresse email."""
    if not isinstance(email, str):
        return False
    
    pattern = VALIDATION_PATTERNS.get('email', r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(re.match(pattern, email))

def validate_uuid(uuid_str: str) -> bool:
    """Valide un UUID."""
    if not isinstance(uuid_str, str):
        return False
    
    pattern = VALIDATION_PATTERNS.get('uuid', r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')
    return bool(re.match(pattern, uuid_str.lower()))

def validate_currency_code(currency: str) -> bool:
    """Valide un code de devise."""
    if not isinstance(currency, str):
        return False
    
    pattern = VALIDATION_PATTERNS.get('currency_code', r'^[A-Z]{3}$')
    return bool(re.match(pattern, currency))

def validate_transaction_type(transaction_type: str) -> bool:
    """Valide un type de transaction."""
    if not isinstance(transaction_type, str):
        return False
    
    allowed_types = {
        'card_payment', 'transfer', 'withdrawal', 'deposit', 
        'direct_debit', 'check', 'fee', 'interest', 'other'
    }
    
    return transaction_type.lower() in allowed_types

# ==================== VALIDATION DES PLAGES ET LIMITES ====================

def validate_amount_range(amount: Union[int, float, str], min_amount: float = 0.0, 
                         max_amount: float = 999999999.99) -> bool:
    """
    Valide qu'un montant est dans une plage acceptable.
    
    Args:
        amount: Montant à valider
        min_amount: Montant minimum
        max_amount: Montant maximum
        
    Returns:
        True si valide, False sinon
    """
    try:
        amount_decimal = Decimal(str(amount))
        return min_amount <= float(amount_decimal) <= max_amount
    except (ValueError, InvalidOperation):
        return False

def validate_date_range(date_value: Union[str, datetime], 
                       min_date: Optional[datetime] = None,
                       max_date: Optional[datetime] = None) -> bool:
    """
    Valide qu'une date est dans une plage acceptable.
    
    Args:
        date_value: Date à valider
        min_date: Date minimum (optionnelle)
        max_date: Date maximum (optionnelle)
        
    Returns:
        True si valide, False sinon
    """
    if not validate_date(date_value):
        return False
    
    if isinstance(date_value, str):
        try:
            date_value = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
        except ValueError:
            return False
    
    if min_date and date_value < min_date:
        return False
    
    if max_date and date_value > max_date:
        return False
    
    return True

def validate_pagination(size: int, from_: int) -> Tuple[bool, Optional[str]]:
    """
    Valide les paramètres de pagination.
    
    Args:
        size: Nombre de résultats
        from_: Offset
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if not isinstance(size, int) or size < 0:
        return False, "Size must be a non-negative integer"
    
    if size > 1000:
        return False, "Size cannot exceed 1000"
    
    if not isinstance(from_, int) or from_ < 0:
        return False, "From must be a non-negative integer"
    
    if from_ > 10000:
        return False, "From cannot exceed 10000"
    
    if from_ + size > 10000:
        return False, "From + size cannot exceed 10000"
    
    return True, None

# ==================== FONCTIONS PRIVÉES ====================

def _validate_query_text(query: str, result: ValidationResult) -> bool:
    """Valide le texte de recherche."""
    if not query:
        result.add_error("Query text is required")
        return False
    
    if not isinstance(query, str):
        result.add_error("Query must be a string")
        return False
    
    query = query.strip()
    if len(query) == 0:
        result.add_error("Query text cannot be empty")
        return False
    
    if len(query) > 1000:
        result.add_error("Query text too long (max 1000 characters)")
        return False
    
    if not is_safe_query(query):
        result.add_error("Query contains potentially dangerous content")
        return False
    
    return True

def _validate_user_id_param(user_id: int, result: ValidationResult) -> bool:
    """Valide le paramètre user_id."""
    if not validate_user_id(user_id):
        result.add_error("Valid user_id is required (positive integer)")
        return False
    
    return True

def _validate_filters_dict(filters: Dict[str, Any], result: ValidationResult) -> bool:
    """Valide le dictionnaire de filtres."""
    if not isinstance(filters, dict):
        result.add_error("Filters must be a dictionary")
        return False
    
    if len(filters) > 20:
        result.add_error("Too many filters (max 20)")
        return False
    
    for field, value in filters.items():
        if field not in ALLOWED_SEARCH_FIELDS:
            result.add_warning(f"Unknown filter field: {field}")
        
        # Validation spécifique par champ
        if field == "user_id" and not validate_user_id(value):
            result.add_error(f"Invalid user_id in filters: {value}")
            return False
        elif field == "amount" and not validate_amount(value):
            result.add_error(f"Invalid amount in filters: {value}")
            return False
        elif field in ["transaction_date", "created_at"] and not validate_date(value):
            result.add_error(f"Invalid date in filters: {value}")
            return False
    
    return True

# ==================== FONCTIONS DE NETTOYAGE ====================

def clean_search_text(text: str) -> str:
    """
    Nettoie un texte de recherche en supprimant les caractères indésirables.
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé
    """
    if not isinstance(text, str):
        return ""
    
    # Suppression des caractères de contrôle
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Suppression des caractères spéciaux dangereux
    cleaned = re.sub(r'[<>"\';\\{}]', '', cleaned)
    
    # Normalisation des espaces et tirets
    cleaned = re.sub(r'[-_]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def normalize_amount(amount: Union[str, int, float]) -> Optional[float]:
    """
    Normalise un montant en float avec 2 décimales.
    
    Args:
        amount: Montant à normaliser
        
    Returns:
        Montant normalisé ou None si invalide
    """
    try:
        amount_decimal = Decimal(str(amount))
        normalized = float(amount_decimal.quantize(Decimal('0.01')))
        return normalized if 0 <= normalized <= 999999999.99 else None
    except (ValueError, InvalidOperation):
        return None

def normalize_date(date_value: Union[str, datetime]) -> Optional[datetime]:
    """
    Normalise une date en objet datetime.
    
    Args:
        date_value: Date à normaliser
        
    Returns:
        Date normalisée ou None si invalide
    """
    if isinstance(date_value, datetime):
        return date_value
    
    if isinstance(date_value, str):
        try:
            # Essayer le format ISO
            return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Essayer d'autres formats courants
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except ValueError:
                        continue
            except ValueError:
                pass
    
    return None

# ==================== EXPORTS ====================

__all__ = [
    'validate_search_request',
    'sanitize_query',
    'is_safe_query',
    'escape_elasticsearch_query',
    'validate_user_id',
    'validate_amount',
    'validate_date',
    'validate_email',
    'validate_uuid',
    'validate_currency_code',
    'validate_transaction_type',
    'validate_amount_range',
    'validate_date_range',
    'validate_pagination',
    'clean_search_text',
    'normalize_amount',
    'normalize_date'
]