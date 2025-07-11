"""
✅ Validators Search Service - Validation et Sécurité
=====================================================

Module de validation stricte pour les requêtes, filtres et paramètres
du Search Service. Assure la sécurité, performance et conformité.

Responsabilités:
- Validation requêtes Elasticsearch et contrats
- Sécurité et isolation utilisateur obligatoire
- Validation performance (limites, timeouts)
- Sanitization des entrées utilisateur
- Validation des champs et types
- Exceptions spécialisées
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, date
from decimal import Decimal, InvalidOperation

from search_service.models.service_contracts import (
    SearchServiceQuery, SearchFilter, FilterOperator, 
    AggregationType, QueryType, IntentType
)
from search_service.utils.elasticsearch_helpers import (
    FINANCIAL_FIELDS, FILTERABLE_FIELDS, AGGREGATABLE_FIELDS,
    MAX_QUERY_SIZE, DEFAULT_TIMEOUT_MS, validate_field_name
)

logger = logging.getLogger(__name__)

# =============================================================================
# 🚨 EXCEPTIONS SPÉCIALISÉES
# =============================================================================

class ValidationError(Exception):
    """Erreur de validation générique."""
    pass

class SecurityValidationError(ValidationError):
    """Erreur de validation sécurité (isolation utilisateur, champs interdits)."""
    pass

class PerformanceValidationError(ValidationError):
    """Erreur de validation performance (limites dépassées)."""
    pass

class FieldValidationError(ValidationError):
    """Erreur de validation de champ (type, valeur, accès)."""
    pass

class QueryValidationError(ValidationError):
    """Erreur de validation de requête."""
    pass

# =============================================================================
# 📏 CONSTANTES VALIDATION
# =============================================================================

# Limites de performance strictes
MAX_QUERY_LENGTH = 1000
MAX_RESULTS_LIMIT = 1000
MAX_AGGREGATIONS = 10
MAX_FILTERS = 20
MAX_TIMEOUT_MS = 30000
MIN_USER_ID = 1

# Opérateurs autorisés
ALLOWED_OPERATORS = {
    FilterOperator.EQ, FilterOperator.NE, FilterOperator.GT, FilterOperator.GTE,
    FilterOperator.LT, FilterOperator.LTE, FilterOperator.IN, FilterOperator.NOT_IN,
    FilterOperator.BETWEEN, FilterOperator.EXISTS, FilterOperator.MISSING
}

# Champs interdits (sécurité)
FORBIDDEN_FIELDS = {
    "_id", "_index", "_type", "_score", "_source",
    "password", "token", "secret", "key", "hash"
}

# Champs obligatoires pour isolation
REQUIRED_SECURITY_FIELDS = {"user_id"}

# Patterns validation
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_PATTERN = re.compile(r'^\+?[\d\s\-\(\)]{7,15}$')
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
AMOUNT_PATTERN = re.compile(r'^-?\d+(\.\d{1,2})?$')
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
MONTH_YEAR_PATTERN = re.compile(r'^\d{4}-\d{2}$')

# =============================================================================
# 🛡️ VALIDATORS PRINCIPAUX
# =============================================================================

class ElasticsearchQueryValidator:
    """Validateur pour requêtes Elasticsearch brutes."""
    
    @staticmethod
    def validate_query_structure(query: Dict[str, Any]) -> None:
        """Valide la structure d'une requête Elasticsearch."""
        if not isinstance(query, dict):
            raise QueryValidationError("Query must be a dictionary")
        
        # Vérification taille
        query_str = str(query)
        if len(query_str) > MAX_QUERY_LENGTH * 10:  # Plus large pour ES
            raise PerformanceValidationError(f"Query too large: {len(query_str)} chars")
        
        # Validation clauses dangereuses
        ElasticsearchQueryValidator._check_dangerous_clauses(query)
        
        # Validation limites
        ElasticsearchQueryValidator._validate_query_limits(query)
    
    @staticmethod
    def _check_dangerous_clauses(query: Dict[str, Any]) -> None:
        """Vérifie les clauses dangereuses."""
        query_str = str(query).lower()
        
        dangerous_patterns = [
            'script', 'eval', 'function_score', 'native', 'groovy',
            '_all', 'wildcard', 'regexp', 'fuzzy_like_this'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in query_str:
                raise SecurityValidationError(f"Dangerous clause detected: {pattern}")
    
    @staticmethod
    def _validate_query_limits(query: Dict[str, Any]) -> None:
        """Valide les limites de performance."""
        # Size limit
        size = query.get("size", 10)
        if size > MAX_QUERY_SIZE:
            raise PerformanceValidationError(f"Size too large: {size} > {MAX_QUERY_SIZE}")
        
        # Aggregations limit
        aggs = query.get("aggs", {})
        if len(aggs) > MAX_AGGREGATIONS:
            raise PerformanceValidationError(f"Too many aggregations: {len(aggs)}")
        
        # Timeout validation
        if "timeout" in query:
            timeout = query["timeout"]
            if isinstance(timeout, str) and timeout.endswith("ms"):
                timeout_ms = int(timeout[:-2])
                if timeout_ms > MAX_TIMEOUT_MS:
                    raise PerformanceValidationError(f"Timeout too large: {timeout_ms}ms")

class SearchServiceQueryValidator:
    """Validateur pour contrats SearchServiceQuery."""
    
    @staticmethod
    def validate_complete_query(query: SearchServiceQuery) -> None:
        """Validation complète d'un SearchServiceQuery."""
        # Validation sécurité (priorité #1)
        SearchServiceQueryValidator.validate_security(query)
        
        # Validation métadonnées
        SearchServiceQueryValidator.validate_metadata(query.query_metadata)
        
        # Validation paramètres recherche
        SearchServiceQueryValidator.validate_search_parameters(query.search_parameters)
        
        # Validation filtres
        SearchServiceQueryValidator.validate_filters(query.filters)
        
        # Validation agrégations
        SearchServiceQueryValidator.validate_aggregations(query.aggregations)
        
        # Validation performance globale
        SearchServiceQueryValidator.validate_performance(query)
    
    @staticmethod
    def validate_security(query: SearchServiceQuery) -> None:
        """Validation sécurité OBLIGATOIRE."""
        # Vérification isolation utilisateur MANDATORY
        user_filter_exists = False
        
        for filter_item in query.filters.required:
            if (isinstance(filter_item, dict) and 
                filter_item.get("field") == "user_id" and 
                filter_item.get("operator") in ["eq", FilterOperator.EQ]):
                user_filter_exists = True
                break
        
        if not user_filter_exists:
            raise SecurityValidationError(
                "user_id filter is mandatory for security isolation"
            )
        
        # Validation user_id cohérent
        metadata_user_id = query.query_metadata.user_id
        filter_user_id = None
        
        for filter_item in query.filters.required:
            if (isinstance(filter_item, dict) and 
                filter_item.get("field") == "user_id"):
                filter_user_id = filter_item.get("value")
                break
        
        if metadata_user_id != filter_user_id:
            raise SecurityValidationError(
                f"user_id mismatch: metadata={metadata_user_id}, filter={filter_user_id}"
            )
    
    @staticmethod
    def validate_metadata(metadata) -> None:
        """Valide les métadonnées de requête."""
        # User ID obligatoire et valide
        if not isinstance(metadata.user_id, int) or metadata.user_id < MIN_USER_ID:
            raise ValidationError(f"Invalid user_id: {metadata.user_id}")
        
        # Intent type valide
        if metadata.intent_type not in IntentType:
            raise ValidationError(f"Invalid intent_type: {metadata.intent_type}")
        
        # Confidence valide
        if not (0.0 <= metadata.confidence <= 1.0):
            raise ValidationError(f"Invalid confidence: {metadata.confidence}")
        
        # Agent name requis
        if not metadata.agent_name or len(metadata.agent_name.strip()) == 0:
            raise ValidationError("agent_name is required")
    
    @staticmethod
    def validate_search_parameters(params) -> None:
        """Valide les paramètres de recherche."""
        # Query type valide
        if params.query_type not in QueryType:
            raise ValidationError(f"Invalid query_type: {params.query_type}")
        
        # Fields validation
        for field in params.fields:
            if not validate_field_access(field):
                raise FieldValidationError(f"Invalid or forbidden field: {field}")
        
        # Limits validation
        if params.limit > MAX_RESULTS_LIMIT:
            raise PerformanceValidationError(f"Limit too high: {params.limit}")
        
        if params.offset < 0:
            raise ValidationError(f"Invalid offset: {params.offset}")
        
        # Timeout validation
        if params.timeout_ms > MAX_TIMEOUT_MS:
            raise PerformanceValidationError(f"Timeout too high: {params.timeout_ms}")
    
    @staticmethod
    def validate_filters(filters) -> None:
        """Valide le groupe de filtres."""
        total_filters = (
            len(filters.required) + 
            len(filters.optional) + 
            len(filters.ranges)
        )
        
        if total_filters > MAX_FILTERS:
            raise PerformanceValidationError(f"Too many filters: {total_filters}")
        
        # Validation filtres individuels
        for filter_list in [filters.required, filters.optional, filters.ranges]:
            for filter_item in filter_list:
                FilterValidator.validate_filter(filter_item)
    
    @staticmethod
    def validate_aggregations(aggs) -> None:
        """Valide les agrégations."""
        if not aggs.enabled:
            return
        
        if len(aggs.requests) > MAX_AGGREGATIONS:
            raise PerformanceValidationError(f"Too many aggregations: {len(aggs.requests)}")
        
        for agg_request in aggs.requests:
            if agg_request.agg_type not in AggregationType:
                raise ValidationError(f"Invalid aggregation type: {agg_request.agg_type}")
            
            if not validate_field_access(agg_request.field):
                raise FieldValidationError(f"Invalid aggregation field: {agg_request.field}")
    
    @staticmethod
    def validate_performance(query: SearchServiceQuery) -> None:
        """Validation performance globale."""
        complexity_score = 0
        
        # Score des filtres
        complexity_score += len(query.filters.required) * 2
        complexity_score += len(query.filters.optional) * 1
        complexity_score += len(query.filters.ranges) * 3
        
        # Score des agrégations
        if query.aggregations.enabled:
            complexity_score += len(query.aggregations.requests) * 5
        
        # Score de la taille
        if query.search_parameters.limit > 100:
            complexity_score += 3
        
        # Validation complexité
        if complexity_score > 50:
            raise PerformanceValidationError(f"Query too complex: score={complexity_score}")

class FilterValidator:
    """Validateur pour filtres individuels."""
    
    @staticmethod
    def validate_filter(filter_item: Union[Dict, SearchFilter]) -> None:
        """Valide un filtre individuel."""
        if isinstance(filter_item, dict):
            # Format dict basique
            FilterValidator._validate_dict_filter(filter_item)
        elif isinstance(filter_item, SearchFilter):
            # Objet SearchFilter
            FilterValidator._validate_search_filter(filter_item)
        else:
            raise ValidationError(f"Invalid filter type: {type(filter_item)}")
    
    @staticmethod
    def _validate_dict_filter(filter_dict: Dict[str, Any]) -> None:
        """Valide un filtre format dictionnaire."""
        required_keys = {"field", "operator", "value"}
        if not all(key in filter_dict for key in required_keys):
            raise ValidationError(f"Filter missing required keys: {required_keys}")
        
        field = filter_dict["field"]
        operator = filter_dict["operator"]
        value = filter_dict["value"]
        
        # Validation champ
        if not validate_field_access(field):
            raise FieldValidationError(f"Invalid field: {field}")
        
        # Validation opérateur
        if isinstance(operator, str):
            operator = FilterOperator(operator)
        if operator not in ALLOWED_OPERATORS:
            raise ValidationError(f"Invalid operator: {operator}")
        
        # Validation valeur selon le type de champ
        FilterValidator._validate_field_value(field, operator, value)
    
    @staticmethod
    def _validate_search_filter(search_filter: SearchFilter) -> None:
        """Valide un objet SearchFilter."""
        # Validation champ
        if not validate_field_access(search_filter.field):
            raise FieldValidationError(f"Invalid field: {search_filter.field}")
        
        # Validation opérateur
        if search_filter.operator not in ALLOWED_OPERATORS:
            raise ValidationError(f"Invalid operator: {search_filter.operator}")
        
        # Validation valeur
        FilterValidator._validate_field_value(
            search_filter.field, 
            search_filter.operator, 
            search_filter.value
        )
    
    @staticmethod
    def _validate_field_value(field: str, operator: FilterOperator, value: Any) -> None:
        """Valide la valeur d'un filtre selon le type de champ."""
        base_field = field.split('.')[0]
        
        # Validation selon le type de champ
        if base_field == "user_id":
            if not isinstance(value, int) or value < MIN_USER_ID:
                raise ValidationError(f"Invalid user_id: {value}")
        
        elif base_field in ["amount", "amount_abs"]:
            FilterValidator._validate_amount_value(operator, value)
        
        elif base_field == "date":
            FilterValidator._validate_date_value(operator, value)
        
        elif base_field == "month_year":
            FilterValidator._validate_month_year_value(value)
        
        elif base_field in ["category_name", "merchant_name"]:
            FilterValidator._validate_string_value(operator, value)
    
    @staticmethod
    def _validate_amount_value(operator: FilterOperator, value: Any) -> None:
        """Valide une valeur de montant."""
        if operator == FilterOperator.BETWEEN:
            if not isinstance(value, list) or len(value) != 2:
                raise ValidationError("BETWEEN operator requires list of 2 values")
            for v in value:
                if not isinstance(v, (int, float, Decimal)):
                    raise ValidationError(f"Invalid amount value: {v}")
        elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(value, list):
                raise ValidationError(f"{operator} requires list value")
            for v in value:
                if not isinstance(v, (int, float, Decimal)):
                    raise ValidationError(f"Invalid amount value: {v}")
        else:
            if not isinstance(value, (int, float, Decimal)):
                raise ValidationError(f"Invalid amount value: {value}")
    
    @staticmethod
    def _validate_date_value(operator: FilterOperator, value: Any) -> None:
        """Valide une valeur de date."""
        if operator == FilterOperator.BETWEEN:
            if not isinstance(value, list) or len(value) != 2:
                raise ValidationError("BETWEEN operator requires list of 2 values")
            for v in value:
                if not _is_valid_date_string(v):
                    raise ValidationError(f"Invalid date value: {v}")
        elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(value, list):
                raise ValidationError(f"{operator} requires list value")
            for v in value:
                if not _is_valid_date_string(v):
                    raise ValidationError(f"Invalid date value: {v}")
        else:
            if not _is_valid_date_string(value):
                raise ValidationError(f"Invalid date value: {value}")
    
    @staticmethod
    def _validate_month_year_value(value: Any) -> None:
        """Valide une valeur month_year."""
        if not isinstance(value, str) or not MONTH_YEAR_PATTERN.match(value):
            raise ValidationError(f"Invalid month_year format: {value}")
    
    @staticmethod
    def _validate_string_value(operator: FilterOperator, value: Any) -> None:
        """Valide une valeur string."""
        if operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(value, list):
                raise ValidationError(f"{operator} requires list value")
            for v in value:
                if not isinstance(v, str) or len(v.strip()) == 0:
                    raise ValidationError(f"Invalid string value: {v}")
        else:
            if not isinstance(value, str) or len(value.strip()) == 0:
                raise ValidationError(f"Invalid string value: {value}")

# =============================================================================
# 🔍 VALIDATION SPÉCIALISÉE
# =============================================================================

def validate_user_isolation(query: SearchServiceQuery) -> bool:
    """
    Vérifie que l'isolation utilisateur est correctement appliquée.
    
    Args:
        query: Requête à valider
        
    Returns:
        True si l'isolation est correcte
        
    Raises:
        SecurityValidationError: Si l'isolation est manquante ou incorrecte
    """
    try:
        SearchServiceQueryValidator.validate_security(query)
        return True
    except SecurityValidationError:
        raise
    except Exception as e:
        raise SecurityValidationError(f"User isolation validation failed: {e}")

def validate_financial_query(query: SearchServiceQuery) -> bool:
    """
    Valide qu'une requête respecte les contraintes financières.
    
    Args:
        query: Requête à valider
        
    Returns:
        True si la requête est valide
        
    Raises:
        ValidationError: Si la requête est invalide
    """
    # Validation champs financiers uniquement
    for field in query.search_parameters.fields:
        base_field = field.split('.')[0]
        if base_field not in FINANCIAL_FIELDS:
            raise FieldValidationError(f"Non-financial field not allowed: {field}")
    
    # Validation intentions financières
    financial_intents = {
        IntentType.SEARCH_BY_CATEGORY, IntentType.SEARCH_BY_MERCHANT,
        IntentType.SEARCH_BY_AMOUNT, IntentType.SEARCH_BY_DATE,
        IntentType.COUNT_OPERATIONS, IntentType.TEMPORAL_SPENDING_ANALYSIS
    }
    
    if query.query_metadata.intent_type not in financial_intents:
        raise ValidationError(f"Non-financial intent: {query.query_metadata.intent_type}")
    
    return True

def validate_search_parameters(params) -> bool:
    """Valide les paramètres de recherche."""
    try:
        SearchServiceQueryValidator.validate_search_parameters(params)
        return True
    except ValidationError:
        raise

def validate_filter_security(filter_item: Union[Dict, SearchFilter]) -> bool:
    """
    Valide qu'un filtre respecte les règles de sécurité.
    
    Args:
        filter_item: Filtre à valider
        
    Returns:
        True si le filtre est sécurisé
        
    Raises:
        SecurityValidationError: Si le filtre est dangereux
    """
    if isinstance(filter_item, dict):
        field = filter_item.get("field")
        value = filter_item.get("value")
    elif isinstance(filter_item, SearchFilter):
        field = filter_item.field
        value = filter_item.value
    else:
        raise ValidationError(f"Invalid filter type: {type(filter_item)}")
    
    # Vérification champs interdits
    if field in FORBIDDEN_FIELDS:
        raise SecurityValidationError(f"Forbidden field: {field}")
    
    # Vérification valeurs dangereuses
    if isinstance(value, str):
        dangerous_patterns = ['<script', 'javascript:', 'eval(', 'function(']
        for pattern in dangerous_patterns:
            if pattern.lower() in value.lower():
                raise SecurityValidationError(f"Dangerous value detected: {pattern}")
    
    return True

def validate_aggregation_request(agg_request) -> bool:
    """Valide une demande d'agrégation."""
    if not validate_field_access(agg_request.field):
        raise FieldValidationError(f"Invalid aggregation field: {agg_request.field}")
    
    # Validation selon le type d'agrégation
    if agg_request.agg_type in [AggregationType.SUM, AggregationType.AVG]:
        base_field = agg_request.field.split('.')[0]
        if base_field not in ["amount", "amount_abs"]:
            raise ValidationError(f"Sum/Avg only allowed on amount fields: {agg_request.field}")
    
    return True

def validate_query_complexity(query: SearchServiceQuery) -> str:
    """
    Évalue et valide la complexité d'une requête.
    
    Args:
        query: Requête à évaluer
        
    Returns:
        Niveau de complexité ('simple', 'medium', 'complex')
        
    Raises:
        PerformanceValidationError: Si la requête est trop complexe
    """
    complexity_score = 0
    
    # Score des filtres
    complexity_score += len(query.filters.required) * 2
    complexity_score += len(query.filters.optional) * 1
    complexity_score += len(query.filters.ranges) * 3
    
    # Score text search
    if query.filters.text_search and query.filters.text_search.query:
        complexity_score += 3
    
    # Score des agrégations
    if query.aggregations.enabled:
        complexity_score += len(query.aggregations.requests) * 5
        
        # Bonus pour agrégations complexes
        for agg in query.aggregations.requests:
            if agg.agg_type in [AggregationType.DATE_HISTOGRAM, AggregationType.STATS]:
                complexity_score += 2
    
    # Score de la taille
    if query.search_parameters.limit > 100:
        complexity_score += 3
    elif query.search_parameters.limit > 50:
        complexity_score += 1
    
    # Classification
    if complexity_score <= 5:
        level = "simple"
    elif complexity_score <= 15:
        level = "medium"
    elif complexity_score <= 30:
        level = "complex"
    else:
        raise PerformanceValidationError(f"Query too complex: score={complexity_score}")
    
    return level

# =============================================================================
# 🔍 VALIDATION CHAMPS
# =============================================================================

def validate_field_access(field_name: str) -> bool:
    """
    Valide qu'un champ peut être accédé.
    
    Args:
        field_name: Nom du champ à valider
        
    Returns:
        True si le champ est accessible
    """
    if not field_name or not isinstance(field_name, str):
        return False
    
    # Nettoyage et extraction du champ de base
    clean_field = field_name.strip()
    base_field = clean_field.split('.')[0]
    
    # Vérification champs interdits
    if base_field in FORBIDDEN_FIELDS:
        return False
    
    # Vérification champs autorisés
    if base_field not in FINANCIAL_FIELDS:
        return False
    
    # Validation format nom de champ
    return validate_field_name(clean_field)

def validate_field_types(field_name: str, expected_type: str) -> bool:
    """Valide le type d'un champ."""
    from search_service.utils.elasticsearch_helpers import get_field_mapping
    
    actual_type = get_field_mapping(field_name)
    if actual_type is None:
        return False
    
    # Mapping des types compatibles
    compatible_types = {
        "text": ["text", "keyword"],
        "keyword": ["keyword", "text"],
        "long": ["long", "integer", "double", "float"],
        "double": ["double", "float", "long", "integer"],
        "date": ["date"],
    }
    
    return expected_type in compatible_types.get(actual_type, [actual_type])

def validate_field_values(field_name: str, values: List[Any]) -> bool:
    """Valide les valeurs d'un champ."""
    base_field = field_name.split('.')[0]
    
    for value in values:
        if base_field == "user_id":
            if not isinstance(value, int) or value < MIN_USER_ID:
                return False
        elif base_field in ["amount", "amount_abs"]:
            if not isinstance(value, (int, float, Decimal)):
                return False
        elif base_field == "date":
            if not _is_valid_date_string(value):
                return False
        elif base_field in ["category_name", "merchant_name"]:
            if not isinstance(value, str) or len(value.strip()) == 0:
                return False
    
    return True

def validate_date_ranges(start_date: str, end_date: str) -> bool:
    """Valide une plage de dates."""
    if not _is_valid_date_string(start_date) or not _is_valid_date_string(end_date):
        return False
    
    # Vérification ordre chronologique
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return start <= end
    except ValueError:
        return False

def validate_amount_ranges(min_amount: float, max_amount: float) -> bool:
    """Valide une plage de montants."""
    if not isinstance(min_amount, (int, float)) or not isinstance(max_amount, (int, float)):
        return False
    
    return min_amount <= max_amount and min_amount >= 0

def validate_text_search(query_text: str) -> bool:
    """Valide un texte de recherche."""
    if not isinstance(query_text, str):
        return False
    
    clean_text = query_text.strip()
    
    # Longueur minimale et maximale
    if len(clean_text) < 1 or len(clean_text) > 500:
        return False
    
    # Caractères dangereux
    dangerous_patterns = ['<script', 'javascript:', 'eval(', '../../']
    for pattern in dangerous_patterns:
        if pattern.lower() in clean_text.lower():
            return False
    
    return True

# =============================================================================
# 🔍 VALIDATION PERFORMANCE
# =============================================================================

def validate_query_limits(limit: int, offset: int) -> bool:
    """Valide les limites de requête."""
    if not isinstance(limit, int) or not isinstance(offset, int):
        return False
    
    if limit <= 0 or limit > MAX_RESULTS_LIMIT:
        return False
    
    if offset < 0 or offset > 100000:  # Limite pagination profonde
        return False
    
    return True

def validate_timeout_settings(timeout_ms: int) -> bool:
    """Valide les paramètres de timeout."""
    if not isinstance(timeout_ms, int):
        return False
    
    return 100 <= timeout_ms <= MAX_TIMEOUT_MS

def validate_cache_settings(cache_enabled: bool, cache_ttl: Optional[int] = None) -> bool:
    """Valide les paramètres de cache."""
    if not isinstance(cache_enabled, bool):
        return False
    
    if cache_ttl is not None:
        if not isinstance(cache_ttl, int) or cache_ttl < 0:
            return False
    
    return True

# =============================================================================
# 🧹 SANITIZATION
# =============================================================================

def sanitize_user_input(user_input: str) -> str:
    """Nettoie une entrée utilisateur."""
    if not isinstance(user_input, str):
        return ""
    
    # Nettoyage de base
    sanitized = user_input.strip()
    
    # Suppression caractères de contrôle
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    
    # Échappement caractères dangereux
    dangerous_chars = ['<', '>', '"', "'", '&', '\\']
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, f'\\{char}')
    
    # Limitation longueur
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000]
    
    return sanitized

def sanitize_elasticsearch_query(query_text: str) -> str:
    """Nettoie un texte pour requête Elasticsearch."""
    from search_service.utils.elasticsearch_helpers import escape_elasticsearch_query
    
    if not isinstance(query_text, str):
        return ""
    
    # Nettoyage de base
    clean_text = query_text.strip()
    
    # Échappement spécifique Elasticsearch
    escaped = escape_elasticsearch_query(clean_text)
    
    return escaped

def sanitize_field_values(values: List[Any]) -> List[Any]:
    """Nettoie une liste de valeurs."""
    sanitized = []
    
    for value in values:
        if isinstance(value, str):
            clean_value = sanitize_user_input(value)
            if clean_value:  # Garde seulement les valeurs non-vides
                sanitized.append(clean_value)
        elif isinstance(value, (int, float, bool)):
            sanitized.append(value)
        # Ignore les autres types
    
    return sanitized

# =============================================================================
# 🔧 HELPERS VALIDATION
# =============================================================================

def is_valid_user_id(user_id: Any) -> bool:
    """Vérifie si un user_id est valide."""
    return isinstance(user_id, int) and user_id >= MIN_USER_ID

def is_valid_field_name(field_name: str) -> bool:
    """Vérifie si un nom de champ est valide."""
    return validate_field_name(field_name)

def is_safe_query_value(value: Any) -> bool:
    """Vérifie si une valeur de requête est sécurisée."""
    if isinstance(value, str):
        return validate_text_search(value)
    elif isinstance(value, (int, float, bool)):
        return True
    elif isinstance(value, list):
        return all(is_safe_query_value(v) for v in value)
    else:
        return False

def _is_valid_date_string(date_str: Any) -> bool:
    """Vérifie si une string est une date valide."""
    if not isinstance(date_str, str):
        return False
    
    if not DATE_PATTERN.match(date_str):
        return False
    
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def _is_valid_email(email: str) -> bool:
    """Vérifie si un email est valide."""
    if not isinstance(email, str):
        return False
    return EMAIL_PATTERN.match(email) is not None

def _is_valid_phone(phone: str) -> bool:
    """Vérifie si un téléphone est valide."""
    if not isinstance(phone, str):
        return False
    return PHONE_PATTERN.match(phone) is not None

def _is_valid_uuid(uuid_str: str) -> bool:
    """Vérifie si un UUID est valide."""
    if not isinstance(uuid_str, str):
        return False
    return UUID_PATTERN.match(uuid_str) is not None

def _is_valid_amount(amount_str: str) -> bool:
    """Vérifie si un montant est valide."""
    if not isinstance(amount_str, str):
        return False
    return AMOUNT_PATTERN.match(amount_str) is not None

def _is_numeric_value(value: Any) -> bool:
    """Vérifie si une valeur est numérique."""
    return isinstance(value, (int, float, Decimal))

def _is_string_value(value: Any) -> bool:
    """Vérifie si une valeur est une string non-vide."""
    return isinstance(value, str) and len(value.strip()) > 0

def _is_list_value(value: Any) -> bool:
    """Vérifie si une valeur est une liste."""
    return isinstance(value, list) and len(value) > 0

def _contains_sql_injection(text: str) -> bool:
    """Détecte les tentatives d'injection SQL."""
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    sql_patterns = [
        'union select', 'drop table', 'delete from', 'insert into',
        'update set', 'alter table', 'create table', '--', '/*', '*/',
        'xp_cmdshell', 'sp_executesql', 'exec(', 'execute('
    ]
    
    return any(pattern in text_lower for pattern in sql_patterns)

def _contains_xss_attempt(text: str) -> bool:
    """Détecte les tentatives XSS."""
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    xss_patterns = [
        '<script', '</script>', 'javascript:', 'onerror=', 'onload=',
        'onclick=', 'onmouseover=', 'onfocus=', 'alert(', 'confirm(',
        'prompt(', 'document.cookie', 'window.location'
    ]
    
    return any(pattern in text_lower for pattern in xss_patterns)

def _sanitize_special_characters(text: str) -> str:
    """Nettoie les caractères spéciaux."""
    if not isinstance(text, str):
        return ""
    
    # Caractères à remplacer
    replacements = {
        '\x00': '',  # NULL byte
        '\r': '',    # Carriage return
        '\n': ' ',   # Newline to space
        '\t': ' ',   # Tab to space
        '\b': '',    # Backspace
        '\f': '',    # Form feed
        '\v': '',    # Vertical tab
    }
    
    cleaned = text
    for old_char, new_char in replacements.items():
        cleaned = cleaned.replace(old_char, new_char)
    
    return cleaned

def _normalize_whitespace(text: str) -> str:
    """Normalise les espaces."""
    if not isinstance(text, str):
        return ""
    
    # Remplace plusieurs espaces par un seul
    normalized = re.sub(r'\s+', ' ', text.strip())
    return normalized

def _validate_string_length(text: str, min_length: int = 0, max_length: int = 1000) -> bool:
    """Valide la longueur d'une string."""
    if not isinstance(text, str):
        return False
    
    length = len(text.strip())
    return min_length <= length <= max_length

def _validate_numeric_range(value: Union[int, float], min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
    """Valide qu'un nombre est dans une plage."""
    if not isinstance(value, (int, float)):
        return False
    
    if min_value is not None and value < min_value:
        return False
    
    if max_value is not None and value > max_value:
        return False
    
    return True

def _validate_list_contents(values: List[Any], validator_func: callable) -> bool:
    """Valide le contenu d'une liste avec une fonction."""
    if not isinstance(values, list):
        return False
    
    return all(validator_func(value) for value in values)

def _extract_field_base_name(field_name: str) -> str:
    """Extrait le nom de base d'un champ (sans .keyword, etc.)."""
    if not isinstance(field_name, str):
        return ""
    
    return field_name.split('.')[0]

def _get_field_type_from_mapping(field_name: str) -> Optional[str]:
    """Récupère le type d'un champ depuis le mapping."""
    base_field = _extract_field_base_name(field_name)
    
    # Types des champs financiers
    field_types = {
        "user_id": "long",
        "transaction_id": "keyword",
        "account_id": "long",
        "amount": "double",
        "amount_abs": "double",
        "currency_code": "keyword",
        "date": "date",
        "month_year": "keyword",
        "weekday": "keyword",
        "primary_description": "text",
        "merchant_name": "text",
        "category_name": "text",
        "transaction_type": "keyword",
        "operation_type": "keyword",
        "searchable_text": "text"
    }
    
    return field_types.get(base_field)

def _validate_field_value_type(field_name: str, value: Any) -> bool:
    """Valide qu'une valeur correspond au type attendu du champ."""
    field_type = _get_field_type_from_mapping(field_name)
    if not field_type:
        return False
    
    if field_type in ["long", "double"]:
        return isinstance(value, (int, float))
    elif field_type in ["text", "keyword"]:
        return isinstance(value, str)
    elif field_type == "date":
        return isinstance(value, str) and _is_valid_date_string(value)
    elif field_type == "boolean":
        return isinstance(value, bool)
    else:
        return True  # Type inconnu, on laisse passer

# =============================================================================
# 🔧 VALIDATION AVANCÉE BUSINESS
# =============================================================================

def validate_financial_amount(amount: Union[int, float, str]) -> bool:
    """Valide un montant financier."""
    if isinstance(amount, str):
        if not _is_valid_amount(amount):
            return False
        try:
            amount = float(amount)
        except ValueError:
            return False
    
    if not isinstance(amount, (int, float)):
        return False
    
    # Validation plage réaliste pour transactions
    return -1000000 <= amount <= 1000000  # -1M à +1M

def validate_transaction_date(date_value: Union[str, date, datetime]) -> bool:
    """Valide une date de transaction."""
    if isinstance(date_value, str):
        if not _is_valid_date_string(date_value):
            return False
        try:
            parsed_date = datetime.strptime(date_value, "%Y-%m-%d").date()
        except ValueError:
            return False
    elif isinstance(date_value, datetime):
        parsed_date = date_value.date()
    elif isinstance(date_value, date):
        parsed_date = date_value
    else:
        return False
    
    # Validation plage réaliste (pas de dates futures, pas trop anciennes)
    today = date.today()
    min_date = date(2000, 1, 1)  # Pas de transactions avant 2000
    
    return min_date <= parsed_date <= today

def validate_user_context(user_id: int, query_user_id: int) -> bool:
    """Valide que l'utilisateur peut accéder aux données demandées."""
    # Isolation stricte : un utilisateur ne peut accéder qu'à ses données
    return user_id == query_user_id and user_id > 0

def validate_query_intent_coherence(intent: IntentType, query_params: Dict[str, Any]) -> bool:
    """Valide que l'intention correspond aux paramètres de la requête."""
    if intent == IntentType.SEARCH_BY_CATEGORY:
        return "category" in query_params or "category_name" in query_params
    
    elif intent == IntentType.SEARCH_BY_MERCHANT:
        return "merchant" in query_params or "merchant_name" in query_params
    
    elif intent == IntentType.SEARCH_BY_AMOUNT:
        return any(key in query_params for key in ["amount", "amount_min", "amount_max"])
    
    elif intent == IntentType.SEARCH_BY_DATE:
        return any(key in query_params for key in ["date", "start_date", "end_date", "month_year"])
    
    elif intent == IntentType.TEXT_SEARCH:
        return "query" in query_params or "search_text" in query_params
    
    # Pour les autres intentions, on accepte
    return True

def validate_aggregation_coherence(agg_type: AggregationType, field: str) -> bool:
    """Valide que l'agrégation est cohérente avec le champ."""
    base_field = _extract_field_base_name(field)
    
    if agg_type in [AggregationType.SUM, AggregationType.AVG]:
        # Sum/Avg seulement sur champs numériques
        return base_field in ["amount", "amount_abs"]
    
    elif agg_type == AggregationType.DATE_HISTOGRAM:
        # Histogramme seulement sur champs date
        return base_field in ["date"]
    
    elif agg_type == AggregationType.TERMS:
        # Terms sur champs catégoriels
        return base_field in ["category_name", "merchant_name", "transaction_type", "month_year"]
    
    elif agg_type in [AggregationType.COUNT, AggregationType.MIN, AggregationType.MAX]:
        # Accepté sur tous les champs
        return True
    
    return False

# =============================================================================
# 📊 CONSTANTES EXPORT
# =============================================================================

__all__ = [
    # Validators principaux
    "ElasticsearchQueryValidator", "SearchServiceQueryValidator", "FilterValidator",
    # Validation spécialisée
    "validate_user_isolation", "validate_financial_query", "validate_search_parameters",
    "validate_filter_security", "validate_aggregation_request", "validate_query_complexity",
    # Validation champs
    "validate_field_access", "validate_field_types", "validate_field_values",
    "validate_date_ranges", "validate_amount_ranges", "validate_text_search",
    # Validation performance
    "validate_query_limits", "validate_timeout_settings", "validate_cache_settings",
    # Sanitization
    "sanitize_user_input", "sanitize_elasticsearch_query", "sanitize_field_values",
    # Helpers
    "is_valid_user_id", "is_valid_field_name", "is_safe_query_value",
    # Validation business
    "validate_financial_amount", "validate_transaction_date", "validate_user_context",
    "validate_query_intent_coherence", "validate_aggregation_coherence",
    # Exceptions
    "ValidationError", "SecurityValidationError", "PerformanceValidationError",
    "FieldValidationError", "QueryValidationError",
    # Constantes
    "MAX_QUERY_LENGTH", "MAX_RESULTS_LIMIT", "ALLOWED_OPERATORS", "FORBIDDEN_FIELDS",
    "REQUIRED_SECURITY_FIELDS", "MIN_USER_ID", "MAX_AGGREGATIONS", "MAX_FILTERS",
    "EMAIL_PATTERN", "PHONE_PATTERN", "UUID_PATTERN", "AMOUNT_PATTERN", 
    "DATE_PATTERN", "MONTH_YEAR_PATTERN"
]