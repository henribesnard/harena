"""
Configuration et constantes pour les validateurs.

Ce module centralise toutes les constantes et configurations
utilisées par les différents validateurs.
"""

from typing import Set, Dict, List

# ==================== CHAMPS AUTORISÉS ====================

# Champs autorisés pour les recherches
ALLOWED_SEARCH_FIELDS: Set[str] = {
    'searchable_text', 'primary_description', 'clean_description',
    'provider_description', 'merchant_name', 'category_id',
    'transaction_date', 'amount', 'user_id', 'account_id',
    'transaction_type', 'currency_code', 'operation_type',
    'reference_number', 'external_id', 'tags'
}

# Champs sensibles qui nécessitent une validation stricte
SENSITIVE_FIELDS: Set[str] = {
    'user_id', 'account_id', 'amount', 'external_id', 'reference_number'
}

# Champs requis pour une recherche de base
REQUIRED_SEARCH_FIELDS: Set[str] = {
    'user_id'
}

# ==================== TYPES DE REQUÊTES ====================

# Types de requêtes Elasticsearch autorisés
ALLOWED_QUERY_TYPES: Set[str] = {
    'match', 'match_all', 'match_phrase', 'match_phrase_prefix',
    'multi_match', 'term', 'terms', 'range', 'exists',
    'wildcard', 'regexp', 'fuzzy', 'prefix', 'bool',
    'simple_query_string', 'function_score', 'more_like_this',
    'constant_score'
}

# Types de requêtes dangereuses qui nécessitent une validation stricte
DANGEROUS_QUERY_TYPES: Set[str] = {
    'script', 'script_score', 'percolate'
}

# ==================== FILTRES ====================

# Types de filtres autorisés
ALLOWED_FILTER_TYPES: Set[str] = {
    'term', 'terms', 'range', 'exists', 'bool', 'match'
}

# Opérateurs de comparaison autorisés
ALLOWED_COMPARISON_OPERATORS: Set[str] = {
    'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in', 'not_in', 'exists', 'not_exists'
}

# ==================== PATTERNS DE SÉCURITÉ ====================

# Patterns de sécurité dangereux (importés de base.py mais réexportés ici)
DANGEROUS_PATTERNS: List[str] = [
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
    r'alter\s+table',
    r'create\s+table',
    r'truncate\s+table'
]

# Caractères spéciaux Elasticsearch
ES_SPECIAL_CHARS: str = r'+-=&|><!(){}[]^"~*?:\/\\'

# ==================== LIMITES ET CONTRAINTES ====================

# Limites de validation
VALIDATION_LIMITS: Dict[str, int] = {
    'max_query_length': 1000,
    'min_query_length': 1,
    'max_filter_values': 100,
    'max_nested_depth': 10,
    'max_bool_clauses': 50,
    'max_wildcard_queries': 10,
    'max_regexp_queries': 5,
    'max_fuzzy_queries': 20,
    'max_results_limit': 1000,
    'default_results_limit': 20,
    'max_aggregation_buckets': 10000,
    'max_scroll_size': 1000,
    'max_timeout_ms': 30000,
    'default_timeout_ms': 5000
}

# Limites pour les types de données
DATA_TYPE_LIMITS: Dict[str, Dict[str, any]] = {
    'user_id': {
        'min_value': 1,
        'max_value': 999999999
    },
    'amount': {
        'min_value': 0.0,
        'max_value': 999999999.99,
        'decimal_places': 2
    },
    'string_field': {
        'min_length': 0,
        'max_length': 1000
    },
    'description': {
        'min_length': 0,
        'max_length': 5000
    }
}

# ==================== CONFIGURATION VALIDATION ====================

# Configuration par niveau de validation
VALIDATION_CONFIG: Dict[str, Dict[str, any]] = {
    'basic': {
        'check_security_patterns': False,
        'sanitize_input': True,
        'validate_types': True,
        'validate_ranges': False,
        'check_complexity': False
    },
    'standard': {
        'check_security_patterns': True,
        'sanitize_input': True,
        'validate_types': True,
        'validate_ranges': True,
        'check_complexity': True
    },
    'strict': {
        'check_security_patterns': True,
        'sanitize_input': True,
        'validate_types': True,
        'validate_ranges': True,
        'check_complexity': True,
        'block_dangerous_patterns': True
    },
    'paranoid': {
        'check_security_patterns': True,
        'sanitize_input': True,
        'validate_types': True,
        'validate_ranges': True,
        'check_complexity': True,
        'block_dangerous_patterns': True,
        'whitelist_only': True
    }
}

# ==================== PATTERNS DE VALIDATION ====================

# Patterns regex pour validation
VALIDATION_PATTERNS: Dict[str, str] = {
    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    'user_id': r'^\d+$',
    'amount': r'^\d+(\.\d{1,2})?$',
    'currency_code': r'^[A-Z]{3}$',
    'transaction_type': r'^[a-zA-Z_]+$',
    'safe_string': r'^[a-zA-Z0-9\s\-_.]+$'
}

# ==================== MESSAGES D'ERREUR ====================

# Messages d'erreur standardisés
ERROR_MESSAGES: Dict[str, str] = {
    'required_field': "Le champ '{field}' est requis",
    'invalid_type': "Le champ '{field}' doit être de type {expected_type}",
    'invalid_length': "Le champ '{field}' doit avoir entre {min_length} et {max_length} caractères",
    'invalid_range': "Le champ '{field}' doit être entre {min_value} et {max_value}",
    'invalid_pattern': "Le champ '{field}' a un format invalide",
    'security_violation': "Contenu potentiellement dangereux détecté dans le champ '{field}'",
    'query_too_complex': "La requête est trop complexe (score: {complexity_score})",
    'too_many_results': "Trop de résultats demandés (max: {max_limit})",
    'timeout_exceeded': "Timeout de validation dépassé"
}

# ==================== EXPORTS ====================

__all__ = [
    'ALLOWED_SEARCH_FIELDS',
    'SENSITIVE_FIELDS',
    'REQUIRED_SEARCH_FIELDS',
    'ALLOWED_QUERY_TYPES',
    'DANGEROUS_QUERY_TYPES',
    'ALLOWED_FILTER_TYPES',
    'ALLOWED_COMPARISON_OPERATORS',
    'DANGEROUS_PATTERNS',
    'ES_SPECIAL_CHARS',
    'VALIDATION_LIMITS',
    'DATA_TYPE_LIMITS',
    'VALIDATION_CONFIG',
    'VALIDATION_PATTERNS',
    'ERROR_MESSAGES'
]