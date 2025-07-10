"""
Configuration et constantes pour les validateurs.

Ce module centralise toutes les constantes et configurations
utilisées par les différents validateurs.
"""

from typing import Set

# Champs autorisés pour les recherches
ALLOWED_SEARCH_FIELDS: Set[str] = {
    'searchable_text', 'primary_description', 'clean_description',
    'provider_description', 'merchant_name', 'category_id',
    'transaction_date', 'amount', 'user_id', 'account_id',
    'transaction_type', 'currency_code', 'operation_type'
}

# Champs sensibles qui nécessitent une validation stricte
SENSITIVE_FIELDS: Set[str] = {'user_id', 'account_id', 'amount'}

# Types de requêtes Elasticsearch autorisés
ALLOWED_QUERY_TYPES: Set[str] = {
    'match', 'match_all', 'match_phrase', 'match_phrase_prefix',
    'multi_match', 'term', 'terms', 'range', 'exists',
    'wildcard', 'regexp', 'fuzzy', 'prefix', 'bool',
    'simple_query_string', 'function_score', 'more_like_this'
}

# Patterns de sécurité dangereux (déjà définis dans base.py mais réexportés ici)
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
]

# Caractères spéciaux Elasticsearch
ES_SPECIAL_CHARS = r'+-=&|><!(){}[]^"~*?:\/\\'