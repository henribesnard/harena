"""
🔍 Elasticsearch Helpers - Utilitaires Spécialisés Bonsai
==========================================================

Utilitaires spécialisés pour la construction, optimisation et traitement 
des requêtes Elasticsearch/Bonsai pour les transactions financières.

Responsabilités:
- Construction requêtes Elasticsearch optimisées
- Filtres spécialisés financiers (user, category, merchant, amount, date)
- Agrégations financières (sommes, moyennes, histogrammes temporels)
- Optimisation performance et cache
- Parsing et formatage réponses
- Validation et sécurité requêtes
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date
from decimal import Decimal
import re

from search_service.models.service_contracts import (
    SearchFilter, FilterOperator, AggregationType, TextSearchOperator
)

logger = logging.getLogger(__name__)

# =============================================================================
# 📊 CONSTANTES FINANCIÈRES
# =============================================================================

# Champs financiers de l'index harena_transactions
FINANCIAL_FIELDS = {
    # Identifiants
    "transaction_id", "user_id", "account_id",
    # Montants et devises
    "amount", "amount_abs", "currency_code",
    # Temporel
    "date", "month_year", "weekday",
    # Descriptifs
    "primary_description", "merchant_name", "category_name",
    # Types et statuts
    "transaction_type", "operation_type",
    # Recherche
    "searchable_text"
}

# Champs pour recherche textuelle (BM25)
SEARCHABLE_FIELDS = [
    "searchable_text^2.0",      # Boost principal
    "primary_description^1.5",   # Boost description
    "merchant_name^1.8",         # Boost marchand
    "category_name^1.2"          # Boost catégorie
]

# Champs pour filtrage exact
FILTERABLE_FIELDS = {
    "user_id": "long",
    "category_name": "keyword", 
    "merchant_name": "keyword",
    "transaction_type": "keyword",
    "currency_code": "keyword",
    "operation_type": "keyword",
    "month_year": "keyword",
    "weekday": "keyword"
}

# Champs pour agrégations
AGGREGATABLE_FIELDS = {
    "amount": "double",
    "amount_abs": "double", 
    "date": "date",
    "month_year": "keyword",
    "category_name": "keyword",
    "merchant_name": "keyword",
    "transaction_id": "keyword"
}

# Limites performance
MAX_QUERY_SIZE = 10000
DEFAULT_TIMEOUT_MS = 5000
CACHE_TTL_SECONDS = 300
MAX_AGGREGATION_BUCKETS = 1000

# Patterns validation
FIELD_NAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_\.]*$')
UNSAFE_CHARS_PATTERN = re.compile(r'[<>"\'\\\x00-\x1f]')

# =============================================================================
# 🔨 CONSTRUCTION REQUÊTES PRINCIPALES
# =============================================================================

def build_bool_query(
    must: List[Dict] = None,
    should: List[Dict] = None,
    must_not: List[Dict] = None,
    filter_clauses: List[Dict] = None,
    minimum_should_match: Optional[Union[int, str]] = None
) -> Dict[str, Any]:
    """
    Construit une requête bool Elasticsearch optimisée.
    
    Args:
        must: Clauses obligatoires (score calculé)
        should: Clauses optionnelles (boost score)
        must_not: Clauses d'exclusion
        filter_clauses: Filtres (pas de score, cache)
        minimum_should_match: Minimum de clauses should à matcher
        
    Returns:
        Dict représentant la requête bool
    """
    bool_query = {}
    
    if must:
        bool_query["must"] = must
    if should:
        bool_query["should"] = should
        if minimum_should_match:
            bool_query["minimum_should_match"] = minimum_should_match
    if must_not:
        bool_query["must_not"] = must_not
    if filter_clauses:
        bool_query["filter"] = filter_clauses
    
    # Optimisation : si seulement des filtres, pas de scoring
    if not must and not should and filter_clauses:
        return {"bool": bool_query}
    
    return {"bool": bool_query}

def build_text_search_query(
    query_text: str,
    fields: List[str] = None,
    operator: TextSearchOperator = TextSearchOperator.MATCH,
    fuzziness: Optional[str] = None,
    boost: float = 1.0
) -> Dict[str, Any]:
    """
    Construit une requête de recherche textuelle optimisée.
    
    Args:
        query_text: Texte à rechercher
        fields: Champs à rechercher (défaut: SEARCHABLE_FIELDS)
        operator: Type de recherche (match, phrase, fuzzy)
        fuzziness: Tolérance fuzzy ('AUTO', '1', '2')
        boost: Boost de score
        
    Returns:
        Dict représentant la requête textuelle
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty")
    
    # Nettoyage du texte
    clean_text = escape_elasticsearch_query(query_text.strip())
    search_fields = fields or SEARCHABLE_FIELDS
    
    if operator == TextSearchOperator.MATCH:
        return {
            "multi_match": {
                "query": clean_text,
                "fields": search_fields,
                "type": "best_fields",
                "operator": "and",
                "boost": boost,
                **({"fuzziness": fuzziness} if fuzziness else {})
            }
        }
    
    elif operator == TextSearchOperator.PHRASE:
        return {
            "multi_match": {
                "query": clean_text,
                "fields": search_fields,
                "type": "phrase",
                "boost": boost
            }
        }
    
    elif operator == TextSearchOperator.FUZZY:
        return {
            "multi_match": {
                "query": clean_text,
                "fields": search_fields,
                "type": "best_fields",
                "fuzziness": fuzziness or "AUTO",
                "boost": boost
            }
        }
    
    else:
        raise ValueError(f"Unsupported text search operator: {operator}")

def build_filter_query(search_filter: SearchFilter) -> Dict[str, Any]:
    """
    Construit une clause de filtre Elasticsearch selon l'opérateur.
    
    Args:
        search_filter: Filtre à appliquer
        
    Returns:
        Dict représentant le filtre Elasticsearch
    """
    field = search_filter.field
    operator = search_filter.operator
    value = search_filter.value
    
    # Validation sécurité
    if not validate_field_name(field):
        raise ValueError(f"Invalid field name: {field}")
    
    # Construction selon l'opérateur
    if operator == FilterOperator.EQ:
        return {"term": {f"{field}": value}}
    
    elif operator == FilterOperator.NE:
        return {"bool": {"must_not": {"term": {f"{field}": value}}}}
    
    elif operator == FilterOperator.GT:
        return {"range": {field: {"gt": value}}}
    
    elif operator == FilterOperator.GTE:
        return {"range": {field: {"gte": value}}}
    
    elif operator == FilterOperator.LT:
        return {"range": {field: {"lt": value}}}
    
    elif operator == FilterOperator.LTE:
        return {"range": {field: {"lte": value}}}
    
    elif operator == FilterOperator.IN:
        if not isinstance(value, list):
            raise ValueError("IN operator requires list value")
        return {"terms": {field: value}}
    
    elif operator == FilterOperator.NOT_IN:
        if not isinstance(value, list):
            raise ValueError("NOT_IN operator requires list value")
        return {"bool": {"must_not": {"terms": {field: value}}}}
    
    elif operator == FilterOperator.BETWEEN:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("BETWEEN operator requires list of 2 values")
        return {"range": {field: {"gte": value[0], "lte": value[1]}}}
    
    elif operator == FilterOperator.EXISTS:
        return {"exists": {"field": field}}
    
    elif operator == FilterOperator.MISSING:
        return {"bool": {"must_not": {"exists": {"field": field}}}}
    
    else:
        raise ValueError(f"Unsupported filter operator: {operator}")

def build_aggregation_query(
    agg_name: str,
    agg_type: AggregationType,
    field: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Construit une agrégation Elasticsearch.
    
    Args:
        agg_name: Nom de l'agrégation
        agg_type: Type d'agrégation
        field: Champ à agréger
        **kwargs: Paramètres additionnels
        
    Returns:
        Dict représentant l'agrégation
    """
    if not validate_field_name(field):
        raise ValueError(f"Invalid field name: {field}")
    
    if agg_type == AggregationType.SUM:
        return {agg_name: {"sum": {"field": field}}}
    
    elif agg_type == AggregationType.COUNT:
        return {agg_name: {"value_count": {"field": field}}}
    
    elif agg_type == AggregationType.AVG:
        return {agg_name: {"avg": {"field": field}}}
    
    elif agg_type == AggregationType.MIN:
        return {agg_name: {"min": {"field": field}}}
    
    elif agg_type == AggregationType.MAX:
        return {agg_name: {"max": {"field": field}}}
    
    elif agg_type == AggregationType.TERMS:
        size = kwargs.get("size", 100)
        return {
            agg_name: {
                "terms": {
                    "field": field,
                    "size": min(size, MAX_AGGREGATION_BUCKETS)
                }
            }
        }
    
    elif agg_type == AggregationType.DATE_HISTOGRAM:
        interval = kwargs.get("interval", "month")
        return {
            agg_name: {
                "date_histogram": {
                    "field": field,
                    "calendar_interval": interval,
                    "format": "yyyy-MM"
                }
            }
        }
    
    elif agg_type == AggregationType.STATS:
        return {agg_name: {"stats": {"field": field}}}
    
    else:
        raise ValueError(f"Unsupported aggregation type: {agg_type}")

# =============================================================================
# 🔍 FILTRES SPÉCIALISÉS FINANCIERS
# =============================================================================

def build_user_filter(user_id: int) -> Dict[str, Any]:
    """Construit le filtre obligatoire d'isolation utilisateur."""
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError(f"Invalid user_id: {user_id}")
    
    return {"term": {"user_id": user_id}}

def build_category_filter(category: str, exact: bool = True) -> Dict[str, Any]:
    """Construit un filtre par catégorie."""
    if not category or not category.strip():
        raise ValueError("Category cannot be empty")
    
    clean_category = category.strip().lower()
    
    if exact:
        return {"term": {"category_name.keyword": clean_category}}
    else:
        return {"match": {"category_name": clean_category}}

def build_merchant_filter(merchant: str, exact: bool = True) -> Dict[str, Any]:
    """Construit un filtre par marchand."""
    if not merchant or not merchant.strip():
        raise ValueError("Merchant cannot be empty")
    
    clean_merchant = merchant.strip()
    
    if exact:
        return {"term": {"merchant_name.keyword": clean_merchant}}
    else:
        return {"match": {"merchant_name": clean_merchant}}

def build_amount_filter(
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    exact_amount: Optional[float] = None,
    use_absolute: bool = True
) -> Dict[str, Any]:
    """Construit un filtre par montant."""
    field = "amount_abs" if use_absolute else "amount"
    
    if exact_amount is not None:
        return {"term": {field: exact_amount}}
    
    range_filter = {}
    if min_amount is not None:
        range_filter["gte"] = min_amount
    if max_amount is not None:
        range_filter["lte"] = max_amount
    
    if not range_filter:
        raise ValueError("At least one amount condition required")
    
    return {"range": {field: range_filter}}

def build_date_filter(
    start_date: Optional[Union[str, date, datetime]] = None,
    end_date: Optional[Union[str, date, datetime]] = None,
    exact_date: Optional[Union[str, date, datetime]] = None,
    month_year: Optional[str] = None
) -> Dict[str, Any]:
    """Construit un filtre par date."""
    
    if exact_date is not None:
        date_str = _format_date(exact_date)
        return {"term": {"date": date_str}}
    
    if month_year is not None:
        if not re.match(r'^\d{4}-\d{2}$', month_year):
            raise ValueError("month_year must be in format YYYY-MM")
        return {"term": {"month_year": month_year}}
    
    range_filter = {}
    if start_date is not None:
        range_filter["gte"] = _format_date(start_date)
    if end_date is not None:
        range_filter["lte"] = _format_date(end_date)
    
    if not range_filter:
        raise ValueError("At least one date condition required")
    
    return {"range": {"date": range_filter}}

def build_text_filter(
    search_text: str,
    fields: List[str] = None
) -> Dict[str, Any]:
    """Construit un filtre de recherche textuelle."""
    if not search_text or not search_text.strip():
        raise ValueError("Search text cannot be empty")
    
    clean_text = escape_elasticsearch_query(search_text.strip())
    search_fields = fields or ["searchable_text", "primary_description"]
    
    return {
        "multi_match": {
            "query": clean_text,
            "fields": search_fields,
            "type": "cross_fields",
            "operator": "and"
        }
    }

# =============================================================================
# 📊 AGRÉGATIONS FINANCIÈRES SPÉCIALISÉES
# =============================================================================

def build_sum_aggregation(field: str, name: str = None) -> Dict[str, Any]:
    """Construit une agrégation de somme."""
    agg_name = name or f"sum_{field}"
    return {agg_name: {"sum": {"field": field}}}

def build_count_aggregation(field: str, name: str = None) -> Dict[str, Any]:
    """Construit une agrégation de comptage."""
    agg_name = name or f"count_{field}"
    return {agg_name: {"value_count": {"field": field}}}

def build_avg_aggregation(field: str, name: str = None) -> Dict[str, Any]:
    """Construit une agrégation de moyenne."""
    agg_name = name or f"avg_{field}"
    return {agg_name: {"avg": {"field": field}}}

def build_date_histogram(
    field: str = "date",
    interval: str = "month",
    format_str: str = "yyyy-MM",
    name: str = None
) -> Dict[str, Any]:
    """Construit un histogramme temporel."""
    agg_name = name or f"histogram_{field}"
    return {
        agg_name: {
            "date_histogram": {
                "field": field,
                "calendar_interval": interval,
                "format": format_str,
                "min_doc_count": 1
            }
        }
    }

def build_terms_aggregation(
    field: str,
    size: int = 100,
    name: str = None,
    order_by: str = "_count",
    order_direction: str = "desc"
) -> Dict[str, Any]:
    """Construit une agrégation par termes (top N)."""
    agg_name = name or f"terms_{field}"
    return {
        agg_name: {
            "terms": {
                "field": field,
                "size": min(size, MAX_AGGREGATION_BUCKETS),
                "order": {order_by: order_direction}
            }
        }
    }

def build_stats_aggregation(field: str, name: str = None) -> Dict[str, Any]:
    """Construit une agrégation de statistiques complètes."""
    agg_name = name or f"stats_{field}"
    return {
        agg_name: {
            "stats": {
                "field": field
            }
        }
    }

# =============================================================================
# ⚡ OPTIMISATION PERFORMANCE
# =============================================================================

def optimize_query_performance(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimise une requête Elasticsearch pour les performances.
    
    Args:
        query: Requête Elasticsearch à optimiser
        
    Returns:
        Requête optimisée
    """
    optimized = query.copy()
    
    # Ajout du cache pour les filtres
    if "bool" in optimized and "filter" in optimized["bool"]:
        for filter_clause in optimized["bool"]["filter"]:
            if "term" in filter_clause or "terms" in filter_clause:
                filter_clause["_cache"] = True
    
    # Optimisation des agrégations
    if "aggs" in optimized:
        for agg_name, agg_def in optimized["aggs"].items():
            if "terms" in agg_def and "size" not in agg_def["terms"]:
                agg_def["terms"]["size"] = 100
    
    # Limitation automatique des résultats
    if "size" not in optimized:
        optimized["size"] = 20
    elif optimized["size"] > MAX_QUERY_SIZE:
        optimized["size"] = MAX_QUERY_SIZE
        logger.warning(f"Query size limited to {MAX_QUERY_SIZE}")
    
    return optimized

def add_query_cache(query: Dict[str, Any], cache_key: str = None) -> Dict[str, Any]:
    """Ajoute les paramètres de cache à une requête."""
    cached_query = query.copy()
    
    # Cache de préférence
    cached_query["preference"] = cache_key or "_local"
    
    # Cache pour les filtres
    if "bool" in cached_query and "filter" in cached_query["bool"]:
        for filter_clause in cached_query["bool"]["filter"]:
            filter_clause["_cache"] = True
    
    return cached_query

def calculate_query_complexity(query: Dict[str, Any]) -> str:
    """
    Calcule la complexité d'une requête.
    
    Returns:
        'simple', 'medium', 'complex'
    """
    complexity_score = 0
    
    # Score selon les clauses
    if "bool" in query:
        bool_clauses = query["bool"]
        complexity_score += len(bool_clauses.get("must", []))
        complexity_score += len(bool_clauses.get("should", []))
        complexity_score += len(bool_clauses.get("filter", []))
        complexity_score += len(bool_clauses.get("must_not", []))
    
    # Score selon les agrégations
    if "aggs" in query:
        complexity_score += len(query["aggs"]) * 2
    
    # Score selon la taille
    size = query.get("size", 10)
    if size > 100:
        complexity_score += 2
    elif size > 50:
        complexity_score += 1
    
    # Classification
    if complexity_score <= 3:
        return "simple"
    elif complexity_score <= 8:
        return "medium"
    else:
        return "complex"

# =============================================================================
# 🛡️ VALIDATION ET SÉCURITÉ
# =============================================================================

def escape_elasticsearch_query(query_text: str) -> str:
    """Échappe les caractères spéciaux pour Elasticsearch."""
    if not query_text:
        return ""
    
    # Caractères spéciaux Elasticsearch à échapper
    special_chars = ['\\', '+', '-', '=', '&&', '||', '>', '<', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '/']
    
    escaped = query_text
    for char in special_chars:
        escaped = escaped.replace(char, f'\\{char}')
    
    # Nettoyage caractères dangereux
    escaped = UNSAFE_CHARS_PATTERN.sub('', escaped)
    
    return escaped.strip()

def validate_field_name(field_name: str) -> bool:
    """Valide qu'un nom de champ est sécurisé."""
    if not field_name or not isinstance(field_name, str):
        return False
    
    # Validation pattern
    if not FIELD_NAME_PATTERN.match(field_name):
        return False
    
    # Vérification champ autorisé
    base_field = field_name.split('.')[0]  # Enlève .keyword
    return base_field in FINANCIAL_FIELDS

def format_elasticsearch_error(error: Exception) -> str:
    """Formate une erreur Elasticsearch de manière lisible."""
    error_str = str(error)
    
    # Extraction du message principal
    if "RequestError" in error_str:
        try:
            # Parse JSON error si possible
            import json
            if '"error":' in error_str:
                start = error_str.find('"error":')
                end = error_str.find('"}', start) + 2
                error_json = json.loads('{' + error_str[start:end] + '}')
                return error_json.get("error", {}).get("reason", error_str)
        except:
            pass
    
    return error_str

# =============================================================================
# 📊 PARSING ET FORMATAGE RÉPONSES
# =============================================================================

def parse_elasticsearch_response(response: Dict[str, Any]) -> Tuple[List[Dict], Dict, Dict]:
    """
    Parse une réponse Elasticsearch complète.
    
    Args:
        response: Réponse brute Elasticsearch
        
    Returns:
        Tuple (hits, aggregations, metadata)
    """
    # Extraction des hits
    hits_data = response.get("hits", {})
    hits = []
    
    for hit in hits_data.get("hits", []):
        source = hit.get("_source", {})
        source["_score"] = hit.get("_score")
        source["_id"] = hit.get("_id")
        hits.append(source)
    
    # Extraction des agrégations
    aggregations = extract_aggregation_results(response.get("aggregations", {}))
    
    # Métadonnées
    metadata = {
        "total_hits": hits_data.get("total", {}).get("value", 0),
        "max_score": hits_data.get("max_score"),
        "took": response.get("took", 0),
        "timed_out": response.get("timed_out", False),
        "shards": response.get("_shards", {})
    }
    
    return hits, aggregations, metadata

def extract_aggregation_results(aggs: Dict[str, Any]) -> Dict[str, Any]:
    """Extrait et formate les résultats d'agrégations."""
    results = {}
    
    for agg_name, agg_data in aggs.items():
        if "buckets" in agg_data:
            # Agrégation à buckets (terms, date_histogram)
            results[agg_name] = {
                "buckets": agg_data["buckets"],
                "doc_count_error_upper_bound": agg_data.get("doc_count_error_upper_bound", 0),
                "sum_other_doc_count": agg_data.get("sum_other_doc_count", 0)
            }
        elif "value" in agg_data:
            # Agrégation métrique simple (sum, avg, count)
            results[agg_name] = {
                "value": agg_data["value"]
            }
        elif all(k in agg_data for k in ["min", "max", "avg", "sum", "count"]):
            # Agrégation stats
            results[agg_name] = {
                "min": agg_data["min"],
                "max": agg_data["max"],
                "avg": agg_data["avg"],
                "sum": agg_data["sum"],
                "count": agg_data["count"]
            }
        else:
            # Agrégation complexe ou inconnue
            results[agg_name] = agg_data
    
    return results

# =============================================================================
# 🔧 HELPERS UTILITAIRES
# =============================================================================

def _format_date(date_input: Union[str, date, datetime]) -> str:
    """Formate une date pour Elasticsearch."""
    if isinstance(date_input, str):
        # Vérification format ISO
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_input):
            return date_input
        else:
            raise ValueError(f"Invalid date format: {date_input}")
    
    elif isinstance(date_input, (date, datetime)):
        return date_input.strftime("%Y-%m-%d")
    
    else:
        raise ValueError(f"Unsupported date type: {type(date_input)}")

def generate_query_hash(query: Dict[str, Any]) -> str:
    """Génère un hash unique pour une requête (cache key)."""
    query_str = json.dumps(query, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(query_str.encode('utf-8')).hexdigest()[:16]

def get_field_mapping(field_name: str) -> Optional[str]:
    """Récupère le type de mapping d'un champ."""
    base_field = field_name.split('.')[0]
    
    if base_field in FILTERABLE_FIELDS:
        return FILTERABLE_FIELDS[base_field]
    elif base_field in AGGREGATABLE_FIELDS:
        return AGGREGATABLE_FIELDS[base_field]
    elif base_field in FINANCIAL_FIELDS:
        return "text"  # Champ texte par défaut
    else:
        return None

def is_numeric_field(field_name: str) -> bool:
    """Vérifie si un champ est numérique."""
    mapping_type = get_field_mapping(field_name)
    return mapping_type in ["long", "integer", "double", "float"]

def is_date_field(field_name: str) -> bool:
    """Vérifie si un champ est une date."""
    mapping_type = get_field_mapping(field_name)
    return mapping_type == "date"

def is_text_field(field_name: str) -> bool:
    """Vérifie si un champ est textuel."""
    mapping_type = get_field_mapping(field_name)
    return mapping_type in ["text", "keyword"]

# =============================================================================
# 📊 CONSTANTES MÉTADONNÉES EXPORT
# =============================================================================

# Export pour autres modules
__all__ = [
    # Construction requêtes
    "build_bool_query", "build_text_search_query", "build_filter_query", "build_aggregation_query",
    # Filtres spécialisés
    "build_user_filter", "build_category_filter", "build_merchant_filter", 
    "build_amount_filter", "build_date_filter", "build_text_filter",
    # Agrégations financières
    "build_sum_aggregation", "build_count_aggregation", "build_avg_aggregation",
    "build_date_histogram", "build_terms_aggregation", "build_stats_aggregation",
    # Optimisation
    "optimize_query_performance", "add_query_cache", "calculate_query_complexity",
    # Validation
    "escape_elasticsearch_query", "validate_field_name", "format_elasticsearch_error",
    # Parsing
    "parse_elasticsearch_response", "extract_aggregation_results",
    # Helpers
    "generate_query_hash", "get_field_mapping", "is_numeric_field", "is_date_field", "is_text_field",
    # Constantes
    "FINANCIAL_FIELDS", "SEARCHABLE_FIELDS", "FILTERABLE_FIELDS", "AGGREGATABLE_FIELDS",
    "MAX_QUERY_SIZE", "DEFAULT_TIMEOUT_MS", "CACHE_TTL_SECONDS"
]