"""
Fonctions utilitaires Elasticsearch de haut niveau.

Ce module fournit des fonctions utilitaires simples pour
les opérations Elasticsearch courantes sans instancier des classes.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)

def format_search_results(
    es_results: Dict[str, Any],
    user_preferences: Optional[Dict[str, Any]] = None,
    **formatter_kwargs
) -> Dict[str, Any]:
    """
    Formate des résultats Elasticsearch avec optimisations financières.
    
    Args:
        es_results: Résultats bruts d'Elasticsearch
        user_preferences: Préférences utilisateur
        **formatter_kwargs: Arguments pour le ResultFormatter
        
    Returns:
        Résultats formatés
    """
    from .formatters import ResultFormatter
    formatter = ResultFormatter(**formatter_kwargs)
    return formatter.format_financial_results(es_results, user_preferences)

def extract_highlights(
    es_results: Dict[str, Any],
    original_query: str = "",
    **processor_kwargs
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extrait et optimise les highlights de résultats Elasticsearch.
    
    Args:
        es_results: Résultats d'Elasticsearch
        original_query: Requête originale pour optimisation
        **processor_kwargs: Arguments pour le HighlightProcessor
        
    Returns:
        Highlights optimisés par hit
    """
    from .highlights import HighlightProcessor
    processor = HighlightProcessor(**processor_kwargs)
    highlights_by_hit = {}
    
    hits = es_results.get('hits', {}).get('hits', [])
    for hit in hits:
        hit_id = hit.get('_id', '')
        if 'highlight' in hit:
            processed = processor.process_highlights(hit['highlight'], original_query)
            if processed:
                highlights_by_hit[hit_id] = processed
    
    return highlights_by_hit

def calculate_relevance_score(
    elasticsearch_score: float,
    transaction_date: datetime,
    merchant_name: Optional[str] = None,
    amount: Optional[Decimal] = None,
    user_preferences: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calcule un score de pertinence amélioré pour une transaction.
    
    Args:
        elasticsearch_score: Score Elasticsearch de base
        transaction_date: Date de la transaction
        merchant_name: Nom du marchand
        amount: Montant de la transaction
        user_preferences: Préférences utilisateur
        
    Returns:
        Score de pertinence amélioré
    """
    from .scoring import ScoreCalculator
    calculator = ScoreCalculator()
    score_components = calculator.calculate_enhanced_score(
        elasticsearch_score=elasticsearch_score,
        transaction_date=transaction_date,
        merchant_name=merchant_name,
        amount=amount,
        user_preferences=user_preferences
    )
    return score_components.final_score

def optimize_query_for_performance(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimise une requête Elasticsearch pour les performances.
    
    Args:
        query: Requête à optimiser
        
    Returns:
        Requête optimisée
    """
    optimized = query.copy()
    
    # Ajouter un timeout si pas présent
    if "timeout" not in optimized:
        optimized["timeout"] = "30s"
    
    # Limiter la taille des résultats
    if "size" in optimized and optimized["size"] > 100:
        optimized["size"] = 100
        logger.warning("Limited query size to 100 for performance")
    
    # Optimiser les wildcards coûteux
    def optimize_wildcards(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "wildcard" and isinstance(value, dict):
                    for field, wildcard_spec in value.items():
                        if isinstance(wildcard_spec, dict) and "value" in wildcard_spec:
                            wildcard_value = wildcard_spec["value"]
                            # Limiter les wildcards trop génériques
                            if wildcard_value.count('*') > 2:
                                # Convertir en recherche fuzzy moins coûteuse
                                clean_value = wildcard_value.replace('*', '')
                                if len(clean_value) > 2:
                                    node[key] = {
                                        field: {
                                            "value": clean_value,
                                            "fuzziness": "AUTO",
                                            "boost": wildcard_spec.get("boost", 1.0) * 0.8
                                        }
                                    }
                else:
                    optimize_wildcards(value)
        elif isinstance(node, list):
            for item in node:
                optimize_wildcards(item)
    
    optimize_wildcards(optimized)
    
    return optimized

def build_suggestion_query(
    partial_query: str,
    user_id: int,
    suggestion_types: List[str] = None
) -> Dict[str, Any]:
    """
    Construit une requête optimisée pour les suggestions d'auto-complétion.
    
    Args:
        partial_query: Début de requête
        user_id: ID utilisateur
        suggestion_types: Types de suggestions ("merchant", "description")
        
    Returns:
        Requête de suggestion Elasticsearch
    """
    if suggestion_types is None:
        suggestion_types = ["merchant", "description"]
    
    should_clauses = []
    aggs = {}
    
    # Suggestions de marchands
    if "merchant" in suggestion_types:
        should_clauses.append({
            "prefix": {
                "merchant_name.keyword": {
                    "value": partial_query,
                    "boost": 3.0
                }
            }
        })
        
        aggs["merchants"] = {
            "terms": {
                "field": "merchant_name.keyword",
                "size": 10,
                "include": f".*{re.escape(partial_query)}.*"
            }
        }
    
    # Suggestions de descriptions
    if "description" in suggestion_types:
        should_clauses.append({
            "match_phrase_prefix": {
                "primary_description": {
                    "query": partial_query,
                    "boost": 2.0
                }
            }
        })
        
        aggs["descriptions"] = {
            "terms": {
                "field": "primary_description.keyword",
                "size": 10,
                "include": f".*{re.escape(partial_query)}.*"
            }
        }
    
    return {
        "query": {
            "bool": {
                "must": [{"term": {"user_id": user_id}}],
                "should": should_clauses,
                "minimum_should_match": 0 if should_clauses else 1
            }
        },
        "size": 0,
        "aggs": aggs
    }

def validate_query_structure(query: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valide la structure d'une requête Elasticsearch.
    
    Args:
        query: Requête à valider
        
    Returns:
        Tuple (is_valid, errors)
    """
    errors = []
    
    # Vérifications de base
    if not isinstance(query, dict):
        errors.append("Query must be a dictionary")
        return False, errors
    
    if "query" not in query:
        errors.append("Missing 'query' field")
    
    if "size" in query:
        size = query["size"]
        if not isinstance(size, int) or size < 0:
            errors.append("'size' must be a non-negative integer")
        elif size > 1000:
            errors.append("'size' is too large (max 1000)")
    
    if "from" in query:
        from_val = query["from"]
        if not isinstance(from_val, int) or from_val < 0:
            errors.append("'from' must be a non-negative integer")
        elif from_val > 10000:
            errors.append("'from' is too large (max 10000)")
    
    # Vérification de la complexité
    complexity_score = _calculate_query_complexity(query)
    if complexity_score > 100:
        errors.append(f"Query too complex (score: {complexity_score})")
    
    return len(errors) == 0, errors

def _calculate_query_complexity(query: Dict[str, Any]) -> int:
    """Calcule un score de complexité pour une requête."""
    complexity = 0
    
    def analyze_node(node, depth=0):
        nonlocal complexity
        
        if depth > 10:
            complexity += 50  # Pénalité pour profondeur excessive
            return
        
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ["bool", "function_score"]:
                    complexity += 5
                elif key in ["wildcard", "regexp"]:
                    complexity += 15
                elif key in ["should", "must", "must_not"]:
                    if isinstance(value, list):
                        complexity += len(value) * 2
                
                analyze_node(value, depth + 1)
        elif isinstance(node, list):
            complexity += len(node)
            for item in node:
                analyze_node(item, depth + 1)
    
    analyze_node(query.get("query", {}))
    return complexity