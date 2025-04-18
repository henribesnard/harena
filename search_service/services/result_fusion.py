"""
Module de fusion des résultats de recherche.

Ce module combine intelligemment les résultats lexicaux et vectoriels
en utilisant l'algorithme Reciprocal Rank Fusion (RRF).
"""
import logging
from typing import List, Dict, Any, Set

from search_service.schemas.response import SearchResult, ResultType, MatchDetails

logger = logging.getLogger(__name__)

async def fuse_results(
    lexical_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Fusionne les résultats lexicaux et vectoriels en utilisant RRF.
    
    Args:
        lexical_results: Résultats de la recherche lexicale
        vector_results: Résultats de la recherche vectorielle
        lexical_weight: Poids à appliquer aux résultats lexicaux
        semantic_weight: Poids à appliquer aux résultats vectoriels
        
    Returns:
        Liste fusionnée des résultats
    """
    # Normaliser les poids
    total_weight = lexical_weight + semantic_weight
    normalized_lexical_weight = lexical_weight / total_weight
    normalized_semantic_weight = semantic_weight / total_weight
    
    logger.debug(f"Fusion de {len(lexical_results)} résultats lexicaux et {len(vector_results)} résultats vectoriels")
    logger.debug(f"Poids normalisés: lexical={normalized_lexical_weight:.2f}, semantic={normalized_semantic_weight:.2f}")
    
    # Appliquer l'algorithme Reciprocal Rank Fusion (RRF)
    fused_results = rrf_fusion(
        lexical_results=lexical_results,
        vector_results=vector_results,
        k=60,  # Constante RRF standard
        lexical_weight=normalized_lexical_weight,
        semantic_weight=normalized_semantic_weight
    )
    
    return fused_results

def rrf_fusion(
    lexical_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Implémentation de l'algorithme Reciprocal Rank Fusion.
    
    Args:
        lexical_results: Résultats de la recherche lexicale
        vector_results: Résultats de la recherche vectorielle
        k: Constante pour pénaliser les rangs inférieurs (défaut: 60)
        lexical_weight: Poids à appliquer aux scores lexicaux
        semantic_weight: Poids à appliquer aux scores sémantiques
        
    Returns:
        Liste fusionnée de résultats triés par score RRF
    """
    # Map pour stocker les scores RRF et les résultats complets
    rrf_scores = {}
    result_map = {}  # Map des résultats par ID
    
    # Traiter les résultats lexicaux
    for rank, result in enumerate(lexical_results):
        result_id = result["id"]
        rrf_score = lexical_weight * (1.0 / (k + rank + 1))
        
        if result_id in rrf_scores:
            rrf_scores[result_id] += rrf_score
        else:
            rrf_scores[result_id] = rrf_score
            result_map[result_id] = result
    
    # Traiter les résultats vectoriels
    for rank, result in enumerate(vector_results):
        result_id = result["id"]
        rrf_score = semantic_weight * (1.0 / (k + rank + 1))
        
        if result_id in rrf_scores:
            rrf_scores[result_id] += rrf_score
            
            # Fusionner les scores de match s'ils existent
            if "match_details" in result and "match_details" in result_map[result_id]:
                result_map[result_id]["match_details"]["semantic_score"] = result["match_details"].get("semantic_score")
            
            # Conserver les highlights s'ils existent
            if "highlight" in result and "highlight" not in result_map[result_id]:
                result_map[result_id]["highlight"] = result["highlight"]
            
        else:
            rrf_scores[result_id] = rrf_score
            result_map[result_id] = result
    
    # Trier par score RRF et créer la liste fusionnée
    sorted_results = []
    for result_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        result = result_map[result_id]
        
        # Ajouter le score RRF au résultat
        result["score"] = score
        
        # S'assurer que les details de match sont complets
        if "match_details" not in result:
            result["match_details"] = {}
        
        result["match_details"]["rrf_score"] = score
        
        sorted_results.append(result)
    
    logger.info(f"Fusion RRF: {len(sorted_results)} résultats après fusion et déduplication")
    return sorted_results

def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Supprime les doublons des résultats.
    
    Args:
        results: Liste des résultats à dédupliquer
        
    Returns:
        Liste de résultats sans doublons
    """
    seen_ids: Set[str] = set()
    deduplicated_results = []
    
    for result in results:
        result_id = result["id"]
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            deduplicated_results.append(result)
    
    return deduplicated_results

def normalize_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalise les scores pour qu'ils soient sur une échelle de 0 à 1.
    
    Args:
        results: Liste des résultats à normaliser
        
    Returns:
        Liste de résultats avec scores normalisés
    """
    if not results:
        return []
    
    # Trouver le score maximum
    max_score = max(result["score"] for result in results)
    
    # Normaliser tous les scores
    if max_score > 0:
        for result in results:
            result["score"] = result["score"] / max_score
            
            # Normaliser aussi les détails de match si présents
            if "match_details" in result:
                if "lexical_score" in result["match_details"] and result["match_details"]["lexical_score"] is not None:
                    result["match_details"]["lexical_score"] /= max_score
                if "semantic_score" in result["match_details"] and result["match_details"]["semantic_score"] is not None:
                    result["match_details"]["semantic_score"] /= max_score
    
    return results