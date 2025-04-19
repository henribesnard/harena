"""
Module de reranking des résultats de recherche.

Ce module implémente la réévaluation précise des paires requête-résultat
en utilisant l'API Cohere Rerank pour améliorer la précision du classement.
"""
import logging
from typing import List, Dict, Any

import cohere
from config_service.config import settings
from search_service.storage.memory_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

# Tentative d'import du SDK Cohere, avec fallback sur un reranking basique
try:
    cohere_client = cohere.ClientV2(settings.COHERE_KEY)
    COHERE_AVAILABLE = True
    logger.info("Client Cohere initialisé avec succès")
except (ImportError, Exception) as e:
    COHERE_AVAILABLE = False
    logger.warning(f"SDK Cohere non disponible ou erreur d'initialisation: {e}. Le reranking avancé est désactivé.")

async def rerank_results(
    query_text: str,
    results: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Réévalue la pertinence des résultats pour la requête.
    
    Args:
        query_text: Texte de la requête
        results: Liste des résultats à reclasser
        top_k: Nombre de résultats à conserver
        
    Returns:
        Liste des résultats reclassés
    """
    # Si aucun résultat, retourner une liste vide
    if not results:
        return []
    
    # Limite le nombre de résultats à reclasser
    results_to_rerank = results[:min(100, len(results))]  # Pour éviter de reclasser trop de résultats
    
    logger.debug(f"Reranking de {len(results_to_rerank)} résultats pour la requête: {query_text}")
    
    # Génération d'une clé de cache pour cette requête et ces résultats
    cache_key = f"rerank:{query_text}:{','.join(sorted([r['id'] for r in results_to_rerank]))}"
    cached_results = await get_cache(cache_key)
    
    if cached_results:
        logger.debug(f"Résultats de reranking récupérés du cache")
        return cached_results[:top_k]
    
    # Reclassement en fonction de la disponibilité de Cohere
    if COHERE_AVAILABLE:
        reranked_results = await rerank_with_cohere(query_text, results_to_rerank, top_k)
    else:
        reranked_results = await rerank_basic(query_text, results_to_rerank)
    
    # Limiter aux top_k résultats
    top_results = reranked_results[:top_k]
    
    # Mettre en cache les résultats
    await set_cache(cache_key, top_results, ttl=3600)  # Cache pour 1 heure
    
    logger.info(f"Reranking terminé: {len(top_results)} résultats retenus")
    return top_results

async def rerank_with_cohere(
    query_text: str,
    results: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Reclasse les résultats en utilisant l'API Cohere Rerank via le SDK officiel.
    
    Args:
        query_text: Texte de la requête
        results: Liste des résultats à reclasser
        top_k: Nombre maximum de résultats à retourner
        
    Returns:
        Liste des résultats reclassés
    """
    # Préparer les documents à reclasser
    documents = []
    for result in results:
        # Extraire les champs pertinents pour l'évaluation
        content = result["content"]
        merchant_name = content.get("merchant_name", "")
        description = content.get("description", "")
        clean_description = content.get("clean_description", "")
        amount = str(content.get("amount", 0))
        date = content.get("transaction_date", "")
        
        # Construire un texte représentatif de la transaction
        transaction_text = f"{merchant_name} {clean_description or description} {date} {amount}"
        documents.append(transaction_text)
    
    try:
        # Utiliser le SDK Cohere pour le reranking
        rerank_response = cohere_client.rerank(
            model="rerank-v3.5",  # Version la plus récente du modèle
            query=query_text,
            documents=documents,
            top_n=top_k
        )
        
        # Construire la liste des résultats reclassés
        reranked_results = []
        
        # Utiliser les résultats dans l'ordre retourné par Cohere
        for cohere_result in rerank_response.results:
            idx = cohere_result.index
            relevance_score = cohere_result.relevance_score
            
            if idx < len(results):
                result = results[idx].copy()
                result["score"] = relevance_score
                if "match_details" not in result:
                    result["match_details"] = {}
                result["match_details"]["reranking_score"] = relevance_score
                reranked_results.append(result)
        
        # S'assurer que tous les résultats sont inclus (même ceux non retournés par Cohere)
        if len(reranked_results) < len(results):
            # Filtrer les IDs déjà inclus
            included_ids = {r["id"] for r in reranked_results}
            # Ajouter les résultats manquants à la fin
            for result in results:
                if result["id"] not in included_ids:
                    reranked_results.append(result)
        
        logger.debug(f"Reranking via Cohere SDK réussi: {len(reranked_results)} résultats traités")
        return reranked_results
        
    except Exception as e:
        logger.error(f"Erreur lors du reranking avec Cohere SDK: {str(e)}", exc_info=True)
        # Fallback sur le reranking basique en cas d'erreur
        return await rerank_basic(query_text, results)

async def rerank_basic(
    query_text: str,
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Reclasse les résultats avec une méthode basique en cas d'indisponibilité de Cohere.
    
    Args:
        query_text: Texte de la requête
        results: Liste des résultats à reclasser
        
    Returns:
        Liste des résultats reclassés
    """
    # Préparation des termes de la requête pour la correspondance
    query_terms = set(query_text.lower().split())
    
    # Fonction de calcul de score basique
    def calculate_basic_score(result):
        content = result["content"]
        merchant_name = str(content.get("merchant_name", "")).lower()
        description = str(content.get("description", "")).lower()
        
        # Calculer le chevauchement des termes
        merchant_terms = set(merchant_name.split())
        description_terms = set(description.split())
        all_terms = merchant_terms.union(description_terms)
        
        # Nombre de termes communs avec la requête
        common_terms = query_terms.intersection(all_terms)
        overlap_score = len(common_terms) / max(1, len(query_terms))
        
        # Combiner avec le score original
        original_score = result.get("score", 0)
        combined_score = 0.7 * original_score + 0.3 * overlap_score
        
        return combined_score
    
    # Calculer les nouveaux scores
    for result in results:
        basic_score = calculate_basic_score(result)
        result["score"] = basic_score
        if "match_details" not in result:
            result["match_details"] = {}
        result["match_details"]["reranking_score"] = basic_score
    
    # Trier par score décroissant
    reranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    return reranked_results