"""
Routes API pour le service de recherche.

Ce module définit les endpoints pour la recherche hybride de transactions.
"""
import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from db_service.session import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from search_service.models import SearchQuery, SearchResponse, SearchType
from search_service.core.search_engine import SearchEngine
from search_service.utils.cache import SearchCache
from search_service.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)
router = APIRouter()

# Instances globales (initialisées dans main.py)
elastic_client = None
qdrant_client = None
search_cache = None
metrics_collector = None


def get_search_engine() -> SearchEngine:
    """Crée une instance du moteur de recherche."""
    if not elastic_client and not qdrant_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not available"
        )
    
    return SearchEngine(
        elastic_client=elastic_client,
        qdrant_client=qdrant_client,
        cache=search_cache
    )


@router.post("/search", response_model=SearchResponse)
async def search_transactions(
    query: SearchQuery,
    current_user: User = Depends(get_current_active_user)
):
    """
    Effectue une recherche hybride dans les transactions.
    
    Ce endpoint combine recherche lexicale (Elasticsearch) et sémantique (Qdrant)
    avec reranking optionnel pour optimiser la pertinence.
    
    Args:
        query: Requête de recherche avec filtres et paramètres
        current_user: Utilisateur authentifié
        
    Returns:
        SearchResponse: Résultats de recherche avec scores et métadonnées
    """
    start_time = time.time()
    
    # Vérifier que l'utilisateur ne peut chercher que ses propres transactions
    if query.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot search other users' transactions"
        )
    
    try:
        # Obtenir le moteur de recherche
        search_engine = get_search_engine()
        
        # Vérifier si on peut utiliser le cache
        cache_key = None
        if search_cache and not query.include_explanations:
            cache_key = search_cache.generate_key(query)
            cached_response = await search_cache.get(cache_key)
            
            if cached_response:
                logger.info(f"Cache hit for query: {query.query[:50]}...")
                if metrics_collector:
                    metrics_collector.record_cache_hit()
                return cached_response
        
        # Exécuter la recherche
        logger.info(f"Executing {query.search_type} search for user {query.user_id}: {query.query}")
        
        response = await search_engine.search(query)
        
        # Mettre en cache si approprié
        if cache_key and response.results:
            await search_cache.set(cache_key, response, ttl=300)  # 5 minutes
        
        # Enregistrer les métriques
        if metrics_collector:
            metrics_collector.record_search(
                search_type=query.search_type,
                results_count=len(response.results),
                duration=time.time() - start_time
            )
        
        return response
        
    except ValueError as e:
        logger.error(f"Invalid search query: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/search/suggest")
async def get_search_suggestions(
    prefix: str = Query(..., min_length=2, description="Préfixe de recherche"),
    limit: int = Query(10, ge=1, le=20, description="Nombre de suggestions"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Obtient des suggestions de recherche basées sur un préfixe.
    
    Args:
        prefix: Début du terme de recherche
        limit: Nombre maximum de suggestions
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Liste des suggestions
    """
    try:
        # Pour l'instant, on retourne des suggestions statiques
        # TODO: Implémenter avec Elasticsearch suggest API
        
        static_suggestions = [
            "restaurant",
            "supermarché",
            "carburant",
            "pharmacie",
            "virement",
            "retrait",
            "abonnement",
            "transport",
            "courses",
            "shopping"
        ]
        
        # Filtrer par préfixe
        suggestions = [
            s for s in static_suggestions 
            if s.lower().startswith(prefix.lower())
        ][:limit]
        
        return {
            "prefix": prefix,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get suggestions: {str(e)}"
        )


@router.get("/search/stats")
async def get_search_stats(
    current_user: User = Depends(get_current_active_user)
):
    """
    Obtient les statistiques de recherche pour l'utilisateur.
    
    Args:
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Statistiques de recherche
    """
    try:
        # Obtenir les statistiques depuis le collecteur de métriques
        if not metrics_collector:
            return {
                "message": "Metrics collector not available",
                "user_id": current_user.id
            }
        
        stats = metrics_collector.get_user_stats(current_user.id)
        
        return {
            "user_id": current_user.id,
            "total_searches": stats.get("total_searches", 0),
            "search_types": stats.get("search_types", {}),
            "avg_results_count": stats.get("avg_results_count", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0),
            "recent_queries": stats.get("recent_queries", [])
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.post("/search/feedback")
async def submit_search_feedback(
    transaction_id: int,
    query: str,
    relevant: bool,
    feedback: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Soumet un feedback sur la pertinence d'un résultat de recherche.
    
    Args:
        transaction_id: ID de la transaction
        query: Requête de recherche originale
        relevant: Si le résultat était pertinent
        feedback: Commentaire optionnel
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Confirmation du feedback
    """
    try:
        # TODO: Stocker le feedback pour améliorer le modèle
        logger.info(
            f"Feedback from user {current_user.id}: "
            f"transaction {transaction_id} is {'relevant' if relevant else 'not relevant'} "
            f"for query '{query}'"
        )
        
        if metrics_collector:
            metrics_collector.record_feedback(
                user_id=current_user.id,
                query=query,
                transaction_id=transaction_id,
                relevant=relevant
            )
        
        return {
            "status": "success",
            "message": "Feedback recorded",
            "transaction_id": transaction_id,
            "relevant": relevant
        }
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}"
        )


@router.delete("/search/cache")
async def clear_search_cache(
    current_user: User = Depends(get_current_active_user)
):
    """
    Vide le cache de recherche pour l'utilisateur.
    
    Args:
        current_user: Utilisateur authentifié
        
    Returns:
        Dict: Confirmation de la suppression
    """
    try:
        if not search_cache:
            return {
                "status": "warning",
                "message": "Cache not available"
            }
        
        # Vider le cache pour l'utilisateur
        cleared = await search_cache.clear_user_cache(current_user.id)
        
        return {
            "status": "success",
            "message": f"Cache cleared for user {current_user.id}",
            "entries_cleared": cleared
        }
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )