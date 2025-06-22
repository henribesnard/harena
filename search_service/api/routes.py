"""
Routes API pour le service de recherche - VERSION CORRIGÉE.

Ce module définit les endpoints pour la recherche hybride de transactions.
Fix: Correction de l'attribut 'rerank' vers 'use_reranking' et ajustement de l'interface SearchEngine.
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
embedding_service = None
reranker_service = None
search_cache = None
metrics_collector = None


def get_search_engine() -> SearchEngine:
    """Crée une instance du moteur de recherche avec vérification robuste."""
    # Vérification détaillée des clients
    elastic_available = elastic_client is not None
    qdrant_available = qdrant_client is not None
    
    # Log de diagnostic détaillé
    logger.info(f"🔍 Vérification des clients:")
    logger.info(f"   - elastic_client: {type(elastic_client).__name__ if elastic_client else 'None'}")
    logger.info(f"   - qdrant_client: {type(qdrant_client).__name__ if qdrant_client else 'None'}")
    logger.info(f"   - elastic_available: {elastic_available}")
    logger.info(f"   - qdrant_available: {qdrant_available}")
    
    # Vérification de l'état d'initialisation des clients
    elastic_initialized = False
    qdrant_initialized = False
    
    if elastic_client:
        elastic_initialized = hasattr(elastic_client, '_initialized') and elastic_client._initialized
        logger.info(f"   - elastic_initialized: {elastic_initialized}")
    
    if qdrant_client:
        qdrant_initialized = hasattr(qdrant_client, '_initialized') and qdrant_client._initialized
        logger.info(f"   - qdrant_initialized: {qdrant_initialized}")
    
    # Au moins un client doit être disponible ET initialisé
    if not ((elastic_available and elastic_initialized) or (qdrant_available and qdrant_initialized)):
        error_details = []
        
        if not elastic_available:
            error_details.append("Elasticsearch client not injected")
        elif not elastic_initialized:
            error_details.append("Elasticsearch client not initialized")
            
        if not qdrant_available:
            error_details.append("Qdrant client not injected")
        elif not qdrant_initialized:
            error_details.append("Qdrant client not initialized")
        
        error_message = f"Search service not available: {'; '.join(error_details)}"
        logger.error(f"❌ {error_message}")
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_message
        )
    
    # Log des clients disponibles
    available_services = []
    if elastic_available and elastic_initialized:
        available_services.append("Elasticsearch")
    if qdrant_available and qdrant_initialized:
        available_services.append("Qdrant")
    
    logger.info(f"✅ Services disponibles: {', '.join(available_services)}")
    
    # Créer le moteur de recherche
    try:
        search_engine = SearchEngine(
            elastic_client=elastic_client if (elastic_available and elastic_initialized) else None,
            qdrant_client=qdrant_client if (qdrant_available and qdrant_initialized) else None,
            cache=search_cache
        )
        logger.info("✅ SearchEngine créé avec succès")
        return search_engine
    except Exception as e:
        logger.error(f"❌ Erreur création SearchEngine: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to create search engine: {str(e)}"
        )


@router.get("/health")
async def search_health():
    """Check de santé détaillé du service de recherche."""
    health_status = {
        "service": "search_service",
        "timestamp": time.time(),
        "clients": {
            "elasticsearch": {
                "injected": elastic_client is not None,
                "initialized": elastic_client is not None and hasattr(elastic_client, '_initialized') and elastic_client._initialized,
                "type": type(elastic_client).__name__ if elastic_client else None
            },
            "qdrant": {
                "injected": qdrant_client is not None,
                "initialized": qdrant_client is not None and hasattr(qdrant_client, '_initialized') and qdrant_client._initialized,
                "type": type(qdrant_client).__name__ if qdrant_client else None
            },
            "embedding_service": {
                "injected": embedding_service is not None,
                "type": type(embedding_service).__name__ if embedding_service else None
            },
            "cache": {
                "injected": search_cache is not None,
                "type": type(search_cache).__name__ if search_cache else None
            }
        }
    }
    
    # Déterminer la santé globale
    elasticsearch_ok = health_status["clients"]["elasticsearch"]["initialized"]
    qdrant_ok = health_status["clients"]["qdrant"]["initialized"]
    
    health_status["healthy"] = elasticsearch_ok or qdrant_ok
    health_status["status"] = "healthy" if health_status["healthy"] else "unhealthy"
    health_status["available_services"] = []
    
    if elasticsearch_ok:
        health_status["available_services"].append("elasticsearch")
    if qdrant_ok:
        health_status["available_services"].append("qdrant")
    
    # Tests de connectivité si possible
    if elasticsearch_ok:
        try:
            is_healthy = await elastic_client.is_healthy()
            health_status["clients"]["elasticsearch"]["connectivity"] = is_healthy
        except Exception as e:
            health_status["clients"]["elasticsearch"]["connectivity_error"] = str(e)
    
    if qdrant_ok:
        try:
            is_healthy = await qdrant_client.is_healthy()
            health_status["clients"]["qdrant"]["connectivity"] = is_healthy
        except Exception as e:
            health_status["clients"]["qdrant"]["connectivity_error"] = str(e)
    
    return health_status


@router.get("/debug/injection")
async def debug_injection():
    """Debug endpoint pour vérifier l'injection des dépendances."""
    import sys
    
    # Informations sur les variables globales
    injection_debug = {
        "module_info": {
            "module_name": __name__,
            "module_file": __file__,
            "module_id": id(sys.modules[__name__])
        },
        "global_variables": {
            "elastic_client": {
                "value": elastic_client,
                "type": type(elastic_client).__name__ if elastic_client else None,
                "id": id(elastic_client),
                "is_none": elastic_client is None
            },
            "qdrant_client": {
                "value": qdrant_client,
                "type": type(qdrant_client).__name__ if qdrant_client else None,
                "id": id(qdrant_client),
                "is_none": qdrant_client is None
            },
            "embedding_service": {
                "value": embedding_service,
                "type": type(embedding_service).__name__ if embedding_service else None,
                "id": id(embedding_service),
                "is_none": embedding_service is None
            },
            "search_cache": {
                "value": search_cache,
                "type": type(search_cache).__name__ if search_cache else None,
                "id": id(search_cache),
                "is_none": search_cache is None
            }
        }
    }
    
    # Vérifier si les clients sont initialisés
    if elastic_client:
        injection_debug["global_variables"]["elastic_client"]["initialized"] = hasattr(elastic_client, '_initialized') and elastic_client._initialized
    
    if qdrant_client:
        injection_debug["global_variables"]["qdrant_client"]["initialized"] = hasattr(qdrant_client, '_initialized') and qdrant_client._initialized
    
    return injection_debug


@router.post("/search", response_model=SearchResponse)
async def search_transactions(
    query: SearchQuery,
    current_user: User = Depends(get_current_active_user)
):
    """
    Effectue une recherche hybride dans les transactions.
    
    Ce endpoint combine recherche lexicale (Elasticsearch) et sémantique (Qdrant)
    avec reranking optionnel pour optimiser la pertinence.
    """
    start_time = time.time()
    
    try:
        # Log de la requête
        logger.info(f"🔍 Nouvelle recherche pour user {current_user.id}")
        logger.info(f"   Query: '{query.query}'")
        logger.info(f"   Type: {query.search_type}")
        logger.info(f"   Limit: {query.limit}")
        logger.info(f"   Use reranking: {query.use_reranking}")  # FIX: utiliser use_reranking
        
        # Mise à jour de l'user_id dans la query si ce n'est pas déjà fait
        query.user_id = current_user.id
        
        # Créer le moteur de recherche (avec vérifications détaillées)
        search_engine = get_search_engine()
        
        # FIX: Passer l'objet SearchQuery complet au lieu de paramètres individuels
        # Le moteur de recherche attend un objet SearchQuery selon les modèles
        search_result = await search_engine.search(query)
        
        search_time = time.time() - start_time
        
        # FIX: search_engine.search() retourne déjà un SearchResponse
        # Nous devons extraire les bonnes propriétés ou adapter selon l'implémentation réelle
        if hasattr(search_result, 'results'):
            # Si search_result est déjà un SearchResponse
            results = search_result.results
            total_found = search_result.total_found
        else:
            # Si search_result est une liste de SearchResult
            results = search_result
            total_found = len(results)
        
        # Préparer la réponse
        response = SearchResponse(
            results=results,
            total_found=total_found,
            search_time=search_time,
            query_info={
                "original_query": query.query,
                "search_type": query.search_type,
                "user_id": current_user.id,
                "limit": query.limit,
                "use_reranking": query.use_reranking  # FIX: utiliser use_reranking
            }
        )
        
        logger.info(f"✅ Recherche terminée en {search_time:.3f}s - {len(results)} résultats")
        
        return response
        
    except HTTPException:
        # Re-lever les HTTPException (comme 503)
        raise
    except Exception as e:
        search_time = time.time() - start_time
        error_msg = f"Search failed: {str(e)}"
        logger.error(f"❌ Erreur recherche après {search_time:.3f}s: {error_msg}")
        logger.error(f"   Type: {type(e).__name__}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


@router.get("/suggest")
async def get_search_suggestions(
    q: str = Query(..., description="Requête pour les suggestions"),
    limit: int = Query(5, description="Nombre de suggestions"),
    current_user: User = Depends(get_current_active_user)
):
    """Retourne des suggestions de recherche basées sur la requête."""
    try:
        # Pour l'instant, retourner des suggestions statiques
        # TODO: Implémenter la logique de suggestions dynamiques
        
        suggestions = [
            f"{q} ce mois-ci",
            f"{q} cette semaine",
            f"Toutes les transactions {q}",
            f"{q} supérieur à 100€",
            f"{q} récent"
        ][:limit]
        
        return {
            "query": q,
            "suggestions": suggestions,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Suggestions failed: {str(e)}"
        )


@router.get("/stats")
async def get_search_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Retourne des statistiques de recherche pour l'utilisateur."""
    try:
        # TODO: Implémenter les vraies statistiques
        return {
            "user_id": current_user.id,
            "total_searches": 0,
            "avg_search_time": 0,
            "most_searched_terms": [],
            "search_service_status": {
                "elasticsearch_available": elastic_client is not None and hasattr(elastic_client, '_initialized') and elastic_client._initialized,
                "qdrant_available": qdrant_client is not None and hasattr(qdrant_client, '_initialized') and qdrant_client._initialized
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats failed: {str(e)}"
        )


@router.post("/feedback")
async def submit_search_feedback(
    feedback_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
):
    """Collecte les retours utilisateur sur les résultats de recherche."""
    try:
        # TODO: Implémenter la collecte de feedback
        logger.info(f"📝 Feedback reçu de user {current_user.id}: {feedback_data}")
        
        return {
            "status": "success",
            "message": "Feedback enregistré",
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback failed: {str(e)}"
        )