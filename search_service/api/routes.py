import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from ..models.request import SearchRequest
from ..models.response import SearchResponse
from ..core.search_engine import SearchEngine
from ..config.settings import settings

logger = logging.getLogger(__name__)

# Router principal
router = APIRouter(prefix="/api/v1", tags=["search"])

# Instance globale du moteur de recherche
search_engine = SearchEngine()

async def get_search_engine() -> SearchEngine:
    """Dependency pour obtenir le moteur de recherche initialisé"""
    if not search_engine.elasticsearch_client:
        # Essayer d'obtenir le client depuis l'app state si disponible
        try:
            from fastapi import Request
            # Cette dépendance sera résolue au runtime quand l'app sera disponible
        except:
            pass
        
        # Si toujours pas de client, lever une erreur
        if not search_engine.elasticsearch_client:
            raise HTTPException(
                status_code=503, 
                detail="Service non disponible - Client Elasticsearch non initialisé"
            )
    
    return search_engine

@router.post("/search", response_model=SearchResponse)
async def search_transactions(
    request: SearchRequest,
    engine: SearchEngine = Depends(get_search_engine)
) -> SearchResponse:
    """
    Endpoint unique pour toutes les recherches de transactions
    
    Gère automatiquement:
    - Recherches textuelles avec scoring BM25
    - Filtres simples (term, range, terms) 
    - Combinaisons texte + filtres
    - Pagination
    - Tri intelligent par pertinence et date
    
    Args:
        request: Requête de recherche unifiée
        
    Returns:
        SearchResponse: Résultats avec métadonnées
        
    Raises:
        HTTPException: En cas d'erreur de validation ou de recherche
    """
    try:
        # Validation sécurité
        if request.user_id <= 0:
            raise HTTPException(
                status_code=400, 
                detail="user_id est obligatoire et doit être positif"
            )
        
        # Log de la requête pour monitoring
        logger.info(
            f"Search request from user {request.user_id}: "
            f"query='{request.query}', filters={len(request.filters)}, limit={request.limit}"
        )
        
        # Recherche via moteur unifié
        results = await engine.search(request)
        
        # Log des résultats
        logger.info(
            f"Search completed for user {request.user_id}: "
            f"{results.returned_hits}/{results.total_hits} results in {results.execution_time_ms}ms"
        )
        
        return results
        
    except HTTPException:
        # Re-raise les erreurs HTTP sans les modifier
        raise
        
    except Exception as e:
        logger.error(f"Search failed for user {request.user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la recherche: {str(e)}"
        )

@router.post("/count")
async def count_transactions(
    request: SearchRequest,
    engine: SearchEngine = Depends(get_search_engine)
) -> Dict[str, Any]:
    """
    Compte le nombre de transactions correspondant aux critères
    
    Args:
        request: Critères de recherche (sans pagination)
        
    Returns:
        Dict avec le nombre de résultats
    """
    try:
        if request.user_id <= 0:
            raise HTTPException(
                status_code=400, 
                detail="user_id est obligatoire et doit être positif"
            )
        
        count = await engine.count(request)
        
        return {
            "count": count,
            "user_id": request.user_id,
            "query": request.query,
            "filters": request.filters
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Count failed for user {request.user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors du comptage: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Vérification de santé du service de recherche
    
    Returns:
        Dict avec le statut du service et ses composants
    """
    health_status = {
        "status": "healthy",
        "service": "search_service",
        "version": settings.api_version,
        "components": {}
    }
    
    # Vérifier le client Elasticsearch
    try:
        if search_engine.elasticsearch_client:
            # Test simple de connectivité
            # On peut adapter selon les méthodes disponibles dans votre client
            health_status["components"]["elasticsearch"] = {
                "status": "connected",
                "index": search_engine.index_name
            }
        else:
            health_status["components"]["elasticsearch"] = {
                "status": "not_initialized"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["elasticsearch"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    return health_status

@router.get("/debug/config")
async def debug_config() -> Dict[str, Any]:
    """
    Informations de configuration pour debugging (en mode debug uniquement)
    
    Returns:
        Dict avec les informations de configuration
    """
    if not settings.debug_mode:
        raise HTTPException(
            status_code=404,
            detail="Endpoint disponible uniquement en mode debug"
        )
    
    return {
        "settings": {
            "bonsai_url_configured": bool(settings.BONSAI_URL),
            "elasticsearch_index": settings.ELASTICSEARCH_INDEX,
            "test_user_id": settings.test_user_id,
            "default_limit": settings.default_limit,
            "max_limit": settings.max_limit
        },
        "search_engine": {
            "client_initialized": search_engine.elasticsearch_client is not None,
            "index_name": search_engine.index_name
        }
    }

# Fonction d'initialisation pour le moteur de recherche
def initialize_search_engine(elasticsearch_client):
    """
    Initialise le moteur de recherche avec le client Elasticsearch
    Appelée depuis main.py au démarrage de l'application
    """
    global search_engine
    search_engine.set_elasticsearch_client(elasticsearch_client)
    logger.info("✅ Search engine initialized with Elasticsearch client")