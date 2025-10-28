import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from search_service.models.request import SearchRequest
from search_service.core.search_engine import SearchEngine, RateLimitExceeded
from search_service.api.deps import get_current_active_user, validate_user_access
from db_service.models.user import User
from config_service.config import settings

logger = logging.getLogger(__name__)

# Router principal sans préfixe (le préfixe sera ajouté par local_app.py)
router = APIRouter(tags=["search"])

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

@router.post("/search")
async def search_transactions(
    request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
    engine: SearchEngine = Depends(get_search_engine)
) -> Dict[str, Any]:
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
        Dict: Réponse structurée avec résultats et métadonnées
        
    Raises:
        HTTPException: En cas d'erreur de validation ou de recherche
    """
    try:
        # Validation sécurité - user_id obligatoire
        if request.user_id <= 0:
            raise HTTPException(
                status_code=400, 
                detail="user_id est obligatoire et doit être positif"
            )
        
        # Contrôle d'accès : vérifier que l'utilisateur peut accéder aux données demandées
        validate_user_access(current_user, request.user_id)
        
        # Log de la requête pour monitoring (inclus l'utilisateur authentifié)
        logger.info(
            f"Search request from authenticated user {current_user.id} (admin: {current_user.is_superuser}) for user_id {request.user_id}: "
            f"query='{request.query}', filters={len(request.filters) if request.filters else 0}, page={request.page}, page_size={request.page_size}"
        )
        
        # Recherche via moteur unifié
        results = await engine.search(request)

        # Vérification du succès de la recherche
        if not results["success"]:
            raise HTTPException(status_code=502, detail=results["error_message"])

        # Log des résultats
        metadata = results.get("response_metadata", {})
        logger.info(
            f"Search completed for user {request.user_id}: "
            f"{metadata.get('returned_results', 0)}/{metadata.get('total_results', 0)} "
            f"results in {metadata.get('processing_time_ms', 0)}ms"
        )

        return results
        
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except HTTPException:
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
    current_user: User = Depends(get_current_active_user),
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
        
        # Contrôle d'accès : vérifier que l'utilisateur peut accéder aux données demandées
        validate_user_access(current_user, request.user_id)
        
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
        "version": "1.0.0",
        "components": {}
    }

    # Vérifier le client Elasticsearch
    try:
        if hasattr(search_engine, 'elasticsearch_client') and search_engine.elasticsearch_client:
            # Test simple de connectivité
            # On peut adapter selon les méthodes disponibles dans votre client
            health_status["components"]["elasticsearch"] = {
                "status": "connected",
                "index": getattr(search_engine, 'index_name', 'unknown')
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
        health_status["status"] = "degraded"

    return health_status

@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Expose cache and rate limiting metrics."""
    return search_engine.get_stats()

@router.delete("/cache/user/{user_id}")
async def clear_user_cache(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    engine: SearchEngine = Depends(get_search_engine)
) -> Dict[str, Any]:
    """
    Invalide le cache de recherche pour un utilisateur spécifique.

    Utilisé après la synchronisation des données pour s'assurer que
    les recherches ultérieures reflètent les nouvelles données indexées.

    Args:
        user_id: ID de l'utilisateur dont le cache doit être invalidé

    Returns:
        Dict avec le nombre d'entrées supprimées

    Raises:
        HTTPException 403: Si l'utilisateur n'a pas les droits d'accès
    """
    # Contrôle d'accès : seul un admin ou l'utilisateur lui-même peut invalider son cache
    validate_user_access(current_user, user_id)

    # Invalider le cache
    entries_deleted = await engine.cache.clear_user(user_id)

    logger.info(f"Cache cleared for user {user_id}: {entries_deleted} entries removed by user {current_user.id}")

    return {
        "success": True,
        "user_id": user_id,
        "entries_deleted": entries_deleted,
        "message": f"Cache invalidé pour l'utilisateur {user_id}"
    }

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
    
    debug_info = {
        "settings": {
            "bonsai_url_configured": bool(settings.BONSAI_URL),
            "elasticsearch_index": settings.ELASTICSEARCH_INDEX,
            "test_user_id": settings.test_user_id,
            "default_limit": settings.default_limit,
            "max_limit": settings.max_limit
        },
        "search_engine": {
            "client_initialized": search_engine.elasticsearch_client is not None,
            "index_name": search_engine.index_name,
            "rate_limit_per_minute": search_engine.requests_per_minute,
            "cache_stats": search_engine.get_stats()["cache"],
        }
    }
    return debug_info

@router.get("/accounts/{user_id}")
async def get_user_accounts(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    engine: SearchEngine = Depends(get_search_engine)
) -> Dict[str, Any]:
    """
    🏦 Récupère tous les comptes d'un utilisateur avec leurs soldes
    
    Args:
        user_id: ID de l'utilisateur
        
    Returns:
        Dict avec la liste des comptes et leurs métadonnées
    """
    try:
        if user_id <= 0:
            raise HTTPException(
                status_code=400, 
                detail="user_id doit être positif"
            )
        
        # Contrôle d'accès : vérifier que l'utilisateur peut accéder aux données demandées  
        validate_user_access(current_user, user_id)
        
        # Récupérer les comptes depuis l'index accounts
        accounts = await engine.elasticsearch_client.get_user_accounts(user_id)
        
        # Calculer des statistiques
        total_balance = sum(acc.get("account_balance", 0) or 0 for acc in accounts)
        active_accounts = len([acc for acc in accounts if acc.get("is_active", True)])
        
        return {
            "success": True,
            "user_id": user_id,
            "accounts": accounts,
            "summary": {
                "total_accounts": len(accounts),
                "active_accounts": active_accounts,
                "total_balance": total_balance,
                "currencies": list(set(acc.get("account_currency", "EUR") for acc in accounts if acc.get("account_currency")))
            },
            "response_metadata": {
                "source": "accounts_index",
                "returned_results": len(accounts)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get accounts failed for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la récupération des comptes: {str(e)}"
        )

@router.get("/accounts/{user_id}/{account_id}/balance")
async def get_account_balance(
    user_id: int,
    account_id: int,
    current_user: User = Depends(get_current_active_user),
    engine: SearchEngine = Depends(get_search_engine)
) -> Dict[str, Any]:
    """
    💰 Récupère le solde d'un compte spécifique (accès direct sans agrégation)
    
    Args:
        user_id: ID de l'utilisateur
        account_id: ID du compte
        
    Returns:
        Dict avec le solde et les métadonnées du compte
    """
    try:
        if user_id <= 0 or account_id <= 0:
            raise HTTPException(
                status_code=400, 
                detail="user_id et account_id doivent être positifs"
            )
        
        # Contrôle d'accès
        validate_user_access(current_user, user_id)
        
        # Récupérer le solde directement
        balance = await engine.elasticsearch_client.get_account_balance(user_id, account_id)
        
        if balance is None:
            raise HTTPException(
                status_code=404,
                detail=f"Compte {account_id} non trouvé pour l'utilisateur {user_id}"
            )
        
        return {
            "success": True,
            "user_id": user_id,
            "account_id": account_id,
            "account_balance": balance,
            "response_metadata": {
                "source": "accounts_index",
                "direct_access": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get balance failed for user {user_id}, account {account_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la récupération du solde: {str(e)}"
        )

# Fonction d'initialisation pour le moteur de recherche
def initialize_search_engine(elasticsearch_client):
    """
    Initialise le moteur de recherche avec le client Elasticsearch
    Appelée depuis main.py au démarrage de l'application
    """
    global search_engine
    search_engine.set_elasticsearch_client(elasticsearch_client)
    logger.info("✅ Search engine initialized with Elasticsearch client")