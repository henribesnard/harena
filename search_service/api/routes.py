# search_service/api/routes.py
"""
Routes API REST simplifiées pour le Search Service
==================================================

Version simplifiée pour démarrage - à enrichir progressivement.
Health check géré par main.py pour éviter les conflits.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# === ROUTEUR PRINCIPAL ===
router = APIRouter(tags=["search"])

# === VARIABLE GLOBALE POUR TRACKER L'INITIALISATION ===
_initialization_attempted = False
_initialization_successful = False
_initialization_error = None

async def _ensure_core_initialization():
    """
    Force l'initialisation du core manager si elle n'a pas été faite
    ✅ CORRECTION: Utilisation de la nouvelle API simplifiée
    """
    global _initialization_attempted, _initialization_successful, _initialization_error
    
    if _initialization_attempted:
        return _initialization_successful, _initialization_error
    
    _initialization_attempted = True
    
    try:
        logger.info("🔧 Tentative d'initialisation forcée du search_service...")
        
        # Vérifier les variables d'environnement
        bonsai_url = os.environ.get("BONSAI_URL")
        
        if not bonsai_url:
            raise ValueError("BONSAI_URL n'est pas configurée")
        
        # Import et initialisation
        from search_service.clients.elasticsearch_client import get_default_client, initialize_default_client
        from search_service.core import core_manager
        
        # ✅ CORRECTION MAJEURE: Utiliser initialize_default_client() 
        # qui gère automatiquement la détection de l'URL
        elasticsearch_client = await initialize_default_client()
        logger.info("✅ Client Elasticsearch initialisé via routes.py")
        
        # Initialiser le core manager
        await core_manager.initialize(elasticsearch_client)
        
        # Vérifier l'initialisation
        if core_manager.is_initialized():
            logger.info("✅ Core manager initialisé avec succès via routes.py")
            _initialization_successful = True
            _initialization_error = None
        else:
            raise RuntimeError("Core manager non initialisé après tentative")
            
    except Exception as e:
        error_msg = f"Échec initialisation search_service: {str(e)}"
        logger.error(f"❌ {error_msg}")
        _initialization_successful = False
        _initialization_error = error_msg
    
    return _initialization_successful, _initialization_error

@router.get("/health", summary="Vérification de l'état du service")
async def health_check(request: Request):
    """
    Endpoint de health check pour vérifier l'état du service de recherche.
    
    Vérifie:
    - L'état de l'initialisation du service
    - La connectivité à Elasticsearch/Bonsai
    - L'état du core manager
    
    Returns:
        JSONResponse: Statut de santé détaillé
    """
    health_status = {
        "service": "search_service",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "unknown",
        "version": "1.0.0",
        "details": {}
    }
    
    try:
        # Vérifier l'état du service depuis main.py
        service_initialized = getattr(request.app.state, 'service_initialized', False)
        elasticsearch_client = getattr(request.app.state, 'elasticsearch_client', None)
        initialization_error = getattr(request.app.state, 'initialization_error', None)
        
        if service_initialized and elasticsearch_client:
            # Service correctement initialisé
            try:
                # Test de santé Elasticsearch
                es_health = await elasticsearch_client.health_check()
                
                # Vérifier le core manager
                core_manager = getattr(request.app.state, 'core_manager', None)
                core_status = "initialized" if (core_manager and core_manager.is_initialized()) else "not_initialized"
                
                health_status.update({
                    "status": "healthy",
                    "details": {
                        "service_initialized": True,
                        "elasticsearch": es_health,
                        "core_manager": core_status,
                        "bonsai_url_configured": bool(os.environ.get("BONSAI_URL"))
                    }
                })
                
                return JSONResponse(content=health_status, status_code=200)
                
            except Exception as e:
                # Erreur lors des tests de santé
                health_status.update({
                    "status": "degraded",
                    "details": {
                        "service_initialized": True,
                        "elasticsearch_error": str(e),
                        "core_manager": "error",
                        "bonsai_url_configured": bool(os.environ.get("BONSAI_URL"))
                    }
                })
                
                return JSONResponse(content=health_status, status_code=503)
        
        elif initialization_error:
            # Erreur d'initialisation connue
            health_status.update({
                "status": "unhealthy",
                "details": {
                    "service_initialized": False,
                    "initialization_error": initialization_error,
                    "bonsai_url_configured": bool(os.environ.get("BONSAI_URL"))
                }
            })
            
            return JSONResponse(content=health_status, status_code=503)
        
        else:
            # Initialisation en cours ou pas encore tentée
            logger.info("🔄 Service non initialisé, tentative d'initialisation forcée...")
            
            # Tentative d'initialisation forcée
            success, error = await _ensure_core_initialization()
            
            if success:
                health_status.update({
                    "status": "healthy",
                    "details": {
                        "service_initialized": True,
                        "elasticsearch": "initialized_via_fallback",
                        "core_manager": "initialized_via_fallback",
                        "bonsai_url_configured": bool(os.environ.get("BONSAI_URL"))
                    }
                })
                return JSONResponse(content=health_status, status_code=200)
            else:
                health_status.update({
                    "status": "unhealthy",
                    "details": {
                        "service_initialized": False,
                        "fallback_initialization_error": error,
                        "bonsai_url_configured": bool(os.environ.get("BONSAI_URL"))
                    }
                })
                return JSONResponse(content=health_status, status_code=503)
    
    except Exception as e:
        # Erreur inattendue dans le health check
        logger.error(f"❌ Erreur inattendue dans health check: {e}")
        health_status.update({
            "status": "error",
            "details": {
                "unexpected_error": str(e),
                "bonsai_url_configured": bool(os.environ.get("BONSAI_URL"))
            }
        })
        
        return JSONResponse(content=health_status, status_code=500)


@router.get("/status", summary="Statut détaillé du service")
async def service_status(request: Request):
    """
    Endpoint pour obtenir un statut détaillé du service.
    
    Returns:
        JSONResponse: Informations détaillées sur l'état du service
    """
    try:
        # Import des utilitaires de configuration
        from search_service.clients.elasticsearch_client import get_client_configuration_info, get_client_metrics
        
        # Informations de base
        status_info = {
            "service": "search_service",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
        
        # État du service depuis main.py
        service_state = {
            "service_initialized": getattr(request.app.state, 'service_initialized', False),
            "elasticsearch_client_available": getattr(request.app.state, 'elasticsearch_client', None) is not None,
            "initialization_error": getattr(request.app.state, 'initialization_error', None)
        }
        
        # Configuration Elasticsearch
        try:
            config_info = get_client_configuration_info()
            status_info["configuration"] = config_info
        except Exception as e:
            status_info["configuration_error"] = str(e)
        
        # Métriques du client
        try:
            metrics = get_client_metrics()
            status_info["metrics"] = metrics
        except Exception as e:
            status_info["metrics_error"] = str(e)
        
        # État global
        status_info["service_state"] = service_state
        
        return JSONResponse(content=status_info, status_code=200)
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Failed to get service status",
                "details": str(e)
            },
            status_code=500
        )


@router.get("/test-connection", summary="Test de connexion Elasticsearch")
async def test_elasticsearch_connection_endpoint():
    """
    Endpoint pour tester la connexion à Elasticsearch/Bonsai.
    
    Returns:
        JSONResponse: Résultat du test de connexion
    """
    try:
        from search_service.clients.elasticsearch_client import test_elasticsearch_connection
        
        result = await test_elasticsearch_connection()
        
        if result.get("connection_test", False):
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=503)
            
    except Exception as e:
        return JSONResponse(
            content={
                "connection_test": False,
                "error": str(e),
                "health_check": {"status": "error", "message": str(e)}
            },
            status_code=500
        )


@router.get("/quick-search", summary="Recherche rapide pour tests")
async def quick_search_endpoint(
    user_id: int = 34,
    query: str = "test",
    limit: int = 5
):
    """
    Endpoint pour effectuer une recherche rapide (utile pour les tests).
    
    Args:
        user_id: ID utilisateur (défaut: 34)
        query: Terme de recherche (défaut: "test")
        limit: Nombre de résultats (défaut: 5)
    
    Returns:
        JSONResponse: Résultats de recherche
    """
    try:
        from search_service.clients.elasticsearch_client import quick_search
        
        result = await quick_search(
            user_id=user_id,
            query=query,
            limit=limit
        )
        
        if "error" in result:
            return JSONResponse(content=result, status_code=503)
        else:
            return JSONResponse(content=result, status_code=200)
            
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Quick search failed",
                "details": str(e),
                "took": 0,
                "hits": {"total": {"value": 0}, "hits": []}
            },
            status_code=500
        )


@router.post("/search", summary="Recherche de transactions")
async def search_transactions(
    request: Request,
    search_request: dict
):
    """
    Endpoint principal pour la recherche de transactions.
    
    Args:
        search_request: Paramètres de recherche
        
    Returns:
        JSONResponse: Résultats de recherche
    """
    try:
        # Vérifier que le service est initialisé
        service_initialized = getattr(request.app.state, 'service_initialized', False)
        core_manager = getattr(request.app.state, 'core_manager', None)
        
        if not service_initialized or not core_manager:
            # Tentative d'initialisation forcée
            success, error = await _ensure_core_initialization()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail=f"Service not initialized: {error}"
                )
        
        # Extraire les paramètres
        user_id = search_request.get("user_id")
        query = search_request.get("query", "")
        filters = search_request.get("filters", {})
        limit = search_request.get("limit", 20)
        offset = search_request.get("offset", 0)
        
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required"
            )
        
        # Effectuer la recherche via le core manager
        if hasattr(request.app.state, 'core_manager'):
            core_manager = request.app.state.core_manager
            
            # Utiliser l'engine de recherche
            search_engine = core_manager.get_search_engine()
            
            result = await search_engine.search_transactions(
                user_id=user_id,
                query=query,
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            return JSONResponse(content=result, status_code=200)
        else:
            raise HTTPException(
                status_code=503,
                detail="Search engine not available"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/config", summary="Configuration du service")
async def get_service_configuration():
    """
    Endpoint pour obtenir la configuration actuelle du service.
    
    Returns:
        JSONResponse: Configuration du service
    """
    try:
        config = {
            "service": "search_service",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": {
                "bonsai_url_configured": bool(os.environ.get("BONSAI_URL")),
                "elasticsearch_index": os.environ.get("ELASTICSEARCH_INDEX", "harena_transactions"),
                "test_user_id": os.environ.get("TEST_USER_ID", "34")
            }
        }
        
        # Ajouter des infos détaillées si disponibles
        try:
            from search_service.clients.elasticsearch_client import get_client_configuration_info
            detailed_config = get_client_configuration_info()
            config["detailed_configuration"] = detailed_config
        except Exception as e:
            config["configuration_error"] = str(e)
        
        return JSONResponse(content=config, status_code=200)
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Failed to get configuration",
                "details": str(e)
            },
            status_code=500
        )


# === ROUTES DE DEBUG/DÉVELOPPEMENT ===

@router.post("/reset-client", summary="Reset du client Elasticsearch (dev only)")
async def reset_elasticsearch_client():
    """
    Endpoint pour réinitialiser le client Elasticsearch.
    Utile pour le développement et les tests.
    """
    try:
        from search_service.clients.elasticsearch_client import reset_default_client
        
        reset_default_client()
        
        # Reset des variables globales
        global _initialization_attempted, _initialization_successful, _initialization_error
        _initialization_attempted = False
        _initialization_successful = False
        _initialization_error = None
        
        return JSONResponse(
            content={
                "message": "Elasticsearch client reset successfully",
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=200
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Failed to reset client",
                "details": str(e)
            },
            status_code=500
        )


@router.get("/metrics", summary="Métriques du service")
async def get_service_metrics():
    """
    Endpoint pour obtenir les métriques du service.
    
    Returns:
        JSONResponse: Métriques détaillées
    """
    try:
        from search_service.clients.elasticsearch_client import get_client_metrics
        
        metrics = get_client_metrics()
        
        # Ajouter timestamp
        metrics["timestamp"] = datetime.utcnow().isoformat()
        metrics["service"] = "search_service"
        
        return JSONResponse(content=metrics, status_code=200)
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Failed to get metrics",
                "details": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )