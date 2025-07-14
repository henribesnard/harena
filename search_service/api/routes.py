# search_service/api/routes.py
"""
Routes API REST simplifiées pour le Search Service
==================================================

Version simplifiée pour démarrage - à enrichir progressivement.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# === ROUTEUR PRINCIPAL ===
router = APIRouter(tags=["search"])


# === ENDPOINT DE SANTÉ SIMPLE ===
@router.get(
    "/health",
    summary="Santé du service de recherche",
    description="Vérification de l'état de santé du Search Service"
)
async def health_check(request: Request) -> JSONResponse:
    """
    Vérification de santé simplifiée du service
    """
    
    start_time = time.time()
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        logger.debug(f"Health check started [{correlation_id}]")
        
        # Vérifications de base
        components = []
        overall_status = "healthy"
        
        # Vérifier les imports core (sans les exécuter)
        try:
            from search_service.core import core_manager
            if hasattr(core_manager, 'is_initialized') and core_manager.is_initialized():
                components.append({
                    "name": "core_engine",
                    "status": "healthy",
                    "last_check": datetime.now().isoformat(),
                    "dependencies": ["elasticsearch", "query_executor"],
                    "metrics": {}
                })
            else:
                components.append({
                    "name": "core_engine", 
                    "status": "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "error_message": "Core manager not initialized",
                    "dependencies": ["elasticsearch", "query_executor"],
                    "metrics": {}
                })
                overall_status = "unhealthy"
        except Exception as e:
            components.append({
                "name": "core_engine",
                "status": "unhealthy", 
                "last_check": datetime.now().isoformat(),
                "error_message": f"Core import failed: {str(e)}",
                "dependencies": ["elasticsearch", "query_executor"],
                "metrics": {}
            })
            overall_status = "unhealthy"
        
        # Vérifier Elasticsearch (basique)
        try:
            # Import conditionnel pour éviter les erreurs
            from search_service.clients.elasticsearch_client import ElasticsearchClient
            components.append({
                "name": "elasticsearch",
                "status": "degraded",  # Pas de vraie vérification pour l'instant
                "last_check": datetime.now().isoformat(),
                "error_message": "Client available but not tested",
                "dependencies": [],
                "metrics": {}
            })
        except Exception as e:
            components.append({
                "name": "elasticsearch",
                "status": "unhealthy",
                "last_check": datetime.now().isoformat(), 
                "error_message": f"Elasticsearch client not available: {str(e)}",
                "dependencies": [],
                "metrics": {}
            })
            overall_status = "unhealthy"
        
        # Vérifier les utilitaires
        try:
            from search_service.utils.cache import CacheManager
            components.append({
                "name": "utils",
                "status": "healthy",
                "last_check": datetime.now().isoformat(),
                "dependencies": ["cache"],
                "metrics": {}
            })
        except Exception as e:
            components.append({
                "name": "utils", 
                "status": "degraded",
                "last_check": datetime.now().isoformat(),
                "error_message": f"Utils partially available: {str(e)}",
                "dependencies": ["cache"],
                "metrics": {}
            })
        
        # Vérifier le cache
        try:
            from search_service.utils.cache import CacheManager
            cache = CacheManager()
            components.append({
                "name": "cache",
                "status": "healthy",
                "last_check": datetime.now().isoformat(),
                "dependencies": [],
                "metrics": {"type": "in_memory", "status": "available"}
            })
        except Exception as e:
            components.append({
                "name": "cache",
                "status": "degraded",
                "last_check": datetime.now().isoformat(),
                "error_message": f"Cache not available: {str(e)}",
                "dependencies": [],
                "metrics": {}
            })
        
        # Métriques système de base
        health_check_duration = (time.time() - start_time) * 1000
        
        # Système de base
        system_health = {
            "overall_status": overall_status,
            "uptime_seconds": time.time() - getattr(request.app.state, 'start_time', time.time()),
            "memory_usage_mb": _get_basic_memory_usage(),
            "cpu_usage_percent": 0.0,  # Placeholder
            "active_connections": 0,
            "total_requests": 0,
            "error_rate_percent": 0.0
        }
        
        # Construire la réponse
        response_data = {
            "system": system_health,
            "components": components,
            "timestamp": datetime.now().isoformat(),
            "service_version": "1.0.0",
            "environment": "development",
            "metadata": {
                "correlation_id": correlation_id,
                "health_check_duration_ms": health_check_duration,
                "components_checked": len(components)
            }
        }
        
        # Déterminer le status code
        status_code = 200 if overall_status in ["healthy", "degraded"] else 503
        
        response = JSONResponse(
            content=response_data,
            status_code=status_code,
            headers={"X-Health-Check-Duration": f"{health_check_duration:.2f}ms"}
        )
        
        logger.info(
            f"Health check completed [{correlation_id}] - "
            f"Status: {overall_status}, Duration: {health_check_duration:.1f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed [{correlation_id}]: {e}", exc_info=True)
        
        # Réponse d'erreur minimale
        error_response_data = {
            "system": {
                "overall_status": "unhealthy",
                "uptime_seconds": 0.0,
                "memory_usage_mb": 0.0,
                "cpu_usage_percent": 0.0,
                "active_connections": 0,
                "total_requests": 0,
                "error_rate_percent": 100.0
            },
            "components": [{
                "name": "system",
                "status": "unhealthy",
                "last_check": datetime.now().isoformat(),
                "error_message": f"Health check process failed: {str(e)}",
                "dependencies": [],
                "metrics": {}
            }],
            "timestamp": datetime.now().isoformat(),
            "service_version": "1.0.0",
            "environment": "development",
            "metadata": {
                "correlation_id": correlation_id,
                "error": str(e)
            }
        }
        
        return JSONResponse(
            content=error_response_data,
            status_code=503
        )


# === ENDPOINTS DE BASE ===
@router.get(
    "/status",
    summary="Statut simple du service",
    description="Statut rapide sans vérifications approfondies"
)
async def simple_status() -> JSONResponse:
    """Statut simple et rapide"""
    
    return JSONResponse(
        content={
            "service": "search-service",
            "status": "running", 
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "message": "Service démarré en mode simplifié"
        }
    )


@router.post(
    "/lexical",
    summary="Recherche lexicale (placeholder)",
    description="Endpoint de recherche - non implémenté dans cette version simplifiée"
)
async def search_lexical_placeholder() -> JSONResponse:
    """Placeholder pour la recherche lexicale"""
    
    return JSONResponse(
        content={
            "error": "Not implemented yet",
            "message": "Search functionality will be implemented in next version",
            "service": "search-service",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        },
        status_code=501
    )


@router.get(
    "/info",
    summary="Informations sur le service",
    description="Informations détaillées sur le Search Service"
)
async def service_info() -> JSONResponse:
    """Informations sur le service"""
    
    return JSONResponse(
        content={
            "service": {
                "name": "search-service",
                "version": "1.0.0",
                "description": "Service de recherche lexicale haute performance",
                "status": "simplified_mode"
            },
            "features": {
                "implemented": [
                    "Health checks basiques",
                    "Status endpoint",
                    "Service info"
                ],
                "planned": [
                    "Recherche lexicale Elasticsearch",
                    "Cache intelligent", 
                    "Métriques détaillées",
                    "Validation de requêtes",
                    "Templates de requêtes"
                ]
            },
            "endpoints": {
                "GET /health": "Vérification de santé",
                "GET /status": "Statut simple",
                "GET /info": "Informations service",
                "POST /lexical": "Recherche lexicale (à venir)"
            },
            "environment": "development",
            "timestamp": datetime.now().isoformat()
        }
    )


# === FONCTIONS UTILITAIRES SIMPLIFIÉES ===

def _get_basic_memory_usage() -> float:
    """Récupère l'usage mémoire de base"""
    try:
        import psutil
        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        # psutil pas disponible
        return 0.0
    except Exception:
        return 0.0


# === GESTIONNAIRE D'ERREURS SIMPLE ===

@router.get("/test-error")
async def test_error():
    """Endpoint de test pour vérifier la gestion d'erreur"""
    raise HTTPException(status_code=500, detail="Test error endpoint")


# === EXPORTS ===

__all__ = [
    "router",
    "health_check",
    "simple_status", 
    "search_lexical_placeholder",
    "service_info"
]

logger.info("Routes API simplifiées chargées - prêt pour démarrage")