"""
Routeur centralisé pour l'API du service de synchronisation.

Ce module s'occupe de l'enregistrement des différents routeurs d'endpoints
et de la configuration des middlewares API pour le service de synchronisation.
"""
import logging
import time
from typing import Callable, List, Dict, Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRouter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour journaliser les requêtes HTTP et leur temps de traitement."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Extraire l'IP client (en tenant compte des proxys)
        client_host = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else client_host
        
        # Journaliser le début de la requête
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"from {client_ip} - Query: {request.query_params}"
        )
        
        try:
            # Traiter la requête
            response = await call_next(request)
            
            # Calculer le temps de traitement
            process_time = time.time() - start_time
            
            # Journaliser la fin de la requête
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Duration: {process_time:.3f}s"
            )
            
            # Ajouter le temps de traitement dans les en-têtes (utile pour le débogage)
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Journaliser l'erreur
            logger.error(
                f"Request failed: {request.method} {request.url.path} - Error: {str(e)}",
                exc_info=True
            )
            # Re-lever l'exception pour que FastAPI puisse la gérer
            raise

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware pour limiter le taux de requêtes."""
    
    def __init__(self, app: ASGIApp, requests_limit: int = 60, period_seconds: int = 60):
        """
        Initialise le middleware avec les limites de taux.
        
        Args:
            app: Application ASGI
            requests_limit: Nombre maximum de requêtes autorisées par période
            period_seconds: Durée de la période en secondes
        """
        super().__init__(app)
        self.rate_limits = {}  # Pour stocker les limites par IP
        self.requests_limit = requests_limit
        self.period_seconds = period_seconds
        logger.info(f"Rate limiting configured: {requests_limit} requests per {period_seconds} seconds")
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extraction de l'IP ou de l'identifiant utilisateur
        client_ip = self._get_client_ip(request)
        
        # TODO: Implémenter la logique de limite de taux réelle
        # Pour l'instant, le middleware est passif et enregistre seulement
        logger.debug(f"Rate limit check for IP: {client_ip}")
        
        # Laisser passer la requête
        response = await call_next(request)
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extraire l'adresse IP du client en tenant compte des proxys."""
        client_host = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        return forwarded_for.split(",")[0].strip() if forwarded_for else client_host

def register_routers(app: FastAPI) -> None:
    """
    Enregistre tous les routeurs d'API du service de synchronisation.
    
    Args:
        app: Application FastAPI principale
    """
    # Structure pour suivre l'enregistrement des routeurs
    registered_endpoints = {
        "api": [],
        "webhooks": []
    }
    
    # Création du routeur API principal avec préfixe global
    api_router = APIRouter(prefix="/api/v1")
    
    # Définition des routeurs à enregistrer avec leur préfixe
    api_routers_config = [
        {
            "module": "sync_service.api.endpoints.sync",
            "attr": "router",
            "prefix": "/sync",
            "tags": ["synchronization"]
        },
        {
            "module": "sync_service.api.endpoints.transactions",
            "attr": "router",
            "prefix": "/transactions",
            "tags": ["transactions"]
        },
        {
            "module": "sync_service.api.endpoints.accounts",
            "attr": "router",
            "prefix": "/accounts",
            "tags": ["accounts"]
        },
        {
            "module": "sync_service.api.endpoints.items",
            "attr": "router",
            "prefix": "/items",
            "tags": ["items"]
        },
        {
            "module": "sync_service.api.endpoints.stocks",
            "attr": "router",
            "prefix": "/stocks",
            "tags": ["stocks"]
        },
        {
            "module": "sync_service.api.endpoints.categories",
            "attr": "router",
            "prefix": "/categories",
            "tags": ["categories"]
        },
        {
            "module": "sync_service.api.endpoints.insights",
            "attr": "router",
            "prefix": "/insights",
            "tags": ["insights"]
        }
    ]
    
    # Inclusion des routeurs API avec gestion des erreurs individuelle
    for router_config in api_routers_config:
        try:
            # Import dynamique du module
            module_path = router_config["module"]
            attr_name = router_config["attr"]
            
            module = __import__(module_path, fromlist=[attr_name])
            router = getattr(module, attr_name)
            
            # Inclusion dans le routeur API
            api_router.include_router(
                router,
                prefix=router_config["prefix"],
                tags=router_config.get("tags", [])
            )
            
            registered_endpoints["api"].append({
                "module": module_path,
                "prefix": router_config["prefix"],
                "status": "registered"
            })
            
            logger.info(f"API router registered: {module_path} at {router_config['prefix']}")
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to register API router {router_config['module']}: {str(e)}")
            registered_endpoints["api"].append({
                "module": router_config["module"],
                "prefix": router_config["prefix"],
                "status": "failed",
                "error": str(e)
            })
    
    # Enregistrer le routeur API principal
    app.include_router(api_router)
    logger.info(f"Main API router registered with prefix /api/v1")
    
    # Enregistrement séparé pour les webhooks
    try:
        # Import du routeur de webhooks
        from sync_service.api.endpoints.webhooks import router as webhooks_router
        
        # Enregistrement directement sur l'app (sans préfixe API)
        app.include_router(
            webhooks_router,
            prefix="/webhooks",
            tags=["webhooks"]
        )
        
        registered_endpoints["webhooks"].append({
            "module": "sync_service.api.endpoints.webhooks",
            "prefix": "/webhooks",
            "status": "registered"
        })
        
        logger.info("Webhooks router registered at /webhooks")
    except ImportError as e:
        logger.warning(f"Failed to register webhooks router: {str(e)}")
        registered_endpoints["webhooks"].append({
            "module": "sync_service.api.endpoints.webhooks",
            "prefix": "/webhooks",
            "status": "failed",
            "error": str(e)
        })
    
    # Résumé des enregistrements
    successful_apis = [ep for ep in registered_endpoints["api"] if ep["status"] == "registered"]
    failed_apis = [ep for ep in registered_endpoints["api"] if ep["status"] == "failed"]
    
    logger.info(f"Router registration summary: {len(successful_apis)} APIs successful, {len(failed_apis)} APIs failed")
    
    # Journaliser les échecs en détail
    for failed in failed_apis:
        logger.warning(f"Failed router: {failed['module']} at {failed['prefix']} - Error: {failed.get('error', 'unknown')}")
    
    # Journaliser les webhooks séparément
    webhooks_status = registered_endpoints["webhooks"][0]["status"] if registered_endpoints["webhooks"] else "not attempted"
    logger.info(f"Webhooks router status: {webhooks_status}")


def setup_middleware(app: FastAPI) -> None:
    """
    Configure les middlewares spécifiques au service de synchronisation.
    
    Args:
        app: Application FastAPI principale
    """
    # Ajouter le middleware de journalisation en premier
    app.add_middleware(LoggingMiddleware)
    logger.info("Logging middleware configured")
    
    # Ajouter le middleware de limitation de taux conditionnellement
    from config_service.config import settings
    
    if getattr(settings, "RATE_LIMIT_ENABLED", False):
        rate_limit = getattr(settings, "RATE_LIMIT_REQUESTS", 60)
        rate_period = getattr(settings, "RATE_LIMIT_PERIOD", 60)
        
        app.add_middleware(
            RateLimitingMiddleware,
            requests_limit=rate_limit,
            period_seconds=rate_period
        )
        logger.info(f"Rate limiting middleware configured: {rate_limit} requests per {rate_period} seconds")
    else:
        logger.info("Rate limiting disabled in configuration")


def create_error_handlers(app: FastAPI) -> None:
    """
    Configure les gestionnaires d'erreurs personnalisés.
    
    Args:
        app: Application FastAPI principale
    """
    from fastapi import Request, status
    from fastapi.responses import JSONResponse
    import traceback
    
    @app.exception_handler(status.HTTP_404_NOT_FOUND)
    async def not_found_handler(request: Request, exc):
        """Gestionnaire pour les erreurs 404 Not Found."""
        logger.info(f"Resource not found: {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "status": "error",
                "code": "not_found",
                "message": "The requested resource was not found",
                "path": request.url.path
            }
        )
    
    @app.exception_handler(status.HTTP_500_INTERNAL_SERVER_ERROR)
    async def server_error_handler(request: Request, exc):
        """Gestionnaire pour les erreurs 500 Internal Server Error."""
        # Journaliser l'erreur avec stack trace
        logger.error(
            f"Internal server error: {request.method} {request.url.path} - Error: {str(exc)}",
            exc_info=True
        )
        
        # Déterminer le niveau de détail selon l'environnement
        from config_service.config import settings
        is_dev = getattr(settings, "ENVIRONMENT", "production").lower() in ["development", "dev", "local"]
        
        error_content = {
            "status": "error",
            "code": "internal_server_error",
            "message": "An internal server error occurred",
            "path": request.url.path
        }
        
        # Ajouter des détails supplémentaires en dev
        if is_dev:
            error_content["error_details"] = str(exc)
            error_content["traceback"] = traceback.format_exc()
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_content
        )
    
    @app.exception_handler(Exception)
    async def catch_all_handler(request: Request, exc):
        """Gestionnaire pour capturer toute exception non gérée."""
        logger.error(
            f"Unhandled exception: {request.method} {request.url.path} - Error: {str(exc)}",
            exc_info=True
        )
        
        # Déterminer le niveau de détail selon l'environnement
        from config_service.config import settings
        is_dev = getattr(settings, "ENVIRONMENT", "production").lower() in ["development", "dev", "local"]
        
        error_content = {
            "status": "error",
            "code": "unhandled_exception",
            "message": "An unexpected error occurred" if not is_dev else str(exc),
            "path": request.url.path
        }
        
        # Ajouter des détails supplémentaires en dev
        if is_dev:
            error_content["error_type"] = exc.__class__.__name__
            error_content["traceback"] = traceback.format_exc()
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_content
        )
    
    logger.info("Custom error handlers configured")