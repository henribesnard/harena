"""
🔍 Search Service - Point d'entrée FastAPI

Service de recherche lexicale haute performance avec Elasticsearch.
Point d'entrée principal de l'application FastAPI.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# === IMPORTS INTERNES ===
from .config import settings
from .api import api_manager
from .core import CoreManager
from .utils import (
    MetricsCollector,
    get_system_metrics,
    cleanup_old_metrics,
    get_utils_health
)
from .templates import QueryTemplateEngine, FinancialAggregationEngine

# === CONFIGURATION LOGGING ===
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === GESTIONNAIRES GLOBAUX ===
core_manager = CoreManager()
metrics_collector = MetricsCollector()

# === CONTEXTE APPLICATION ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    # Démarrage
    logger.info("🚀 Démarrage Search Service...")
    try:
        await api_manager.initialize()
        logger.info("✅ Search Service démarré avec succès")
        yield
    except Exception as e:
        logger.error(f"❌ Erreur démarrage Search Service: {e}")
        raise
    finally:
        # Arrêt
        logger.info("🛑 Arrêt Search Service...")
        try:
            await api_manager.shutdown()
            cleanup_old_metrics(hours=1)
            logger.info("✅ Search Service arrêté proprement")
        except Exception as e:
            logger.error(f"❌ Erreur arrêt Search Service: {e}")


def create_app(
    environment: str = None,
    debug: bool = None,
    cors_enabled: bool = None,
    trusted_hosts: list = None
) -> FastAPI:
    """
    Créer une application FastAPI configurée
    
    Args:
        environment: Environnement (development/production/testing)
        debug: Mode debug
        cors_enabled: Activer CORS
        trusted_hosts: Hôtes de confiance
        
    Returns:
        FastAPI: Application configurée
    """
    # Configuration par défaut ou depuis settings
    env = environment or settings.environment
    debug_mode = debug if debug is not None else (env == "development")
    cors_enabled = cors_enabled if cors_enabled is not None else settings.cors_enabled
    
    # Créer l'application
    app = FastAPI(
        title="🔍 Search Service",
        description="Service de recherche lexicale haute performance",
        version="1.0.0",
        debug=debug_mode,
        lifespan=lifespan,
        docs_url="/docs" if debug_mode else None,
        redoc_url="/redoc" if debug_mode else None
    )
    
    # === MIDDLEWARE ===
    
    # CORS
    if cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
    
    # Hôtes de confiance
    if trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts
        )
    
    # === ROUTES ===
    
    # Inclure les routes API
    app.include_router(
        api_manager.router,
        prefix="/api/v1",
        tags=["Search API"]
    )
    
    # Routes admin (si développement)
    if debug_mode:
        app.include_router(
            api_manager.admin_router,
            prefix="/admin",
            tags=["Admin"]
        )
    
    # === ROUTES SYSTÈME ===
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Vérification santé globale du service"""
        try:
            health_data = await get_api_health()
            status_code = 200 if health_data.get("healthy", False) else 503
            return JSONResponse(content=health_data, status_code=status_code)
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return JSONResponse(
                content={
                    "healthy": False, 
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                status_code=503
            )
    
    @app.get("/info", tags=["Info"])
    async def app_info():
        """Informations détaillées sur l'application"""
        try:
            return await get_api_info()
        except Exception as e:
            logger.error(f"Erreur récupération info: {e}")
            return JSONResponse(
                content={
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                status_code=500
            )
    
    # === GESTIONNAIRE D'ERREURS GLOBAL ===
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Gestionnaire d'erreurs global"""
        logger.error(f"Erreur non gérée: {exc}", exc_info=True)
        return JSONResponse(
            content={
                "error": "Erreur interne du serveur",
                "detail": str(exc) if debug_mode else "Une erreur inattendue s'est produite",
                "path": str(request.url),
                "method": request.method,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )
    
    logger.info("✅ Application FastAPI créée avec succès")
    return app


def create_development_app() -> FastAPI:
    """
    Créer application de développement
    
    Returns:
        FastAPI: Application configurée pour développement
    """
    return create_app(
        environment="development",
        debug=True,
        cors_enabled=True,
        trusted_hosts=["localhost", "127.0.0.1", "*.localhost"]
    )


def create_production_app() -> FastAPI:
    """
    Créer application de production
    
    Returns:
        FastAPI: Application configurée pour production
    """
    return create_app(
        environment="production",
        debug=False,
        cors_enabled=settings.cors_enabled,
        trusted_hosts=settings.trusted_hosts
    )


def create_testing_app() -> FastAPI:
    """
    Créer application de test
    
    Returns:
        FastAPI: Application configurée pour tests
    """
    return create_app(
        environment="testing",
        debug=True,
        cors_enabled=True
    )


async def get_api_health() -> Dict[str, Any]:
    """
    Obtenir l'état de santé de l'API
    
    Returns:
        Dict: Informations de santé détaillées
    """
    try:
        # Vérifier composants principaux
        health_data = {
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "environment": settings.environment,
            "service": "search_service",
            "version": "1.0.0"
        }
        
        # Santé des utilitaires
        utils_health = get_utils_health()
        health_data.update(utils_health)
        
        # Vérifier API manager
        if hasattr(api_manager, 'health_check'):
            api_health = await api_manager.health_check()
            health_data["api_manager"] = api_health
        
        # Métriques système
        system_metrics = get_system_metrics()
        health_data["system_metrics"] = system_metrics
        
        # Déterminer santé globale
        health_data["healthy"] = all([
            utils_health.get("healthy", False),
            system_metrics.get("healthy", False)
        ])
        
        return health_data
        
    except Exception as e:
        logger.error(f"Erreur health check API: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def get_api_info() -> Dict[str, Any]:
    """
    Obtenir informations détaillées de l'API
    
    Returns:
        Dict: Informations détaillées
    """
    try:
        info_data = {
            "service": "search_service",
            "version": "1.0.0",
            "description": "Service de recherche lexicale haute performance",
            "environment": settings.environment,
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "health": "/health",
                "info": "/info",
                "api": "/api/v1",
                "docs": "/docs" if settings.environment == "development" else None,
                "admin": "/admin" if settings.environment == "development" else None
            },
            "features": [
                "Recherche lexicale Elasticsearch",
                "Cache LRU intelligent",
                "Validation stricte des requêtes",
                "Métriques de performance",
                "Templates de requêtes",
                "Agrégations financières"
            ]
        }
        
        # Ajouter métriques système
        system_metrics = get_system_metrics()
        info_data["system"] = system_metrics
        
        return info_data
        
    except Exception as e:
        logger.error(f"Erreur récupération info API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """
    Point d'entrée principal pour l'exécution directe
    """
    import uvicorn
    
    # Configuration selon l'environnement
    if settings.environment == "development":
        app = create_development_app()
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    elif settings.environment == "production":
        app = create_production_app()
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=settings.port,
            workers=settings.workers,
            log_level="warning"
        )
    else:
        # Testing ou autre
        app = create_testing_app()
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="debug"
        )


# === APPLICATION PAR DÉFAUT ===
app = create_app()


if __name__ == "__main__":
    main()