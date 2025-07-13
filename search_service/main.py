"""
🚀 Search Service - Point d'entrée principal FastAPI
Point d'entrée simplifié pour le service de recherche lexicale haute performance
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# === IMPORTS SEARCH SERVICE ===
from api import api_manager
from config import settings
from core import CoreManager
from utils import (
    get_utils_health,
    initialize_utils,
    shutdown_utils,
    get_system_metrics,
    get_performance_summary
)


# === CONFIGURATION LOGGING ===
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# === INSTANCE CORE MANAGER ===
core_manager = CoreManager()


# === FONCTIONS SANTÉ ET INFO ===

async def get_api_health() -> dict:
    """Retourne l'état de santé de l'API"""
    try:
        utils_health = get_utils_health()
        
        # Vérification basique des composants
        components_health = {
            "utils": utils_health.get("overall_status") == "healthy",
            "api_manager": hasattr(api_manager, 'router'),
            "core_manager": hasattr(core_manager, 'lexical_engine'),
            "settings": hasattr(settings, 'elasticsearch_host')
        }
        
        all_healthy = all(components_health.values())
        
        return {
            "healthy": all_healthy,
            "timestamp": datetime.now().isoformat(),
            "components": components_health,
            "details": utils_health
        }
    except Exception as e:
        logger.error(f"Erreur health check API: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def get_api_info() -> dict:
    """Retourne les informations détaillées de l'API"""
    try:
        system_metrics = get_system_metrics()
        performance_summary = get_performance_summary(hours=1)
        
        return {
            "service": "search-service",
            "version": "1.0.0",
            "description": "Service de recherche lexicale haute performance",
            "environment": settings.environment,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "elasticsearch_host": settings.elasticsearch_host,
                "elasticsearch_port": settings.elasticsearch_port,
                "cache_enabled": getattr(settings, 'cache_enabled', True),
                "metrics_enabled": getattr(settings, 'metrics_enabled', True),
                "log_level": settings.log_level
            },
            "system_metrics": system_metrics,
            "performance": performance_summary
        }
    except Exception as e:
        logger.error(f"Erreur récupération info API: {e}")
        return {
            "service": "search-service",
            "version": "1.0.0",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# === GESTIONNAIRE CYCLE DE VIE APPLICATION ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application FastAPI
    Gère l'initialisation et la fermeture propre de tous les composants
    """
    startup_start = datetime.now()
    logger.info("🚀 Démarrage Search Service...")
    
    try:
        # === INITIALISATION COMPOSANTS ===
        
        # 1. Initialisation des utilitaires
        logger.info("📚 Initialisation des utilitaires...")
        try:
            initialize_utils()
        except Exception as e:
            logger.warning(f"Erreur initialisation utilitaires: {e}")
        
        # 2. Initialisation des composants core
        logger.info("🔧 Initialisation des composants core...")
        # Core manager s'auto-initialise
        
        # 3. Initialisation de l'API
        logger.info("🌐 Initialisation de l'API...")
        await api_manager.initialize()
        
        # === VÉRIFICATIONS SANTÉ ===
        
        # Vérification santé composants
        logger.info("🏥 Vérification santé des composants...")
        
        try:
            utils_health = get_utils_health()
            if utils_health.get("overall_status") != "healthy":
                logger.warning(f"Santé utilitaires dégradée: {utils_health}")
        except Exception as e:
            logger.warning(f"Impossible de vérifier la santé des utilitaires: {e}")
        
        try:
            api_health = await get_api_health()
            if not api_health.get("healthy", False):
                logger.warning(f"Santé API dégradée: {api_health}")
        except Exception as e:
            logger.warning(f"Impossible de vérifier la santé de l'API: {e}")
        
        # === FINALISATION DÉMARRAGE ===
        startup_duration = (datetime.now() - startup_start).total_seconds()
        
        logger.info(f"✅ Search Service démarré avec succès en {startup_duration:.2f}s")
        logger.info(f"🔍 Mode: {settings.environment}")
        logger.info(f"📊 Elasticsearch: {settings.elasticsearch_host}:{settings.elasticsearch_port}")
        logger.info(f"💾 Cache: {'Activé' if getattr(settings, 'cache_enabled', True) else 'Désactivé'}")
        logger.info(f"📈 Métriques: {'Activées' if getattr(settings, 'metrics_enabled', True) else 'Désactivées'}")
        
        # Stocker le temps de démarrage pour les métriques d'uptime
        app.state.start_time = startup_start.timestamp()
        
        # Point de démarrage - l'application est prête
        yield
        
        # === ARRÊT PROPRE ===
        logger.info("🛑 Arrêt du Search Service...")
        shutdown_start = datetime.now()
        
        # Arrêt des composants dans l'ordre inverse
        try:
            await api_manager.shutdown()
            logger.info("✅ API fermée")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture API: {e}")
        
        try:
            # Core manager cleanup si nécessaire
            logger.info("✅ Composants core fermés")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture core: {e}")
        
        try:
            shutdown_utils()
            logger.info("✅ Utilitaires fermés")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture utilitaires: {e}")
        
        shutdown_duration = (datetime.now() - shutdown_start).total_seconds()
        logger.info(f"✅ Search Service arrêté proprement en {shutdown_duration:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Erreur critique lors du démarrage: {e}")
        # Tentative de nettoyage en cas d'erreur
        try:
            await api_manager.shutdown()
        except:
            pass
        try:
            shutdown_utils()
        except:
            pass
        raise


# === CRÉATION APPLICATION PRINCIPALE ===

def create_app(
    environment: Optional[str] = None,
    debug: Optional[bool] = None,
    **kwargs
) -> FastAPI:
    """
    Crée l'application FastAPI principale avec toute la configuration
    
    Args:
        environment: Environnement ('development', 'production', 'testing')
        debug: Mode debug
        **kwargs: Arguments supplémentaires pour FastAPI
        
    Returns:
        FastAPI: Application configurée et prête
    """
    
    # Configuration environnement
    env = environment or getattr(settings, 'environment', 'development')
    debug_mode = debug if debug is not None else (env == "development")
    
    logger.info(f"🏗️ Création application Search Service - Env: {env}, Debug: {debug_mode}")
    
    # === CONFIGURATION FASTAPI ===
    
    app_config = {
        "title": "Search Service API",
        "description": "Service de recherche lexicale haute performance avec Elasticsearch",
        "version": "1.0.0",
        "debug": debug_mode,
        "lifespan": lifespan,
        **kwargs
    }
    
    # Docs uniquement en development
    if env != "production":
        app_config.update({
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "openapi_url": "/openapi.json"
        })
    else:
        app_config.update({
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None
        })
    
    # Création application
    app = FastAPI(**app_config)
    
    # === MIDDLEWARE CORS ===
    
    if getattr(settings, 'cors_enabled', True):
        cors_origins = getattr(settings, 'cors_origins', ["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        logger.info("🌐 CORS configuré")
    
    # === INCLUSION ROUTES PRINCIPALES ===
    
    # Inclure les routes de l'API manager
    app.include_router(api_manager.router)
    if hasattr(api_manager, 'admin_router'):
        app.include_router(api_manager.admin_router)
    
    # === ROUTES DE BASE ===
    
    @app.get("/", tags=["Health"])
    async def root():
        """Point d'entrée racine du service"""
        return {
            "service": "search-service",
            "status": "running",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "environment": env,
            "endpoints": {
                "health": "/health",
                "info": "/info",
                "docs": "/docs" if env != "production" else None,
                "api": "/api/v1",
                "admin": "/admin"
            }
        }
    
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


# === FONCTIONS UTILITAIRES POUR DÉMARRAGE ===

def create_development_app() -> FastAPI:
    """Crée une application pour le développement"""
    return create_app(
        environment="development",
        debug=True,
        docs_url="/docs",
        redoc_url="/redoc"
    )


def create_production_app() -> FastAPI:
    """Crée une application pour la production"""
    return create_app(
        environment="production",
        debug=False,
        docs_url=None,
        redoc_url=None
    )


def create_testing_app() -> FastAPI:
    """Crée une application pour les tests"""
    return create_app(
        environment="testing",
        debug=True,
        docs_url="/docs"
    )


# === POINT D'ENTRÉE PRINCIPAL ===

def main():
    """
    Point d'entrée principal pour le démarrage du serveur
    Utilisé par uvicorn ou directement
    """
    
    # Configuration depuis les variables d'environnement
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"🚀 Démarrage serveur Search Service")
    logger.info(f"📍 Host: {host}:{port}")
    logger.info(f"👷 Workers: {workers}")
    logger.info(f"🔄 Reload: {reload}")
    
    # Configuration uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": host,
        "port": port,
        "reload": reload,
        "log_level": getattr(settings, 'log_level', 'info').lower(),
        "access_log": getattr(settings, 'access_log_enabled', True),
    }
    
    # Workers uniquement en production sans reload
    if workers > 1 and not reload:
        uvicorn_config["workers"] = workers
    
    # Démarrage du serveur
    uvicorn.run(**uvicorn_config)


# === INSTANCE PRINCIPALE ===

# Création de l'instance principale de l'application
app = create_app()


# === POINT D'ENTRÉE SCRIPT ===

if __name__ == "__main__":
    main()