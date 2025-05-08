"""
Point d'entrée principal pour l'application Harena.

Ce module initialise et démarre tous les services de la plateforme financière Harena:
- User Service: Gestion des utilisateurs et authentification
- Sync Service: Synchronisation des données bancaires via Bridge API
- Analytics Service: Analyse des données financières (optionnel)
"""

import logging
import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("harena")

# ======== GESTION DU CYCLE DE VIE DE L'APPLICATION ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application.
    Initialise les ressources au démarrage et les libère à l'arrêt.
    """
    # Initialization code
    logger.info("Application Harena en démarrage...")
    
    # Vérification des variables d'environnement critiques
    required_env_vars = ["BRIDGE_CLIENT_ID", "BRIDGE_CLIENT_SECRET"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Variables d'environnement manquantes: {', '.join(missing_vars)}")
        logger.warning("Certaines fonctionnalités peuvent ne pas fonctionner correctement.")
    
    yield  # L'application s'exécute ici
    
    # Cleanup code
    logger.info("Application Harena en arrêt...")

# ======== CRÉATION DE L'APPLICATION ========

# Création de l'application FastAPI principale
app = FastAPI(
    title="Harena Finance API",
    description="API pour les services financiers Harena",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configuration CORS
origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,https://app.harena.finance").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Liste pour suivre les services disponibles
available_services = {}

# ======== IMPORTATION DES SERVICES ========

# User Service
try:
    from user_service.main import create_app as create_user_app
    user_app = create_user_app()
    from config_service.config import settings as user_settings
    logger.info("User Service importé avec succès")
    available_services["user_service"] = {
        "app": user_app,
        "settings": user_settings,
        "api_prefix": user_settings.API_V1_STR
    }
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du User Service: {e}")
    available_services["user_service"] = None

# Sync Service
try:
    from sync_service.main import create_app as create_sync_app
    sync_app = create_sync_app()
    logger.info("Sync Service importé avec succès")
    available_services["sync_service"] = {
        "app": sync_app,
        "api_prefix": "/api/v1/sync"
    }
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Sync Service: {e}")
    available_services["sync_service"] = None

# ======== MONTAGE DES APPLICATIONS ========

# Monter les applications sur des préfixes spécifiques
mount_paths = {
    "user_service": "/user",
    "sync_service": "/sync",
}

# Monter les applications et leurs API
for service_name, service_info in available_services.items():
    if service_info is not None:
        # Monter l'application sur son chemin dédié
        service_path = mount_paths.get(service_name)
        if service_path:
            app.mount(service_path, service_info["app"])
            logger.info(f"Service {service_name} monté sur {service_path}")
        
        # Monter également l'API à la racine API pour un accès unifié
        if "api_prefix" in service_info:
            api_prefix = service_info["api_prefix"]
            app.mount(api_prefix, service_info["app"])
            logger.info(f"API {service_name} montée sur {api_prefix}")

# ======== ENDPOINTS DE BASE ========

@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    Retourne un statut et des informations basiques sur l'application.
    """
    active_services = [name for name, info in available_services.items() if info is not None]
    
    return {
        "status": "ok",
        "application": "Harena Finance API",
        "version": "1.0.0",
        "services": active_services
    }

@app.get("/health", tags=["health"])
async def health_check():
    """
    Vérification de l'état de santé de tous les services.
    Interroge chaque service et renvoie leur état.
    """
    services_status = {
        "main": "ok",
    }
    
    # Vérifier l'état de chaque service
    for service_name, service_info in available_services.items():
        if service_info:
            services_status[service_name] = "ok"
        else:
            services_status[service_name] = "unavailable"
    
    # Déterminer l'état global
    overall_status = "ok" if all(status == "ok" for status in services_status.values()) else "degraded"
        
    return {
        "status": overall_status,
        "services": services_status,
        "version": "1.0.0",
        "timestamp": str(datetime.now())
    }

# ======== GESTIONNAIRE D'EXCEPTIONS ========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions pour toute l'application.
    Capture et formate les erreurs non gérées.
    """
    logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
    
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    error_detail = str(exc) if debug_mode else "Contactez l'administrateur pour plus d'informations."
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Une erreur interne est survenue",
            "detail": error_detail
        }
    )

# ======== LANCEMENT DE L'APPLICATION ========

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    # Ajouter l'environnement comme variable globale
    os.environ["ENVIRONMENT"] = os.getenv("ENVIRONMENT", "development")
    
    logger.info(f"Démarrage de l'application Harena sur {host}:{port} (debug={debug}, env={os.environ['ENVIRONMENT']})")
    logger.info(f"Services disponibles: {[name for name, info in available_services.items() if info is not None]}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        workers=1
    )