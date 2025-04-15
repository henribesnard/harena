"""
Point d'entrée complet pour l'application Harena sur Heroku.

Ce module initialise et monte tous les services de la plateforme financière Harena:
- User Service: Gestion des utilisateurs et authentification
- Sync Service: Synchronisation des données bancaires via Bridge API
- Transaction Vector Service: Recherche et analyse vectorielle des transactions
- Conversation Service: Interface conversationnelle intelligente
"""

import logging
import os
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("harena")

# Création de l'application FastAPI principale
app = FastAPI(
    title="Harena Finance API",
    description="API pour les services financiers Harena",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Liste pour suivre les services disponibles
available_services = {}

# ======== IMPORTATION DES SERVICES ========

# User Service
try:
    from user_service.main import app as user_app
    from user_service.core.config import settings as user_settings
    logger.info("User Service importé avec succès")
    available_services["user_service"] = {
        "app": user_app,
        "settings": user_settings,
        "api_prefix": user_settings.API_V1_STR
    }
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du User Service: {str(e)}")
    available_services["user_service"] = None

# Sync Service
try:
    from sync_service.main import app as sync_app
    logger.info("Sync Service importé avec succès")
    available_services["sync_service"] = {
        "app": sync_app,
        "api_prefix": "/api/v1/sync"
    }
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Sync Service: {str(e)}")
    available_services["sync_service"] = None

# Transaction Vector Service
try:
    from transaction_vector_service.main import app as transaction_vector_app
    from transaction_vector_service.config.settings import settings as transaction_settings
    logger.info("Transaction Vector Service importé avec succès")
    available_services["transaction_vector_service"] = {
        "app": transaction_vector_app,
        "settings": transaction_settings,
        "api_prefix": transaction_settings.API_V1_STR
    }
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Transaction Vector Service: {str(e)}")
    available_services["transaction_vector_service"] = None

# Conversation Service
try:
    from conversation_service.main import app as conversation_app
    from conversation_service.config.settings import settings as conversation_settings
    logger.info("Conversation Service importé avec succès")
    available_services["conversation_service"] = {
        "app": conversation_app,
        "settings": conversation_settings,
        "api_prefix": conversation_settings.API_PREFIX
    }
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Conversation Service: {str(e)}")
    available_services["conversation_service"] = None

# ======== MONTAGE DES APPLICATIONS ========

# Monter les applications sur des préfixes spécifiques
mount_paths = {
    "user_service": "/user",
    "sync_service": "/sync",
    "transaction_vector_service": "/transactions",
    "conversation_service": "/conversations"
}

# Monter les applications et leurs API
for service_name, service_info in available_services.items():
    if service_info is not None:
        try:
            # Monter l'application sur son chemin dédié
            service_path = mount_paths.get(service_name)
            if service_path:
                app.mount(service_path, service_info["app"])
                logger.info(f"Service {service_name} monté sur {service_path}")
            
            # Monter également l'API à la racine API pour un accès unifié
            if "api_prefix" in service_info:
                api_prefix = service_info["api_prefix"]
                # Éviter de monter deux fois au même endroit
                if api_prefix != service_path:
                    app.mount(api_prefix, service_info["app"])
                    logger.info(f"API {service_name} montée sur {api_prefix}")
        except Exception as e:
            logger.error(f"Erreur lors du montage du service {service_name}: {str(e)}")

# ======== ENDPOINTS DE BASE ========

@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    Retourne un statut et des informations basiques sur l'application.
    """
    # Collecter les services actifs
    active_services = [name for name, info in available_services.items() if info is not None]
    
    return {
        "status": "ok",
        "application": "Harena Finance API",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "production"),
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
    
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    error_detail = str(exc) if debug_mode else "Contactez l'administrateur pour plus d'informations."
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Une erreur interne est survenue",
            "detail": error_detail
        }
    )

# Vérifier si exécuté directement (développement local)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    logger.info(f"Démarrage de l'application Harena sur {host}:{port} (debug={debug})")
    logger.info(f"Services disponibles: {[name for name, info in available_services.items() if info is not None]}")
    
    uvicorn.run(
        "heroku_app:app",
        host=host,
        port=port,
        reload=debug,
        workers=1
    )