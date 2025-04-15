"""
Version plus simple de l'application Harena pour le déploiement Heroku.
Cette version se concentre uniquement sur l'intégration du service utilisateur.
"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("harena")

# Création de l'application FastAPI simplifiée
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    """
    return {
        "status": "ok",
        "application": "Harena Finance API",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "production")
    }

@app.get("/health", tags=["health"])
async def health_check():
    """
    Vérification de l'état de santé simplifiée.
    """
    from datetime import datetime
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": str(datetime.now())
    }

# Tenter d'importer seulement le service utilisateur
user_service_available = False
try:
    from user_service.api.endpoints import users as users_endpoints
    from user_service.core.config import settings as user_settings
    
    # Inclure les routes utilisateur directement plutôt que de monter l'application
    app.include_router(
        users_endpoints.router, 
        prefix=f"{user_settings.API_V1_STR}/users", 
        tags=["users"]
    )
    
    user_service_available = True
    logger.info("User Service routes intégrées avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du User Service: {str(e)}")

@app.get("/debug", tags=["debug"])
async def debug_info():
    """
    Endpoint de débogage pour vérifier les configurations.
    """
    import sys
    import os
    
    return {
        "python_version": sys.version,
        "user_service_available": user_service_available,
        "environment_variables": {k: v for k, v in os.environ.items() if not k.startswith(('AWS', 'SECRET'))},
        "sys_path": sys.path
    }