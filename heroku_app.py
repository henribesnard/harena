"""
Version simplifiée de l'application Harena pour le déploiement Heroku.
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
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    Retourne un statut et des informations basiques sur l'application.
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

# Tenter d'importer et de monter le service utilisateur (pour démonstration)
try:
    # Cette partie peut être activée progressivement une fois l'application de base fonctionnelle
    # from user_service.main import app as user_app
    # app.mount("/user", user_app)
    # logger.info("User Service monté avec succès")
    pass
except ImportError as e:
    logger.warning(f"User Service n'a pas pu être monté: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=port)