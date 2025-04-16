# sync_service/main.py
"""
Point d'entrée principal pour le service de synchronisation.

Ce module initialise le service de synchronisation Harena, qui gère la synchronisation
des données bancaires via l'API Bridge et la mise à jour des transactions.
"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from user_service.core.config import settings
from sync_service.utils.logging import setup_structured_logging

# Configuration du logging structuré - changé à DEBUG
os.environ["LOG_LEVEL"] = "DEBUG"
logger = setup_structured_logging(level="DEBUG")

# Création de l'application FastAPI
app = FastAPI(
    title="Harena Sync Service",
    description="Service de synchronisation pour les données bancaires via Bridge API",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importation des routeurs
from sync_service.api.endpoints import sync as sync_router
from sync_service.api.endpoints import webhooks as webhooks_router

# Inclusion des routeurs
app.include_router(
    sync_router.router,
    prefix=f"{settings.API_V1_STR}/sync",
    tags=["synchronization"]
)

app.include_router(
    webhooks_router.router,
    prefix="/webhooks",
    tags=["webhooks"]
)

@app.get("/health")
async def health_check():
    """
    Vérification de l'état de santé du service.
    """
    return {
        "status": "ok",
        "service": "Harena Sync Service",
        "version": "1.0.0",
        "log_level": "DEBUG"
    }

@app.on_event("startup")
async def startup_event():
    """
    Actions à exécuter au démarrage du service.
    """
    logger.info("Starting Harena Sync Service with DEBUG log level")
    
    # Vérifier les configurations essentielles
    if not settings.BRIDGE_CLIENT_ID or not settings.BRIDGE_CLIENT_SECRET:
        logger.warning("Bridge API credentials are missing. Synchronization might not work correctly.")
    
    if not settings.BRIDGE_WEBHOOK_SECRET:
        logger.warning("Bridge webhook secret is missing. Webhook verification will be disabled.")
    
    # Initialiser les services supplémentaires si nécessaire

@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions à exécuter à l'arrêt du service.
    """
    logger.info("Shutting down Harena Sync Service")

# Exporter l'application pour être montée dans l'application principale
# ou démarrée directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sync_service.main:app", host="0.0.0.0", port=8003, reload=True)