"""
Point d'entrée principal pour l'application Harena.

Ce module initialise et démarre tous les services :
- User Service
- Sync Service
- Transaction Vector Service
- Conversation Service
"""

import logging
import uvicorn
import os
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

# Importation des routeurs de chaque service
try:
    from user_service.main import app as user_app
    from user_service.core.config import settings as user_settings
    logger.info("User Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du User Service: {e}")
    user_app = None

try:
    from sync_service.api.endpoints.sync import router as sync_router
    from sync_service.api.endpoints.webhooks import router as webhooks_router
    logger.info("Sync Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Sync Service: {e}")
    sync_router = None
    webhooks_router = None

try:
    from transaction_vector_service.main import app as transaction_vector_app
    from transaction_vector_service.config.settings import settings as transaction_settings
    logger.info("Transaction Vector Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Transaction Vector Service: {e}")
    transaction_vector_app = None

try:
    from conversation_service.main import app as conversation_app
    from conversation_service.config.settings import settings as conversation_settings
    logger.info("Conversation Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Conversation Service: {e}")
    conversation_app = None

# Inclure les routeurs de User Service
if user_app:
    # Monter l'application User Service
    app.mount("/user", user_app)
    # Rendre les routes User Service disponibles à la racine aussi
    app.mount(user_settings.API_V1_STR, user_app)

# Inclure les routeurs de Sync Service
if sync_router and webhooks_router:
    app.include_router(sync_router, prefix="/api/v1/sync", tags=["synchronization"])
    app.include_router(webhooks_router, prefix="/webhooks", tags=["webhooks"])

# Inclure Transaction Vector Service
if transaction_vector_app:
    app.mount("/transactions", transaction_vector_app)
    # Monter les endpoints à la racine API aussi
    app.mount(transaction_settings.API_V1_STR, transaction_vector_app)

# Inclure Conversation Service
if conversation_app:
    app.mount("/conversations", conversation_app)
    # Monter les endpoints à la racine API aussi
    app.mount(conversation_settings.API_PREFIX, conversation_app)

# Point d'entrée de base pour vérifier que l'application fonctionne
@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    """
    return {
        "status": "ok",
        "application": "Harena Finance API",
        "version": "1.0.0"
    }

# Point d'entrée pour vérifier la santé de tous les services
@app.get("/health", tags=["health"])
async def health_check():
    """
    Vérification de l'état de santé de tous les services.
    Renvoie le statut de chaque service.
    """
    services_status = {
        "main": "ok",
    }
    
    # Vérifier User Service
    if user_app:
        services_status["user_service"] = "ok"
    else:
        services_status["user_service"] = "unavailable"
        
    # Vérifier Sync Service
    if sync_router and webhooks_router:
        services_status["sync_service"] = "ok"
    else:
        services_status["sync_service"] = "unavailable"
        
    # Vérifier Transaction Vector Service
    if transaction_vector_app:
        services_status["transaction_vector_service"] = "ok"
    else:
        services_status["transaction_vector_service"] = "unavailable"
        
    # Vérifier Conversation Service
    if conversation_app:
        services_status["conversation_service"] = "ok"
    else:
        services_status["conversation_service"] = "unavailable"
        
    return {
        "status": "ok" if all(status == "ok" for status in services_status.values()) else "degraded",
        "services": services_status,
        "version": "1.0.0"
    }

# Gestionnaire d'exceptions global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions pour toute l'application.
    """
    logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Une erreur interne est survenue",
            "detail": str(exc) if os.getenv("DEBUG", "False").lower() == "true" else None
        }
    )

# Pour exécuter l'application en mode développement
if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Démarrage de l'application Harena sur {host}:{port} (debug={debug})")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        workers=1
    )