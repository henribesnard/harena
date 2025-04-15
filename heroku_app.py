"""
Application Harena complète pour le déploiement Heroku.
Intégration de tous les services via inclusion directe des routeurs.
"""

import logging
import os
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("harena")

# Création de l'application FastAPI
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

# Suivi des services intégrés
service_status = {
    "user_service": False,
    "sync_service": False,
    "transaction_vector_service": False,
    "conversation_service": False
}

# ======== USER SERVICE ========
try:
    # Importer les dépendances de base de données
    from user_service.db.session import get_db
    
    # Importer les routes utilisateur
    from user_service.api.endpoints import users as users_endpoints
    from user_service.core.config import settings as user_settings
    
    # Inclure les routes utilisateur avec le bon préfixe 
    app.include_router(
        users_endpoints.router, 
        prefix=f"{user_settings.API_V1_STR}/users", 
        tags=["users"]
    )
    
    # Vérifier si les utilisateurs reçoivent aussi les dépendances de base de données
    # Get_db est disponible pour les routes utilisateur
    logger.info(f"Dépendances DB importées avec succès")
    
    service_status["user_service"] = True
    logger.info("User Service routes intégrées avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du User Service: {str(e)}")

# ======== SYNC SERVICE ========
try:
    # Tenter d'importer les endpoints de synchronisation
    from sync_service.api.endpoints import sync as sync_endpoints
    from sync_service.api.endpoints import webhooks as webhooks_endpoints
    
    # Inclure les routes de synchronisation
    app.include_router(
        sync_endpoints.router, 
        prefix="/api/v1/sync", 
        tags=["synchronization"]
    )
    
    # Inclure les routes de webhooks
    app.include_router(
        webhooks_endpoints.router,
        prefix="/webhooks",
        tags=["webhooks"]
    )
    
    service_status["sync_service"] = True
    logger.info("Sync Service routes intégrées avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Sync Service: {str(e)}")

# ======== TRANSACTION VECTOR SERVICE ========
try:
    # Tenter d'importer les endpoints de transactions
    from transaction_vector_service.api.endpoints.transactions import router as transactions_router
    
    # Inclure les routes de transactions
    app.include_router(
        transactions_router,
        prefix="/api/v1/transactions",
        tags=["transactions"]
    )
    
    service_status["transaction_vector_service"] = True
    logger.info("Transaction Vector Service routes intégrées avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Transaction Vector Service: {str(e)}")

# ======== CONVERSATION SERVICE ========
try:
    # Tenter d'importer les endpoints de conversation
    from conversation_service.api.endpoints import router as conversation_router
    
    # Inclure les routes de conversation
    app.include_router(
        conversation_router,
        prefix="/api/v1/conversations",
        tags=["conversations"]
    )
    
    service_status["conversation_service"] = True
    logger.info("Conversation Service routes intégrées avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Conversation Service: {str(e)}")

# ======== ENDPOINTS DE BASE ========

@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    """
    # Collecter les services actifs
    active_services = [name for name, status in service_status.items() if status]
    
    return {
        "status": "ok",
        "application": "Harena Finance API",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "active_services": active_services
    }

@app.get("/health", tags=["health"])
async def health_check():
    """
    Vérification de l'état de santé de tous les services.
    """
    from datetime import datetime
    
    return {
        "status": "ok",
        "services": service_status,
        "version": "1.0.0",
        "timestamp": str(datetime.now())
    }

@app.get("/debug", tags=["debug"])
async def debug_info():
    """
    Endpoint de débogage pour vérifier les configurations.
    """
    import sys
    
    # Récupérer les routeurs FastAPI
    routes_info = []
    for route in app.routes:
        methods = getattr(route, "methods", set())
        if methods:
            methods_str = list(methods)
        else:
            methods_str = []
            
        route_info = {
            "path": getattr(route, "path", "unknown"),
            "name": getattr(route, "name", "unnamed"),
            "methods": methods_str,
        }
        routes_info.append(route_info)
    
    # Trier les routes par chemin
    routes_info.sort(key=lambda x: x["path"])
    
    return {
        "python_version": sys.version,
        "service_status": service_status,
        "routes_count": len(routes_info),
        "routes": routes_info,
        "database_available": "get_db" in globals()
    }

# ======== GESTIONNAIRE D'EXCEPTIONS ========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions.
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