from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from user_service.api.endpoints import users
from user_service.core.config import settings

# Importation des endpoints de synchronisation
try:
    from sync_service.api.endpoints import sync, webhooks
    sync_module_available = True
except ImportError:
    sync_module_available = False
    logging.warning("Sync service module not found, synchronization endpoints will not be available")

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Création de l'application FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="API pour le service utilisateur et la synchronisation bancaire de Harena",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À remplacer par les domaines autorisés en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes utilisateur
app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])

# Inclusion des routes de synchronisation si disponibles
if sync_module_available:
    # Routes pour la synchronisation (avec le préfixe API)
    app.include_router(
        sync.router, 
        prefix=f"{settings.API_V1_STR}/sync", 
        tags=["synchronization"]
    )
    
    # Route pour les webhooks (sans préfixe API_V1_STR pour faciliter la configuration)
    app.include_router(
        webhooks.router,
        prefix="/webhooks",
        tags=["webhooks"]
    )
    logging.info("Sync service endpoints registered successfully")

@app.get("/health")
def health_check():
    """
    Vérification de l'état de santé de l'API.
    Renvoie un statut OK si l'API est opérationnelle.
    """
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "sync_service_available": sync_module_available
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)