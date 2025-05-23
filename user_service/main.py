# user_service/main.py
"""
Module principal du service utilisateur.

Ce module initialise et configure le service utilisateur de la plateforme Harena,
gérant l'authentification, les utilisateurs et leurs connexions bancaires.
"""
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from user_service.api.endpoints import users
from config_service.config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
)

logger = logging.getLogger("user_service")

# ======== GESTION DU CYCLE DE VIE DE L'APPLICATION ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service utilisateur."""
    # Initialisation
    logger.info("Démarrage du service utilisateur")
    
    # Vérification des configurations critiques
    if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
        logger.warning("SECRET_KEY non définie ou trop courte. La sécurité des tokens JWT peut être compromise.")
    
    if not settings.BRIDGE_CLIENT_ID or not settings.BRIDGE_CLIENT_SECRET:
        logger.warning("Identifiants Bridge API manquants. Les fonctionnalités de connexion bancaire ne fonctionneront pas.")
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("Arrêt du service utilisateur")


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service utilisateur."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        description="API pour la gestion des utilisateurs et l'authentification de Harena",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # À configurer en production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Inclusion des routes utilisateur
    app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])
    
    # Ajout de l'endpoint de santé
    @app.get("/health")
    def health_check():
        """Vérification de l'état de santé du service utilisateur."""
        return {
            "status": "ok",
            "service": settings.PROJECT_NAME,
            "version": "1.0.0",
            "api_prefix": settings.API_V1_STR,
            "bridge_api_configured": bool(settings.BRIDGE_CLIENT_ID and settings.BRIDGE_CLIENT_SECRET)
        }
    
    # Réglage du niveau de log pour les modules tiers trop verbeux
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    return app


# Pour les tests/développement
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)