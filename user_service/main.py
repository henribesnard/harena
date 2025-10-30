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

from user_service.api.endpoints import users, metrics
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
    
    # Configuration CORS - Activée en dev, désactivée en prod (gérée par Nginx)
    ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
    if ENVIRONMENT == "dev":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5174",  # Frontend Docker
                "http://localhost:5173",  # Frontend Vite direct
                "http://localhost:3000",  # Autre port dev
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Inclusion des routes utilisateur
    app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])

    # Inclusion des routes métriques
    app.include_router(metrics.router, prefix=f"{settings.API_V1_STR}/metrics", tags=["metrics"])
    
    # Ajout de l'endpoint de santé
    @app.get("/health")
    def health_check():
        """Vérification de l'état de santé du service utilisateur avec check DB."""
        from db_service.health import check_database_health
        from fastapi import status
        from fastapi.responses import JSONResponse

        # Vérifier la connexion à la base de données
        db_healthy, db_message = check_database_health()

        # Préparer la réponse
        health_status = {
            "status": "healthy" if db_healthy else "unhealthy",
            "service": settings.PROJECT_NAME,
            "version": "1.0.0",
            "api_prefix": settings.API_V1_STR,
            "bridge_api_configured": bool(settings.BRIDGE_CLIENT_ID and settings.BRIDGE_CLIENT_SECRET),
            "database": {
                "healthy": db_healthy,
                "message": db_message
            }
        }

        # Retourner 503 si la DB n'est pas accessible
        if not db_healthy:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_status
            )

        return health_status
    
    # Réglage du niveau de log pour les modules tiers trop verbeux
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    return app


# Pour les tests/développement
app = create_app()

if __name__ == "__main__":
    import os, sys
    allow = os.getenv("HARENA_STANDALONE", "").lower() == "true"
    if not allow:
        print("Standalone server disabled. Use local_app.py (port 8000) or set HARENA_STANDALONE=true")
        sys.exit(0)
    import uvicorn
    uvicorn.run("user_service.main:app", host="0.0.0.0", port=8000, reload=True)
