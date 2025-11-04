"""
Module principal du service de synchronisation.

Ce module initialise et configure le service de synchronisation de la plateforme Harena,
gérant la collecte et le stockage des données financières depuis Bridge API.
"""
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from sync_service.api.router import register_routers, setup_middleware
from config_service.config import settings
from sync_service.utils.logging import setup_logging

# Configuration du logging
setup_logging(
    level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE if settings.LOG_TO_FILE else None
)

logger = logging.getLogger("sync_service")

# ======== GESTION DU CYCLE DE VIE DE L'APPLICATION ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service de synchronisation."""
    # Initialisation
    logger.info("Démarrage du service de synchronisation")

    # Mode strict pour validation de configuration (défaut: False)
    STRICT_CONFIG_CHECK = os.getenv("STRICT_CONFIG_CHECK", "false").lower() == "true"

    # Vérification des configurations critiques
    if not settings.BRIDGE_CLIENT_ID or not settings.BRIDGE_CLIENT_SECRET:
        error_msg = "Identifiants Bridge API manquants (BRIDGE_CLIENT_ID et BRIDGE_CLIENT_SECRET)"
        if STRICT_CONFIG_CHECK:
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.warning(f"{error_msg}. Les fonctionnalités de synchronisation ne fonctionneront pas.")

    if not settings.BRIDGE_WEBHOOK_SECRET:
        error_msg = "Secret webhook Bridge non défini (BRIDGE_WEBHOOK_SECRET)"
        if STRICT_CONFIG_CHECK:
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.warning(f"{error_msg}. La validation des webhooks sera désactivée.")

    logger.info("Configuration validée avec succès")

    yield  # L'application s'exécute ici

    # Nettoyage
    logger.info("Arrêt du service de synchronisation")


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service de synchronisation."""
    app = FastAPI(
        title="Harena Sync Service",
        description="Service de synchronisation et stockage des données bancaires pour Harena",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Enregistrer les routeurs API
    register_routers(app)
    
    # Configurer les middlewares spécifiques
    setup_middleware(app)
    
    # Ajout de l'endpoint de santé
    @app.get("/health")
    def health_check():
        """Vérification de l'état de santé du service de synchronisation avec check DB."""
        from db_service.health import check_database_health
        from fastapi import status
        from fastapi.responses import JSONResponse

        # Vérifier la connexion à la base de données
        db_healthy, db_message = check_database_health()

        # Préparer la réponse
        health_status = {
            "status": "healthy" if db_healthy else "unhealthy",
            "service": "sync_service",
            "version": "1.0.0",
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
    uvicorn.run("sync_service.main:app", host="0.0.0.0", port=8002, reload=True)
