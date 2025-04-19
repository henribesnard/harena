# sync_service/main.py
"""
Module principal du service de synchronisation.

Ce module initialise et configure le service de synchronisation de la plateforme Harena,
gérant la récupération et le stockage des données bancaires.
"""
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config_service.config import settings
from sync_service.utils.logging import setup_structured_logging

# Configuration du logging structuré
logger = setup_structured_logging()

# ======== GESTION DU CYCLE DE VIE DE L'APPLICATION ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service de synchronisation."""
    # Initialisation
    logger.info("Démarrage du service de synchronisation")
    
    # Vérification des configurations critiques
    if not settings.BRIDGE_CLIENT_ID or not settings.BRIDGE_CLIENT_SECRET:
        logger.warning("Identifiants Bridge API manquants. Les fonctionnalités de synchronisation ne fonctionneront pas.")
    
    if not settings.BRIDGE_WEBHOOK_SECRET:
        logger.warning("Secret de webhook Bridge non configuré. La vérification des webhooks sera désactivée.")
    
    # Initialisation du stockage vectoriel
    try:
        from sync_service.services.vector_storage import VectorStorageService
        vector_service = VectorStorageService()
        if vector_service.client:
            logger.info("Connexion au stockage vectoriel établie avec succès")
        else:
            logger.warning("Service de stockage vectoriel non initialisé")
    except ImportError:
        logger.warning("Module de stockage vectoriel non disponible")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du stockage vectoriel: {e}")
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("Arrêt du service de synchronisation")


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service de synchronisation."""
    app = FastAPI(
        title="Harena Sync Service",
        description="Service de synchronisation pour les données bancaires via Bridge API",
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
    
    # Importation des routeurs après la création de l'app pour éviter les imports circulaires
    from sync_service.api.endpoints import sync as sync_router
    from sync_service.api.endpoints import webhooks as webhooks_router
    
    # Inclusion des routeurs
    app.include_router(
        sync_router.router,
        prefix="/api/v1/sync", 
        tags=["synchronization"]
    )
    
    app.include_router(
        webhooks_router.router,
        prefix="/webhooks",
        tags=["webhooks"]
    )
    
    # Ajout de l'endpoint de santé
    @app.get("/health")
    async def health_check():
        """Vérification de l'état de santé du service de synchronisation."""
        vector_status = "unknown"
        try:
            from sync_service.services.vector_storage import VectorStorageService
            vector_service = VectorStorageService()
            vector_status = "connected" if vector_service.client else "disconnected"
        except ImportError:
            vector_status = "module_not_available"
        except Exception:
            vector_status = "error"
            
        return {
            "status": "ok",
            "service": "Harena Sync Service",
            "version": "1.0.0",
            "vector_storage": vector_status,
            "bridge_api_configured": bool(settings.BRIDGE_CLIENT_ID and settings.BRIDGE_CLIENT_SECRET),
            "webhooks_verified": bool(settings.BRIDGE_WEBHOOK_SECRET)
        }
    
    return app


# Pour les tests/développement
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sync_service.main:app", host="0.0.0.0", port=8003, reload=True)