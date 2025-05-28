"""
Module principal du service d'enrichissement.

Ce module initialise et configure le service d'enrichissement de la plateforme Harena,
responsable de la structuration et du stockage vectoriel des données financières.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from enrichment_service.api.routes import router
from enrichment_service.storage.qdrant import QdrantStorage
from config_service.config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
)
logger = logging.getLogger("enrichment_service")

# Instance globale du storage Qdrant
qdrant_storage = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service d'enrichissement."""
    global qdrant_storage
    
    # Initialisation
    logger.info("Démarrage du service d'enrichissement")
    
    # Vérification des configurations critiques
    if not settings.QDRANT_URL:
        logger.warning("QDRANT_URL non définie. Le stockage vectoriel ne fonctionnera pas.")
    
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY non définie. La génération d'embeddings ne fonctionnera pas.")
    
    # Initialisation du storage Qdrant
    try:
        qdrant_storage = QdrantStorage()
        await qdrant_storage.initialize()
        logger.info("Qdrant storage initialisé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de Qdrant: {e}")
        qdrant_storage = None
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    if qdrant_storage:
        await qdrant_storage.close()
    logger.info("Arrêt du service d'enrichissement")

def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service d'enrichissement."""
    app = FastAPI(
        title="Harena Enrichment Service",
        description="Service d'enrichissement et de stockage vectoriel des données financières",
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
    
    # Enregistrement des routes
    app.include_router(router, prefix="/api/v1", tags=["enrichment"])
    
    # Endpoint de santé
    @app.get("/health")
    def health_check():
        """Vérification de l'état de santé du service d'enrichissement."""
        return {
            "status": "ok",
            "service": "enrichment_service",
            "version": "1.0.0",
            "qdrant_configured": bool(settings.QDRANT_URL),
            "openai_configured": bool(settings.OPENAI_API_KEY),
            "qdrant_ready": qdrant_storage is not None and qdrant_storage.client is not None
        }
    
    return app

# Pour les tests/développement
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)