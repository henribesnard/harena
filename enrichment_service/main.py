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
from enrichment_service.core.processor import TransactionProcessor
from enrichment_service.core.embeddings import embedding_service
from config_service.config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
)
logger = logging.getLogger("enrichment_service")

# Instances globales
qdrant_storage = None
transaction_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service d'enrichissement."""
    global qdrant_storage, transaction_processor
    
    # Initialisation
    logger.info("Démarrage du service d'enrichissement")
    
    # Vérification des configurations critiques
    if not settings.QDRANT_URL:
        logger.warning("QDRANT_URL non définie. Le stockage vectoriel ne fonctionnera pas.")
    
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY non définie. La génération d'embeddings ne fonctionnera pas.")
    
    # Initialisation du service d'embeddings
    try:
        await embedding_service.initialize()
        logger.info("Service d'embeddings initialisé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du service d'embeddings: {e}")
    
    # Initialisation du storage Qdrant
    try:
        qdrant_storage = QdrantStorage()
        await qdrant_storage.initialize()
        logger.info("Qdrant storage initialisé avec succès")
        
        # Créer le transaction processor
        transaction_processor = TransactionProcessor(qdrant_storage)
        logger.info("Transaction processor créé avec succès")
        
        # Injecter les instances dans le module routes
        import enrichment_service.api.routes as routes
        routes.qdrant_storage = qdrant_storage
        routes.transaction_processor = transaction_processor
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de Qdrant: {e}")
        qdrant_storage = None
        transaction_processor = None
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    if qdrant_storage:
        await qdrant_storage.close()
    if embedding_service:
        await embedding_service.close()
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
        qdrant_ready = qdrant_storage is not None and hasattr(qdrant_storage, 'client') and qdrant_storage.client is not None
        embedding_ready = embedding_service is not None and hasattr(embedding_service, 'client') and embedding_service.client is not None
        
        return {
            "status": "ok",
            "service": "enrichment_service",
            "version": "1.0.0",
            "qdrant_configured": bool(settings.QDRANT_URL),
            "openai_configured": bool(settings.OPENAI_API_KEY),
            "qdrant_ready": qdrant_ready,
            "embedding_ready": embedding_ready
        }
    
    return app

# Pour les tests/développement
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)