"""
Module principal du service de recherche.

Ce module initialise et configure le service de recherche de la plateforme Harena,
fournissant des capacités de recherche hybride avancée pour les données financières.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from search_service.api.endpoints import search, health
from config_service.config import settings
from search_service.utils.logging import setup_structured_logging

# Configuration du logging
logger = setup_structured_logging()

# Gestion du cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service de recherche."""
    # Initialisation
    logger.info("Démarrage du service de recherche")
    
    # Initialisation des moteurs de recherche
    from search_service.storage.unified_engine import get_unified_engine
    from search_service.storage.qdrant import init_qdrant
    
    # Initialiser le moteur unifié
    unified_engine = get_unified_engine()
    logger.info(f"Moteur de recherche unifié initialisé avec {unified_engine.primary_engine_type} comme moteur principal")
    
    # Initialiser Qdrant pour la recherche vectorielle (si disponible)
    qdrant_client = await init_qdrant()
    if qdrant_client:
        logger.info("Client Qdrant initialisé pour la recherche vectorielle")
    else:
        logger.warning("Client Qdrant non disponible. La recherche vectorielle sera limitée.")
    
    # Vérification de la configuration DeepSeek
    if settings.DEEPSEEK_API_KEY:
        logger.info(f"DeepSeek configuré avec les modèles: Chat={settings.DEEPSEEK_CHAT_MODEL}, Reasoner={settings.DEEPSEEK_REASONER_MODEL}")
    else:
        logger.warning("Clé API DeepSeek manquante. Le traitement avancé des requêtes sera limité.")
    
    # Planification de l'indexation en arrière-plan (optionnel)
    if settings.ENABLE_BACKGROUND_INDEXING:
        from search_service.utils.indexer import schedule_background_indexing
        import asyncio
        
        # Planifier l'indexation en arrière-plan dans une tâche asyncio
        asyncio.create_task(
            schedule_background_indexing(
                interval_seconds=settings.BACKGROUND_INDEXING_INTERVAL
            )
        )
        logger.info(f"Indexation en arrière-plan planifiée toutes les {settings.BACKGROUND_INDEXING_INTERVAL} secondes")
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("Arrêt du service de recherche")

def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service de recherche."""
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride pour les données financières",
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
    
    # Inclusion des routeurs
    app.include_router(
        search.router,
        prefix="/api/v1/search",
        tags=["search"]
    )
    
    app.include_router(
        health.router,
        prefix="/api/v1/search/health",
        tags=["health"]
    )
    
    return app

# Pour les tests/développement
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_service.main:app", host="0.0.0.0", port=8002, reload=True)