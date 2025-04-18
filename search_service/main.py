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
from search_service.core.config import settings
from search_service.utils.logging import setup_structured_logging

# Configuration du logging
logger = setup_structured_logging()

# Gestion du cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service de recherche."""
    # Initialisation
    logger.info("Démarrage du service de recherche")
    
    # Initialisation des connexions Elasticsearch et Qdrant
    from search_service.storage.elasticsearch import init_elasticsearch
    from search_service.storage.qdrant import init_qdrant
    
    es_client = await init_elasticsearch()
    qdrant_client = await init_qdrant()
    
    # Vérification de la configuration des embeddings
    from search_service.services.embedding_service import EmbeddingService
    embedding_service = EmbeddingService()
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("Arrêt du service de recherche")
    # Fermeture des connexions
    await es_client.close()

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