"""
Main FastAPI application for conversation_service_v3
Architecture basée sur agents LangChain autonomes avec auto-correction
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
# IMPORTANT: load_dotenv() doit être appelé AVANT d'importer settings
load_dotenv()

from .config.settings import settings
from .api.v3.endpoints import conversation

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    logger.info(f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    logger.info(f"Search service URL: {settings.SEARCH_SERVICE_URL}")
    logger.info(f"LLM model: {settings.LLM_MODEL}")
    logger.info(f"Max correction attempts: {settings.MAX_CORRECTION_ATTEMPTS}")

    yield

    logger.info(f"Shutting down {settings.SERVICE_NAME}")

    # Fermer l'orchestrateur si initialisé
    if conversation.orchestrator:
        await conversation.orchestrator.close()


# Créer l'application FastAPI
app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description="""
    Conversation Service v3 - Architecture avec Agents LangChain Autonomes

    ## Nouveautés v3:
    - **Agents autonomes**: Chaque agent (Analyzer, Builder, Generator) utilise LangChain
    - **Auto-correction**: Correction automatique des queries Elasticsearch en cas d'erreur
    - **Connaissance du schéma**: Les agents comprennent la structure Elasticsearch
    - **Pipeline optimisé**: Agrégations + résumé + 50 premières transactions

    ## Pipeline:
    1. **QueryAnalyzerAgent**: Analyse la question et extrait les entités
    2. **ElasticsearchBuilderAgent**: Traduit en query Elasticsearch valide
    3. **Exécution**: Appel à search_service avec retry + correction
    4. **ResponseGeneratorAgent**: Génère une réponse naturelle avec insights

    ## Auto-correction:
    - Si la query échoue, l'agent ElasticsearchBuilder analyse l'erreur
    - Il génère une query corrigée automatiquement
    - Maximum 2 tentatives de correction
    """,
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routes - Le router a déjà le prefix /api/v1/conversation
app.include_router(conversation.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "architecture": "langchain_autonomous_agents",
        "features": [
            "autonomous_agents",
            "auto_correction",
            "elasticsearch_schema_aware",
            "aggregations_support"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
