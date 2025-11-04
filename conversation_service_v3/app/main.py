"""
Main FastAPI application for conversation_service_v3
Architecture basée sur agents LangChain autonomes avec auto-correction
"""
import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
# IMPORTANT: load_dotenv() doit être appelé AVANT d'importer settings
load_dotenv()

from .config.settings import settings
from .api.v3.endpoints import conversation
from .api.middleware.auth_middleware import JWTAuthMiddleware

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")

    # Mode strict pour validation de configuration (défaut: False)
    STRICT_CONFIG_CHECK = os.getenv("STRICT_CONFIG_CHECK", "false").lower() == "true"

    # Vérification des configurations critiques
    config_issues = []

    if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
        config_issues.append("SECRET_KEY non définie ou trop courte (min 32 caractères)")

    if not settings.SEARCH_SERVICE_URL:
        config_issues.append("SEARCH_SERVICE_URL non définie")

    # Vérifier les clés API LLM
    if settings.LLM_PRIMARY_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        config_issues.append("OPENAI_API_KEY manquante pour provider OpenAI")
    elif settings.LLM_PRIMARY_PROVIDER == "deepseek" and not settings.DEEPSEEK_API_KEY:
        config_issues.append("DEEPSEEK_API_KEY manquante pour provider DeepSeek")

    if config_issues:
        for issue in config_issues:
            logger.warning(f"⚠️ {issue}")

        if STRICT_CONFIG_CHECK:
            error_msg = f"Configuration critique manquante: {', '.join(config_issues)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    logger.info(f"✅ Configuration validée avec succès")
    logger.info(f"🔍 Search service URL: {settings.SEARCH_SERVICE_URL}")
    logger.info(f"🤖 LLM primary: {settings.LLM_PRIMARY_PROVIDER} / {settings.LLM_MODEL}")
    logger.info(f"🔄 Fallback: {settings.LLM_FALLBACK_PROVIDER if settings.LLM_FALLBACK_ENABLED else 'disabled'}")
    logger.info(f"🔧 Max correction attempts: {settings.MAX_CORRECTION_ATTEMPTS}")

    yield

    logger.info(f"🛑 Shutting down {settings.SERVICE_NAME}")

    # Fermer l'orchestrateur si initialisé
    if conversation._orchestrator:
        await conversation._orchestrator.close()
        logger.info("✅ HTTP client and resources properly closed")


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

# IMPORTANT: JWT Middleware ajouté EN PREMIER (s'exécute en dernier)
app.add_middleware(JWTAuthMiddleware)
logger.info("JWT middleware configured - user_service compatible")

# Configuration CORS - Activée en dev, ajoutée EN DERNIER (s'exécute en premier)
# Les middlewares FastAPI sont exécutés dans l'ordre INVERSE de leur ajout
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

# Inclure les routes - Le router a déjà le prefix /api/v3/conversation
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
