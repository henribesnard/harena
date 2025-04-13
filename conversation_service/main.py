"""
Point d'entrée principal pour le service de conversation.

Ce module initialise et démarre le serveur FastAPI pour le service de conversation.
Il peut être exécuté directement pour le développement ou importé par une application parent.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from conversation_service.api.router import router as api_router
from conversation_service.config.settings import settings
from conversation_service.config.logging import setup_logging
from conversation_service.db.session import engine, Base


def create_app() -> FastAPI:
    """
    Crée et configure l'application FastAPI.
    
    Returns:
        FastAPI: Application FastAPI configurée
    """
    # Configurer le logging
    setup_logging()
    
    # Créer l'application FastAPI
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Service de conversation intelligent pour Harena",
        version="1.0.0",
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
        openapi_url=f"{settings.API_PREFIX}/openapi.json"
    )
    
    # Configurer CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Ajouter les routes API
    app.include_router(api_router, prefix=settings.API_PREFIX)
    
    # Ajouter les événements de démarrage et d'arrêt
    @app.on_event("startup")
    async def startup_event():
        # Créer les tables en développement (en production, utiliser Alembic)
        if settings.DEBUG:
            Base.metadata.create_all(bind=engine)
    
    return app


app = create_app()


if __name__ == "__main__":
    """Exécute l'application en mode développement."""
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS
    )