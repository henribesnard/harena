# search_service/main.py
"""
Module principal du service de recherche.

Ce module initialise et configure le service de recherche lexicale de la plateforme Harena,
gérant les requêtes Elasticsearch et la mise en cache des résultats.
"""
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from search_service.api.routes import router

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
)

logger = logging.getLogger("search_service")

# ======== GESTION DU CYCLE DE VIE DE L'APPLICATION ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application.
    Initialise les ressources au démarrage et les libère à l'arrêt.
    """
    # Initialization code
    logger.info("🚀 Search Service en démarrage...")
    
    # Vérification des variables d'environnement critiques
    required_env_vars = ["ELASTICSEARCH_URL"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Variables d'environnement manquantes: {', '.join(missing_vars)}")
        logger.warning("Certaines fonctionnalités peuvent ne pas fonctionner correctement.")
    
    # Initialisation des composants
    try:
        # Import dynamique pour éviter les erreurs de démarrage
        from search_service.clients.elasticsearch_client import ElasticsearchClient
        from search_service.core import core_manager
        
        # Initialiser le client Elasticsearch
        elasticsearch_client = ElasticsearchClient()
        await elasticsearch_client.initialize()
        logger.info("✅ Client Elasticsearch initialisé")
        
        # Initialiser le core manager avec le client ES
        await core_manager.initialize(elasticsearch_client)
        logger.info("✅ Core manager initialisé")
        
        # Stocker le client dans l'app state pour le cleanup
        app.state.elasticsearch_client = elasticsearch_client
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {str(e)}")
        # Ne pas faire planter l'application, juste logger l'erreur
        # L'application démarrera en mode dégradé
        app.state.elasticsearch_client = None
    
    yield  # L'application s'exécute ici
    
    # Cleanup code
    logger.info("🛑 Search Service en arrêt...")
    
    # Nettoyage propre
    try:
        if hasattr(app.state, 'elasticsearch_client') and app.state.elasticsearch_client:
            if hasattr(app.state.elasticsearch_client, 'close'):
                await app.state.elasticsearch_client.close()
                logger.info("✅ Client Elasticsearch fermé")
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage: {str(e)}")


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service de recherche."""
    app = FastAPI(
        title="Search Service",
        openapi_url="/api/v1/search/openapi.json",
        description="API pour la recherche lexicale dans les données financières de Harena",
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
    
    # Inclusion des routes de recherche
    app.include_router(router, prefix="/api/v1/search", tags=["search"])
    
    # Ajout de l'endpoint de santé
    @app.get("/health")
    async def health_check():
        """Vérification de l'état de santé du service de recherche."""
        try:
            # Import dynamique pour éviter les erreurs de démarrage
            from search_service.core import core_manager
            
            # Vérifier si le core manager est initialisé
            if not core_manager.is_initialized():
                return {
                    "status": "unhealthy",
                    "service": "search-service",
                    "version": "1.0.0",
                    "error": "Core manager not initialized",
                    "elasticsearch_configured": bool(os.environ.get("ELASTICSEARCH_URL"))
                }
            
            # Effectuer le health check complet
            health_status = await core_manager.health_check()
            
            return {
                "status": "healthy" if health_status.get("status") == "healthy" else "unhealthy",
                "service": "search-service",
                "version": "1.0.0",
                "components": health_status.get("components", []),
                "elasticsearch_configured": bool(os.environ.get("ELASTICSEARCH_URL"))
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "search-service",
                "version": "1.0.0",
                "error": str(e),
                "elasticsearch_configured": bool(os.environ.get("ELASTICSEARCH_URL"))
            }
    
    # Réglage du niveau de log pour les modules tiers trop verbeux
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    
    return app


# Pour les tests/développement
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)