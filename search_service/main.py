# search_service/main.py
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config_service.config import settings
from .api import router, initialize_search_engine

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Variables globales pour le state de l'application
_service_initialized = False
_initialization_error = None
_elasticsearch_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    global _service_initialized, _initialization_error, _elasticsearch_client
    
    logger.info("🚀 Démarrage du Search Service...")
    logger.info("📋 Configuration: API v1.0.0")
    
    try:
        # Initialisation du client Elasticsearch unifié
        logger.info("📡 Initialisation du client Elasticsearch/Bonsai...")
        
        from .core import initialize_default_client
        elasticsearch_client = await initialize_default_client()
        logger.info("✅ Client Elasticsearch/Bonsai initialisé")
        
        # Initialisation du moteur de recherche
        logger.info("🔍 Initialisation du moteur de recherche...")
        initialize_search_engine(elasticsearch_client)
        
        # Test de connectivité
        logger.info("🩺 Test de connectivité...")
        health = await elasticsearch_client.health_check()
        logger.info(f"📊 Santé Elasticsearch: {health}")
        
        # Marquer le service comme initialisé
        _service_initialized = True
        _initialization_error = None
        _elasticsearch_client = elasticsearch_client
        
        # State pour les routes
        app.state.service_initialized = True
        app.state.elasticsearch_client = elasticsearch_client
        
        logger.info("🎉 Search Service initialisé avec succès!")
        logger.info("📋 Composants initialisés:")
        logger.info("   ✅ Client Elasticsearch/Bonsai")
        logger.info("   ✅ Moteur de recherche")
        logger.info("   ✅ Routes API")
        
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE lors de l'initialisation")
        logger.error(f"📋 Type d'erreur: {type(e).__name__}")
        logger.error(f"📝 Message d'erreur: {str(e)}")
        
        # Log détaillé pour debugging
        import traceback
        logger.error(f"📄 STACK TRACE:\n{traceback.format_exc()}")
        
        logger.error("🚨 Le service démarrera en mode dégradé")
        
        # Marquer l'échec d'initialisation
        _service_initialized = False
        _initialization_error = str(e)
        _elasticsearch_client = None
        
        # App state pour compatibilité
        app.state.service_initialized = False
        app.state.elasticsearch_client = None
        app.state.initialization_error = str(e)
    
    yield  # L'application s'exécute ici
    
    # Cleanup code
    logger.info("🛑 Search Service en arrêt...")
    
    try:
        if _elasticsearch_client:
            from .core import shutdown_default_client
            await shutdown_default_client()
            logger.info("✅ Client Elasticsearch fermé proprement")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage: {str(e)}")

def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI"""
    
    app = FastAPI(
        title="Search Service API",
        version="1.0.0",
        description="API de recherche pour transactions et comptes",
        lifespan=lifespan
    )
    
    # Middleware CORS - Désactivée car gérée par Nginx
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=["*"],  # À configurer selon vos besoins
    #     allow_credentials=True,
    #     allow_methods=["*"],
    #     allow_headers=["*"],
    # )
    
    # Routes principales avec préfixe API v1
    app.include_router(router, prefix="/api/v1/search")
    
    # Route de santé globale
    @app.get("/")
    async def root():
        return {
            "service": "search_service",
            "version": "1.0.0",
            "status": "running",
            "initialized": _service_initialized,
            "error": _initialization_error
        }

    # Health check endpoint for Docker
    @app.get("/health")
    async def health():
        """Vérification de l'état de santé du search service avec check DB."""
        from db_service.health import check_database_health
        from fastapi import status
        from fastapi.responses import JSONResponse

        # Vérifier la connexion à la base de données
        db_healthy, db_message = check_database_health()

        # Vérifier si le service Elasticsearch est initialisé
        es_healthy = _service_initialized and _elasticsearch_client is not None

        # Le service est healthy si DB ET Elasticsearch sont OK
        overall_healthy = db_healthy and es_healthy

        # Préparer la réponse
        health_status = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "service": "search_service",
            "version": "1.0.0",
            "database": {
                "healthy": db_healthy,
                "message": db_message
            },
            "elasticsearch": {
                "initialized": es_healthy,
                "error": _initialization_error
            }
        }

        # Retourner 503 si un composant n'est pas accessible
        if not overall_healthy:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_status
            )

        return health_status

    return app

# Instance de l'application
app = create_app()

if __name__ == "__main__":
    import os, sys
    allow = os.getenv("HARENA_STANDALONE", "").lower() == "true"
    if not allow:
        print("Standalone server disabled. Use local_app.py (port 8000) or set HARENA_STANDALONE=true")
        sys.exit(0)
    import uvicorn
    uvicorn.run(
        "search_service.main:app",
        host="0.0.0.0",
        port=8005,
        reload=False,
        log_level="info"
    )
