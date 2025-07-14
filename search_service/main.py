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
    required_env_vars = ["BONSAI_URL"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"❌ Variables d'environnement manquantes: {', '.join(missing_vars)}")
        logger.error("Le service ne peut pas démarrer sans configuration Elasticsearch/Bonsai.")
        # Marquer l'application comme non initialisée
        app.state.initialization_failed = True
        app.state.elasticsearch_client = None
        yield
        return
    
    # Initialisation des composants
    try:
        # Import dynamique pour éviter les erreurs de démarrage
        logger.info("📦 Import des modules...")
        from search_service.clients.elasticsearch_client import ElasticsearchClient
        from search_service.core import core_manager
        logger.info("✅ Modules importés avec succès")
        
        logger.info("📡 Initialisation du client Elasticsearch/Bonsai...")
        logger.info(f"🔍 Configuration détectée:")
        logger.info(f"   - BONSAI_URL: {'✅ SET' if os.environ.get('BONSAI_URL') else '❌ NOT SET'}")
        logger.info(f"   - ELASTICSEARCH_URL: {'✅ SET' if os.environ.get('ELASTICSEARCH_URL') else '❌ NOT SET'}")
        
        # Initialiser le client Elasticsearch avec l'URL Bonsai
        try:
            elasticsearch_client = ElasticsearchClient()
            logger.info("📊 Client Elasticsearch créé, tentative d'initialisation...")
            
            await elasticsearch_client.initialize()
            logger.info("✅ Client Elasticsearch/Bonsai initialisé et connecté")
            
            # Test de connectivité
            logger.info("🩺 Test de connectivité Elasticsearch...")
            health = await elasticsearch_client.health_check()
            logger.info(f"📊 Santé Elasticsearch: {health}")
            
        except Exception as es_error:
            logger.error(f"❌ Erreur initialisation client Elasticsearch: {str(es_error)}")
            logger.error(f"📋 Type d'erreur: {type(es_error).__name__}")
            import traceback
            logger.error(f"📄 Trace complète:\n{traceback.format_exc()}")
            raise
        
        # Initialiser le core manager avec le client ES
        logger.info("🔧 Initialisation du core manager...")
        try:
            await core_manager.initialize(elasticsearch_client)
            logger.info("✅ Core manager initialisé avec succès")
            
            # Vérification état du core manager
            logger.info("🔍 Vérification état core manager...")
            is_init = core_manager.is_initialized()
            logger.info(f"📊 Core manager initialisé: {is_init}")
            
        except Exception as core_error:
            logger.error(f"❌ Erreur initialisation core manager: {str(core_error)}")
            logger.error(f"📋 Type d'erreur: {type(core_error).__name__}")
            import traceback
            logger.error(f"📄 Trace complète:\n{traceback.format_exc()}")
            raise
        
        # Stocker le client dans l'app state pour le cleanup
        app.state.elasticsearch_client = elasticsearch_client
        app.state.initialization_failed = False
        
        # Effectuer un test de santé initial
        logger.info("🏥 Test de santé initial...")
        health_status = await core_manager.health_check()
        if health_status.get("status") == "healthy":
            logger.info("✅ Service de recherche opérationnel")
        else:
            logger.warning(f"⚠️ Service en mode dégradé: {health_status.get('error', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE lors de l'initialisation")
        logger.error(f"📋 Type d'erreur: {type(e).__name__}")
        logger.error(f"📝 Message d'erreur: {str(e)}")
        
        # Log détaillé des variables d'environnement pour debug
        logger.error("🔍 DIAGNOSTIC ENVIRONNEMENT:")
        logger.error(f"   - BONSAI_URL présent: {bool(os.environ.get('BONSAI_URL'))}")
        logger.error(f"   - ELASTICSEARCH_URL présent: {bool(os.environ.get('ELASTICSEARCH_URL'))}")
        
        if os.environ.get('BONSAI_URL'):
            bonsai_url = os.environ.get('BONSAI_URL')
            logger.error(f"   - BONSAI_URL format: {bonsai_url[:20]}..." if len(bonsai_url) > 20 else f"   - BONSAI_URL: {bonsai_url}")
        
        # Stack trace complète pour debugging
        import traceback
        logger.error(f"📄 STACK TRACE COMPLÈTE:\n{traceback.format_exc()}")
        
        logger.error("🚨 Le service démarrera en mode dégradé")
        # Marquer l'échec d'initialisation
        app.state.initialization_failed = True
        app.state.elasticsearch_client = None
        app.state.initialization_error = str(e)
    
    yield  # L'application s'exécute ici
    
    # Cleanup code
    logger.info("🛑 Search Service en arrêt...")
    
    # Nettoyage propre
    try:
        if hasattr(app.state, 'elasticsearch_client') and app.state.elasticsearch_client:
            if hasattr(app.state.elasticsearch_client, 'close'):
                await app.state.elasticsearch_client.close()
                logger.info("✅ Client Elasticsearch fermé proprement")
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
    
    # Ajout de l'endpoint de santé amélioré
    @app.get("/health")
    async def health_check():
        """Vérification de l'état de santé du service de recherche."""
        try:
            # Vérifier si l'initialisation a échoué
            if getattr(app.state, 'initialization_failed', True):
                error_details = {
                    "status": "unhealthy",
                    "service": "search-service",
                    "version": "1.0.0",
                    "error": "Service initialization failed",
                    "bonsai_configured": bool(os.environ.get("BONSAI_URL")),
                    "elasticsearch_configured": bool(os.environ.get("ELASTICSEARCH_URL")),
                    "initialization_error": getattr(app.state, 'initialization_error', 'Unknown error')
                }
                
                # Log détaillé pour debugging
                logger.error("🏥 HEALTH CHECK - Initialization failed:")
                logger.error(f"   - BONSAI_URL: {'SET' if error_details['bonsai_configured'] else 'NOT SET'}")
                logger.error(f"   - ELASTICSEARCH_URL: {'SET' if error_details['elasticsearch_configured'] else 'NOT SET'}")
                logger.error(f"   - Error: {error_details['initialization_error']}")
                
                return error_details
            
            # Import dynamique pour éviter les erreurs de démarrage
            from search_service.core import core_manager
            
            # Vérifier si le core manager est initialisé
            if not core_manager.is_initialized():
                return {
                    "status": "unhealthy",
                    "service": "search-service",
                    "version": "1.0.0",
                    "error": "Core manager not initialized",
                    "bonsai_configured": bool(os.environ.get("BONSAI_URL")),
                    "elasticsearch_configured": bool(os.environ.get("ELASTICSEARCH_URL"))
                }
            
            # Effectuer le health check complet
            health_status = await core_manager.health_check()
            
            return {
                "status": "healthy" if health_status.get("status") == "healthy" else "unhealthy",
                "service": "search-service",
                "version": "1.0.0",
                "components": health_status.get("components", []),
                "bonsai_configured": bool(os.environ.get("BONSAI_URL")),
                "elasticsearch_configured": bool(os.environ.get("ELASTICSEARCH_URL")),
                "uptime_seconds": health_status.get("uptime_seconds", 0),
                "initialization_status": "success" if not getattr(app.state, 'initialization_failed', True) else "failed"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "search-service",
                "version": "1.0.0",
                "error": str(e),
                "bonsai_configured": bool(os.environ.get("BONSAI_URL")),
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