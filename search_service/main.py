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

# Variables globales pour l'état du service
_service_initialized = False
_initialization_error = None
_elasticsearch_client = None

# ======== GESTION DU CYCLE DE VIE DE L'APPLICATION ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application.
    Initialise les ressources au démarrage et les libère à l'arrêt.
    """
    global _service_initialized, _initialization_error, _elasticsearch_client
    
    # Initialization code
    logger.info("🚀 Search Service en démarrage...")
    
    # Vérification des variables d'environnement critiques
    bonsai_url = os.environ.get("BONSAI_URL")
    
    if not bonsai_url:
        logger.error("❌ BONSAI_URL n'est pas configurée")
        logger.error("💡 Veuillez définir BONSAI_URL dans votre fichier .env")
        logger.error("   Exemple: BONSAI_URL=https://your-cluster.eu-west-1.bonsaisearch.net:443")
        _initialization_error = "Variable d'environnement manquante: BONSAI_URL"
        _service_initialized = False
        yield
        return
    
    logger.info(f"🔍 Configuration détectée:")
    logger.info(f"   - BONSAI_URL: ✅ SET")
    logger.info(f"   - URL utilisée: {bonsai_url[:50]}...")
    
    # Initialisation des composants
    try:
        # Import dynamique pour éviter les erreurs de démarrage
        logger.info("📦 Import des modules...")
        from search_service.clients.elasticsearch_client import get_default_client, initialize_default_client
        from search_service.core import core_manager
        logger.info("✅ Modules importés avec succès")
        
        logger.info("📡 Initialisation du client Elasticsearch/Bonsai...")
        
        # ✅ CORRECTION MAJEURE: Plus besoin de passer bonsai_url
        # Le client auto-détecte l'URL depuis BONSAI_URL
        try:
            elasticsearch_client = await initialize_default_client()
            logger.info("✅ Client Elasticsearch/Bonsai initialisé et connecté")
            
            # Test de connectivité
            logger.info("🩺 Test de connectivité Elasticsearch...")
            health = await elasticsearch_client.health_check()
            logger.info(f"📊 Santé Elasticsearch: {health}")
            
            # Test de connexion basique
            connection_test = await elasticsearch_client.test_connection()
            if connection_test:
                logger.info("✅ Test de connexion Elasticsearch réussi")
            else:
                logger.warning("⚠️ Test de connexion Elasticsearch échoué")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du client Elasticsearch: {e}")
            raise
        
        # Initialisation du core manager
        logger.info("🧠 Initialisation du core manager...")
        try:
            await core_manager.initialize(elasticsearch_client)
            
            if core_manager.is_initialized():
                logger.info("✅ Core manager initialisé avec succès")
            else:
                raise RuntimeError("Core manager non initialisé après tentative")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du core manager: {e}")
            raise
        
        # Marquer le service comme initialisé
        _service_initialized = True
        _initialization_error = None
        _elasticsearch_client = elasticsearch_client
        
        # State pour les routes
        app.state.service_initialized = True
        app.state.elasticsearch_client = elasticsearch_client
        app.state.core_manager = core_manager
        
        logger.info("🎉 Search Service initialisé avec succès!")
        logger.info("📋 Composants initialisés:")
        logger.info("   ✅ Client Elasticsearch/Bonsai")
        logger.info("   ✅ Core Manager")
        logger.info("   ✅ Routes API")
        
        # Informations de configuration pour debugging
        try:
            from search_service.clients.elasticsearch_client import get_client_configuration_info
            config_info = get_client_configuration_info()
            logger.info(f"🔧 Configuration utilisée: {config_info}")
        except Exception as e:
            logger.debug(f"Cannot get config info: {e}")
        
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE lors de l'initialisation")
        logger.error(f"📋 Type d'erreur: {type(e).__name__}")
        logger.error(f"📝 Message d'erreur: {str(e)}")
        
        # Log détaillé des variables d'environnement pour debug
        logger.error("🔍 DIAGNOSTIC ENVIRONNEMENT:")
        logger.error(f"   - BONSAI_URL présent: {bool(os.environ.get('BONSAI_URL'))}")
        
        if os.environ.get('BONSAI_URL'):
            bonsai_url = os.environ.get('BONSAI_URL')
            logger.error(f"   - BONSAI_URL format: {bonsai_url[:30]}..." if len(bonsai_url) > 30 else f"   - BONSAI_URL: {bonsai_url}")
        
        # Stack trace complète pour debugging
        import traceback
        logger.error(f"📄 STACK TRACE COMPLÈTE:\n{traceback.format_exc()}")
        
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
    
    # Nettoyage propre
    try:
        if _elasticsearch_client:
            # Utiliser la fonction de shutdown globale
            from search_service.clients.elasticsearch_client import shutdown_default_client
            await shutdown_default_client()
            logger.info("✅ Client Elasticsearch fermé proprement")
        
        # Nettoyage du core manager
        if hasattr(app.state, 'core_manager') and app.state.core_manager:
            try:
                await app.state.core_manager.shutdown()
                logger.info("✅ Core manager fermé proprement")
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors du nettoyage du core manager: {e}")
                
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
    
    # Inclusion des routes de recherche (avec health check)
    app.include_router(router, prefix="/api/v1/search", tags=["search"])
    
    # Réglage du niveau de log pour les modules tiers trop verbeux
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    
    return app


# === FONCTIONS UTILITAIRES POUR STATUS ===

def get_service_status() -> dict:
    """
    Retourne le statut du service pour debugging
    """
    return {
        "service_initialized": _service_initialized,
        "initialization_error": _initialization_error,
        "elasticsearch_client_available": _elasticsearch_client is not None
    }


def is_service_ready() -> bool:
    """
    Vérifie si le service est prêt à traiter des requêtes
    """
    return _service_initialized and _elasticsearch_client is not None


# Pour les tests/développement
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    # Vérifier la configuration avant de démarrer
    if not os.environ.get("BONSAI_URL"):
        print("❌ ERREUR: BONSAI_URL n'est pas configurée")
        print("💡 Veuillez définir BONSAI_URL dans votre fichier .env")
        print("   Exemple: BONSAI_URL=https://your-cluster.eu-west-1.bonsaisearch.net:443")
        exit(1)
    
    print("🚀 Démarrage du Search Service...")
    print(f"🔗 BONSAI_URL configurée: {os.environ.get('BONSAI_URL')[:50]}...")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)