"""
Module principal du service de recherche avec logging amélioré.

Ce module initialise et configure le service de recherche hybride de Harena,
combinant recherche lexicale, sémantique et reranking.
"""
import logging
import time
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from search_service.api.routes import router
from search_service.core.embeddings import embedding_service
from search_service.core.reranker import reranker_service
from search_service.storage.elastic_client import ElasticClient
from search_service.storage.qdrant_client import QdrantClient
from search_service.utils.cache import SearchCache
from search_service.utils.metrics import MetricsCollector
from config_service.config import settings

# Configuration du logging amélioré
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("search_service")

# Réduire le bruit des libs externes
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Instances globales
elastic_client = None
qdrant_client = None
search_cache = None
metrics_collector = None
startup_time = None


def log_configuration_status():
    """Log détaillé de l'état des configurations."""
    logger.info("🔧 === VÉRIFICATION DES CONFIGURATIONS ===")
    
    config_status = []
    
    # Elasticsearch
    if settings.SEARCHBOX_URL:
        logger.info("✅ ELASTICSEARCH: SearchBox URL configurée")
        # Masquer les credentials
        safe_url = settings.SEARCHBOX_URL.split('@')[-1] if '@' in settings.SEARCHBOX_URL else settings.SEARCHBOX_URL
        logger.info(f"📡 SearchBox endpoint: {safe_url}")
        config_status.append(("elasticsearch", "searchbox", True))
    elif settings.BONSAI_URL:
        logger.info("✅ ELASTICSEARCH: Bonsai URL configurée")
        safe_url = settings.BONSAI_URL.split('@')[-1] if '@' in settings.BONSAI_URL else settings.BONSAI_URL
        logger.info(f"📡 Bonsai endpoint: {safe_url}")
        config_status.append(("elasticsearch", "bonsai", True))
    else:
        logger.error("❌ ELASTICSEARCH: Aucune URL configurée (SEARCHBOX_URL/BONSAI_URL)")
        logger.error("   La recherche lexicale sera INDISPONIBLE")
        config_status.append(("elasticsearch", "none", False))
    
    # Qdrant
    if settings.QDRANT_URL:
        logger.info("✅ QDRANT: URL configurée")
        logger.info(f"📡 Qdrant endpoint: {settings.QDRANT_URL}")
        if settings.QDRANT_API_KEY:
            logger.info("🔑 Qdrant: API Key configurée")
        else:
            logger.info("🔓 Qdrant: Connexion sans API Key")
        config_status.append(("qdrant", "configured", True))
    else:
        logger.error("❌ QDRANT: URL non configurée (QDRANT_URL)")
        logger.error("   La recherche sémantique sera INDISPONIBLE")
        config_status.append(("qdrant", "none", False))
    
    # Services externes
    if settings.COHERE_KEY:
        logger.info("✅ COHERE: Clé API configurée (reranking disponible)")
        config_status.append(("cohere", "configured", True))
    else:
        logger.warning("⚠️ COHERE: Clé non configurée (reranking indisponible)")
        config_status.append(("cohere", "none", False))
    
    if settings.OPENAI_API_KEY:
        logger.info("✅ OPENAI: Clé API configurée (embeddings disponibles)")
        config_status.append(("openai", "configured", True))
    else:
        logger.warning("⚠️ OPENAI: Clé non configurée (embeddings indisponibles)")
        config_status.append(("openai", "none", False))
    
    # Résumé
    critical_services = [status for status in config_status 
                        if status[0] in ["elasticsearch", "qdrant"] and status[2]]
    
    if len(critical_services) == 2:
        logger.info("🎉 Configuration: Tous les services critiques configurés")
    elif len(critical_services) == 1:
        logger.warning("⚠️ Configuration: Service de recherche PARTIELLEMENT configuré")
    else:
        logger.error("🚨 Configuration: AUCUN service de recherche configuré")
    
    return config_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service de recherche avec diagnostic complet."""
    global elastic_client, qdrant_client, search_cache, metrics_collector, startup_time
    
    # =============================================================================
    # PHASE D'INITIALISATION
    # =============================================================================
    
    logger.info("🚀 === DÉMARRAGE DU SERVICE DE RECHERCHE HYBRIDE ===")
    startup_time = time.time()
    
    # Log de la configuration
    config_status = log_configuration_status()
    
    # =============================================================================
    # INITIALISATION DES SERVICES EXTERNES
    # =============================================================================
    
    logger.info("🤖 === INITIALISATION DES SERVICES EXTERNES ===")
    
    # Service d'embeddings
    embedding_success = False
    try:
        logger.info("🤖 Initialisation du service d'embeddings...")
        start_time = time.time()
        await embedding_service.initialize()
        init_time = time.time() - start_time
        logger.info(f"✅ Service d'embeddings initialisé en {init_time:.2f}s")
        embedding_success = True
    except Exception as e:
        logger.error(f"❌ Échec initialisation embeddings: {type(e).__name__}: {str(e)}")
        logger.error("📍 Vérifiez OPENAI_API_KEY et la connectivité internet")
        logger.debug("Détails de l'erreur", exc_info=True)
    
    # Service de reranking
    reranking_success = False
    try:
        logger.info("🎯 Initialisation du service de reranking...")
        start_time = time.time()
        await reranker_service.initialize()
        init_time = time.time() - start_time
        logger.info(f"✅ Service de reranking initialisé en {init_time:.2f}s")
        reranking_success = True
    except Exception as e:
        logger.error(f"❌ Échec initialisation reranking: {type(e).__name__}: {str(e)}")
        logger.error("📍 Vérifiez COHERE_KEY et la connectivité internet")
        logger.debug("Détails de l'erreur", exc_info=True)
    
    # =============================================================================
    # INITIALISATION DES CLIENTS DE STOCKAGE
    # =============================================================================
    
    logger.info("💾 === INITIALISATION DES CLIENTS DE STOCKAGE ===")
    
    # Client Elasticsearch
    elastic_success = False
    try:
        logger.info("🔍 Initialisation du client Elasticsearch...")
        start_time = time.time()
        
        elastic_client = ElasticClient()
        
        # Test de connexion détaillé
        logger.info("🔗 Test de connexion Elasticsearch...")
        await elastic_client.initialize()
        
        if elastic_client._initialized:
            init_time = time.time() - start_time
            logger.info(f"✅ Client Elasticsearch opérationnel en {init_time:.2f}s")
            
            # Test de santé
            health_start = time.time()
            is_healthy = await elastic_client.is_healthy()
            health_time = time.time() - health_start
            
            if is_healthy:
                logger.info(f"🩺 Elasticsearch en bonne santé (ping: {health_time:.3f}s)")
                elastic_success = True
            else:
                logger.error("🚨 Elasticsearch répond mais n'est pas en bonne santé")
        else:
            logger.error("❌ Client Elasticsearch initialisé mais non opérationnel")
            
    except Exception as e:
        init_time = time.time() - start_time
        logger.error(f"💥 Erreur critique Elasticsearch après {init_time:.2f}s:")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error(f"   Message: {str(e)}")
        
        # Diagnostic spécifique
        if "connection" in str(e).lower():
            logger.error("🔌 DIAGNOSTIC: Problème de connexion réseau")
            logger.error("   - Vérifiez l'URL Elasticsearch")
            logger.error("   - Testez la connectivité réseau")
            logger.error("   - Vérifiez les credentials")
        elif "auth" in str(e).lower() or "401" in str(e):
            logger.error("🔑 DIAGNOSTIC: Problème d'authentification")
            logger.error("   - Vérifiez les credentials dans l'URL")
            logger.error("   - Vérifiez les permissions du compte")
        elif "timeout" in str(e).lower():
            logger.error("⏱️ DIAGNOSTIC: Timeout de connexion")
            logger.error("   - Le service Elasticsearch peut être surchargé")
            logger.error("   - Augmentez le timeout de connexion")
        
        logger.debug("Trace complète de l'erreur", exc_info=True)
        elastic_client = None
    
    # Client Qdrant
    qdrant_success = False
    try:
        logger.info("🎯 Initialisation du client Qdrant...")
        start_time = time.time()
        
        qdrant_client = QdrantClient()
        
        # Test de connexion détaillé
        logger.info("🔗 Test de connexion Qdrant...")
        await qdrant_client.initialize()
        
        if qdrant_client._initialized:
            init_time = time.time() - start_time
            logger.info(f"✅ Client Qdrant opérationnel en {init_time:.2f}s")
            
            # Test de santé
            health_start = time.time()
            is_healthy = await qdrant_client.is_healthy()
            health_time = time.time() - health_start
            
            if is_healthy:
                logger.info(f"🩺 Qdrant en bonne santé (ping: {health_time:.3f}s)")
                qdrant_success = True
            else:
                logger.error("🚨 Qdrant répond mais n'est pas en bonne santé")
        else:
            logger.error("❌ Client Qdrant initialisé mais non opérationnel")
            
    except Exception as e:
        init_time = time.time() - start_time
        logger.error(f"💥 Erreur critique Qdrant après {init_time:.2f}s:")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error(f"   Message: {str(e)}")
        
        # Diagnostic spécifique
        if "connection" in str(e).lower():
            logger.error("🔌 DIAGNOSTIC: Problème de connexion réseau")
            logger.error("   - Vérifiez l'URL Qdrant")
            logger.error("   - Testez la connectivité réseau")
        elif "401" in str(e) or "auth" in str(e).lower():
            logger.error("🔑 DIAGNOSTIC: Problème d'authentification")
            logger.error("   - Vérifiez QDRANT_API_KEY")
        elif "404" in str(e):
            logger.error("📂 DIAGNOSTIC: Collection non trouvée")
            logger.error("   - La collection 'financial_transactions' doit être créée")
            logger.error("   - Lancez d'abord l'enrichment_service")
        elif "timeout" in str(e).lower():
            logger.error("⏱️ DIAGNOSTIC: Timeout de connexion")
            logger.error("   - Le service Qdrant peut être indisponible")
        
        logger.debug("Trace complète de l'erreur", exc_info=True)
        qdrant_client = None
    
    # =============================================================================
    # INITIALISATION DU CACHE ET MÉTRIQUES
    # =============================================================================
    
    logger.info("🗃️ === INITIALISATION CACHE ET MÉTRIQUES ===")
    
    try:
        search_cache = SearchCache()
        logger.info("✅ Cache de recherche initialisé")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation cache: {e}")
        search_cache = None
    
    try:
        metrics_collector = MetricsCollector()
        logger.info("✅ Collecteur de métriques initialisé")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation métriques: {e}")
        metrics_collector = None
    
    # =============================================================================
    # INJECTION DES DÉPENDANCES
    # =============================================================================
    
    logger.info("🔗 === INJECTION DES DÉPENDANCES ===")
    
    # Injecter les instances dans le module routes
    import search_service.api.routes as routes
    routes.elastic_client = elastic_client
    routes.qdrant_client = qdrant_client
    routes.search_cache = search_cache
    routes.metrics_collector = metrics_collector
    
    logger.info("✅ Dépendances injectées dans les routes")
    
    # =============================================================================
    # RÉSUMÉ DE DÉMARRAGE
    # =============================================================================
    
    startup_duration = time.time() - startup_time
    
    # Compter les services opérationnels
    operational_services = []
    if elastic_success:
        operational_services.append("Elasticsearch")
    if qdrant_success:
        operational_services.append("Qdrant")
    if embedding_success:
        operational_services.append("Embeddings")
    if reranking_success:
        operational_services.append("Reranking")
    
    # Déterminer l'état global
    if elastic_success and qdrant_success:
        status_icon = "🎉"
        status_msg = "OPÉRATIONNEL COMPLET"
        service_level = "FULL"
    elif elastic_success or qdrant_success:
        status_icon = "⚠️"
        status_msg = "OPÉRATIONNEL PARTIEL"
        service_level = "PARTIAL"
    else:
        status_icon = "🚨"
        status_msg = "NON OPÉRATIONNEL"
        service_level = "FAILED"
    
    logger.info("=" * 80)
    logger.info(f"{status_icon} SERVICE DE RECHERCHE - ÉTAT: {status_msg}")
    logger.info(f"⏱️ Temps de démarrage: {startup_duration:.2f}s")
    logger.info(f"🎯 Services opérationnels ({len(operational_services)}/4): {', '.join(operational_services) if operational_services else 'Aucun'}")
    
    if service_level == "FULL":
        logger.info("✨ Toutes les fonctionnalités de recherche sont disponibles")
    elif service_level == "PARTIAL":
        logger.info("⚠️ Fonctionnalités limitées - vérifiez les erreurs ci-dessus")
    else:
        logger.info("🚨 Service de recherche indisponible - vérifiez la configuration")
    
    logger.info("📊 Health check disponible sur: GET /health")
    logger.info("🔍 API de recherche disponible sur: POST /api/v1/search")
    logger.info("=" * 80)
    
    # =============================================================================
    # EXÉCUTION DE L'APPLICATION
    # =============================================================================
    
    yield  # L'application s'exécute ici
    
    # =============================================================================
    # PHASE D'ARRÊT
    # =============================================================================
    
    logger.info("🛑 === ARRÊT DU SERVICE DE RECHERCHE ===")
    shutdown_start = time.time()
    
    # Fermer les connexions
    if elastic_client:
        logger.info("🔍 Fermeture connexion Elasticsearch...")
        try:
            await elastic_client.close()
            logger.info("✅ Elasticsearch fermé proprement")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Elasticsearch: {e}")
    
    if qdrant_client:
        logger.info("🎯 Fermeture connexion Qdrant...")
        try:
            await qdrant_client.close()
            logger.info("✅ Qdrant fermé proprement")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Qdrant: {e}")
    
    # Arrêter les services
    if embedding_service:
        logger.info("🤖 Arrêt service embeddings...")
        try:
            await embedding_service.close()
            logger.info("✅ Service embeddings arrêté")
        except Exception as e:
            logger.error(f"❌ Erreur arrêt embeddings: {e}")
    
    if reranker_service:
        logger.info("🎯 Arrêt service reranking...")
        try:
            await reranker_service.close()
            logger.info("✅ Service reranking arrêté")
        except Exception as e:
            logger.error(f"❌ Erreur arrêt reranking: {e}")
    
    shutdown_duration = time.time() - shutdown_start
    total_uptime = time.time() - startup_time
    
    logger.info("=" * 60)
    logger.info("✅ ARRÊT PROPRE TERMINÉ")
    logger.info(f"⏱️ Temps d'arrêt: {shutdown_duration:.2f}s")
    logger.info(f"⏱️ Uptime total: {total_uptime:.2f}s")
    logger.info("=" * 60)


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI."""
    
    logger.info("🏗️ Création de l'application FastAPI...")
    
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride pour les transactions financières",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configuration CORS
    logger.info("🌐 Configuration CORS...")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware de métriques amélioré
    @app.middleware("http")
    async def enhanced_metrics_middleware(request, call_next):
        """Middleware de métriques avec logging amélioré."""
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        # Log des requêtes importantes
        if path.startswith("/api/v1/search"):
            logger.info(f"🔍 {method} {path} - Début de traitement")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log des résultats
            if path.startswith("/api/v1/search"):
                status_icon = "✅" if response.status_code < 400 else "❌"
                logger.info(f"{status_icon} {method} {path} - {response.status_code} en {process_time:.3f}s")
            
            # Enregistrer les métriques
            if metrics_collector:
                metrics_collector.record_request(
                    path=path,
                    method=method,
                    status_code=response.status_code,
                    duration=process_time
                )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"💥 {method} {path} - Erreur après {process_time:.3f}s: {type(e).__name__}: {str(e)}")
            raise
    
    # Enregistrement des routes
    logger.info("🛣️ Configuration des routes...")
    app.include_router(router, prefix="/api/v1", tags=["search"])
    
    # Endpoint de santé racine
    @app.get("/")
    def root():
        """Endpoint racine pour vérification rapide."""
        return {
            "service": "search_service",
            "version": "1.0.0",
            "status": "running",
            "timestamp": time.time()
        }
    
    # Endpoint de santé détaillé
    @app.get("/health")
    async def health_check():
        """Vérification détaillée de l'état du service."""
        logger.debug("🩺 Health check demandé")
        
        try:
            # Vérifier l'état des composants
            elasticsearch_ok = elastic_client is not None and await elastic_client.is_healthy()
            qdrant_ok = qdrant_client is not None and await qdrant_client.is_healthy()
            
            # État des services externes
            embedding_ok = embedding_service is not None and hasattr(embedding_service, 'client')
            reranking_ok = reranker_service is not None and hasattr(reranker_service, 'client')
            
            # Calculer l'état global
            all_ok = elasticsearch_ok and qdrant_ok and embedding_ok and reranking_ok
            some_ok = elasticsearch_ok or qdrant_ok
            
            if all_ok:
                status = "healthy"
            elif some_ok:
                status = "degraded"
            else:
                status = "unhealthy"
            
            # Calculer l'uptime
            uptime = time.time() - startup_time if startup_time else 0
            
            health_data = {
                "service": "search_service",
                "version": "1.0.0",
                "status": status,
                "timestamp": time.time(),
                "uptime_seconds": uptime,
                "components": {
                    "elasticsearch": {
                        "available": elastic_client is not None,
                        "healthy": elasticsearch_ok,
                        "status": "ok" if elasticsearch_ok else "error"
                    },
                    "qdrant": {
                        "available": qdrant_client is not None,
                        "healthy": qdrant_ok,
                        "status": "ok" if qdrant_ok else "error"
                    },
                    "embeddings": {
                        "available": embedding_service is not None,
                        "initialized": embedding_ok,
                        "status": "ok" if embedding_ok else "error"
                    },
                    "reranking": {
                        "available": reranker_service is not None,
                        "initialized": reranking_ok,
                        "status": "ok" if reranking_ok else "error"
                    }
                }
            }
            
            logger.debug(f"🩺 Health check: {status}")
            return health_data
            
        except Exception as e:
            logger.error(f"❌ Erreur dans health check: {e}")
            return {
                "service": "search_service",
                "version": "1.0.0", 
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    logger.info("✅ Application FastAPI configurée")
    return app


# Création de l'application
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Démarrage du serveur uvicorn...")
    
    try:
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=8004, 
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"💥 Erreur fatale du serveur: {e}")
        sys.exit(1)