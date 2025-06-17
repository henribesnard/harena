"""
Module principal amélioré du service de recherche avec logging complet.

Cette version intègre tous les améliorations de logging, monitoring et diagnostic.
"""
import logging
import time
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Imports du service de recherche
from search_service.api.routes import router
from search_service.core.embeddings import embedding_service
from search_service.core.reranker import reranker_service
from search_service.storage.elastic_client import ElasticClient  # Version améliorée
from search_service.storage.qdrant_client import QdrantClient   # Version améliorée
from search_service.utils.cache import SearchCache

# Imports des nouveaux modules de monitoring
from search_service.monitoring.search_monitor import search_monitor
from search_service.monitoring.middleware import setup_middleware, setup_metrics_logging
from search_service.monitoring.diagnostic import setup_diagnostic_routes

from config_service.config import settings

# Configuration du logging principal
def setup_logging():
    """Configure le système de logging complet."""
    
    # Configuration du format de base
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(module)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Configuration du logger racine
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Loggers spécialisés avec niveaux différents
    loggers_config = {
        "search_service": logging.INFO,
        "search_service.elasticsearch": logging.INFO,
        "search_service.qdrant": logging.INFO,
        "search_service.monitoring": logging.INFO,
        "search_service.middleware": logging.INFO,
        "search_service.diagnostic": logging.INFO,
        "search_service.metrics": logging.INFO,
        "search_service.access": logging.INFO,
    }
    
    # Configurer chaque logger
    for logger_name, level in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    # Logger de débogage (seulement si DEBUG activé)
    if settings.DEBUG:
        logging.getLogger("search_service").setLevel(logging.DEBUG)
        logging.getLogger("search_service.elasticsearch").setLevel(logging.DEBUG)
        logging.getLogger("search_service.qdrant").setLevel(logging.DEBUG)
    
    # Configurer les loggers externes (réduire le bruit)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    print("📊 Système de logging configuré")


# Configuration du logging dès l'import
setup_logging()
logger = logging.getLogger("search_service.main")

# Instances globales
elastic_client = None
qdrant_client = None
search_cache = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie amélioré avec monitoring complet."""
    global elastic_client, qdrant_client, search_cache, startup_time
    
    # =============================================================================
    # PHASE D'INITIALISATION
    # =============================================================================
    
    logger.info("🚀 === DÉMARRAGE DU SERVICE DE RECHERCHE HYBRIDE ===")
    startup_time = time.time()
    
    # Configuration des métriques
    setup_metrics_logging()
    
    # Vérification des configurations critiques
    logger.info("🔧 Vérification des configurations...")
    config_issues = []
    
    if not settings.SEARCHBOX_URL and not settings.BONSAI_URL:
        config_issues.append("❌ ELASTICSEARCH: Aucune URL configurée (SEARCHBOX_URL/BONSAI_URL)")
    else:
        logger.info("✅ ELASTICSEARCH: URL configurée")
    
    if not settings.QDRANT_URL:
        config_issues.append("❌ QDRANT: URL non configurée (QDRANT_URL)")
    else:
        logger.info("✅ QDRANT: URL configurée")
    
    if not settings.COHERE_KEY:
        config_issues.append("⚠️ COHERE: Clé non configurée (reranking indisponible)")
    else:
        logger.info("✅ COHERE: Clé configurée")
    
    if not settings.OPENAI_API_KEY:
        config_issues.append("⚠️ OPENAI: Clé non configurée (embeddings indisponibles)")
    else:
        logger.info("✅ OPENAI: Clé configurée")
    
    # Log des problèmes de configuration
    if config_issues:
        logger.warning("⚠️ Problèmes de configuration détectés:")
        for issue in config_issues:
            logger.warning(f"  {issue}")
    
    # =============================================================================
    # INITIALISATION DES SERVICES EXTERNES
    # =============================================================================
    
    logger.info("🔌 Initialisation des services externes...")
    
    # Service d'embeddings
    try:
        logger.info("🤖 Initialisation service d'embeddings...")
        await embedding_service.initialize()
        logger.info("✅ Service d'embeddings prêt")
    except Exception as e:
        logger.error(f"❌ Échec initialisation embeddings: {e}")
        logger.error(f"📍 Détails", exc_info=True)
    
    # Service de reranking
    try:
        logger.info("🎯 Initialisation service de reranking...")
        await reranker_service.initialize()
        logger.info("✅ Service de reranking prêt")
    except Exception as e:
        logger.error(f"❌ Échec initialisation reranking: {e}")
        logger.error(f"📍 Détails", exc_info=True)
    
    # =============================================================================
    # INITIALISATION DES CLIENTS DE STOCKAGE
    # =============================================================================
    
    logger.info("💾 Initialisation des clients de stockage...")
    
    # Client Elasticsearch (avec logging amélioré)
    try:
        logger.info("🔍 Initialisation client Elasticsearch...")
        elastic_client = ElasticClient()  # Version améliorée
        await elastic_client.initialize()
        
        if elastic_client._initialized:
            logger.info("✅ Client Elasticsearch opérationnel")
        else:
            logger.error("❌ Client Elasticsearch non opérationnel")
            
    except Exception as e:
        logger.error(f"💥 Erreur critique Elasticsearch: {e}")
        logger.error(f"📍 Détails", exc_info=True)
        elastic_client = None
    
    # Client Qdrant (avec logging amélioré)
    try:
        logger.info("🎯 Initialisation client Qdrant...")
        qdrant_client = QdrantClient()  # Version améliorée
        await qdrant_client.initialize()
        
        if qdrant_client._initialized:
            logger.info("✅ Client Qdrant opérationnel")
        else:
            logger.error("❌ Client Qdrant non opérationnel")
            
    except Exception as e:
        logger.error(f"💥 Erreur critique Qdrant: {e}")
        logger.error(f"📍 Détails", exc_info=True)
        qdrant_client = None
    
    # =============================================================================
    # INITIALISATION DU CACHE ET MONITORING
    # =============================================================================
    
    logger.info("🗃️ Initialisation cache et monitoring...")
    
    # Cache de recherche
    search_cache = SearchCache()
    logger.info("✅ Cache de recherche initialisé")
    
    # Configuration du monitor avec les clients
    search_monitor.set_clients(
        elasticsearch_client=elastic_client,
        qdrant_client=qdrant_client
    )
    
    # Démarrage du monitoring en arrière-plan
    await search_monitor.start_monitoring()
    logger.info("✅ Monitoring démarré")
    
    # =============================================================================
    # INJECTION DES DÉPENDANCES
    # =============================================================================
    
    logger.info("🔗 Injection des dépendances...")
    
    # Injecter les instances dans le module routes
    import search_service.api.routes as routes
    routes.elastic_client = elastic_client
    routes.qdrant_client = qdrant_client
    routes.search_cache = search_cache
    routes.metrics_collector = search_monitor  # Le monitor fait office de collector
    
    logger.info("✅ Dépendances injectées")
    
    # =============================================================================
    # TESTS DE SANTÉ INITIAUX
    # =============================================================================
    
    logger.info("🩺 Tests de santé initiaux...")
    
    # Test Elasticsearch
    if elastic_client:
        es_healthy = await elastic_client.is_healthy()
        logger.info(f"🔍 Elasticsearch: {'✅ Sain' if es_healthy else '❌ Problème'}")
    else:
        logger.warning("🔍 Elasticsearch: ❌ Non disponible")
    
    # Test Qdrant
    if qdrant_client:
        qdrant_healthy = await qdrant_client.is_healthy()
        logger.info(f"🎯 Qdrant: {'✅ Sain' if qdrant_healthy else '❌ Problème'}")
    else:
        logger.warning("🎯 Qdrant: ❌ Non disponible")
    
    # =============================================================================
    # RÉSUMÉ DE DÉMARRAGE
    # =============================================================================
    
    startup_duration = time.time() - startup_time
    
    # Déterminer l'état global
    services_available = []
    if elastic_client and elastic_client._initialized:
        services_available.append("Elasticsearch")
    if qdrant_client and qdrant_client._initialized:
        services_available.append("Qdrant")
    
    if len(services_available) == 2:
        status_icon = "🎉"
        status_msg = "COMPLET"
    elif len(services_available) == 1:
        status_icon = "⚠️"
        status_msg = "PARTIEL"
    else:
        status_icon = "🚨"
        status_msg = "DÉGRADÉ"
    
    logger.info("=" * 70)
    logger.info(f"{status_icon} SERVICE DE RECHERCHE DÉMARRÉ - ÉTAT: {status_msg}")
    logger.info(f"⏱️ Temps de démarrage: {startup_duration:.2f}s")
    logger.info(f"🔧 Services disponibles: {', '.join(services_available) if services_available else 'Aucun'}")
    logger.info("📊 Monitoring actif: Métriques et alertes disponibles")
    logger.info("🩺 Endpoints de diagnostic: /diagnostic/health")
    logger.info("=" * 70)
    
    # Marquer le temps de démarrage pour le monitor
    search_monitor._start_time = startup_time
    
    # =============================================================================
    # EXÉCUTION DE L'APPLICATION
    # =============================================================================
    
    yield  # L'application s'exécute ici
    
    # =============================================================================
    # PHASE D'ARRÊT
    # =============================================================================
    
    logger.info("🛑 === ARRÊT DU SERVICE DE RECHERCHE ===")
    shutdown_start = time.time()
    
    # Arrêter le monitoring
    logger.info("📊 Arrêt du monitoring...")
    await search_monitor.stop_monitoring()
    
    # Fermer les connexions
    if elastic_client:
        logger.info("🔍 Fermeture connexion Elasticsearch...")
        await elastic_client.close()
    
    if qdrant_client:
        logger.info("🎯 Fermeture connexion Qdrant...")
        await qdrant_client.close()
    
    # Arrêter les services
    if embedding_service:
        logger.info("🤖 Arrêt service embeddings...")
        await embedding_service.close()
    
    if reranker_service:
        logger.info("🎯 Arrêt service reranking...")
        await reranker_service.close()
    
    shutdown_duration = time.time() - shutdown_start
    total_uptime = time.time() - startup_time
    
    logger.info("=" * 50)
    logger.info("✅ ARRÊT PROPRE TERMINÉ")
    logger.info(f"⏱️ Temps d'arrêt: {shutdown_duration:.2f}s")
    logger.info(f"⏱️ Uptime total: {total_uptime:.2f}s")
    logger.info("=" * 50)


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI avec monitoring complet."""
    
    logger.info("🏗️ Création de l'application FastAPI...")
    
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride pour les transactions financières",
        version="2.0.0",  # Version avec monitoring
        lifespan=lifespan
    )
    
    # =============================================================================
    # CONFIGURATION CORS
    # =============================================================================
    
    logger.info("🌐 Configuration CORS...")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # =============================================================================
    # CONFIGURATION DES MIDDLEWARES (avec monitoring)
    # =============================================================================
    
    logger.info("🔧 Configuration des middlewares...")
    setup_middleware(app, monitor=search_monitor)
    
    # =============================================================================
    # CONFIGURATION DES ROUTES
    # =============================================================================
    
    logger.info("🛣️ Configuration des routes...")
    
    # Routes principales
    app.include_router(router, prefix="/api/v1")
    
    # Routes de diagnostic
    setup_diagnostic_routes(app, monitor=search_monitor)
    
    # Endpoint de santé simple (compatible avec les load balancers)
    @app.get("/")
    async def root():
        """Endpoint racine simple."""
        return {
            "service": "harena-search-service",
            "version": "2.0.0",
            "status": "running",
            "timestamp": time.time()
        }
    
    # Endpoint de ping simple
    @app.get("/ping")
    async def ping():
        """Ping simple pour les health checks."""
        return {"status": "pong", "timestamp": time.time()}
    
    # Endpoint de santé détaillé (utilise le monitor)
    @app.get("/health")
    async def health_check():
        """Vérification détaillée de l'état du service."""
        try:
            health_summary = search_monitor.get_health_summary()
            
            # Calculer l'uptime
            uptime = time.time() - startup_time if startup_time else 0
            
            return {
                "service": "harena-search-service",
                "version": "2.0.0",
                "status": health_summary["overall_status"],
                "timestamp": time.time(),
                "uptime_seconds": uptime,
                "components": {
                    "elasticsearch": {
                        "status": health_summary["elasticsearch"]["status"],
                        "response_time_ms": health_summary["elasticsearch"]["response_time_ms"]
                    },
                    "qdrant": {
                        "status": health_summary["qdrant"]["status"],
                        "response_time_ms": health_summary["qdrant"]["response_time_ms"]
                    }
                },
                "metrics": health_summary["search_metrics"],
                "alerts_count": len(health_summary["active_alerts"])
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur health check: {e}")
            return {
                "service": "harena-search-service",
                "version": "2.0.0",
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    logger.info("✅ Application FastAPI configurée")
    return app


# =============================================================================
# CRÉATION DE L'APPLICATION
# =============================================================================

app = create_app()

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Démarrage du serveur uvicorn...")
    
    # Configuration du serveur
    server_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": 8004,
        "reload": settings.DEBUG,
        "log_level": "info",
        "access_log": True,
    }
    
    # Log de la configuration
    logger.info(f"🌐 Serveur: {server_config['host']}:{server_config['port']}")
    logger.info(f"🔄 Reload: {server_config['reload']}")
    logger.info(f"📊 Log level: {server_config['log_level']}")
    
    try:
        uvicorn.run(**server_config)
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"💥 Erreur fatale du serveur: {e}")
        sys.exit(1)