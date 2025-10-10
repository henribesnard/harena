"""
Module principal du service d'enrichissement - Elasticsearch uniquement.

Ce module initialise et configure le service d'enrichissement de la plateforme Harena,
responsable de la structuration et de l'indexation des données financières
dans Elasticsearch.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from pythonjsonlogger import jsonlogger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from enrichment_service.api.routes import router
from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
from enrichment_service.core.processor import ElasticsearchTransactionProcessor
from config_service.config import settings

# Configuration du logging JSON
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s"
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("enrichment_service")

# Instances globales
elasticsearch_client = None
elasticsearch_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service d'enrichissement Elasticsearch."""
    global elasticsearch_client, elasticsearch_processor
    
    # Initialisation
    logger.info("🚀 Démarrage du service d'enrichissement Elasticsearch")
    
    # Vérification des configurations critiques
    config_issues = []

    if not settings.BONSAI_URL and not getattr(settings, 'ELASTICSEARCH_URL', None):
        config_issues.append("BONSAI_URL ou ELASTICSEARCH_URL non définie")
        logger.warning("⚠️ BONSAI_URL non définie. Tentative avec ELASTICSEARCH_URL...")

    if config_issues and not getattr(settings, 'ELASTICSEARCH_URL', None):
        logger.error(f"❌ Problèmes de configuration détectés: {', '.join(config_issues)}")
        logger.error("💡 Service désactivé car Elasticsearch non configuré")
        # Ne pas raise l'erreur, juste logger et continuer sans Elasticsearch
        elasticsearch_client = None
        elasticsearch_processor = None
    
    # 1. Initialisation du client Elasticsearch (si configuré)
    elasticsearch_success = False
    if not config_issues or getattr(settings, 'ELASTICSEARCH_URL', None):
        try:
            elasticsearch_client = ElasticsearchClient()
            await elasticsearch_client.initialize()
            elasticsearch_success = True
            logger.info("✅ Elasticsearch client initialisé avec succès")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de l'initialisation d'Elasticsearch: {e}")
            logger.warning("💡 Le service continuera sans Elasticsearch")
            elasticsearch_client = None

    # 2. Création du processeur Elasticsearch (si client disponible)
    processor_success = False
    if elasticsearch_client:
        try:
            elasticsearch_processor = ElasticsearchTransactionProcessor(elasticsearch_client)
            processor_success = True
            logger.info("✅ Elasticsearch processor créé avec succès")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de la création du processeur: {e}")
            elasticsearch_processor = None
    
    # 3. Injection des instances dans le module routes
    try:
        import enrichment_service.api.routes as routes
        routes.elasticsearch_client = elasticsearch_client
        routes.elasticsearch_processor = elasticsearch_processor
        logger.info("✅ Instances injectées dans les routes")
    except Exception as e:
        logger.error(f"❌ Erreur injection dans routes: {e}")
        raise Exception(f"Failed to inject instances: {e}")
    
    # 4. Résumé de l'initialisation
    logger.info("📊 RÉSUMÉ DE L'INITIALISATION:")
    logger.info(f"   🔍 Elasticsearch: {'✅' if elasticsearch_success else '❌'}")
    logger.info(f"   🔄 Processor: {'✅' if processor_success else '❌'}")
    
    if processor_success and elasticsearch_success:
        logger.info("🎉 Service d'enrichissement Elasticsearch prêt!")
    else:
        logger.error("🚨 Service d'enrichissement en échec critique")
        raise Exception("Service initialization failed")
    
    # 5. Affichage des endpoints disponibles
    logger.info("🌐 ENDPOINTS DISPONIBLES:")
    logger.info("   Elasticsearch Processing:")
    logger.info("     POST /api/v1/enrichment/elasticsearch/sync-user/{user_id}")
    logger.info("     DELETE /api/v1/enrichment/elasticsearch/user-data/{user_id}")
    logger.info("   Monitoring & Diagnostics:")
    logger.info("     GET  /api/v1/enrichment/elasticsearch/health")
    logger.info("     GET  /api/v1/enrichment/elasticsearch/user-stats/{user_id}")
    logger.info("     GET  /api/v1/enrichment/elasticsearch/cluster-info")
    logger.info("   Utilities:")
    logger.info("     POST /api/v1/enrichment/elasticsearch/reindex-user/{user_id}")
    logger.info("     GET  /api/v1/enrichment/elasticsearch/document-exists/{user_id}/{transaction_id}")
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("🛑 Arrêt du service d'enrichissement...")
    
    if elasticsearch_client:
        try:
            await elasticsearch_client.close()
            logger.info("✅ Elasticsearch client fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Elasticsearch: {e}")
    
    logger.info("👋 Arrêt du service d'enrichissement terminé")

def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service d'enrichissement Elasticsearch."""
    
    app = FastAPI(
        title="Harena Enrichment Service - Elasticsearch",
        description="Service d'enrichissement et d'indexation des données financières dans Elasticsearch",
        version="2.0.0-elasticsearch",
        lifespan=lifespan
    )
    
    # Configuration CORS
    if settings.CORS_ORIGINS:
        origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Inclusion des routes avec le bon préfixe
    app.include_router(router, prefix="/api/v1/enrichment", tags=["enrichment"])
    
    return app

# Création de l'application
app = create_app()

# Health check endpoint au niveau racine
@app.get("/health")
async def health_check():
    """Point de santé du service d'enrichissement Elasticsearch."""
    global elasticsearch_client, elasticsearch_processor
    
    # Vérifier l'état des composants
    elasticsearch_available = elasticsearch_client is not None
    elasticsearch_initialized = getattr(elasticsearch_client, '_initialized', False) if elasticsearch_client else False
    processor_available = elasticsearch_processor is not None
    
    # Calculer le statut global
    overall_healthy = elasticsearch_available and elasticsearch_initialized and processor_available
    
    return {
        "service": "enrichment_service_elasticsearch",
        "version": "2.0.0-elasticsearch",
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "storage_systems": {
            "elasticsearch": {
                "available": elasticsearch_available,
                "initialized": elasticsearch_initialized,
                "url_configured": bool(settings.BONSAI_URL),
                "index_name": elasticsearch_client.index_name if elasticsearch_client else None
            }
        },
        "processors": {
            "elasticsearch_processor": processor_available
        },
        "capabilities": {
            "transaction_processing": processor_available,
            "batch_processing": processor_available,
            "user_sync": processor_available,
            "lexical_indexing": elasticsearch_available and elasticsearch_initialized,
            "statistics": processor_available
        }
    }

@app.get("/")
async def root():
    """Endpoint racine avec informations du service."""
    return {
        "service": "Harena Enrichment Service - Elasticsearch",
        "version": "2.0.0-elasticsearch",
        "description": "Service d'enrichissement avec indexation Elasticsearch",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "api": "/api/v1/enrichment/elasticsearch/",
            "diagnostic": "/diagnostic"
        },
        "features": [
            "Transaction enrichment and indexation",
            "Elasticsearch lexical indexing",
            "Batch processing optimization",
            "User-specific data management",
            "Real-time health monitoring",
            "Advanced diagnostics"
        ],
        "architecture": {
            "storage": "Elasticsearch only",
            "search_capabilities": "Lexical/text search",
            "optimized_for": "High-volume transaction indexing"
        }
    }

# Endpoint pour diagnostics avancés
@app.get("/diagnostic")
async def advanced_diagnostic():
    """Diagnostic avancé du service Elasticsearch."""
    global elasticsearch_client, elasticsearch_processor
    
    diagnostic = {
        "timestamp": datetime.now().isoformat(),
        "service_info": {
            "name": "enrichment_service_elasticsearch",
            "version": "2.0.0-elasticsearch",
            "python_version": f"{__import__('sys').version}",
            "environment": settings.ENVIRONMENT
        },
        "configuration": {
            "bonsai_url_set": bool(settings.BONSAI_URL),
            "cors_origins_set": bool(settings.CORS_ORIGINS)
        },
        "elasticsearch_status": {},
        "processor_status": {}
    }
    
    # Diagnostic Elasticsearch détaillé
    if elasticsearch_client:
        try:
            # Test de connectivité basique
            diagnostic["elasticsearch_status"] = {
                "client_available": True,
                "initialized": elasticsearch_client._initialized,
                "index_name": elasticsearch_client.index_name,
                "base_url": elasticsearch_client.base_url
            }
            
            # Tenter d'obtenir des infos cluster si possible
            try:
                cluster_info = await elasticsearch_client.get_cluster_info()
                diagnostic["elasticsearch_status"]["cluster_info"] = cluster_info
            except Exception as cluster_e:
                diagnostic["elasticsearch_status"]["cluster_error"] = str(cluster_e)
                
        except Exception as e:
            diagnostic["elasticsearch_status"] = {
                "client_available": True,
                "error": str(e)
            }
    else:
        diagnostic["elasticsearch_status"] = {
            "client_available": False,
            "reason": "Client not initialized"
        }
    
    # Diagnostic du processeur
    if elasticsearch_processor:
        try:
            processor_health = await elasticsearch_processor.health_check()
            diagnostic["processor_status"] = {
                "processor_available": True,
                "health_check": processor_health
            }
        except Exception as e:
            diagnostic["processor_status"] = {
                "processor_available": True,
                "health_check_error": str(e)
            }
    else:
        diagnostic["processor_status"] = {
            "processor_available": False,
            "reason": "Processor not initialized"
        }
    
    return diagnostic

# Endpoint pour exporter les métriques Prometheus
@app.get("/metrics")
async def get_metrics():
    """Expose les métriques au format Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Démarrage direct du service d'enrichissement Elasticsearch")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )