"""
Module principal du service d'enrichissement avec dual storage.

Ce module initialise et configure le service d'enrichissement de la plateforme Harena,
responsable de la structuration et du stockage vectoriel des données financières
dans Qdrant ET Elasticsearch pour assurer la cohérence des données.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from enrichment_service.api.routes import router
from enrichment_service.storage.qdrant import QdrantStorage
from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
from enrichment_service.core.processor import TransactionProcessor, DualStorageTransactionProcessor
from enrichment_service.core.embeddings import embedding_service
from config_service.config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
)
logger = logging.getLogger("enrichment_service")

# Instances globales
qdrant_storage = None
elasticsearch_client = None
transaction_processor = None
dual_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service d'enrichissement avec dual storage."""
    global qdrant_storage, elasticsearch_client, transaction_processor, dual_processor
    
    # Initialisation
    logger.info("🚀 Démarrage du service d'enrichissement avec dual storage")
    
    # Vérification des configurations critiques
    config_issues = []
    
    if not settings.QDRANT_URL:
        config_issues.append("QDRANT_URL non définie")
        logger.warning("⚠️ QDRANT_URL non définie. Le stockage vectoriel ne fonctionnera pas.")
    
    if not settings.BONSAI_URL:
        config_issues.append("BONSAI_URL non définie")
        logger.warning("⚠️ BONSAI_URL non définie. L'indexation Elasticsearch ne fonctionnera pas.")
    
    if not settings.OPENAI_API_KEY:
        config_issues.append("OPENAI_API_KEY non définie")
        logger.warning("⚠️ OPENAI_API_KEY non définie. La génération d'embeddings ne fonctionnera pas.")
    
    if config_issues:
        logger.error(f"❌ Problèmes de configuration détectés: {', '.join(config_issues)}")
        logger.error("💡 Le service fonctionnera en mode dégradé")
    
    # 1. Initialisation du service d'embeddings
    embedding_success = False
    try:
        await embedding_service.initialize()
        embedding_success = True
        logger.info("✅ Service d'embeddings initialisé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation du service d'embeddings: {e}")
    
    # 2. Initialisation du storage Qdrant
    qdrant_success = False
    try:
        if settings.QDRANT_URL:
            qdrant_storage = QdrantStorage()
            await qdrant_storage.initialize()
            qdrant_success = True
            logger.info("✅ Qdrant storage initialisé avec succès")
        else:
            logger.warning("⚠️ Qdrant non configuré, ignoré")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de Qdrant: {e}")
        qdrant_storage = None
    
    # 3. Initialisation du client Elasticsearch
    elasticsearch_success = False
    try:
        if settings.BONSAI_URL:
            elasticsearch_client = ElasticsearchClient()
            await elasticsearch_client.initialize()
            elasticsearch_success = True
            logger.info("✅ Elasticsearch client initialisé avec succès")
        else:
            logger.warning("⚠️ Elasticsearch non configuré, ignoré")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation d'Elasticsearch: {e}")
        elasticsearch_client = None
    
    # 4. Création des processeurs
    processor_success = False
    try:
        # Processeur legacy (Qdrant uniquement) pour compatibilité
        if qdrant_storage:
            transaction_processor = TransactionProcessor(qdrant_storage)
            logger.info("✅ Transaction processor (legacy) créé")
        
        # Processeur dual storage (Qdrant + Elasticsearch)
        if qdrant_storage and elasticsearch_client:
            dual_processor = DualStorageTransactionProcessor(qdrant_storage, elasticsearch_client)
            processor_success = True
            logger.info("🎉 Dual storage processor créé avec succès")
        elif qdrant_storage or elasticsearch_client:
            logger.warning("⚠️ Dual processor partiellement disponible")
            # Créer quand même le dual processor avec les clients disponibles
            dual_processor = DualStorageTransactionProcessor(
                qdrant_storage, 
                elasticsearch_client
            )
            processor_success = True
        else:
            logger.error("❌ Aucun système de stockage disponible")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création des processeurs: {e}")
    
    # 5. Injection des instances dans le module routes
    try:
        import enrichment_service.api.routes as routes
        routes.qdrant_storage = qdrant_storage
        routes.elasticsearch_client = elasticsearch_client
        routes.transaction_processor = transaction_processor
        routes.dual_processor = dual_processor
        logger.info("✅ Instances injectées dans les routes")
    except Exception as e:
        logger.error(f"❌ Erreur injection dans routes: {e}")
    
    # 6. Résumé de l'initialisation
    logger.info("📊 RÉSUMÉ DE L'INITIALISATION:")
    logger.info(f"   🧠 Embeddings: {'✅' if embedding_success else '❌'}")
    logger.info(f"   🎯 Qdrant: {'✅' if qdrant_success else '❌'}")
    logger.info(f"   🔍 Elasticsearch: {'✅' if elasticsearch_success else '❌'}")
    logger.info(f"   🔄 Legacy processor: {'✅' if transaction_processor else '❌'}")
    logger.info(f"   🔄 Dual processor: {'✅' if processor_success else '❌'}")
    
    if processor_success and qdrant_success and elasticsearch_success:
        logger.info("🎉 Service d'enrichissement prêt avec dual storage complet!")
    elif processor_success:
        logger.warning("⚠️ Service d'enrichissement en mode partiel")
    else:
        logger.error("🚨 Service d'enrichissement en mode dégradé critique")
    
    # 7. Affichage des endpoints disponibles
    logger.info("🌐 ENDPOINTS DISPONIBLES:")
    logger.info("   Legacy (Qdrant only):")
    logger.info("     POST /api/v1/enrichment/enrich/transaction")
    logger.info("     POST /api/v1/enrichment/sync/user/{user_id}")
    logger.info("   Dual Storage (Qdrant + Elasticsearch):")
    logger.info("     POST /api/v1/enrichment/dual/sync-user")
    logger.info("     GET  /api/v1/enrichment/dual/sync-status/{user_id}")
    logger.info("     POST /api/v1/enrichment/dual/enrich-transaction")
    logger.info("     GET  /api/v1/enrichment/dual/health")
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("🛑 Arrêt du service d'enrichissement...")
    
    if elasticsearch_client:
        try:
            await elasticsearch_client.close()
            logger.info("✅ Elasticsearch client fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Elasticsearch: {e}")
    
    if qdrant_storage:
        try:
            await qdrant_storage.close()
            logger.info("✅ Qdrant storage fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Qdrant: {e}")
    
    if embedding_service:
        try:
            await embedding_service.close()
            logger.info("✅ Service d'embeddings fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture embeddings: {e}")
    
    logger.info("👋 Arrêt du service d'enrichissement terminé")

def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service d'enrichissement avec dual storage."""
    
    app = FastAPI(
        title="Harena Enrichment Service",
        description="Service d'enrichissement et de stockage vectoriel des données financières avec dual storage (Qdrant + Elasticsearch)",
        version="2.0.0",  # Version mise à jour pour dual storage
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
    """Point de santé du service d'enrichissement avec informations détaillées."""
    global qdrant_storage, elasticsearch_client, transaction_processor, dual_processor
    
    # Vérifier l'état de chaque composant
    qdrant_available = qdrant_storage is not None
    qdrant_initialized = getattr(qdrant_storage, 'client', None) is not None if qdrant_storage else False
    
    elasticsearch_available = elasticsearch_client is not None
    elasticsearch_initialized = getattr(elasticsearch_client, '_initialized', False) if elasticsearch_client else False
    
    embedding_available = embedding_service is not None
    embedding_initialized = getattr(embedding_service, 'client', None) is not None if embedding_service else False
    
    # Calculer le statut global
    overall_healthy = (
        (qdrant_available and qdrant_initialized) or 
        (elasticsearch_available and elasticsearch_initialized)
    ) and embedding_initialized
    
    return {
        "service": "enrichment_service",
        "version": "2.0.0",
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "storage_systems": {
            "qdrant": {
                "available": qdrant_available,
                "initialized": qdrant_initialized,
                "url_configured": bool(settings.QDRANT_URL)
            },
            "elasticsearch": {
                "available": elasticsearch_available,
                "initialized": elasticsearch_initialized,
                "url_configured": bool(settings.BONSAI_URL)
            }
        },
        "services": {
            "embeddings": {
                "available": embedding_available,
                "initialized": embedding_initialized,
                "api_key_configured": bool(settings.OPENAI_API_KEY)
            }
        },
        "processors": {
            "legacy_processor": transaction_processor is not None,
            "dual_processor": dual_processor is not None
        },
        "capabilities": {
            "legacy_sync": transaction_processor is not None,
            "dual_storage_sync": dual_processor is not None,
            "semantic_search": qdrant_available and qdrant_initialized,
            "lexical_indexing": elasticsearch_available and elasticsearch_initialized
        }
    }

@app.get("/")
async def root():
    """Endpoint racine avec informations du service."""
    return {
        "service": "Harena Enrichment Service",
        "version": "2.0.0",
        "description": "Service d'enrichissement avec dual storage (Qdrant + Elasticsearch)",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "legacy_api": "/api/v1/enrichment/",
            "dual_storage_api": "/api/v1/enrichment/dual/"
        },
        "features": [
            "Transaction enrichment and vectorization",
            "Dual storage (Qdrant + Elasticsearch)",
            "Semantic search capabilities",
            "Lexical indexing for hybrid search",
            "Batch processing optimization"
        ]
    }

# Endpoint pour diagnostics avancés (superuser uniquement)
@app.get("/diagnostic")
async def advanced_diagnostic():
    """Diagnostic avancé du service (informations techniques détaillées)."""
    global qdrant_storage, elasticsearch_client, dual_processor
    
    diagnostic = {
        "timestamp": datetime.now().isoformat(),
        "service_info": {
            "name": "enrichment_service",
            "version": "2.0.0",
            "python_version": f"{__import__('sys').version}",
            "environment": settings.ENVIRONMENT
        },
        "configuration": {
            "qdrant_url_set": bool(settings.QDRANT_URL),
            "qdrant_api_key_set": bool(settings.QDRANT_API_KEY),
            "bonsai_url_set": bool(settings.BONSAI_URL),
            "openai_api_key_set": bool(settings.OPENAI_API_KEY),
            "embedding_model": settings.EMBEDDING_MODEL
        },
        "storage_status": {},
        "processor_status": {}
    }
    
    # Diagnostic Qdrant
    if qdrant_storage:
        try:
            collection_info = await qdrant_storage.get_collection_info()
            diagnostic["storage_status"]["qdrant"] = {
                "client_available": True,
                "collection_exists": collection_info is not None,
                "collection_name": qdrant_storage.collection_name,
                "points_count": collection_info.points_count if collection_info else 0
            }
        except Exception as e:
            diagnostic["storage_status"]["qdrant"] = {
                "client_available": True,
                "error": str(e)
            }
    else:
        diagnostic["storage_status"]["qdrant"] = {
            "client_available": False,
            "reason": "Not configured or initialization failed"
        }
    
    # Diagnostic Elasticsearch
    if elasticsearch_client:
        try:
            diagnostic["storage_status"]["elasticsearch"] = {
                "client_available": True,
                "initialized": elasticsearch_client._initialized,
                "index_name": elasticsearch_client.index_name
            }
        except Exception as e:
            diagnostic["storage_status"]["elasticsearch"] = {
                "client_available": True,
                "error": str(e)
            }
    else:
        diagnostic["storage_status"]["elasticsearch"] = {
            "client_available": False,
            "reason": "Not configured or initialization failed"
        }
    
    # Diagnostic des processeurs
    diagnostic["processor_status"] = {
        "legacy_processor_available": transaction_processor is not None,
        "dual_processor_available": dual_processor is not None
    }
    
    return diagnostic

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Démarrage direct du service d'enrichissement")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )