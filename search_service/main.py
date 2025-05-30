"""
Module principal du service de recherche.

Ce module initialise et configure le service de recherche hybride de Harena,
combinant recherche lexicale, sémantique et reranking.
"""
import logging
import time
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

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
)
logger = logging.getLogger("search_service")

# Instances globales
elastic_client = None
qdrant_client = None
search_cache = None
metrics_collector = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service de recherche."""
    global elastic_client, qdrant_client, search_cache, metrics_collector, startup_time
    
    # Initialisation
    logger.info("Démarrage du service de recherche hybride")
    startup_time = time.time()
    
    # Vérification des configurations
    if not settings.SEARCHBOX_URL and not settings.BONSAI_URL:
        logger.warning("Aucune URL Elasticsearch configurée. La recherche lexicale ne fonctionnera pas.")
    
    if not settings.QDRANT_URL:
        logger.warning("QDRANT_URL non définie. La recherche sémantique ne fonctionnera pas.")
    
    if not settings.COHERE_KEY:
        logger.warning("COHERE_KEY non définie. Le reranking ne fonctionnera pas.")
    
    # Initialisation des services
    try:
        # Service d'embeddings
        await embedding_service.initialize()
        logger.info("Service d'embeddings initialisé")
        
        # Service de reranking
        await reranker_service.initialize()
        logger.info("Service de reranking initialisé")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des services: {e}")
    
    # Initialisation des clients de stockage
    try:
        # Client Elasticsearch
        elastic_client = ElasticClient()
        await elastic_client.initialize()
        logger.info("Client Elasticsearch initialisé")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation Elasticsearch: {e}")
        elastic_client = None
    
    try:
        # Client Qdrant
        qdrant_client = QdrantClient()
        await qdrant_client.initialize()
        logger.info("Client Qdrant initialisé")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation Qdrant: {e}")
        qdrant_client = None
    
    # Initialisation du cache et des métriques
    search_cache = SearchCache()
    metrics_collector = MetricsCollector()
    
    # Injecter les instances dans le module routes
    import search_service.api.routes as routes
    routes.elastic_client = elastic_client
    routes.qdrant_client = qdrant_client
    routes.search_cache = search_cache
    routes.metrics_collector = metrics_collector
    
    logger.info("Service de recherche prêt")
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("Arrêt du service de recherche")
    
    if elastic_client:
        await elastic_client.close()
    if qdrant_client:
        await qdrant_client.close()
    if embedding_service:
        await embedding_service.close()
    if reranker_service:
        await reranker_service.close()
    
    logger.info("Service de recherche arrêté")


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service de recherche."""
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride pour les transactions financières",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware de métriques
    @app.middleware("http")
    async def add_metrics(request, call_next):
        """Collecte des métriques de performance."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Enregistrer les métriques
        if metrics_collector:
            metrics_collector.record_request(
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                duration=process_time
            )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Enregistrement des routes
    app.include_router(router, prefix="/api/v1", tags=["search"])
    
    # Endpoint de santé racine
    @app.get("/")
    def root():
        """Endpoint racine pour vérification rapide."""
        return {
            "service": "search_service",
            "version": "1.0.0",
            "status": "running"
        }
    
    # Endpoint de santé détaillé
    @app.get("/health")
    async def health_check():
        """Vérification détaillée de l'état du service."""
        from search_service.models import HealthStatus
        
        # Vérifier l'état des composants
        elasticsearch_ok = elastic_client is not None and await elastic_client.is_healthy()
        qdrant_ok = qdrant_client is not None and await qdrant_client.is_healthy()
        cohere_ok = reranker_service is not None and reranker_service.is_initialized()
        openai_ok = embedding_service is not None and embedding_service.is_initialized()
        
        # Calculer l'état global
        all_ok = elasticsearch_ok and qdrant_ok and cohere_ok and openai_ok
        some_ok = elasticsearch_ok or qdrant_ok
        
        if all_ok:
            status = "healthy"
        elif some_ok:
            status = "degraded"
        else:
            status = "unhealthy"
        
        # Obtenir les métriques
        metrics = None
        if metrics_collector:
            metrics = metrics_collector.get_summary()
        
        # Calculer l'uptime
        uptime = time.time() - startup_time if startup_time else 0
        
        return HealthStatus(
            status=status,
            elasticsearch_status=elasticsearch_ok,
            qdrant_status=qdrant_ok,
            cohere_status=cohere_ok,
            openai_status=openai_ok,
            response_time_ms=metrics.get("avg_response_time_ms") if metrics else None,
            requests_per_minute=metrics.get("requests_per_minute") if metrics else None,
            cache_hit_rate=search_cache.get_hit_rate() if search_cache else None,
            version="1.0.0",
            uptime_seconds=uptime
        )
    
    return app


# Création de l'application
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True)