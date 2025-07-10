"""
Module principal du Search Service.

Ce module initialise et configure le service de recherche lexicale de la plateforme Harena,
spécialisé dans la recherche de transactions financières via Elasticsearch.

ARCHITECTURE:
- Service de recherche lexicale pure (sans IA)
- API REST FastAPI avec endpoints optimisés
- Cache LRU pour les performances
- Métriques et monitoring intégrés
- Validation et sécurité des requêtes

RESPONSABILITÉS:
✅ Recherche lexicale dans les transactions
✅ Filtrage avancé par utilisateur, dates, montants
✅ Templates de requêtes optimisés
✅ Cache des résultats fréquents
✅ Métriques de performance
✅ Health checks et monitoring
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Imports avec gestion d'erreur gracieuse
try:
    from .config.settings import get_settings, SearchServiceSettings
    CONFIG_AVAILABLE = True
except ImportError:
    logging.warning("Configuration non disponible - utilisation des valeurs par défaut")
    CONFIG_AVAILABLE = False
    
    # Configuration par défaut minimaliste
    class MockSettings:
        PROJECT_NAME = "Search Service"
        API_V1_STR = "/api/v1"
        CORS_ORIGINS = "*"
        LOG_LEVEL = "INFO"
        ELASTICSEARCH_HOST = "localhost"
        ELASTICSEARCH_PORT = 9200
        SEARCH_CACHE_SIZE = 1000
        SEARCH_CACHE_TTL = 300
    
    def get_settings():
        return MockSettings()

try:
    from .api.routes import router as search_router
    from .api.middleware import setup_middleware
    API_AVAILABLE = True
except ImportError:
    logging.warning("Modules API non disponibles")
    API_AVAILABLE = False
    search_router = None
    setup_middleware = lambda app: None

try:
    from .core.lexical_engine import LexicalEngineFactory
    from .clients.elasticsearch_client import ElasticsearchClient
    CORE_AVAILABLE = True
except ImportError:
    logging.warning("Modules core non disponibles")
    CORE_AVAILABLE = False
    LexicalEngineFactory = None
    ElasticsearchClient = None

# Configuration du logging
def setup_logging():
    """Configure le logging pour le service."""
    settings = get_settings()
    log_level = getattr(settings, 'LOG_LEVEL', 'INFO')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('search_service.log') if hasattr(settings, 'LOG_TO_FILE') and settings.LOG_TO_FILE else logging.NullHandler()
        ]
    )

setup_logging()
logger = logging.getLogger("search_service")

# ==================== VARIABLES GLOBALES ====================

# Instance globale du moteur de recherche (initialisée au démarrage)
lexical_engine = None
elasticsearch_client = None

# ==================== CYCLE DE VIE DE L'APPLICATION ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie du Search Service.
    
    Gère l'initialisation et la fermeture propre des composants.
    """
    global lexical_engine, elasticsearch_client
    
    # === DÉMARRAGE ===
    logger.info("🚀 Démarrage du Search Service")
    
    settings = get_settings()
    
    # Vérification des configurations critiques
    if not hasattr(settings, 'ELASTICSEARCH_HOST'):
        logger.warning("Configuration Elasticsearch manquante - certaines fonctionnalités peuvent être limitées")
    
    # Initialisation du client Elasticsearch
    if CORE_AVAILABLE and ElasticsearchClient:
        try:
            elasticsearch_client = ElasticsearchClient({
                'host': getattr(settings, 'ELASTICSEARCH_HOST', 'localhost'),
                'port': getattr(settings, 'ELASTICSEARCH_PORT', 9200),
                'timeout': getattr(settings, 'ELASTICSEARCH_TIMEOUT', 30)
            })
            
            # Test de connexion
            await elasticsearch_client.connect()
            health = await elasticsearch_client.health_check()
            
            if health.get('status') == 'healthy':
                logger.info("✅ Connexion Elasticsearch établie")
            else:
                logger.warning("⚠️ Elasticsearch accessible mais en état dégradé")
                
        except Exception as e:
            logger.error(f"❌ Échec de connexion Elasticsearch: {e}")
            elasticsearch_client = None
    
    # Initialisation du moteur lexical
    if CORE_AVAILABLE and LexicalEngineFactory and elasticsearch_client:
        try:
            lexical_engine = LexicalEngineFactory.create(
                elasticsearch_client=elasticsearch_client,
                config=settings.__dict__ if hasattr(settings, '__dict__') else {}
            )
            logger.info("✅ Moteur de recherche lexicale initialisé")
        except Exception as e:
            logger.error(f"❌ Échec d'initialisation du moteur lexical: {e}")
            lexical_engine = None
    
    # Vérification de l'état général
    if lexical_engine:
        logger.info("🎉 Search Service complètement opérationnel")
    else:
        logger.warning("⚠️ Search Service en mode dégradé - fonctionnalités limitées")
    
    yield  # L'application s'exécute ici
    
    # === ARRÊT ===
    logger.info("🛑 Arrêt du Search Service")
    
    # Fermeture du moteur lexical
    if lexical_engine:
        try:
            await lexical_engine.close()
            logger.info("✅ Moteur lexical fermé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture du moteur lexical: {e}")
    
    # Fermeture du client Elasticsearch
    if elasticsearch_client:
        try:
            await elasticsearch_client.close()
            logger.info("✅ Connexion Elasticsearch fermée")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture Elasticsearch: {e}")
    
    logger.info("👋 Search Service arrêté proprement")

# ==================== CRÉATION DE L'APPLICATION ====================

def create_app() -> FastAPI:
    """
    Crée et configure l'application FastAPI du Search Service.
    
    Returns:
        Application FastAPI configurée
    """
    settings = get_settings()
    
    # Création de l'application avec le cycle de vie
    app = FastAPI(
        title=getattr(settings, 'PROJECT_NAME', 'Harena Search Service'),
        description="Service de recherche lexicale pour les transactions financières",
        version="1.0.0",
        openapi_url=f"{getattr(settings, 'API_V1_STR', '/api/v1')}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # === CONFIGURATION CORS ===
    cors_origins = getattr(settings, 'CORS_ORIGINS', '*')
    if isinstance(cors_origins, str):
        cors_origins = cors_origins.split(',') if ',' in cors_origins else [cors_origins]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"]
    )
    
    # === MIDDLEWARE DE COMPRESSION ===
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000  # Compresse si > 1KB
    )
    
    # === MIDDLEWARE PERSONNALISÉS ===
    if API_AVAILABLE:
        setup_middleware(app)
    
    # === ENREGISTREMENT DES ROUTES ===
    if API_AVAILABLE and search_router:
        app.include_router(
            search_router,
            prefix=getattr(settings, 'API_V1_STR', '/api/v1'),
            tags=["search"]
        )
        logger.info(f"Routes de recherche enregistrées sur {getattr(settings, 'API_V1_STR', '/api/v1')}")
    else:
        logger.warning("Routes de recherche non disponibles - mode dégradé")
    
    # === ENDPOINTS DE BASE ===
    
    @app.get("/health")
    async def health_check():
        """
        Endpoint de vérification de l'état de santé du service.
        
        Returns:
            Statut détaillé du service et de ses composants
        """
        global lexical_engine, elasticsearch_client
        
        health_status = {
            "service": "search_service",
            "version": "1.0.0",
            "status": "healthy",
            "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None)),
            "components": {
                "config": CONFIG_AVAILABLE,
                "api": API_AVAILABLE,
                "core": CORE_AVAILABLE,
                "elasticsearch": False,
                "lexical_engine": False
            },
            "metrics": {
                "uptime_seconds": 0  # TODO: implémenter le tracking d'uptime
            }
        }
        
        # Vérification Elasticsearch
        if elasticsearch_client:
            try:
                es_health = await elasticsearch_client.health_check()
                health_status["components"]["elasticsearch"] = es_health.get("status") == "healthy"
                health_status["elasticsearch_cluster"] = es_health.get("cluster_name", "unknown")
            except Exception as e:
                logger.error(f"Erreur health check Elasticsearch: {e}")
                health_status["components"]["elasticsearch"] = False
        
        # Vérification moteur lexical
        if lexical_engine:
            try:
                # TODO: ajouter méthode health_check au moteur lexical
                health_status["components"]["lexical_engine"] = True
            except Exception as e:
                logger.error(f"Erreur health check moteur lexical: {e}")
                health_status["components"]["lexical_engine"] = False
        
        # Détermination du statut global
        critical_components = ["elasticsearch", "lexical_engine"]
        if any(not health_status["components"].get(comp, False) for comp in critical_components):
            health_status["status"] = "degraded"
        
        # Code de retour HTTP basé sur le statut
        status_code = 200 if health_status["status"] == "healthy" else 503
        
        return JSONResponse(
            content=health_status,
            status_code=status_code
        )
    
    @app.get("/")
    async def root():
        """Endpoint racine avec informations de base."""
        return {
            "service": "Harena Search Service",
            "version": "1.0.0",
            "description": "Service de recherche lexicale pour les transactions financières",
            "docs": "/docs",
            "health": "/health",
            "api": getattr(settings, 'API_V1_STR', '/api/v1')
        }
    
    @app.get("/metrics")
    async def metrics():
        """
        Endpoint pour les métriques de base du service.
        
        Returns:
            Métriques de performance et d'utilisation
        """
        # TODO: intégrer avec le système de métriques complet
        return {
            "service": "search_service",
            "metrics": {
                "requests_total": 0,  # TODO: implémenter compteur
                "requests_errors": 0,  # TODO: implémenter compteur
                "cache_hits": 0,      # TODO: récupérer du cache
                "cache_misses": 0,    # TODO: récupérer du cache
                "avg_response_time_ms": 0  # TODO: calculer moyenne
            },
            "status": "metrics_basic"
        }
    
    # === GESTIONNAIRE D'ERREUR GLOBAL ===
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Gestionnaire personnalisé pour les erreurs HTTP."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "http_error",
                    "status_code": exc.status_code,
                    "message": exc.detail,
                    "path": str(request.url.path),
                    "method": request.method
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Gestionnaire pour les erreurs générales non gérées."""
        logger.error(f"Erreur non gérée: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "status_code": 500,
                    "message": "Erreur interne du serveur",
                    "path": str(request.url.path),
                    "method": request.method
                }
            }
        )
    
    logger.info("✅ Application FastAPI créée et configurée")
    return app

# ==================== POINT D'ENTRÉE ====================

# Création de l'application pour le déploiement
app = create_app()

# ==================== FONCTIONS UTILITAIRES ====================

def get_lexical_engine():
    """
    Récupère l'instance globale du moteur lexical.
    
    Returns:
        Instance du moteur lexical ou None si non disponible
    """
    global lexical_engine
    return lexical_engine

def get_elasticsearch_client():
    """
    Récupère l'instance globale du client Elasticsearch.
    
    Returns:
        Instance du client Elasticsearch ou None si non disponible
    """
    global elasticsearch_client
    return elasticsearch_client

# ==================== POINT D'ENTRÉE POUR DÉVELOPPEMENT ====================

if __name__ == "__main__":
    """Point d'entrée pour le développement local."""
    import uvicorn
    
    logger.info("🚀 Démarrage en mode développement")
    
    # Configuration pour le développement
    uvicorn.run(
        "search_service.main:app",
        host="0.0.0.0",
        port=8003,  # Port dédié au Search Service
        reload=True,
        log_level="info",
        access_log=True
    )