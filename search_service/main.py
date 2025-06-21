"""
Service de recherche Harena - Point d'entrée principal.

Ce module configure et démarre le service de recherche hybride combinant
Elasticsearch (Bonsai) pour la recherche lexicale et Qdrant pour la recherche sémantique.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration du logging avant les autres imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger("search_service.main")

# ==================== IMPORTS DES MODULES ====================

try:
    from config_service.config import settings
    from search_service.storage.elastic_client_hybrid import HybridElasticClient
    from search_service.storage.qdrant_client import QdrantClient
    from search_service.utils.initialization import (
        initialize_search_clients,
        log_initialization_summary,
        run_startup_diagnostics,
        log_diagnostic_summary,
        create_collections_if_needed,
        validate_clients_functionality
    )
    
    logger.info("✅ Imports réussis")
    
except ImportError as e:
    logger.critical(f"💥 Erreur d'import critique: {e}")
    raise

# Imports optionnels pour les services IA et utilitaires
try:
    from search_service.core.embeddings import EmbeddingService
    EMBEDDING_SERVICE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ EmbeddingService non disponible")
    EMBEDDING_SERVICE_AVAILABLE = False
    EmbeddingService = None

try:
    from search_service.core.reranker import RerankerService
    RERANKER_SERVICE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ RerankerService non disponible")
    RERANKER_SERVICE_AVAILABLE = False
    RerankerService = None

try:
    from search_service.utils.cache import SearchCache
    CACHE_SERVICE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ SearchCache non disponible")
    CACHE_SERVICE_AVAILABLE = False
    SearchCache = None

# Le module MetricsCollector n'existe pas encore
METRICS_SERVICE_AVAILABLE = False
MetricsCollector = None
logger.info("ℹ️ MetricsCollector non implémenté - désactivé")

try:
    from search_service.api.routes import router
    ROUTES_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Routes API non disponibles")
    ROUTES_AVAILABLE = False
    router = None

# ==================== VARIABLES GLOBALES ====================

# Variables de l'application
startup_time = None
startup_diagnostics = {}
full_diagnostic_report = {}

# Clients de stockage
elastic_client: Optional[HybridElasticClient] = None
qdrant_client: Optional[QdrantClient] = None

# Services IA
embedding_service: Optional[Any] = None
reranker_service: Optional[Any] = None

# Services utilitaires
search_cache: Optional[Any] = None
metrics_collector: Optional[Any] = None

# ==================== FONCTIONS D'INITIALISATION ====================

def log_startup_banner():
    """Affiche une bannière de démarrage."""
    logger.info("=" * 100)
    logger.info("🚀 DÉMARRAGE DU SERVICE DE RECHERCHE HARENA")
    logger.info("=" * 100)
    logger.info("🔍 Service: Recherche hybride (lexicale + sémantique)")
    logger.info("📊 Moteurs: Elasticsearch (Bonsai) + Qdrant")
    logger.info("🤖 IA: OpenAI Embeddings + Cohere Reranking")
    logger.info("🔧 Mode: Client hybride avec fallback Bonsai HTTP")
    logger.info("📋 Version: 2.0.0")
    logger.info("=" * 100)


async def initialize_ai_services() -> Dict[str, Any]:
    """Initialise les services IA (embeddings et reranking)."""
    global embedding_service, reranker_service
    
    logger.info("🤖 === INITIALISATION DES SERVICES IA ===")
    ai_diagnostics = {
        "embeddings": {"initialized": False, "available": EMBEDDING_SERVICE_AVAILABLE, "error": None},
        "reranking": {"initialized": False, "available": RERANKER_SERVICE_AVAILABLE, "error": None}
    }
    
    # Initialisation du service d'embeddings
    if EMBEDDING_SERVICE_AVAILABLE:
        try:
            if settings.OPENAI_API_KEY:
                embedding_service = EmbeddingService()
                await embedding_service.initialize()
                ai_diagnostics["embeddings"]["initialized"] = True
                logger.info("✅ Service d'embeddings OpenAI initialisé")
            else:
                logger.warning("⚠️ OPENAI_API_KEY non configurée - service d'embeddings désactivé")
                ai_diagnostics["embeddings"]["error"] = "OPENAI_API_KEY not configured"
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation embeddings: {e}")
            ai_diagnostics["embeddings"]["error"] = str(e)
    else:
        ai_diagnostics["embeddings"]["error"] = "Service not available"
    
    # Initialisation du service de reranking
    if RERANKER_SERVICE_AVAILABLE:
        try:
            if getattr(settings, 'COHERE_KEY', None):
                reranker_service = RerankerService()
                await reranker_service.initialize()
                ai_diagnostics["reranking"]["initialized"] = True
                logger.info("✅ Service de reranking Cohere initialisé")
            else:
                logger.warning("⚠️ COHERE_KEY non configurée - service de reranking désactivé")
                ai_diagnostics["reranking"]["error"] = "COHERE_KEY not configured"
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation reranking: {e}")
            ai_diagnostics["reranking"]["error"] = str(e)
    else:
        ai_diagnostics["reranking"]["error"] = "Service not available"
    
    return ai_diagnostics


def initialize_cache_and_metrics() -> Dict[str, Any]:
    """Initialise le cache et le collecteur de métriques."""
    global search_cache, metrics_collector
    
    logger.info("🛠️ === INITIALISATION CACHE ET MÉTRIQUES ===")
    utils_diagnostics = {
        "cache": {"initialized": False, "available": CACHE_SERVICE_AVAILABLE, "error": None},
        "metrics": {"initialized": False, "available": METRICS_SERVICE_AVAILABLE, "error": None}
    }
    
    # Initialisation du cache
    if CACHE_SERVICE_AVAILABLE:
        try:
            search_cache = SearchCache()
            utils_diagnostics["cache"]["initialized"] = True
            logger.info("✅ Cache de recherche initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation cache: {e}")
            utils_diagnostics["cache"]["error"] = str(e)
    else:
        utils_diagnostics["cache"]["error"] = "Service not available"
    
    # Initialisation du collecteur de métriques
    if METRICS_SERVICE_AVAILABLE:
        try:
            metrics_collector = MetricsCollector()
            utils_diagnostics["metrics"]["initialized"] = True
            logger.info("✅ Collecteur de métriques initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation métriques: {e}")
            utils_diagnostics["metrics"]["error"] = str(e)
    else:
        utils_diagnostics["metrics"]["error"] = "Service not implemented yet"
        logger.info("ℹ️ Collecteur de métriques désactivé (non implémenté)")
    
    return utils_diagnostics


def inject_dependencies():
    """Injecte les dépendances dans les modules qui en ont besoin."""
    logger.info("🔗 === INJECTION DES DÉPENDANCES ===")
    
    if not ROUTES_AVAILABLE:
        logger.warning("⚠️ Routes non disponibles - injection ignorée")
        return
    
    try:
        import search_service.api.routes as routes
        
        # Injecter seulement les clients qui ont été initialisés avec succès
        if elastic_client and hasattr(elastic_client, '_initialized') and elastic_client._initialized:
            routes.elastic_client = elastic_client
            logger.info("✅ HybridElasticClient injecté dans routes")
        else:
            routes.elastic_client = None
            logger.warning("⚠️ HybridElasticClient non disponible - non injecté")
        
        if qdrant_client and hasattr(qdrant_client, '_initialized') and qdrant_client._initialized:
            routes.qdrant_client = qdrant_client
            logger.info("✅ QdrantClient injecté dans routes")
        else:
            routes.qdrant_client = None
            logger.warning("⚠️ QdrantClient non disponible - non injecté")
        
        # Injecter les services IA
        routes.embedding_service = embedding_service
        routes.reranker_service = reranker_service
        
        # Injecter les services utilitaires
        routes.search_cache = search_cache
        routes.metrics_collector = metrics_collector
        
        logger.info("✅ Injection des dépendances terminée")
        
        # Log du statut final des services
        services_status = {
            "elasticsearch": elastic_client is not None and getattr(elastic_client, '_initialized', False),
            "qdrant": qdrant_client is not None and getattr(qdrant_client, '_initialized', False),
            "embeddings": embedding_service is not None,
            "reranking": reranker_service is not None,
            "cache": search_cache is not None,
            "metrics": metrics_collector is not None
        }
        
        active_services = sum(services_status.values())
        logger.info(f"📊 Services actifs: {active_services}/6")
        
        for service_name, is_active in services_status.items():
            status_icon = "✅" if is_active else "❌"
            logger.info(f"   {status_icon} {service_name}: {'Actif' if is_active else 'Inactif'}")
            
    except Exception as e:
        logger.error(f"💥 Erreur lors de l'injection des dépendances: {e}")
        # En cas d'erreur, injecter au moins None pour éviter les erreurs d'import
        try:
            import search_service.api.routes as routes
            routes.elastic_client = None
            routes.qdrant_client = None
            routes.embedding_service = None
            routes.reranker_service = None
            routes.search_cache = None
            routes.metrics_collector = None
            logger.warning("⚠️ Services par défaut (None) injectés suite à l'erreur")
        except Exception as fallback_error:
            logger.error(f"💥 Impossible d'injecter même les valeurs par défaut: {fallback_error}")


def generate_startup_summary() -> Dict[str, Any]:
    """Génère un résumé complet du démarrage."""
    global startup_diagnostics, elastic_client, qdrant_client, full_diagnostic_report
    
    startup_duration = time.time() - startup_time
    
    # Compter les services opérationnels
    operational_services = []
    failed_services = []
    
    # Services critiques
    if elastic_client and getattr(elastic_client, '_initialized', False):
        client_type = getattr(elastic_client, 'client_type', 'unknown')
        operational_services.append(f"Elasticsearch ({client_type})")
    else:
        failed_services.append("Elasticsearch")
    
    if qdrant_client and getattr(qdrant_client, '_initialized', False):
        operational_services.append("Qdrant")
    else:
        failed_services.append("Qdrant")
    
    # Services IA
    ai_diag = startup_diagnostics.get("ai_services", {})
    if ai_diag.get("embeddings", {}).get("initialized"):
        operational_services.append("OpenAI Embeddings")
    else:
        failed_services.append("OpenAI Embeddings")
    
    if ai_diag.get("reranking", {}).get("initialized"):
        operational_services.append("Cohere Reranking")
    else:
        failed_services.append("Cohere Reranking")
    
    # Services utilitaires
    if search_cache:
        operational_services.append("Cache")
    if metrics_collector:
        operational_services.append("Métriques")
    
    # Déterminer l'état global du service
    critical_services_ok = (elastic_client and getattr(elastic_client, '_initialized', False) and 
                           qdrant_client and getattr(qdrant_client, '_initialized', False))
    
    if critical_services_ok:
        if len(operational_services) >= 4:  # Elasticsearch + Qdrant + au moins 2 autres
            status = "OPTIMAL"
        else:
            status = "DEGRADED"
    else:
        if len(operational_services) >= 1:
            status = "DEGRADED"
        else:
            status = "FAILED"
    
    summary = {
        "startup_duration": round(startup_duration, 2),
        "status": status,
        "operational_services": operational_services,
        "failed_services": failed_services,
        "total_services": len(operational_services) + len(failed_services),
        "success_rate": len(operational_services) / (len(operational_services) + len(failed_services)) * 100,
        "diagnostics": startup_diagnostics,
        "full_diagnostic": full_diagnostic_report
    }
    
    # Log du résumé
    logger.info("🎊 === RÉSUMÉ DU DÉMARRAGE ===")
    logger.info(f"⏱️ Durée totale: {startup_duration:.2f}s")
    logger.info(f"📊 Statut: {status}")
    logger.info(f"✅ Services opérationnels ({len(operational_services)}):")
    for service in operational_services:
        logger.info(f"   ✓ {service}")
    
    if failed_services:
        logger.info(f"❌ Services en échec ({len(failed_services)}):")
        for service in failed_services:
            logger.info(f"   ✗ {service}")
    
    # Messages selon le statut
    if status == "OPTIMAL":
        logger.info("🎉 Service de recherche OPTIMAL")
        logger.info("   ✓ Recherche lexicale ET sémantique")
        logger.info("   ✓ Toutes les fonctionnalités disponibles")
    elif status == "DEGRADED":
        logger.warning("⚠️ Service de recherche en mode dégradé")
        if elastic_client and getattr(elastic_client, '_initialized', False) and not (qdrant_client and getattr(qdrant_client, '_initialized', False)):
            logger.warning("   ✓ Recherche lexicale disponible")
            logger.warning("   ❌ Recherche sémantique indisponible")
        elif qdrant_client and getattr(qdrant_client, '_initialized', False) and not (elastic_client and getattr(elastic_client, '_initialized', False)):
            logger.warning("   ❌ Recherche lexicale indisponible")
            logger.warning("   ✓ Recherche sémantique disponible")
    else:
        logger.error("🚨 Service de recherche indisponible")
        logger.error("   ❌ Vérifiez la configuration BONSAI_URL et QDRANT_URL")
        logger.error("   ❌ Vérifiez la connectivité réseau")
    
    logger.info("📊 Endpoints disponibles:")
    if ROUTES_AVAILABLE:
        logger.info("   GET  /health - Vérification de santé")
        logger.info("   GET  /diagnostics - Diagnostics détaillés")
        logger.info("   POST /api/v1/search - Recherche de transactions")
        logger.info("   GET  /api/v1/search/suggest - Suggestions de recherche")
    else:
        logger.info("   GET  /health - Vérification de santé (basique)")
        logger.info("   GET  /diagnostics - Diagnostics détaillés")
    logger.info("=" * 100)
    
    return summary

# ==================== CYCLE DE VIE DE L'APPLICATION ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application avec diagnostics complets."""
    global startup_time, startup_diagnostics, elastic_client, qdrant_client, full_diagnostic_report
    
    try:
        # === PHASE DE DÉMARRAGE ===
        log_startup_banner()
        startup_time = time.time()
        
        # 1. Diagnostic complet de démarrage
        logger.info("🔬 Lancement du diagnostic complet...")
        full_diagnostic_report = await run_startup_diagnostics()
        log_diagnostic_summary(full_diagnostic_report)
        
        # 2. Utilisation du système d'initialisation
        logger.info("🚀 Utilisation du système d'initialisation hybride...")
        initialization_report = await initialize_search_clients()
        startup_diagnostics.update(initialization_report)
        
        # Récupérer les clients initialisés
        elastic_client = initialization_report.get("clients", {}).get("elasticsearch")
        qdrant_client = initialization_report.get("clients", {}).get("qdrant")
        
        # Log du rapport d'initialisation
        log_initialization_summary(initialization_report)
        
        # 3. Création des collections Qdrant si nécessaire
        if qdrant_client and qdrant_client._initialized:
            logger.info("🏗️ Vérification des collections Qdrant...")
            collections_created = await create_collections_if_needed(qdrant_client)
            if collections_created:
                logger.info("✅ Collections Qdrant prêtes")
            else:
                logger.warning("⚠️ Problème avec les collections Qdrant")
        
        # 4. Initialisation des services IA
        ai_diagnostics = await initialize_ai_services()
        startup_diagnostics["ai_services"] = ai_diagnostics
        
        # 5. Initialisation du cache et des métriques
        utils_diagnostics = initialize_cache_and_metrics()
        startup_diagnostics["utilities"] = utils_diagnostics
        
        # 6. Injection des dépendances APRÈS initialisation
        inject_dependencies()
        
        # 7. Validation finale de la fonctionnalité
        logger.info("🧪 Validation finale de la fonctionnalité...")
        functionality_validation = await validate_clients_functionality(elastic_client, qdrant_client)
        startup_diagnostics["functionality_validation"] = functionality_validation
        
        # 8. Génération du résumé final
        summary = generate_startup_summary()
        
        # Point de contrôle pour la santé générale
        if summary["status"] == "FAILED":
            logger.error("🚨 ATTENTION: Aucun service critique disponible")
            logger.error("💡 Le service démarre quand même pour permettre le debugging")
        
        logger.info("🎯 Service de recherche Harena démarré et prêt")
        
        # === PHASE D'EXÉCUTION ===
        yield
        
    except Exception as e:
        logger.error(f"💥 Erreur critique lors du démarrage: {e}", exc_info=True)
        raise
    
    finally:
        # === PHASE D'ARRÊT ===
        logger.info("🔄 Arrêt du service de recherche...")
        
        # Fermeture des clients
        if elastic_client:
            try:
                await elastic_client.close()
                logger.info("✅ Client Elasticsearch fermé")
            except Exception as e:
                logger.error(f"❌ Erreur fermeture Elasticsearch: {e}")
        
        if qdrant_client:
            try:
                await qdrant_client.close()
                logger.info("✅ Client Qdrant fermé")
            except Exception as e:
                logger.error(f"❌ Erreur fermeture Qdrant: {e}")
        
        # Fermeture des services IA
        if embedding_service:
            try:
                if hasattr(embedding_service, 'close'):
                    await embedding_service.close()
                logger.info("✅ Service embeddings fermé")
            except Exception as e:
                logger.error(f"❌ Erreur fermeture embeddings: {e}")
        
        if reranker_service:
            try:
                if hasattr(reranker_service, 'close'):
                    await reranker_service.close()
                logger.info("✅ Service reranking fermé")
            except Exception as e:
                logger.error(f"❌ Erreur fermeture reranking: {e}")
        
        logger.info("🏁 Service de recherche arrêté")

# ==================== CRÉATION DE L'APPLICATION ====================

def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI."""
    
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride pour transactions financières",
        version="2.0.0",
        docs_url="/docs" if ROUTES_AVAILABLE else None,
        redoc_url="/redoc" if ROUTES_AVAILABLE else None,
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # À restreindre en production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Inclusion des routes si disponibles
    if ROUTES_AVAILABLE and router:
        app.include_router(router, prefix="/api/v1")
        logger.info("✅ Routes API incluses")
    else:
        logger.warning("⚠️ Routes API non disponibles - endpoints de base uniquement")
    
    # Route de santé simple
    @app.get("/health")
    async def health_check():
        """Check de santé simple du service."""
        global elastic_client, qdrant_client
        
        health_status = {
            "service": "search_service",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "uptime_seconds": (time.time() - startup_time) if startup_time else 0,
            "components": {
                "elasticsearch": {
                    "available": elastic_client is not None and getattr(elastic_client, '_initialized', False),
                    "client_type": getattr(elastic_client, 'client_type', None) if elastic_client else None
                },
                "qdrant": {
                    "available": qdrant_client is not None and getattr(qdrant_client, '_initialized', False)
                },
                "ai_services": {
                    "embeddings": embedding_service is not None,
                    "reranking": reranker_service is not None
                },
                "utilities": {
                    "cache": search_cache is not None,
                    "metrics": metrics_collector is not None
                }
            }
        }
        
        # Déterminer le statut global
        critical_ok = (health_status["components"]["elasticsearch"]["available"] and 
                      health_status["components"]["qdrant"]["available"])
        
        if critical_ok:
            health_status["status"] = "healthy"
        elif (health_status["components"]["elasticsearch"]["available"] or 
              health_status["components"]["qdrant"]["available"]):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "unhealthy"
        
        return health_status
    
    # Route de diagnostics détaillés
    @app.get("/diagnostics")
    async def get_diagnostics():
        """Retourne les diagnostics détaillés du démarrage."""
        global startup_diagnostics, full_diagnostic_report
        
        return {
            "service": "search_service",
            "timestamp": datetime.now().isoformat(),
            "startup_diagnostics": startup_diagnostics,
            "full_diagnostic_report": full_diagnostic_report,
            "current_status": {
                "elasticsearch_initialized": elastic_client is not None and getattr(elastic_client, '_initialized', False),
                "qdrant_initialized": qdrant_client is not None and getattr(qdrant_client, '_initialized', False),
                "elasticsearch_type": getattr(elastic_client, 'client_type', None) if elastic_client else None,
                "uptime_seconds": (time.time() - startup_time) if startup_time else 0,
                "services_available": {
                    "routes": ROUTES_AVAILABLE,
                    "embedding_service": EMBEDDING_SERVICE_AVAILABLE,
                    "reranker_service": RERANKER_SERVICE_AVAILABLE,
                    "cache_service": CACHE_SERVICE_AVAILABLE,
                    "metrics_service": METRICS_SERVICE_AVAILABLE
                }
            }
        }
    
    # Route de test de connectivité
    @app.get("/connectivity")
    async def test_connectivity():
        """Teste la connectivité des services en temps réel."""
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "elasticsearch": {"available": False, "healthy": False, "error": None},
            "qdrant": {"available": False, "healthy": False, "error": None}
        }
        
        # Test Elasticsearch
        if elastic_client and elastic_client._initialized:
            try:
                test_results["elasticsearch"]["available"] = True
                test_results["elasticsearch"]["healthy"] = await elastic_client.is_healthy()
            except Exception as e:
                test_results["elasticsearch"]["error"] = str(e)
        else:
            test_results["elasticsearch"]["error"] = "Client not initialized"
        
        # Test Qdrant
        if qdrant_client and qdrant_client._initialized:
            try:
                test_results["qdrant"]["available"] = True
                test_results["qdrant"]["healthy"] = await qdrant_client.is_healthy()
            except Exception as e:
                test_results["qdrant"]["error"] = str(e)
        else:
            test_results["qdrant"]["error"] = "Client not initialized"
        
        return test_results
    
    # Route d'informations système
    @app.get("/info")
    async def get_system_info():
        """Retourne les informations système du service."""
        import platform
        
        return {
            "service": "harena_search_service",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (time.time() - startup_time) if startup_time else 0,
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture()
            },
            "configuration": {
                "bonsai_configured": bool(settings.BONSAI_URL),
                "qdrant_configured": bool(settings.QDRANT_URL),
                "openai_configured": bool(settings.OPENAI_API_KEY),
                "cohere_configured": bool(getattr(settings, 'COHERE_KEY', None))
            },
            "features": {
                "lexical_search": elastic_client is not None and getattr(elastic_client, '_initialized', False),
                "semantic_search": qdrant_client is not None and getattr(qdrant_client, '_initialized', False),
                "hybrid_search": (elastic_client is not None and getattr(elastic_client, '_initialized', False) and
                                qdrant_client is not None and getattr(qdrant_client, '_initialized', False)),
                "api_routes": ROUTES_AVAILABLE,
                "ai_embeddings": embedding_service is not None,
                "ai_reranking": reranker_service is not None
            }
        }
    
    # Gestionnaire d'erreur global
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Gestionnaire d'erreur global pour toutes les exceptions."""
        logger.error(f"Erreur non gérée: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "Une erreur inattendue s'est produite",
                "timestamp": datetime.now().isoformat(),
                "service": "search_service"
            }
        )
    
    return app

# ==================== POINT D'ENTRÉE ====================

# Créer l'application
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    # Configuration du serveur
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Démarrage du serveur sur {host}:{port}")
    
    # Démarrer le serveur
    uvicorn.run(
        "search_service.main:app",
        host=host,
        port=port,
        reload=False,  # Désactivé en production
        access_log=True,
        log_level="info"
    )