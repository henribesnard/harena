"""
Application Harena complète pour déploiement Heroku avec focus sur le Search Service.

Module optimisé pour le déploiement sur Heroku, avec diagnostic détaillé du service de recherche
au démarrage car c'est le service central de la plateforme.
"""

import logging
import os
import sys
import traceback
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configuration du logging AVANT tout autre import
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("heroku_startup")

try:
    logger.info("🚀 === DÉMARRAGE HARENA APP COMPLÈTE (FOCUS SEARCH SERVICE) ===")
    
    # Affichage de la bannière de démarrage
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🏦 HARENA FINANCE PLATFORM 🏦                      ║
║                        Application Complète - Version Heroku                 ║
║                          🔍 Focus: Search Service Central 🔍                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Log des variables d'environnement critiques (sans exposer les secrets)
    logger.info("🔧 === VÉRIFICATION CONFIGURATION CRITIQUE ===")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'not_set')}")
    logger.info(f"DATABASE_URL configured: {bool(os.environ.get('DATABASE_URL'))}")
    
    # Variables critiques pour le Search Service
    search_critical_vars = {
        "BONSAI_URL": "Elasticsearch via Bonsai (recherche lexicale)",
        "QDRANT_URL": "Qdrant (recherche sémantique)",
        "OPENAI_API_KEY": "OpenAI (génération d'embeddings)", 
        "COHERE_KEY": "Cohere (reranking intelligent)"
    }
    
    logger.info("🔍 === CONFIGURATION SEARCH SERVICE ===")
    search_config_status = {}
    for var, description in search_critical_vars.items():
        is_configured = bool(os.environ.get(var))
        search_config_status[var] = is_configured
        if is_configured:
            if "URL" in var:
                # Masquer les credentials dans l'URL
                safe_value = os.environ.get(var, "").split('@')[-1] if '@' in os.environ.get(var, "") else os.environ.get(var, "")
                logger.info(f"✅ {var}: {safe_value}")
            else:
                logger.info(f"✅ {var}: Configuré")
        else:
            logger.warning(f"⚠️ {var}: NON CONFIGURÉ - {description}")
    
    # Évaluation de la configuration du Search Service
    critical_search_vars = ["BONSAI_URL", "QDRANT_URL"]
    critical_configured = sum(1 for var in critical_search_vars if search_config_status.get(var))
    
    if critical_configured == 2:
        logger.info("🎉 SEARCH SERVICE: Configuration complète (recherche hybride disponible)")
    elif critical_configured == 1:
        logger.warning("⚠️ SEARCH SERVICE: Configuration partielle (recherche limitée)")
    else:
        logger.error("🚨 SEARCH SERVICE: Configuration insuffisante (service non opérationnel)")
    
    # Autres variables importantes
    other_vars = {
        "BRIDGE_CLIENT_ID": "Synchronisation bancaire",
        "DEEPSEEK_API_KEY": "Assistant conversationnel"
    }
    
    logger.info("🔧 === AUTRES SERVICES ===")
    for var, description in other_vars.items():
        is_configured = bool(os.environ.get(var))
        if is_configured:
            logger.info(f"✅ {var}: Configuré - {description}")
        else:
            logger.info(f"📋 {var}: Non configuré - {description}")
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Correction de l'URL de base de données pour Heroku
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
        os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        logger.info("✅ DATABASE_URL corrigé pour SQLAlchemy 1.4+")

    # Définir l'environnement global
    os.environ["ENVIRONMENT"] = os.getenv("ENVIRONMENT", "production")

    # S'assurer que tous les modules sont accessibles
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logger.info(f"✅ Répertoire courant ajouté au sys.path: {current_dir}")

    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import asyncio

    logger.info("✅ Imports de base réussis")

    # ======== VARIABLES GLOBALES ========

    startup_time = None
    search_service_status = {}
    global_diagnostics = {}

    # ======== REGISTRE DE SERVICES ========

    class ServiceRegistry:
        """Gestionnaire central de tous les services avec focus sur Search Service."""
        
        def __init__(self):
            self.services = {}
            self.failed_services = {}
            self.service_apps = {}
            self.search_service_health = {}
            
        def register_service(self, name: str, router, prefix: str, description: str = ""):
            """Enregistre un service avec métadonnées."""
            try:
                if router:
                    self.services[name] = {
                        "router": router,
                        "prefix": prefix,
                        "description": description,
                        "status": "ok",
                        "registered_at": datetime.now(),
                        "error": None
                    }
                    logger.info(f"✅ Service '{name}' enregistré: {prefix}")
                    return True
                else:
                    self.failed_services[name] = {
                        "prefix": prefix,
                        "description": description,
                        "error": "Router is None",
                        "failed_at": datetime.now()
                    }
                    logger.error(f"❌ Service '{name}' échoué: Router is None")
                    return False
                    
            except Exception as e:
                self.failed_services[name] = {
                    "prefix": prefix,
                    "description": description,
                    "error": str(e),
                    "failed_at": datetime.now()
                }
                logger.error(f"❌ Service '{name}' échoué: {e}")
                return False
        
        def get_service_status(self) -> Dict[str, str]:
            """Retourne le statut de tous les services."""
            status = {}
            for name, info in self.services.items():
                status[name] = info["status"]
            for name in self.failed_services:
                status[name] = "failed"
            return status
        
        def get_search_service_priority_status(self) -> Dict[str, Any]:
            """Retourne le statut prioritaire du Search Service."""
            search_status = {
                "service_registered": "search_service" in self.services,
                "service_status": self.services.get("search_service", {}).get("status", "not_registered"),
                "health_check_status": self.search_service_health,
                "configuration_status": search_config_status,
                "overall_assessment": "unknown"
            }
            
            # Évaluation globale du Search Service
            if search_status["service_registered"] and search_status["service_status"] == "ok":
                if search_config_status.get("BONSAI_URL") and search_config_status.get("QDRANT_URL"):
                    search_status["overall_assessment"] = "fully_operational"
                elif search_config_status.get("BONSAI_URL") or search_config_status.get("QDRANT_URL"):
                    search_status["overall_assessment"] = "partially_operational"
                else:
                    search_status["overall_assessment"] = "configuration_incomplete"
            else:
                search_status["overall_assessment"] = "service_failed"
            
            return search_status

    service_registry = ServiceRegistry()

    # ======== FONCTIONS UTILITAIRES ========

    def _check_database_connection() -> bool:
        """Teste la connexion à la base de données."""
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception:
            return False

    async def check_search_service_health() -> Dict[str, Any]:
        """Effectue un health check détaillé du Search Service."""
        logger.info("🔍 === HEALTH CHECK SEARCH SERVICE ===")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "service_available": False,
            "health_endpoint_responsive": False,
            "search_capabilities": {},
            "detailed_status": {},
            "response_time": None,
            "error": None
        }
        
        try:
            # Tentative d'import du search service pour vérifier sa disponibilité
            try:
                from search_service.main import app as search_app
                from search_service.api.routes import elastic_client, qdrant_client
                health_status["service_available"] = True
                logger.info("✅ Search Service importé avec succès")
                
                # Vérifier l'état des clients
                health_status["detailed_status"] = {
                    "elastic_client_initialized": elastic_client is not None,
                    "qdrant_client_initialized": qdrant_client is not None,
                    "both_clients_available": (elastic_client is not None and qdrant_client is not None)
                }
                
                # Test des capacités de recherche
                if elastic_client:
                    try:
                        elastic_healthy = await elastic_client.is_healthy()
                        health_status["search_capabilities"]["lexical_search"] = elastic_healthy
                        logger.info(f"🔍 Elasticsearch (Bonsai): {'✅ Opérationnel' if elastic_healthy else '❌ Non opérationnel'}")
                    except Exception as e:
                        health_status["search_capabilities"]["lexical_search"] = False
                        logger.error(f"❌ Erreur test Elasticsearch: {e}")
                else:
                    health_status["search_capabilities"]["lexical_search"] = False
                    logger.warning("⚠️ Client Elasticsearch non initialisé")
                
                if qdrant_client:
                    try:
                        qdrant_healthy = await qdrant_client.is_healthy()
                        health_status["search_capabilities"]["semantic_search"] = qdrant_healthy
                        logger.info(f"🎯 Qdrant: {'✅ Opérationnel' if qdrant_healthy else '❌ Non opérationnel'}")
                    except Exception as e:
                        health_status["search_capabilities"]["semantic_search"] = False
                        logger.error(f"❌ Erreur test Qdrant: {e}")
                else:
                    health_status["search_capabilities"]["semantic_search"] = False
                    logger.warning("⚠️ Client Qdrant non initialisé")
                
                # Calculer les capacités hybrides
                health_status["search_capabilities"]["hybrid_search"] = (
                    health_status["search_capabilities"].get("lexical_search", False) and
                    health_status["search_capabilities"].get("semantic_search", False)
                )
                
                health_status["health_endpoint_responsive"] = True
                
            except ImportError as e:
                logger.error(f"❌ Impossible d'importer le Search Service: {e}")
                health_status["error"] = f"Import failed: {e}"
            except Exception as e:
                logger.error(f"❌ Erreur lors de la vérification du Search Service: {e}")
                health_status["error"] = f"Health check failed: {e}"
        
        except Exception as e:
            logger.error(f"💥 Erreur critique lors du health check: {e}")
            health_status["error"] = f"Critical error: {e}"
        
        # Mise à jour du registre
        service_registry.search_service_health = health_status
        
        # Résumé du statut
        if health_status["search_capabilities"].get("hybrid_search"):
            logger.info("🎉 SEARCH SERVICE: Complètement opérationnel (recherche hybride)")
        elif health_status["search_capabilities"].get("lexical_search") or health_status["search_capabilities"].get("semantic_search"):
            logger.warning("⚠️ SEARCH SERVICE: Partiellement opérationnel")
        else:
            logger.error("🚨 SEARCH SERVICE: Non opérationnel")
        
        return health_status

    def _count_configured_external_services() -> int:
        """Compte le nombre de services externes configurés."""
        external_vars = [
            "BRIDGE_CLIENT_ID", "BRIDGE_CLIENT_SECRET",
            "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
            "QDRANT_URL", "COHERE_KEY", "BONSAI_URL"
        ]
        return len([var for var in external_vars if os.environ.get(var)])

    # ======== FONCTION DU CYCLE DE VIE ========

    async def startup():
        """Fonction d'initialisation avec focus sur le Search Service."""
        global startup_time, global_diagnostics
        logger.info("📋 === DÉMARRAGE APPLICATION HARENA COMPLÈTE ===")
        startup_time = time.time()
        
        # Vérification des variables d'environnement critiques
        required_env_vars = ["DATABASE_URL"]
        missing_required = [var for var in required_env_vars if not os.environ.get(var)]
        if missing_required:
            logger.error(f"❌ Variables d'environnement critiques manquantes: {', '.join(missing_required)}")
            raise RuntimeError(f"Missing required environment variables: {missing_required}")
        
        # Health check prioritaire du Search Service
        logger.info("🔍 === DIAGNOSTIC PRIORITAIRE: SEARCH SERVICE ===")
        search_health = await check_search_service_health()
        global_diagnostics["search_service_health"] = search_health
        
        # Évaluation de la criticité
        search_assessment = service_registry.get_search_service_priority_status()
        global_diagnostics["search_service_assessment"] = search_assessment
        
        if search_assessment["overall_assessment"] == "fully_operational":
            logger.info("🎉 ✅ SEARCH SERVICE: Prêt pour la production (recherche hybride)")
        elif search_assessment["overall_assessment"] == "partially_operational":
            logger.warning("⚠️ 🔧 SEARCH SERVICE: Fonctionnel mais capacités limitées")
        else:
            logger.error("🚨 ❌ SEARCH SERVICE: Non opérationnel - Impact critique sur la plateforme")
        
        logger.info("✅ Initialisation des services critiques terminée")

    async def shutdown():
        """Fonction de nettoyage lors de l'arrêt."""
        logger.info("⏹️ === ARRÊT APPLICATION HARENA ===")
        
        # Nettoyage spécifique du Search Service
        try:
            from search_service.core.embeddings import embedding_service
            if hasattr(embedding_service, 'close'):
                await embedding_service.close()
                logger.info("✅ Search Service - EmbeddingService fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Search EmbeddingService: {e}")
        
        try:
            from search_service.core.reranker import reranker_service
            if hasattr(reranker_service, 'close'):
                await reranker_service.close()
                logger.info("✅ Search Service - RerankerService fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture RerankerService: {e}")

    # ======== GESTIONNAIRE DE CYCLE DE VIE ========

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Gestionnaire du cycle de vie de l'application."""
        try:
            await startup()
            yield
        except Exception as e:
            logger.error(f"❌ Erreur durant le startup: {e}")
            raise
        finally:
            await shutdown()

    # ======== CRÉATION DE L'APPLICATION ========

    logger.info("🏗️ === CRÉATION APPLICATION FASTAPI ===")

    app = FastAPI(
        title="Harena Finance API (Production Complete - Search Service Focus)",
        description="API complète Harena avec diagnostic prioritaire du Search Service",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    logger.info("✅ Application FastAPI créée")

    # Configuration CORS
    ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "https://app.harena.finance").split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    logger.info("✅ Middleware CORS configuré")

    # ======== MIDDLEWARE DE LOGGING ========

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Middleware pour logger les requêtes importantes."""
        start_time = time.time()
        
        # Logger les requêtes importantes
        if request.url.path not in ["/health", "/", "/docs", "/redoc", "/openapi.json", "/favicon.ico"]:
            logger.info(f"🔄 {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Logger les requêtes lentes
        if process_time > 3.0:
            logger.warning(f"🐌 Requête lente: {request.method} {request.url.path} - {process_time:.2f}s")
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Service-Version"] = "1.0.0-complete"
        response.headers["X-Search-Service-Status"] = service_registry.get_search_service_priority_status()["overall_assessment"]
        return response

    # ======== ENDPOINTS DE BASE AVEC FOCUS SEARCH SERVICE ========

    @app.get("/", tags=["health"])
    async def root():
        """Point d'entrée racine avec statut du Search Service."""
        uptime = time.time() - startup_time if startup_time else 0
        search_status = service_registry.get_search_service_priority_status()
        
        return {
            "message": "Harena Finance API - Tous services avec focus Search Service",
            "status": "online",
            "version": "1.0.0-complete",
            "uptime_seconds": round(uptime, 2),
            "search_service": {
                "status": search_status["overall_assessment"],
                "capabilities": global_diagnostics.get("search_service_health", {}).get("search_capabilities", {}),
                "critical_service": True
            },
            "services_count": len([s for s in service_registry.services.values() if s["status"] == "ok"]),
            "timestamp": datetime.now().isoformat(),
            "environment": os.environ.get("ENVIRONMENT", "production")
        }

    @app.get("/health", tags=["health"])
    async def health_check():
        """Endpoint de santé avec priorité Search Service."""
        service_statuses = service_registry.get_service_status()
        search_priority_status = service_registry.get_search_service_priority_status()
        
        # Calculer des métriques de base
        total_services = len(service_statuses)
        active_services = len([s for s in service_statuses.values() if s == "ok"])
        health_ratio = active_services / total_services if total_services > 0 else 0
        
        uptime = time.time() - startup_time if startup_time else 0
        
        # Statut global basé sur le Search Service
        if search_priority_status["overall_assessment"] == "fully_operational":
            overall_status = "excellent"
        elif search_priority_status["overall_assessment"] == "partially_operational":
            overall_status = "good"
        elif search_priority_status["overall_assessment"] == "configuration_incomplete":
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "search_service_priority": {
                "status": search_priority_status["overall_assessment"],
                "is_critical_service": True,
                "health_details": global_diagnostics.get("search_service_health", {}),
                "configuration": search_priority_status["configuration_status"]
            },
            "service_availability": {
                "active": active_services,
                "total": total_services,
                "percentage": round(health_ratio * 100, 2)
            },
            "uptime": {
                "seconds": uptime,
                "human": str(timedelta(seconds=int(uptime))),
                "hours": round(uptime / 3600, 2)
            },
            "system": {
                "python_version": sys.version.split()[0],
                "environment": os.environ.get("ENVIRONMENT", "production"),
                "platform": "Heroku"
            },
            "database": {
                "connected": _check_database_connection()
            },
            "external_dependencies": {
                "configured_count": _count_configured_external_services(),
                "critical_search_services": {
                    "bonsai_elasticsearch": bool(os.environ.get("BONSAI_URL")),
                    "qdrant_vector_db": bool(os.environ.get("QDRANT_URL")),
                    "openai_embeddings": bool(os.environ.get("OPENAI_API_KEY")),
                    "cohere_reranking": bool(os.environ.get("COHERE_KEY"))
                },
                "other_services": {
                    "bridge_api": bool(os.environ.get("BRIDGE_CLIENT_ID")),
                    "ai_conversation": bool(os.environ.get("DEEPSEEK_API_KEY"))
                }
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/search-service-status", tags=["search"])
    async def search_service_detailed_status():
        """Endpoint dédié au statut détaillé du Search Service."""
        search_health = await check_search_service_health()
        search_assessment = service_registry.get_search_service_priority_status()
        
        return {
            "service": "search_service",
            "priority": "critical",
            "timestamp": datetime.now().isoformat(),
            "overall_assessment": search_assessment["overall_assessment"],
            "service_registration": {
                "registered": search_assessment["service_registered"],
                "status": search_assessment["service_status"]
            },
            "configuration": {
                "bonsai_url_configured": search_config_status.get("BONSAI_URL", False),
                "qdrant_url_configured": search_config_status.get("QDRANT_URL", False),
                "openai_key_configured": search_config_status.get("OPENAI_API_KEY", False),
                "cohere_key_configured": search_config_status.get("COHERE_KEY", False),
                "critical_services_ready": (search_config_status.get("BONSAI_URL", False) and 
                                           search_config_status.get("QDRANT_URL", False))
            },
            "runtime_health": search_health,
            "capabilities": {
                "lexical_search": search_health.get("search_capabilities", {}).get("lexical_search", False),
                "semantic_search": search_health.get("search_capabilities", {}).get("semantic_search", False),
                "hybrid_search": search_health.get("search_capabilities", {}).get("hybrid_search", False),
                "ai_powered_features": search_config_status.get("OPENAI_API_KEY", False) and search_config_status.get("COHERE_KEY", False)
            },
            "recommendations": _get_search_service_recommendations(search_assessment, search_health)
        }

    def _get_search_service_recommendations(assessment: Dict, health: Dict) -> List[str]:
        """Génère des recommandations basées sur l'état du Search Service."""
        recommendations = []
        
        if not assessment["configuration_status"].get("BONSAI_URL"):
            recommendations.append("🔧 Configurez BONSAI_URL pour activer la recherche lexicale")
        
        if not assessment["configuration_status"].get("QDRANT_URL"):
            recommendations.append("🔧 Configurez QDRANT_URL pour activer la recherche sémantique")
        
        if not assessment["configuration_status"].get("OPENAI_API_KEY"):
            recommendations.append("🤖 Configurez OPENAI_API_KEY pour les embeddings automatiques")
        
        if not assessment["configuration_status"].get("COHERE_KEY"):
            recommendations.append("🎯 Configurez COHERE_KEY pour le reranking intelligent")
        
        if not health.get("search_capabilities", {}).get("lexical_search"):
            recommendations.append("🔍 Vérifiez la connectivité à Bonsai Elasticsearch")
        
        if not health.get("search_capabilities", {}).get("semantic_search"):
            recommendations.append("🎯 Vérifiez la connectivité à Qdrant")
        
        if not health.get("service_available"):
            recommendations.append("🚨 Redémarrez l'application pour réinitialiser le Search Service")
        
        if not recommendations:
            recommendations.append("✅ Search Service complètement opérationnel")
        
        return recommendations

    # ======== IMPORTATION ET ENREGISTREMENT DES SERVICES ========

    logger.info("📦 === IMPORTATION DE TOUS LES SERVICES ===")

    # 1. User Service
    try:
        from user_service.api.routes import router as user_router
        success = service_registry.register_service(
            "user_service", 
            user_router, 
            "/api/v1/users",
            "Gestion des utilisateurs et authentification"
        )
        if success:
            app.include_router(user_router, prefix="/api/v1/users", tags=["users"])
    except Exception as e:
        logger.error(f"❌ Erreur User Service: {e}")
        service_registry.register_service("user_service", None, "/api/v1/users", f"Erreur: {e}")

    # 2. Sync Service (tous les sous-modules)
    sync_services = [
        ("sync", "/api/v1/sync", "Synchronisation principale"),
        ("accounts", "/api/v1/accounts", "Gestion des comptes bancaires"),
        ("transactions", "/api/v1/transactions", "Gestion des transactions"),
        ("categories", "/api/v1/categories", "Gestion des catégories"),
        ("banks", "/api/v1/banks", "Gestion des banques"),
        ("items", "/api/v1/items", "Gestion des items Bridge"),
        ("webhooks", "/api/v1/webhooks", "Gestion des webhooks")
    ]

    for service_name, prefix, description in sync_services:
        try:
            module_path = f"sync_service.api.{service_name}"
            module = __import__(module_path, fromlist=["router"])
            router = getattr(module, "router")
            
            success = service_registry.register_service(
                f"sync_{service_name}", 
                router, 
                prefix,
                description
            )
            if success:
                app.include_router(router, prefix=prefix, tags=[service_name])
        except Exception as e:
            logger.error(f"❌ Erreur Sync Service ({service_name}): {e}")
            service_registry.register_service(f"sync_{service_name}", None, prefix, f"Erreur: {e}")

    # 3. Search Service (SERVICE CRITIQUE - TRAITEMENT PRIORITAIRE)
    logger.info("🔍 === ENREGISTREMENT SEARCH SERVICE (PRIORITÉ CRITIQUE) ===")
    try:
        from search_service.api.routes import router as search_router
        success = service_registry.register_service(
            "search_service", 
            search_router, 
            "/api/v1/search",
            "🎯 SERVICE CRITIQUE: Recherche hybride (lexicale + sémantique)"
        )
        if success:
            app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
            logger.info("🎉 ✅ SEARCH SERVICE: Enregistré avec succès - Endpoint /api/v1/search disponible")
        else:
            logger.error("🚨 ❌ SEARCH SERVICE: Échec d'enregistrement - Impact critique")
    except Exception as e:
        logger.error(f"💥 ❌ SEARCH SERVICE: Erreur critique lors de l'importation: {e}")
        logger.error(traceback.format_exc())
        service_registry.register_service("search_service", None, "/api/v1/search", f"Erreur critique: {e}")

    # 4. Enrichment Service
    try:
        from enrichment_service.api.routes import router as enrichment_router
        success = service_registry.register_service(
            "enrichment_service", 
            enrichment_router, 
            "/api/v1/enrichment",
            "Enrichissement et vectorisation des transactions"
        )
        if success:
            app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
    except Exception as e:
        logger.error(f"❌ Erreur Enrichment Service: {e}")
        service_registry.register_service("enrichment_service", None, "/api/v1/enrichment", f"Erreur: {e}")

    # 5. Conversation Service
    try:
        from conversation_service.api.routes import router as conversation_router
        success = service_registry.register_service(
            "conversation_service", 
            conversation_router, 
            "/api/v1/conversation",
            "Assistant conversationnel IA"
        )
        if success:
            app.include_router(conversation_router, prefix="/api/v1/conversation", tags=["conversation"])
    except Exception as e:
        logger.error(f"❌ Erreur Conversation Service: {e}")
        service_registry.register_service("conversation_service", None, "/api/v1/conversation", f"Erreur: {e}")

    # ======== ENDPOINTS D'ADMINISTRATION ET DEBUG ========

    @app.get("/status", tags=["health"])
    async def application_status():
        """Statut global condensé avec focus Search Service."""
        search_status = service_registry.get_search_service_priority_status()
        
        return {
            "version": "1.0.0-complete",
            "build": "heroku-production",
            "critical_service_status": {
                "search_service": search_status["overall_assessment"],
                "impact": "critical" if search_status["overall_assessment"] != "fully_operational" else "none"
            },
            "services": {
                "total": len(service_registry.services) + len(service_registry.failed_services),
                "active": len(service_registry.services),
                "failed": len(service_registry.failed_services)
            },
            "python_version": sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/debug/services", tags=["debug"])
    async def debug_services():
        """Debug des services enregistrés (développement uniquement)."""
        if os.environ.get("ENVIRONMENT", "production").lower() == "production":
            raise HTTPException(status_code=404, detail="Not found")
        
        debug_info = {}
        for name, info in service_registry.services.items():
            debug_info[name] = {
                "status": info["status"],
                "prefix": info["prefix"],
                "description": info["description"],
                "has_router": info["router"] is not None,
                "router_type": type(info["router"]).__name__ if info["router"] else None,
                "registered_at": info["registered_at"].isoformat()
            }
        
        failed_info = {}
        for name, info in service_registry.failed_services.items():
            failed_info[name] = {
                "prefix": info["prefix"],
                "description": info["description"],
                "error": info["error"],
                "failed_at": info["failed_at"].isoformat()
            }
        
        return {
            "successful_services": debug_info,
            "failed_services": failed_info,
            "search_service_priority_status": service_registry.get_search_service_priority_status(),
            "registry_state": {
                "total_services": len(service_registry.services),
                "failed_services": len(service_registry.failed_services)
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/debug/search-service-deep", tags=["debug"])
    async def debug_search_service_deep():
        """Diagnostic approfondi du Search Service (développement uniquement)."""
        if os.environ.get("ENVIRONMENT", "production").lower() == "production":
            raise HTTPException(status_code=404, detail="Not found")
        
        try:
            # Tentative d'accès aux composants internes du Search Service
            diagnostic = {
                "timestamp": datetime.now().isoformat(),
                "import_status": {},
                "client_status": {},
                "configuration_details": {},
                "runtime_checks": {}
            }
            
            # Test d'import de tous les composants
            components = [
                ("search_service.main", "Application principale"),
                ("search_service.api.routes", "Routes API"),
                ("search_service.storage.elastic_client", "Client Elasticsearch"),
                ("search_service.storage.qdrant_client", "Client Qdrant"),
                ("search_service.core.embeddings", "Service embeddings"),
                ("search_service.core.reranker", "Service reranking")
            ]
            
            for module_name, description in components:
                try:
                    __import__(module_name)
                    diagnostic["import_status"][module_name] = {
                        "success": True,
                        "description": description
                    }
                except Exception as e:
                    diagnostic["import_status"][module_name] = {
                        "success": False,
                        "error": str(e),
                        "description": description
                    }
            
            # Vérification des clients
            try:
                from search_service.api.routes import elastic_client, qdrant_client, search_cache, metrics_collector
                diagnostic["client_status"] = {
                    "elastic_client": {
                        "initialized": elastic_client is not None,
                        "type": type(elastic_client).__name__ if elastic_client else None
                    },
                    "qdrant_client": {
                        "initialized": qdrant_client is not None,
                        "type": type(qdrant_client).__name__ if qdrant_client else None
                    },
                    "search_cache": {
                        "initialized": search_cache is not None,
                        "type": type(search_cache).__name__ if search_cache else None
                    },
                    "metrics_collector": {
                        "initialized": metrics_collector is not None,
                        "type": type(metrics_collector).__name__ if metrics_collector else None
                    }
                }
            except Exception as e:
                diagnostic["client_status"]["error"] = str(e)
            
            # Configuration détaillée
            from config_service.config import settings
            diagnostic["configuration_details"] = {
                "bonsai_url_length": len(settings.BONSAI_URL) if settings.BONSAI_URL else 0,
                "qdrant_url_length": len(settings.QDRANT_URL) if settings.QDRANT_URL else 0,
                "openai_key_length": len(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else 0,
                "cohere_key_length": len(settings.COHERE_KEY) if settings.COHERE_KEY else 0,
                "bonsai_configured": bool(settings.BONSAI_URL),
                "qdrant_configured": bool(settings.QDRANT_URL),
                "qdrant_api_key_configured": bool(settings.QDRANT_API_KEY)
            }
            
            return diagnostic
            
        except Exception as e:
            return {
                "error": f"Deep diagnostic failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    # ======== ENDPOINTS POUR LE STATUT GLOBAL ========

    @app.get("/robots.txt", include_in_schema=False)
    async def robots_txt():
        """Fichier robots.txt pour les crawlers."""
        return JSONResponse(
            content="User-agent: *\nDisallow: /",
            media_type="text/plain"
        )

    # ======== RAPPORT FINAL DE DÉMARRAGE ========

    def log_final_startup_report():
        """Affiche le rapport final de démarrage."""
        search_status = service_registry.get_search_service_priority_status()
        total_services = len(service_registry.services)
        failed_services = len(service_registry.failed_services)
        
        # Bannière du rapport final
        logger.info("=" * 100)
        logger.info("🎯 === RAPPORT FINAL DE DÉMARRAGE HARENA ===")
        logger.info("=" * 100)
        
        # Statut du Search Service (prioritaire)
        search_emoji = {
            "fully_operational": "🎉",
            "partially_operational": "⚠️",
            "configuration_incomplete": "🔧",
            "service_failed": "🚨"
        }
        
        search_icon = search_emoji.get(search_status["overall_assessment"], "❓")
        logger.info(f"{search_icon} SEARCH SERVICE (CRITIQUE): {search_status['overall_assessment'].upper()}")
        
        if search_status["overall_assessment"] == "fully_operational":
            logger.info("   ✅ Recherche lexicale (Bonsai Elasticsearch) opérationnelle")
            logger.info("   ✅ Recherche sémantique (Qdrant) opérationnelle")
            logger.info("   ✅ Recherche hybride disponible")
            logger.info("   🚀 Plateforme prête pour la production")
        elif search_status["overall_assessment"] == "partially_operational":
            logger.info("   ⚠️ Au moins un moteur de recherche opérationnel")
            logger.info("   ⚠️ Fonctionnalités limitées")
        else:
            logger.info("   🚨 Service de recherche non opérationnel")
            logger.info("   🚨 Impact critique sur l'expérience utilisateur")
        
        # Statut des autres services
        logger.info(f"📊 AUTRES SERVICES: {total_services} actifs, {failed_services} échoués")
        
        # Services opérationnels
        if service_registry.services:
            logger.info("✅ Services opérationnels:")
            for name, info in service_registry.services.items():
                if name != "search_service":  # Déjà affiché
                    logger.info(f"   - {name}: {info['prefix']}")
        
        # Services échoués
        if service_registry.failed_services:
            logger.info("❌ Services échoués:")
            for name, info in service_registry.failed_services.items():
                logger.info(f"   - {name}: {info['error']}")
        
        # Recommandations
        logger.info("💡 RECOMMANDATIONS:")
        recommendations = _get_search_service_recommendations(search_status, global_diagnostics.get("search_service_health", {}))
        for rec in recommendations:
            logger.info(f"   {rec}")
        
        # URLs importantes
        logger.info("🌐 ENDPOINTS IMPORTANTS:")
        logger.info("   GET  / - Statut général")
        logger.info("   GET  /health - Santé détaillée")
        logger.info("   GET  /search-service-status - Statut Search Service")
        logger.info("   POST /api/v1/search/search - Recherche de transactions")
        
        logger.info("=" * 100)
        logger.info("✅ Application Harena prête pour Heroku")
        logger.info("=" * 100)

    # Affichage du rapport final
    log_final_startup_report()
    logger.info("✅ Application Harena complète configurée")

except Exception as critical_error:
    logger.critical(f"💥 ERREUR CRITIQUE lors de l'initialisation: {critical_error}")
    logger.critical(traceback.format_exc())
    raise

# ======== POINT D'ENTRÉE POUR HEROKU ========

# Cette ligne est cruciale pour Heroku - elle doit être à la racine du module
if 'app' not in locals():
    logger.error("❌ L'application FastAPI n'a pas été créée correctement")
    raise RuntimeError("FastAPI app not created")

logger.info("🎉 heroku_app.py chargé avec succès - Application prête pour déploiement")

# ======== INFORMATIONS DE DÉMARRAGE ========

if __name__ == "__main__":
    # Mode développement local
    import uvicorn
    
    logger.info("🔧 Démarrage en mode développement local")
    uvicorn.run(
        "heroku_app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
        log_level="info"
    )