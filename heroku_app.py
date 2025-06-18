"""
Application Harena pour déploiement Heroku - Version détaillée avec diagnostic complet.

Focus: Diagnostic approfondi de tous les services avec attention particulière au Search Service.
"""

import logging
import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("heroku_app")

# ==================== CONFIGURATION INITIALE ====================

try:
    logger.info("🚀 Démarrage Harena Finance Platform - Version Diagnostic Complète")
    
    # Correction DATABASE_URL pour Heroku
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
        os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        logger.info("✅ DATABASE_URL corrigé pour SQLAlchemy")

    # Ajout du répertoire courant au Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Imports FastAPI
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    logger.info("✅ Imports de base réussis")

    # ==================== VARIABLES GLOBALES ====================

    startup_time = None
    service_health_status = {
        "user_service": {"healthy": False, "details": {}, "error": None},
        "db_service": {"healthy": False, "details": {}, "error": None},
        "sync_service": {"healthy": False, "details": {}, "error": None},
        "enrichment_service": {"healthy": False, "details": {}, "error": None},
        "search_service": {"healthy": False, "details": {}, "error": None},
        "conversation_service": {"healthy": False, "details": {}, "error": None}
    }

    # ==================== FONCTIONS DE DIAGNOSTIC ====================

    async def test_user_service() -> Dict[str, Any]:
        """Test complet du User Service."""
        result = {
            "service": "user_service",
            "importable": False,
            "database_tables": False,
            "security_config": False,
            "routes_available": False,
            "jwt_secret": False,
            "error": None
        }
        
        try:
            # Test d'import avec la vraie structure
            from user_service.api.endpoints.users import router as user_router
            result["importable"] = True
            
            # Test config JWT
            jwt_secret = os.environ.get("JWT_SECRET_KEY")
            result["jwt_secret"] = bool(jwt_secret)
            
            # Test des tables de base de données
            from db_service.session import engine
            from sqlalchemy import text, inspect
            
            with engine.connect() as conn:
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                result["database_tables"] = "users" in tables
            
            # Test sécurité
            try:
                from user_service.core.security import verify_password, get_password_hash
                test_hash = get_password_hash("test")
                result["security_config"] = verify_password("test", test_hash)
            except Exception as e:
                result["security_config"] = False
                
            # Test des routes
            result["routes_available"] = hasattr(user_router, 'routes') and len(user_router.routes) > 0
                
        except Exception as e:
            result["error"] = str(e)
            
        return result

    async def test_db_service() -> Dict[str, Any]:
        """Test complet du DB Service."""
        result = {
            "service": "db_service",
            "connection": False,
            "models_loaded": False,
            "tables_count": 0,
            "session_factory": False,
            "migrations_current": False,
            "error": None
        }
        
        try:
            from db_service.session import engine, SessionLocal
            from sqlalchemy import text, inspect
            
            # Test de connexion
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                result["connection"] = True
            
            # Test factory de session
            db = SessionLocal()
            db.close()
            result["session_factory"] = True
            
            # Compter les tables
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            result["tables_count"] = len(tables)
            
            # Test des modèles
            try:
                from db_service.models import User, BridgeConnection
                result["models_loaded"] = True
            except Exception:
                result["models_loaded"] = False
                
        except Exception as e:
            result["error"] = str(e)
            
        return result

    async def test_sync_service() -> Dict[str, Any]:
        """Test complet du Sync Service."""
        result = {
            "service": "sync_service",
            "modules_imported": {},
            "bridge_config": False,
            "webhook_endpoint": False,
            "transaction_models": False,
            "error": None
        }
        
        sync_modules = [
            "sync_service.api.endpoints.sync",
            "sync_service.api.endpoints.transactions", 
            "sync_service.api.endpoints.accounts",
            "sync_service.api.endpoints.categories",
            "sync_service.api.endpoints.items",
            "sync_service.api.endpoints.webhooks"
        ]
        
        try:
            # Test d'import de chaque module
            for module_path in sync_modules:
                try:
                    module = __import__(module_path, fromlist=["router"])
                    router = getattr(module, "router", None)
                    result["modules_imported"][module_path] = router is not None
                except Exception as e:
                    result["modules_imported"][module_path] = False
            
            # Test config Bridge
            bridge_url = os.environ.get("BRIDGE_BASE_URL")
            bridge_client_id = os.environ.get("BRIDGE_CLIENT_ID") 
            bridge_client_secret = os.environ.get("BRIDGE_CLIENT_SECRET")
            result["bridge_config"] = all([bridge_url, bridge_client_id, bridge_client_secret])
            
            # Test modèles de transaction
            try:
                from db_service.models import RawTransaction, BridgeCategory
                result["transaction_models"] = True
            except Exception:
                result["transaction_models"] = False
                
        except Exception as e:
            result["error"] = str(e)
            
        return result

    async def test_enrichment_service() -> Dict[str, Any]:
        """Test complet de l'Enrichment Service."""
        result = {
            "service": "enrichment_service", 
            "importable": False,
            "openai_config": False,
            "cohere_config": False,
            "deepseek_config": False,
            "routes_available": False,
            "error": None
        }
        
        try:
            # Test d'import
            from enrichment_service.api.routes import router as enrichment_router
            result["importable"] = True
            result["routes_available"] = hasattr(enrichment_router, 'routes') and len(enrichment_router.routes) > 0
            
            # Test des configs API
            result["openai_config"] = bool(os.environ.get("OPENAI_API_KEY"))
            result["cohere_config"] = bool(os.environ.get("COHERE_KEY"))
            result["deepseek_config"] = bool(os.environ.get("DEEPSEEK_API_KEY"))
            
        except Exception as e:
            result["error"] = str(e)
            
        return result

    async def test_search_service() -> Dict[str, Any]:
        """Test approfondi du Search Service."""
        result = {
            "service": "search_service",
            "importable": False,
            "config": {
                "elasticsearch_url": False,
                "qdrant_url": False,
                "openai_key": False,
                "cohere_key": False
            },
            "clients": {
                "elasticsearch_client": False,
                "qdrant_client": False,
                "embedding_service": False
            },
            "connectivity": {
                "elasticsearch_ping": False,
                "qdrant_ping": False,
                "elasticsearch_index_exists": False,
                "qdrant_collection_exists": False
            },
            "capabilities": {
                "lexical_search": False,
                "semantic_search": False,
                "hybrid_search": False
            },
            "routes_available": False,
            "error": None
        }
        
        try:
            # Test d'import
            from search_service.api.routes import router as search_router
            result["importable"] = True
            result["routes_available"] = hasattr(search_router, 'routes') and len(search_router.routes) > 0
            
            # Test de la configuration
            result["config"]["elasticsearch_url"] = bool(os.environ.get("BONSAI_URL"))
            result["config"]["qdrant_url"] = bool(os.environ.get("QDRANT_URL"))  
            result["config"]["openai_key"] = bool(os.environ.get("OPENAI_API_KEY"))
            result["config"]["cohere_key"] = bool(os.environ.get("COHERE_KEY"))
            
            # Test des clients uniquement si la config est OK
            if result["config"]["elasticsearch_url"]:
                try:
                    from search_service.storage.elastic_client import ElasticClient
                    # Le client est injecté dans routes, on teste s'il existe
                    from search_service.api import routes
                    if hasattr(routes, 'elastic_client') and routes.elastic_client:
                        result["clients"]["elasticsearch_client"] = True
                        # Test de ping si le client a cette méthode
                        if hasattr(routes.elastic_client, 'is_healthy'):
                            try:
                                is_healthy = await routes.elastic_client.is_healthy()
                                result["connectivity"]["elasticsearch_ping"] = is_healthy
                                result["capabilities"]["lexical_search"] = is_healthy
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Elasticsearch client test failed: {e}")
            
            if result["config"]["qdrant_url"]:
                try:
                    from search_service.storage.qdrant_client import QdrantClient
                    # Le client est injecté dans routes
                    from search_service.api import routes
                    if hasattr(routes, 'qdrant_client') and routes.qdrant_client:
                        result["clients"]["qdrant_client"] = True
                        # Test de ping si le client a cette méthode
                        if hasattr(routes.qdrant_client, 'is_healthy'):
                            try:
                                is_healthy = await routes.qdrant_client.is_healthy()
                                result["connectivity"]["qdrant_ping"] = is_healthy
                                result["capabilities"]["semantic_search"] = is_healthy
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Qdrant client test failed: {e}")
            
            # Test du service d'embedding
            if result["config"]["openai_key"] or result["config"]["cohere_key"]:
                try:
                    from search_service.core.embeddings import EmbeddingService
                    from search_service.api import routes
                    if hasattr(routes, 'embedding_service') and routes.embedding_service:
                        result["clients"]["embedding_service"] = True
                except Exception as e:
                    logger.warning(f"Embedding service test failed: {e}")
            
            # Capacité de recherche hybride
            result["capabilities"]["hybrid_search"] = (
                result["capabilities"]["lexical_search"] and 
                result["capabilities"]["semantic_search"]
            )
            
        except Exception as e:
            result["error"] = str(e)
            
        return result

    async def test_conversation_service() -> Dict[str, Any]:
        """Test complet du Conversation Service."""
        result = {
            "service": "conversation_service",
            "importable": False,
            "deepseek_config": False,
            "intent_detection": False,
            "response_generation": False,
            "token_counter": False,
            "routes_available": False,
            "error": None
        }
        
        try:
            # Test d'import
            from conversation_service.api.routes import router as conversation_router
            result["importable"] = True
            result["routes_available"] = hasattr(conversation_router, 'routes') and len(conversation_router.routes) > 0
            
            # Test config DeepSeek
            result["deepseek_config"] = bool(os.environ.get("DEEPSEEK_API_KEY"))
            
            # Test des composants
            try:
                from conversation_service.core.intent_detection import IntentDetector
                result["intent_detection"] = True
            except Exception:
                result["intent_detection"] = False
            
            try:
                from conversation_service.core.deepseek_client import DeepSeekClient
                result["response_generation"] = True
            except Exception:
                result["response_generation"] = False
                
            try:
                from conversation_service.utils.token_counter import TokenCounter
                result["token_counter"] = True
            except Exception:
                result["token_counter"] = False
                
        except Exception as e:
            result["error"] = str(e)
            
        return result

    async def run_complete_diagnostic() -> Dict[str, Any]:
        """Lance un diagnostic complet de tous les services."""
        logger.info("🔍 Lancement du diagnostic complet...")
        
        # Tests en parallèle pour plus d'efficacité
        tests = await asyncio.gather(
            test_user_service(),
            test_db_service(), 
            test_sync_service(),
            test_enrichment_service(),
            test_search_service(),
            test_conversation_service(),
            return_exceptions=True
        )
        
        # Mise à jour du statut global
        services = ["user_service", "db_service", "sync_service", "enrichment_service", "search_service", "conversation_service"]
        
        for i, test_result in enumerate(tests):
            if isinstance(test_result, Exception):
                service_health_status[services[i]]["error"] = str(test_result)
                service_health_status[services[i]]["healthy"] = False
            else:
                service_health_status[services[i]]["details"] = test_result
                service_health_status[services[i]]["error"] = test_result.get("error")
                # Un service est considéré comme sain s'il n'y a pas d'erreur ET qu'il est importable
                service_health_status[services[i]]["healthy"] = (
                    not test_result.get("error") and 
                    test_result.get("importable", False)
                )
        
        return service_health_status

    # ==================== REGISTRE DE SERVICES AVANCÉ ====================

    class AdvancedServiceRegistry:
        def __init__(self):
            self.services = {}
            self.failed_services = {}
            self.diagnostic_results = {}
        
        def register(self, name: str, router, prefix: str, description: str = ""):
            try:
                if router:
                    self.services[name] = {
                        "router": router,
                        "prefix": prefix,
                        "description": description,
                        "status": "registered",
                        "routes_count": len(router.routes) if hasattr(router, 'routes') else 0,
                        "registered_at": datetime.now()
                    }
                    logger.info(f"✅ {name}: {prefix} ({self.services[name]['routes_count']} routes)")
                    return True
                else:
                    raise ValueError("Router is None")
            except Exception as e:
                self.failed_services[name] = {
                    "error": str(e),
                    "failed_at": datetime.now()
                }
                logger.error(f"❌ {name}: {e}")
                return False
        
        def update_diagnostic(self, name: str, diagnostic: Dict[str, Any]):
            """Met à jour les résultats de diagnostic pour un service."""
            self.diagnostic_results[name] = diagnostic
            if name in self.services:
                self.services[name]["diagnostic"] = diagnostic
        
        def get_healthy_count(self):
            return len([s for s in self.services.values() if s.get("status") == "registered"])
        
        def get_failed_count(self):
            return len(self.failed_services)
        
        def get_full_status(self):
            """Retourne un statut complet avec diagnostics."""
            return {
                "services": self.services,
                "failed_services": self.failed_services,
                "diagnostic_results": self.diagnostic_results,
                "summary": {
                    "total_registered": len(self.services),
                    "total_failed": len(self.failed_services),
                    "healthy_services": len([d for d in self.diagnostic_results.values() if not d.get("error")]),
                    "services_with_errors": len([d for d in self.diagnostic_results.values() if d.get("error")])
                }
            }

    service_registry = AdvancedServiceRegistry()

    # ==================== CYCLE DE VIE ====================

    async def startup():
        """Initialisation de l'application avec diagnostic complet."""
        global startup_time
        startup_time = time.time()
        logger.info("📋 Démarrage application Harena avec diagnostic complet...")
        
        # Test de connexion DB immédiat
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Base de données connectée")
        except Exception as e:
            logger.error(f"❌ Erreur DB critique: {e}")
            raise RuntimeError("Database connection failed")
        
        # Attendre que tous les services soient enregistrés
        await asyncio.sleep(2.0)
        
        # Lancer le diagnostic complet
        diagnostic_results = await run_complete_diagnostic()
        
        # Mettre à jour le registre avec les diagnostics
        for service_name, diagnostic in diagnostic_results.items():
            service_registry.update_diagnostic(service_name, diagnostic["details"])
        
        total_time = time.time() - startup_time
        logger.info(f"✅ Démarrage terminé en {total_time:.2f}s")

    async def shutdown():
        """Arrêt de l'application."""
        logger.info("⏹️ Arrêt application Harena...")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await startup()
        yield
        await shutdown()

    # ==================== APPLICATION FASTAPI ====================

    app = FastAPI(
        title="Harena Finance Platform",
        description="Plateforme de gestion financière avec recherche hybride",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==================== ENDPOINTS DE DIAGNOSTIC ====================

    @app.get("/")
    async def root():
        """Statut général de l'application."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Statut du Search Service spécialement
        search_status = service_health_status.get("search_service", {})
        search_details = search_status.get("details", {})
        
        return {
            "service": "Harena Finance API",
            "status": "online",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "search_service": {
                "configured": search_details.get("config", {}).get("elasticsearch_url", False) and search_details.get("config", {}).get("qdrant_url", False),
                "healthy": search_status.get("healthy", False),
                "clients_injected": search_details.get("clients", {}).get("elasticsearch_client", False) and search_details.get("clients", {}).get("qdrant_client", False),
                "status": "fully_operational" if search_status.get("healthy") and search_details.get("capabilities", {}).get("hybrid_search") else "degraded"
            },
            "services": {
                "healthy": len([s for s in service_health_status.values() if s.get("healthy")]),
                "failed": len([s for s in service_health_status.values() if not s.get("healthy")])
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health_check():
        """Vérification de santé détaillée."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Relancer le diagnostic pour avoir des données fraîches
        fresh_diagnostic = await run_complete_diagnostic()
        
        health_status = {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }
        
        # Analyser chaque service
        for service_name, status in fresh_diagnostic.items():
            service_health = {
                "healthy": status["healthy"],
                "details": status["details"],
                "error": status["error"]
            }
            
            # Ajouter des recommandations spécifiques
            if service_name == "search_service" and not status["healthy"]:
                service_health["recommendations"] = []
                details = status["details"]
                config = details.get("config", {})
                
                if not config.get("elasticsearch_url"):
                    service_health["recommendations"].append("Configurez BONSAI_URL")
                if not config.get("qdrant_url"):
                    service_health["recommendations"].append("Configurez QDRANT_URL")
                if not details.get("clients", {}).get("elasticsearch_client"):
                    service_health["recommendations"].append("Vérifiez la connectivité Elasticsearch")
                if not details.get("clients", {}).get("qdrant_client"):
                    service_health["recommendations"].append("Vérifiez la connectivité Qdrant")
            
            health_status["services"][service_name] = service_health
            
            if not status["healthy"]:
                health_status["status"] = "degraded"
        
        return health_status

    @app.get("/search-service-status")
    async def search_service_detailed_status():
        """Statut ultra-détaillé du Search Service."""
        search_diagnostic = await test_search_service()
        
        return {
            "service": "search_service",
            "priority": "critical",
            "timestamp": datetime.now().isoformat(),
            "overall_status": "fully_operational" if not search_diagnostic.get("error") and search_diagnostic.get("capabilities", {}).get("hybrid_search") else "degraded",
            "configuration": search_diagnostic.get("config", {}),
            "clients": search_diagnostic.get("clients", {}),
            "connectivity": search_diagnostic.get("connectivity", {}),
            "capabilities": search_diagnostic.get("capabilities", {}),
            "error": search_diagnostic.get("error"),
            "recommendations": _generate_search_recommendations(search_diagnostic),
            "endpoints": [
                "POST /api/v1/search/search - Recherche de transactions",
                "GET /api/v1/search/suggest - Suggestions de recherche"
            ]
        }

    def _generate_search_recommendations(diagnostic: Dict[str, Any]) -> List[str]:
        """Génère des recommandations pour le Search Service."""
        recommendations = []
        
        config = diagnostic.get("config", {})
        clients = diagnostic.get("clients", {})
        connectivity = diagnostic.get("connectivity", {})
        
        if not config.get("elasticsearch_url"):
            recommendations.append("⚠️ Configurez BONSAI_URL pour Elasticsearch")
        if not config.get("qdrant_url"):
            recommendations.append("⚠️ Configurez QDRANT_URL pour Qdrant")
        if not config.get("openai_key") and not config.get("cohere_key"):
            recommendations.append("⚠️ Configurez OPENAI_API_KEY ou COHERE_KEY pour les embeddings")
        
        if config.get("elasticsearch_url") and not clients.get("elasticsearch_client"):
            recommendations.append("🔧 Problème d'initialisation du client Elasticsearch")
        if config.get("qdrant_url") and not clients.get("qdrant_client"):
            recommendations.append("🔧 Problème d'initialisation du client Qdrant")
        
        if clients.get("elasticsearch_client") and not connectivity.get("elasticsearch_ping"):
            recommendations.append("🌐 Vérifiez la connectivité réseau vers Elasticsearch")
        if clients.get("qdrant_client") and not connectivity.get("qdrant_ping"):
            recommendations.append("🌐 Vérifiez la connectivité réseau vers Qdrant")
        
        if not recommendations:
            recommendations.append("✅ Service complètement opérationnel")
        
        return recommendations

    @app.get("/services-registry")
    async def services_registry_status():
        """Statut complet du registre de services."""
        return service_registry.get_full_status()

    # ==================== IMPORTATION DES SERVICES ====================

    logger.info("📦 Importation des services...")

    # 1. User Service
    try:
        from user_service.api.endpoints.users import router as user_router
        if service_registry.register("user_service", user_router, "/api/v1/users", "Gestion utilisateurs"):
            app.include_router(user_router, prefix="/api/v1/users", tags=["users"])
    except Exception as e:
        logger.error(f"❌ User Service: {e}")

    # 2. Sync Service - modules principaux
    sync_modules = [
        ("sync_service.api.endpoints.sync", "/api/v1/sync", "Synchronisation"),
        ("sync_service.api.endpoints.transactions", "/api/v1/transactions", "Transactions"),
        ("sync_service.api.endpoints.accounts", "/api/v1/accounts", "Comptes"),
        ("sync_service.api.endpoints.categories", "/api/v1/categories", "Catégories"),
        ("sync_service.api.endpoints.items", "/api/v1/items", "Items Bridge"),
        ("sync_service.api.endpoints.webhooks", "/webhooks", "Webhooks"),
    ]

    for module_path, prefix, description in sync_modules:
        try:
            module = __import__(module_path, fromlist=["router"])
            router = getattr(module, "router")
            service_name = f"sync_{module_path.split('.')[-1]}"
            if service_registry.register(service_name, router, prefix, description):
                app.include_router(router, prefix=prefix, tags=[module_path.split('.')[-1]])
        except Exception as e:
            logger.error(f"❌ {module_path}: {e}")

    # 3. Enrichment Service
    try:
        from enrichment_service.api.routes import router as enrichment_router
        if service_registry.register("enrichment_service", enrichment_router, "/api/v1/enrichment", "Enrichissement IA"):
            app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
    except Exception as e:
        logger.error(f"❌ Enrichment Service: {e}")

    # 4. Search Service (CRITIQUE)
    try:
        from search_service.api.routes import router as search_router
        if service_registry.register("search_service", search_router, "/api/v1/search", "🔍 CRITIQUE: Recherche hybride"):
            app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
            logger.info("🎉 Search Service enregistré avec succès")
        else:
            logger.error("🚨 Search Service: Échec enregistrement critique")
    except Exception as e:
        logger.error(f"💥 Search Service: {e}")

    # 5. Conversation Service
    try:
        from conversation_service.api.routes import router as conversation_router
        if service_registry.register("conversation_service", conversation_router, "/api/v1/conversation", "Assistant IA"):
            app.include_router(conversation_router, prefix="/api/v1/conversation", tags=["conversation"])
    except Exception as e:
        logger.error(f"❌ Conversation Service: {e}")

    # ==================== ENDPOINTS UTILITAIRES ====================

    @app.get("/version")
    async def version():
        return {
            "version": "1.0.0",
            "build": "heroku-diagnostic-complete",
            "python": sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "diagnostic_version": "2.0"
        }

    @app.get("/robots.txt", include_in_schema=False)
    async def robots():
        return JSONResponse("User-agent: *\nDisallow: /", media_type="text/plain")

    # ==================== GESTIONNAIRE D'ERREURS ====================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        search_status = service_health_status.get("search_service", {})
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "path": request.url.path,
                "search_service_healthy": search_status.get("healthy", False),
                "timestamp": datetime.now().isoformat()
            }
        )

    # ==================== RAPPORT FINAL ====================

    logger.info("=" * 80)
    logger.info("🎯 HARENA FINANCE PLATFORM - VERSION DIAGNOSTIC COMPLÈTE")
    logger.info(f"📊 Services enregistrés: {service_registry.get_healthy_count()}")
    logger.info(f"❌ Services échoués: {service_registry.get_failed_count()}")
    logger.info("🔍 Diagnostic automatique activé au démarrage")
    logger.info("🌐 Endpoints de diagnostic:")
    logger.info("   GET  / - Statut général avec focus Search Service")
    logger.info("   GET  /health - Santé détaillée de tous les services")
    logger.info("   GET  /search-service-status - Diagnostic ultra-détaillé Search Service")
    logger.info("   GET  /services-registry - Statut complet du registre")
    logger.info("🔧 Endpoints principaux:")
    logger.info("   POST /api/v1/search/search - Recherche de transactions")
    logger.info("   POST /api/v1/conversation/chat - Assistant IA")
    logger.info("   GET  /api/v1/sync - Synchronisation Bridge")
    logger.info("=" * 80)
    logger.info("✅ Application Harena prête avec diagnostic complet")

except Exception as critical_error:
    logger.critical(f"💥 ERREUR CRITIQUE: {critical_error}")
    raise

# Point d'entrée Heroku
if 'app' not in locals():
    raise RuntimeError("FastAPI app not created")

# Mode développement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=8000, reload=True)