"""
Application Harena pour déploiement Heroku - Version propre et pérenne.

Architecture: Initialisation séquentielle avec enregistrement immédiat des services prêts.
"""

import logging
import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime
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
    logger.info("🚀 Démarrage Harena Finance Platform - Version Propre")
    
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
    all_services_status = {}

    # ==================== CLASSES UTILITAIRES ====================

    class ServiceRegistry:
        """Registre propre pour la gestion des services."""
        
        def __init__(self):
            self.services = {}
            self.failed_services = {}
        
        def register(self, name: str, router, prefix: str, description: str = "") -> bool:
            """Enregistre un service avec son router."""
            try:
                if router:
                    routes_count = len(router.routes) if hasattr(router, 'routes') else 0
                    self.services[name] = {
                        "router": router,
                        "prefix": prefix,
                        "description": description,
                        "routes_count": routes_count,
                        "registered_at": datetime.now()
                    }
                    logger.info(f"✅ {name}: {prefix} ({routes_count} routes)")
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
        
        def get_summary(self) -> Dict[str, Any]:
            """Retourne un résumé du registre."""
            return {
                "registered": len(self.services),
                "failed": len(self.failed_services),
                "total_routes": sum(s.get("routes_count", 0) for s in self.services.values())
            }

    # ==================== FONCTIONS DE TEST DES SERVICES ====================

    async def test_user_service() -> Dict[str, Any]:
        """Test et initialisation du User Service."""
        logger.info("🔍 Test User Service...")
        result = {
            "service": "user_service",
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "jwt_configured": False
            },
            "error": None
        }
        
        try:
            from user_service.api.endpoints.users import router as user_router
            result["details"]["importable"] = True
            result["details"]["routes_count"] = len(user_router.routes) if hasattr(user_router, 'routes') else 0
            
            # Test config JWT
            jwt_secret = os.environ.get("JWT_SECRET_KEY") or os.environ.get("SECRET_KEY")
            result["details"]["jwt_configured"] = bool(jwt_secret and len(jwt_secret) >= 32)
            
            result["healthy"] = result["details"]["importable"] and result["details"]["routes_count"] > 0
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ User Service: {e}")
            
        return result

    async def test_db_service() -> Dict[str, Any]:
        """Test et initialisation du DB Service."""
        logger.info("🔍 Test DB Service...")
        result = {
            "service": "db_service",
            "healthy": False,
            "details": {
                "connection": False,
                "tables_count": 0
            },
            "error": None
        }
        
        try:
            from db_service.session import engine
            from sqlalchemy import text, inspect
            
            # Test de connexion
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                result["details"]["connection"] = True
            
            # Compter les tables
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            result["details"]["tables_count"] = len(tables)
            
            result["healthy"] = result["details"]["connection"] and result["details"]["tables_count"] > 0
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ DB Service: {e}")
            
        return result

    async def test_sync_service() -> Dict[str, Any]:
        """Test et initialisation du Sync Service."""
        logger.info("🔍 Test Sync Service...")
        result = {
            "service": "sync_service",
            "healthy": False,
            "details": {
                "modules_imported": {},
                "total_routes": 0
            },
            "error": None
        }
        
        sync_modules = [
            ("sync_service.api.endpoints.sync", "/api/v1/sync"),
            ("sync_service.api.endpoints.transactions", "/api/v1/transactions"), 
            ("sync_service.api.endpoints.accounts", "/api/v1/accounts"),
            ("sync_service.api.endpoints.categories", "/api/v1/categories"),
            ("sync_service.api.endpoints.items", "/api/v1/items"),
            ("sync_service.api.endpoints.webhooks", "/webhooks")
        ]
        
        try:
            successful_imports = 0
            total_routes = 0
            
            for module_path, endpoint in sync_modules:
                try:
                    module = __import__(module_path, fromlist=["router"])
                    router = getattr(module, "router", None)
                    if router and hasattr(router, 'routes'):
                        routes_count = len(router.routes)
                        result["details"]["modules_imported"][module_path] = {
                            "success": True,
                            "routes_count": routes_count
                        }
                        successful_imports += 1
                        total_routes += routes_count
                    else:
                        result["details"]["modules_imported"][module_path] = {
                            "success": False,
                            "error": "No router found"
                        }
                except Exception as e:
                    result["details"]["modules_imported"][module_path] = {
                        "success": False,
                        "error": str(e)
                    }
            
            result["details"]["total_routes"] = total_routes
            result["healthy"] = successful_imports >= 4  # Au moins 4 modules sur 6
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Sync Service: {e}")
            
        return result

    async def test_enrichment_service() -> Dict[str, Any]:
        """Test et initialisation de l'Enrichment Service."""
        logger.info("🔍 Test Enrichment Service...")
        result = {
            "service": "enrichment_service", 
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "ai_configs": {
                    "openai": bool(os.environ.get("OPENAI_API_KEY")),
                    "cohere": bool(os.environ.get("COHERE_KEY")),
                    "deepseek": bool(os.environ.get("DEEPSEEK_API_KEY"))
                }
            },
            "error": None
        }
        
        try:
            from enrichment_service.api.routes import router as enrichment_router
            result["details"]["importable"] = True
            result["details"]["routes_count"] = len(enrichment_router.routes) if hasattr(enrichment_router, 'routes') else 0
            
            result["healthy"] = (
                result["details"]["importable"] and 
                any(result["details"]["ai_configs"].values())
            )
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Enrichment Service: {e}")
            
        return result

    async def test_conversation_service() -> Dict[str, Any]:
        """Test et initialisation du Conversation Service."""
        logger.info("🔍 Test Conversation Service...")
        result = {
            "service": "conversation_service",
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "deepseek_config": bool(os.environ.get("DEEPSEEK_API_KEY"))
            },
            "error": None
        }
        
        try:
            from conversation_service.api.routes import router as conversation_router
            result["details"]["importable"] = True
            result["details"]["routes_count"] = len(conversation_router.routes) if hasattr(conversation_router, 'routes') else 0
            
            result["healthy"] = (
                result["details"]["importable"] and 
                result["details"]["deepseek_config"]
            )
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Conversation Service: {e}")
            
        return result

    async def initialize_search_service() -> Dict[str, Any]:
        """Initialisation complète et robuste du Search Service."""
        logger.info("🔧 === INITIALISATION SEARCH SERVICE ===")
        
        result = {
            "service": "search_service",
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "config": {
                    "elasticsearch_url": bool(os.environ.get("BONSAI_URL")),
                    "qdrant_url": bool(os.environ.get("QDRANT_URL"))
                },
                "clients": {
                    "elasticsearch_initialized": False,
                    "qdrant_initialized": False,
                    "elasticsearch_healthy": False,
                    "qdrant_healthy": False
                },
                "capabilities": {
                    "lexical_search": False,
                    "semantic_search": False,
                    "hybrid_search": False
                }
            },
            "error": None
        }
        
        start_time = time.time()
        
        try:
            # Vérification configuration
            if not any([result["details"]["config"]["elasticsearch_url"], result["details"]["config"]["qdrant_url"]]):
                result["error"] = "No search URLs configured"
                return result
            
            # Import des modules
            logger.info("📦 Import modules Search Service...")
            from search_service.storage.elastic_client_hybrid import HybridElasticClient
            from search_service.storage.qdrant_client import QdrantClient
            from config_service.config import settings
            
            elastic_client = None
            qdrant_client = None
            
            # Initialisation Elasticsearch
            if settings.BONSAI_URL:
                logger.info("🔍 Initialisation Elasticsearch...")
                try:
                    elastic_client = HybridElasticClient()
                    if await elastic_client.initialize():
                        result["details"]["clients"]["elasticsearch_initialized"] = True
                        
                        # Test de santé
                        if hasattr(elastic_client, 'is_healthy') and await elastic_client.is_healthy():
                            result["details"]["clients"]["elasticsearch_healthy"] = True
                            result["details"]["capabilities"]["lexical_search"] = True
                            logger.info("✅ Elasticsearch prêt")
                        
                except Exception as e:
                    logger.error(f"❌ Elasticsearch: {e}")
                    elastic_client = None
            
            # Initialisation Qdrant
            if settings.QDRANT_URL:
                logger.info("🎯 Initialisation Qdrant...")
                try:
                    qdrant_client = QdrantClient()
                    if await qdrant_client.initialize():
                        result["details"]["clients"]["qdrant_initialized"] = True
                        
                        # Test de santé
                        if hasattr(qdrant_client, 'is_healthy') and await qdrant_client.is_healthy():
                            result["details"]["clients"]["qdrant_healthy"] = True
                            result["details"]["capabilities"]["semantic_search"] = True
                            logger.info("✅ Qdrant prêt")
                        
                except Exception as e:
                    logger.error(f"❌ Qdrant: {e}")
                    qdrant_client = None
            
            # Calcul des capacités
            result["details"]["capabilities"]["hybrid_search"] = (
                result["details"]["capabilities"]["lexical_search"] and 
                result["details"]["capabilities"]["semantic_search"]
            )
            
            # Injection dans les routes SI au moins un client fonctionne
            if elastic_client or qdrant_client:
                logger.info("🔗 Injection clients dans les routes...")
                try:
                    import search_service.api.routes as routes
                    
                    routes.elastic_client = elastic_client
                    routes.qdrant_client = qdrant_client
                    routes.embedding_service = None
                    routes.reranker_service = None
                    routes.search_cache = None
                    routes.metrics_collector = None
                    
                    # Test d'import des routes
                    from search_service.api.routes import router as search_router
                    result["details"]["importable"] = True
                    result["details"]["routes_count"] = len(search_router.routes) if hasattr(search_router, 'routes') else 0
                    
                    # Service sain si au moins un client fonctionne
                    result["healthy"] = (
                        result["details"]["clients"]["elasticsearch_healthy"] or 
                        result["details"]["clients"]["qdrant_healthy"]
                    )
                    
                    if result["healthy"]:
                        logger.info("✅ Search Service initialisé avec succès")
                    else:
                        logger.warning("⚠️ Search Service partiellement initialisé")
                    
                except Exception as e:
                    logger.error(f"❌ Injection routes: {e}")
                    result["error"] = f"Route injection failed: {e}"
            else:
                result["error"] = "No clients successfully initialized"
            
        except Exception as e:
            logger.error(f"💥 Erreur Search Service: {e}")
            result["error"] = str(e)
        
        initialization_time = time.time() - start_time
        logger.info(f"🏁 Search Service initialisé en {initialization_time:.2f}s")
        
        return result

    # ==================== FONCTION PRINCIPALE DE DIAGNOSTIC ====================

    async def run_services_diagnostic() -> Dict[str, Any]:
        """Lance le diagnostic de tous les services."""
        logger.info("🔍 Diagnostic complet des services...")
        
        # Tests en parallèle des services standards
        tasks = [
            test_user_service(),
            test_db_service(),
            test_sync_service(),
            test_enrichment_service(),
            test_conversation_service(),
            initialize_search_service()  # Search Service avec initialisation spéciale
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        services = ["user_service", "db_service", "sync_service", "enrichment_service", "conversation_service", "search_service"]
        services_status = {}
        
        for i, result in enumerate(results):
            service_name = services[i]
            
            if isinstance(result, Exception):
                services_status[service_name] = {
                    "healthy": False,
                    "error": str(result),
                    "details": {}
                }
                logger.error(f"❌ {service_name}: {result}")
            else:
                services_status[service_name] = result
                status_icon = "✅" if result["healthy"] else "❌"
                logger.info(f"{status_icon} {service_name}: {'Healthy' if result['healthy'] else 'Unhealthy'}")
        
        return services_status

    # ==================== APPLICATION FASTAPI ====================

    # Initialisation globale
    service_registry = ServiceRegistry()

    async def startup():
        """Startup complet avec enregistrement immédiat des services."""
        global startup_time, all_services_status
        startup_time = time.time()
        logger.info("📋 Démarrage application Harena...")
        
        # Test DB critique
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Base de données connectée")
        except Exception as e:
            logger.error(f"❌ Erreur DB critique: {e}")
            raise RuntimeError("Database connection failed")
        
        # Diagnostic complet
        all_services_status = await run_services_diagnostic()
        
        total_time = time.time() - startup_time
        healthy_count = sum(1 for s in all_services_status.values() if s.get("healthy"))
        logger.info(f"✅ Diagnostic terminé en {total_time:.2f}s - {healthy_count}/{len(all_services_status)} services sains")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Cycle de vie de l'application."""
        await startup()
        yield

    # Création de l'application
    app = FastAPI(
        title="Harena Finance Platform",
        description="Plateforme de gestion financière avec services intégrés",
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

    # ==================== ENREGISTREMENT DES SERVICES ====================

    def register_services_after_startup():
        """Enregistre tous les services après le diagnostic."""
        logger.info("📦 Enregistrement des services...")
        
        # 1. User Service
        if all_services_status.get("user_service", {}).get("healthy", False):
            try:
                from user_service.api.endpoints.users import router as user_router
                if service_registry.register("user_service", user_router, "/api/v1/users", "Gestion utilisateurs"):
                    app.include_router(user_router, prefix="/api/v1/users", tags=["users"])
            except Exception as e:
                logger.error(f"❌ User Service registration: {e}")

        # 2. Sync Service - modules
        if all_services_status.get("sync_service", {}).get("healthy", False):
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
                    logger.warning(f"⚠️ {module_path}: {e}")

        # 3. Enrichment Service
        if all_services_status.get("enrichment_service", {}).get("healthy", False):
            try:
                from enrichment_service.api.routes import router as enrichment_router
                if service_registry.register("enrichment_service", enrichment_router, "/api/v1/enrichment", "Enrichissement IA"):
                    app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
            except Exception as e:
                logger.error(f"❌ Enrichment Service registration: {e}")

        # 4. Conversation Service
        if all_services_status.get("conversation_service", {}).get("healthy", False):
            try:
                from conversation_service.api.routes import router as conversation_router
                if service_registry.register("conversation_service", conversation_router, "/api/v1/conversation", "Assistant IA"):
                    app.include_router(conversation_router, prefix="/api/v1/conversation", tags=["conversation"])
            except Exception as e:
                logger.error(f"❌ Conversation Service registration: {e}")

        # 5. Search Service (CRITIQUE)
        search_status = all_services_status.get("search_service", {})
        if search_status.get("healthy", False):
            try:
                from search_service.api.routes import router as search_router
                if service_registry.register("search_service", search_router, "/api/v1/search", "🔍 CRITIQUE: Recherche hybride"):
                    app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
                    
                    capabilities = search_status.get("details", {}).get("capabilities", {})
                    logger.info("🎉 Search Service enregistré avec succès")
                    logger.info(f"   🎯 Capacités: Lexical={capabilities.get('lexical_search')}, Semantic={capabilities.get('semantic_search')}, Hybrid={capabilities.get('hybrid_search')}")
                    
            except Exception as e:
                logger.error(f"❌ Search Service registration: {e}")
        else:
            search_error = search_status.get("error", "Unknown error")
            logger.error(f"🚨 Search Service non disponible: {search_error}")

    # Event de post-startup pour l'enregistrement
    @app.on_event("startup")
    async def post_startup_registration():
        """Enregistrement post-startup des services."""
        # Petit délai pour s'assurer que le diagnostic est terminé
        await asyncio.sleep(0.1)
        register_services_after_startup()

    # ==================== ENDPOINTS DE DIAGNOSTIC ====================

    @app.get("/")
    async def root():
        """Statut général de l'application."""
        uptime = time.time() - startup_time if startup_time else 0
        
        healthy_services = [name for name, status in all_services_status.items() if status.get("healthy")]
        failed_services = [name for name, status in all_services_status.items() if not status.get("healthy")]
        
        # Focus Search Service
        search_status = all_services_status.get("search_service", {})
        search_details = search_status.get("details", {})
        
        return {
            "service": "Harena Finance API",
            "status": "online",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "services": {
                "healthy": len(healthy_services),
                "failed": len(failed_services),
                "healthy_list": healthy_services,
                "failed_list": failed_services
            },
            "search_service": {
                "configured": any(search_details.get("config", {}).values()),
                "healthy": search_status.get("healthy", False),
                "clients_injected": search_details.get("importable", False),
                "elasticsearch_reachable": search_details.get("clients", {}).get("elasticsearch_healthy", False),
                "qdrant_reachable": search_details.get("clients", {}).get("qdrant_healthy", False),
                "status": "fully_operational" if search_status.get("healthy") else "degraded"
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health_check():
        """Check de santé détaillé."""
        return {
            "status": "healthy" if all(s.get("healthy") for s in all_services_status.values()) else "degraded",
            "uptime_seconds": time.time() - startup_time if startup_time else 0,
            "timestamp": datetime.now().isoformat(),
            "registry": service_registry.get_summary(),
            "services": all_services_status
        }

    @app.get("/search-service")
    async def search_service_details():
        """Diagnostic détaillé du Search Service."""
        search_status = all_services_status.get("search_service", {})
        
        return {
            "service": "search_service",
            "timestamp": datetime.now().isoformat(),
            "status": "operational" if search_status.get("healthy") else "degraded",
            "details": search_status.get("details", {}),
            "error": search_status.get("error"),
            "endpoints": [
                "POST /api/v1/search/search - Recherche hybride" if search_status.get("healthy") else "❌ Search endpoints unavailable",
                "GET /api/v1/search/health - Santé du service" if search_status.get("healthy") else "",
                "GET /search-service - Ce diagnostic"
            ]
        }

    @app.get("/version")
    async def version():
        return {
            "version": "1.0.0",
            "build": "heroku-clean-architecture",
            "python": sys.version.split()[0],
            "features": [
                "Sequential service initialization",
                "Immediate route registration after health check",
                "Robust Search Service with direct client injection",
                "Clean service lifecycle management"
            ]
        }

    # ==================== GESTIONNAIRE D'ERREURS ====================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "path": request.url.path,
                "timestamp": datetime.now().isoformat(),
                "search_available": all_services_status.get("search_service", {}).get("healthy", False)
            }
        )

    # ==================== RAPPORT FINAL ====================

    logger.info("=" * 80)
    logger.info("🎯 HARENA FINANCE PLATFORM - ARCHITECTURE PROPRE")
    logger.info("🔧 Fonctionnalités:")
    logger.info("   📋 Diagnostic séquentiel des services")
    logger.info("   🔍 Initialisation robuste Search Service")
    logger.info("   📦 Enregistrement immédiat après diagnostic")
    logger.info("   🌐 Routes disponibles dans /docs")
    logger.info("   ⚡ Performance optimisée")
    logger.info("=" * 80)
    logger.info("✅ Application Harena prête - Architecture propre")

except Exception as critical_error:
    logger.critical(f"💥 ERREUR CRITIQUE: {critical_error}")
    raise

# Point d'entrée
if 'app' not in locals():
    raise RuntimeError("FastAPI app not created")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=8000, reload=True)