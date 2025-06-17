"""
Application Harena pour déploiement Heroku - Version concise et efficace.

Focus: Search Service critique avec diagnostic prioritaire.
"""

import logging
import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
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
    logger.info("🚀 Démarrage Harena Finance Platform")
    
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
    search_service_status = {
        "configured": False,
        "healthy": False,
        "clients_injected": False,
        "last_check": None
    }

    # ==================== FONCTIONS UTILITAIRES ====================

    def check_database_connection() -> bool:
        """Teste la connexion à la base de données."""
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"❌ Erreur DB: {e}")
            return False

    def check_search_config() -> Dict[str, bool]:
        """Vérifie la configuration du Search Service."""
        config = {
            "BONSAI_URL": bool(os.environ.get("BONSAI_URL")),
            "QDRANT_URL": bool(os.environ.get("QDRANT_URL")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "COHERE_KEY": bool(os.environ.get("COHERE_KEY"))
        }
        
        critical_ok = config["BONSAI_URL"] and config["QDRANT_URL"]
        logger.info(f"🔧 Search Config: {sum(config.values())}/4 configurés, Critique: {'✅' if critical_ok else '❌'}")
        
        return config

    async def test_search_service_health() -> Dict[str, Any]:
        """Test de santé du Search Service."""
        logger.info("🔍 Test santé Search Service...")
        
        health = {
            "importable": False,
            "clients_injected": False,
            "elasticsearch_healthy": False,
            "qdrant_healthy": False,
            "overall_status": "failed"
        }
        
        try:
            # Test d'import
            from search_service.api.routes import router as search_router
            health["importable"] = True
            logger.info("   ✅ Search Service importable")
            
            # Attendre que les clients soient injectés
            await asyncio.sleep(1.0)
            
            # Test des clients injectés
            try:
                from search_service.api.routes import elastic_client, qdrant_client
                health["clients_injected"] = elastic_client is not None or qdrant_client is not None
                
                # Test Elasticsearch
                if elastic_client:
                    try:
                        health["elasticsearch_healthy"] = await elastic_client.is_healthy()
                        logger.info(f"   🔍 Elasticsearch: {'✅' if health['elasticsearch_healthy'] else '❌'}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Elasticsearch error: {e}")
                
                # Test Qdrant
                if qdrant_client:
                    try:
                        health["qdrant_healthy"] = await qdrant_client.is_healthy()
                        logger.info(f"   🎯 Qdrant: {'✅' if health['qdrant_healthy'] else '❌'}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Qdrant error: {e}")
                
                logger.info(f"   🔗 Clients injectés: {'✅' if health['clients_injected'] else '❌'}")
                
            except ImportError as e:
                logger.error(f"   ❌ Import clients failed: {e}")
        
        except ImportError as e:
            logger.error(f"   ❌ Import Search Service failed: {e}")
        
        # Déterminer le statut global
        if health["elasticsearch_healthy"] and health["qdrant_healthy"]:
            health["overall_status"] = "fully_operational"
        elif health["elasticsearch_healthy"] or health["qdrant_healthy"]:
            health["overall_status"] = "degraded"
        elif health["clients_injected"]:
            health["overall_status"] = "partial"
        else:
            health["overall_status"] = "failed"
        
        logger.info(f"🎯 Search Service: {health['overall_status'].upper()}")
        return health

    # ==================== REGISTRE DE SERVICES SIMPLE ====================

    class SimpleServiceRegistry:
        def __init__(self):
            self.services = {}
            self.failed_services = {}
        
        def register(self, name: str, router, prefix: str, description: str = ""):
            try:
                if router:
                    self.services[name] = {
                        "router": router,
                        "prefix": prefix,
                        "description": description,
                        "status": "ok"
                    }
                    logger.info(f"✅ {name}: {prefix}")
                    return True
                else:
                    raise ValueError("Router is None")
            except Exception as e:
                self.failed_services[name] = str(e)
                logger.error(f"❌ {name}: {e}")
                return False
        
        def get_healthy_count(self):
            return len(self.services)
        
        def get_failed_count(self):
            return len(self.failed_services)

    service_registry = SimpleServiceRegistry()

    # ==================== CYCLE DE VIE ====================

    async def startup():
        """Initialisation de l'application."""
        global startup_time, search_service_status
        startup_time = time.time()
        logger.info("📋 Démarrage application Harena...")
        
        # Vérifier DB
        if not check_database_connection():
            raise RuntimeError("Database connection failed")
        logger.info("✅ Base de données connectée")
        
        # Vérifier config Search Service
        search_config = check_search_config()
        search_service_status["configured"] = search_config["BONSAI_URL"] and search_config["QDRANT_URL"]
        
        # Test Search Service après enregistrement des services
        await asyncio.sleep(2.0)  # Attendre que tous les services soient enregistrés
        
        search_health = await test_search_service_health()
        search_service_status.update({
            "healthy": search_health["overall_status"] in ["fully_operational", "degraded"],
            "clients_injected": search_health["clients_injected"],
            "last_check": datetime.now()
        })
        
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
        title="Harena Finance API",
        description="API complète avec focus Search Service",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "https://app.harena.finance").split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Middleware simple
    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(round(process_time, 3))
        response.headers["X-Search-Status"] = search_service_status.get("healthy", False) and "ok" or "degraded"
        return response

    # ==================== ENDPOINTS PRINCIPAUX ====================

    @app.get("/")
    async def root():
        """Point d'entrée avec statut Search Service."""
        uptime = time.time() - startup_time if startup_time else 0
        
        return {
            "service": "Harena Finance API",
            "status": "online",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "search_service": {
                "configured": search_service_status["configured"],
                "healthy": search_service_status["healthy"],
                "clients_injected": search_service_status["clients_injected"],
                "status": "operational" if search_service_status["healthy"] else "degraded"
            },
            "services": {
                "healthy": service_registry.get_healthy_count(),
                "failed": service_registry.get_failed_count()
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health_check():
        """Health check détaillé."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Test Search Service en temps réel
        search_health = await test_search_service_health()
        
        # Mettre à jour le statut global
        search_service_status.update({
            "healthy": search_health["overall_status"] in ["fully_operational", "degraded"],
            "clients_injected": search_health["clients_injected"],
            "last_check": datetime.now()
        })
        
        overall_status = "excellent" if search_health["overall_status"] == "fully_operational" else \
                        "good" if search_health["overall_status"] == "degraded" else \
                        "critical"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "uptime": {
                "seconds": round(uptime, 2),
                "human": str(timedelta(seconds=int(uptime)))
            },
            "search_service_priority": {
                "status": search_health["overall_status"],
                "is_critical": True,
                "elasticsearch_healthy": search_health["elasticsearch_healthy"],
                "qdrant_healthy": search_health["qdrant_healthy"],
                "clients_injected": search_health["clients_injected"],
                "capabilities": {
                    "lexical_search": search_health["elasticsearch_healthy"],
                    "semantic_search": search_health["qdrant_healthy"],
                    "hybrid_search": search_health["elasticsearch_healthy"] and search_health["qdrant_healthy"]
                }
            },
            "services": {
                "healthy": service_registry.get_healthy_count(),
                "failed": service_registry.get_failed_count(),
                "failed_list": list(service_registry.failed_services.keys())
            },
            "database": {
                "connected": check_database_connection()
            },
            "configuration": check_search_config()
        }

    @app.get("/search-service-status")
    async def search_service_detailed_status():
        """Statut détaillé du Search Service."""
        search_health = await test_search_service_health()
        config = check_search_config()
        
        return {
            "service": "search_service",
            "priority": "critical",
            "timestamp": datetime.now().isoformat(),
            "status": search_health["overall_status"],
            "configuration": config,
            "runtime_health": {
                "importable": search_health["importable"],
                "clients_injected": search_health["clients_injected"],
                "elasticsearch_healthy": search_health["elasticsearch_healthy"],
                "qdrant_healthy": search_health["qdrant_healthy"]
            },
            "capabilities": {
                "lexical_search": search_health["elasticsearch_healthy"],
                "semantic_search": search_health["qdrant_healthy"],
                "hybrid_search": search_health["elasticsearch_healthy"] and search_health["qdrant_healthy"]
            },
            "recommendations": [
                "Configurez BONSAI_URL" if not config["BONSAI_URL"] else None,
                "Configurez QDRANT_URL" if not config["QDRANT_URL"] else None,
                "Vérifiez la connectivité Elasticsearch" if config["BONSAI_URL"] and not search_health["elasticsearch_healthy"] else None,
                "Vérifiez la connectivité Qdrant" if config["QDRANT_URL"] and not search_health["qdrant_healthy"] else None,
                "Redémarrez l'application" if not search_health["clients_injected"] else None,
                "Service complètement opérationnel" if search_health["overall_status"] == "fully_operational" else None
            ],
            "endpoints": [
                "POST /api/v1/search/search - Recherche de transactions",
                "GET /api/v1/search/suggest - Suggestions de recherche"
            ]
        }

    # ==================== IMPORTATION DES SERVICES ====================

    logger.info("📦 Importation des services...")

    # 1. User Service
    try:
        from user_service.api.routes import router as user_router
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
            "build": "heroku-concise",
            "python": sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production")
        }

    @app.get("/robots.txt", include_in_schema=False)
    async def robots():
        return JSONResponse("User-agent: *\nDisallow: /", media_type="text/plain")

    # ==================== GESTIONNAIRE D'ERREURS ====================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "path": request.url.path,
                "search_service_status": search_service_status.get("healthy", False)
            }
        )

    # ==================== RAPPORT FINAL ====================

    logger.info("=" * 80)
    logger.info("🎯 HARENA FINANCE PLATFORM - VERSION CONCISE")
    logger.info(f"📊 Services chargés: {service_registry.get_healthy_count()}")
    logger.info(f"❌ Services échoués: {service_registry.get_failed_count()}")
    
    search_configured = check_search_config()
    critical_ok = search_configured["BONSAI_URL"] and search_configured["QDRANT_URL"]
    
    if critical_ok:
        logger.info("🎉 Search Service: Configuration critique OK")
    else:
        logger.warning("⚠️ Search Service: Configuration incomplète")
    
    logger.info("🌐 Endpoints clés:")
    logger.info("   GET  / - Statut général")
    logger.info("   GET  /health - Santé détaillée")
    logger.info("   GET  /search-service-status - Search Service")
    logger.info("   POST /api/v1/search/search - Recherche")
    logger.info("=" * 80)
    logger.info("✅ Application Harena prête pour Heroku")

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