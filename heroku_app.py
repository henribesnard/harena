"""
Application Harena complète pour déploiement Heroku.

Module optimisé pour le déploiement sur Heroku, avec gestion de TOUS les services :
- User Service (gestion utilisateurs et authentification)
- Sync Service (synchronisation des données bancaires)
- Enrichment Service (structuration et stockage vectoriel)
- Search Service (recherche hybride)
- Conversation Service (assistant IA conversationnel)
"""

import logging
import os
import sys
import traceback
import time
from pathlib import Path

# Configuration du logging AVANT tout autre import
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("heroku_startup")

try:
    logger.info("🚀 Début de l'importation de heroku_app.py - VERSION COMPLÈTE")
    
    # Log des variables d'environnement critiques (sans exposer les secrets)
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'not_set')}")
    logger.info(f"DATABASE_URL configured: {bool(os.environ.get('DATABASE_URL'))}")
    logger.info(f"BRIDGE_CLIENT_ID configured: {bool(os.environ.get('BRIDGE_CLIENT_ID'))}")
    logger.info(f"OPENAI_API_KEY configured: {bool(os.environ.get('OPENAI_API_KEY'))}")
    logger.info(f"DEEPSEEK_API_KEY configured: {bool(os.environ.get('DEEPSEEK_API_KEY'))}")
    logger.info(f"QDRANT_URL configured: {bool(os.environ.get('QDRANT_URL'))}")
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

    logger.info(f"Python path entries: {len(sys.path)}")

    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from datetime import datetime, timedelta
    from typing import Dict, List, Any, Optional
    import asyncio

    logger.info("✅ Imports de base réussis")

    # ======== DÉFINITION DU REGISTRE DE SERVICES ========

    class ServiceRegistry:
        """Classe pour gérer tous les services disponibles et leurs routeurs."""
        
        def __init__(self):
            self.services = {}
            self.service_apps = {}  # Pour les applications complètes
            self.service_health_checks = {}  # Pour les vérifications de santé
            
        def register_service(self, name: str, router=None, prefix: str = None, status: str = "pending", 
                           app=None, health_check=None):
            """Enregistre un service dans le registre."""
            self.services[name] = {
                "router": router,
                "prefix": prefix,
                "status": status,
                "app": app,
                "error": None
            }
            if app:
                self.service_apps[name] = app
            if health_check:
                self.service_health_checks[name] = health_check
            logger.info(f"Service {name} enregistré avec statut {status}")
            
        def mark_service_failed(self, name: str, error: Exception):
            """Marque un service comme échoué avec l'erreur."""
            if name in self.services:
                self.services[name]["status"] = "failed"
                self.services[name]["error"] = str(error)
                logger.error(f"Service {name} marqué comme échoué: {error}")
            
        def get_service_status(self) -> Dict[str, str]:
            """Retourne le statut de tous les services."""
            return {name: info["status"] for name, info in self.services.items()}
        
        def get_service_errors(self) -> Dict[str, str]:
            """Retourne les erreurs des services échoués."""
            return {name: info["error"] for name, info in self.services.items() 
                   if info["status"] == "failed" and info["error"]}
        
        def get_available_routers(self) -> List[Dict[str, Any]]:
            """Retourne les routeurs disponibles avec leurs préfixes."""
            routers = [
                {"name": name, "router": info["router"], "prefix": info["prefix"]}
                for name, info in self.services.items()
                if info["status"] == "ok" and info["router"] is not None
            ]
            logger.info(f"Nombre de routeurs disponibles: {len(routers)}")
            return routers
        
        def get_service_apps(self) -> Dict[str, Any]:
            """Retourne les applications de service pour montage."""
            return {name: app for name, app in self.service_apps.items()}

    # Création du registre de services
    service_registry = ServiceRegistry()
    logger.info("✅ ServiceRegistry créé")

    # Variables globales
    startup_time = None

    # ======== FONCTION DU CYCLE DE VIE ========

    async def startup():
        """Fonction d'initialisation de l'application"""
        global startup_time
        logger.info("📋 Application Harena complète en démarrage sur Heroku...")
        startup_time = time.time()
        
        # Vérification des variables d'environnement critiques
        required_env_vars = ["DATABASE_URL"]
        optional_env_vars = {
            "BRIDGE_CLIENT_ID": "Fonctionnalités de synchronisation bancaire",
            "BRIDGE_CLIENT_SECRET": "Fonctionnalités de synchronisation bancaire",
            "OPENAI_API_KEY": "Génération d'embeddings et recherche sémantique",
            "DEEPSEEK_API_KEY": "Assistant conversationnel IA",
            "QDRANT_URL": "Stockage vectoriel",
            "COHERE_KEY": "Reranking des résultats de recherche",
            "SEARCHBOX_URL": "Recherche lexicale Elasticsearch",
            "BONSAI_URL": "Recherche lexicale Elasticsearch (alternative)"
        }
        
        missing_required = [var for var in required_env_vars if not os.environ.get(var)]
        if missing_required:
            logger.error(f"❌ Variables d'environnement critiques manquantes: {', '.join(missing_required)}")
            raise RuntimeError(f"Missing required environment variables: {missing_required}")
        
        missing_optional = [var for var in optional_env_vars.keys() if not os.environ.get(var)]
        if missing_optional:
            logger.warning("⚠️ Variables d'environnement optionnelles manquantes:")
            for var in missing_optional:
                logger.warning(f"  - {var}: {optional_env_vars[var]} ne fonctionnera pas")
        
        # Test de la connexion base de données
        try:
            logger.info("🔍 Test de connexion à la base de données...")
            from db_service.session import engine
            from sqlalchemy import text
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("✅ Connexion à la base de données établie avec succès")
        except Exception as db_error:
            logger.error(f"❌ Erreur de connexion à la base de données: {db_error}")
            raise RuntimeError(f"Database connection failed: {db_error}")

    async def shutdown():
        """Fonction de nettoyage lors de l'arrêt de l'application"""
        logger.info("⏹️ Application Harena complète en arrêt sur Heroku...")

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

    logger.info("🏗️ Création de l'application FastAPI...")

    app = FastAPI(
        title="Harena Finance API (Production Complete)",
        description="API complète pour les services financiers Harena - Tous services activés",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    logger.info("✅ Application FastAPI créée")

    # Configuration CORS sécurisée pour production
    ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "https://app.harena.finance").split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    logger.info("✅ Middleware CORS configuré")

    # ======== IMPORTATION DES SERVICES ========

    logger.info("📦 Début de l'importation de TOUS les services...")

    # 1. USER SERVICE
    logger.info("📦 Importation du User Service...")
    try:
        from user_service.api.endpoints import users as users_router
        from config_service.config import settings as user_settings
        service_registry.register_service(
            "user_service", 
            router=users_router.router, 
            prefix=user_settings.API_V1_STR + "/users",
            status="ok"
        )
        logger.info("✅ User Service importé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'importation du User Service: {e}")
        logger.error(traceback.format_exc())
        service_registry.mark_service_failed("user_service", e)

    # 2. SYNC SERVICE - Import de tous les endpoints
    logger.info("📦 Importation du Sync Service...")
    try:
        # Sync Service - Synchronisation principale
        from sync_service.api.endpoints.sync import router as sync_router
        service_registry.register_service(
            "sync_service", 
            router=sync_router,
            prefix="/api/v1/sync",
            status="ok"
        )
        
        # Sync Service - Transactions
        from sync_service.api.endpoints.transactions import router as transactions_router
        service_registry.register_service(
            "transactions_service", 
            router=transactions_router,
            prefix="/api/v1/transactions",
            status="ok"
        )
        
        # Sync Service - Webhooks
        from sync_service.api.endpoints.webhooks import router as webhooks_router
        service_registry.register_service(
            "webhooks_service", 
            router=webhooks_router,
            prefix="/webhooks",
            status="ok"
        )
        
        # Sync Service - Comptes
        from sync_service.api.endpoints.accounts import router as accounts_router
        service_registry.register_service(
            "accounts_service", 
            router=accounts_router,
            prefix="/api/v1/accounts",
            status="ok"
        )
        
        # Sync Service - Items
        from sync_service.api.endpoints.items import router as items_router
        service_registry.register_service(
            "items_service", 
            router=items_router,
            prefix="/api/v1/items",
            status="ok"
        )
        
        # Sync Service - Stocks
        from sync_service.api.endpoints.stocks import router as stocks_router
        service_registry.register_service(
            "stocks_service", 
            router=stocks_router,
            prefix="/api/v1/stocks",
            status="ok"
        )
        
        # Sync Service - Categories
        from sync_service.api.endpoints.categories import router as categories_router
        service_registry.register_service(
            "categories_service", 
            router=categories_router,
            prefix="/api/v1/categories",
            status="ok"
        )
        
        # Sync Service - Insights
        from sync_service.api.endpoints.insights import router as insights_router
        service_registry.register_service(
            "insights_service", 
            router=insights_router,
            prefix="/api/v1/insights",
            status="ok"
        )
        
        logger.info("✅ Sync Service (tous endpoints) importé avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'importation du Sync Service: {e}")
        logger.error(traceback.format_exc())
        service_registry.mark_service_failed("sync_service", e)

    # 3. ENRICHMENT SERVICE
    logger.info("📦 Importation du Enrichment Service...")
    try:
        from enrichment_service.api.routes import router as enrichment_router
        service_registry.register_service(
            "enrichment_service",
            router=enrichment_router,
            prefix="/api/v1/enrich",
            status="ok"
        )
        logger.info("✅ Enrichment Service importé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'importation du Enrichment Service: {e}")
        logger.error(traceback.format_exc())
        service_registry.mark_service_failed("enrichment_service", e)

    # 4. SEARCH SERVICE - MAINTENANT ACTIVÉ
    logger.info("📦 Importation du Search Service...")
    try:
        from search_service.api.routes import router as search_router
        service_registry.register_service(
            "search_service",
            router=search_router,
            prefix="/api/v1/search",
            status="ok"
        )
        logger.info("✅ Search Service importé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'importation du Search Service: {e}")
        logger.error(traceback.format_exc())
        service_registry.mark_service_failed("search_service", e)

    # 5. CONVERSATION SERVICE - MAINTENANT ACTIVÉ
    logger.info("📦 Importation du Conversation Service...")
    try:
        # Routes REST
        from conversation_service.api.routes import router as conversation_router
        service_registry.register_service(
            "conversation_service",
            router=conversation_router,
            prefix="/api/v1/conversation",
            status="ok"
        )
        
        # Routes WebSocket
        from conversation_service.api.websocket import websocket_router
        service_registry.register_service(
            "conversation_websocket",
            router=websocket_router,
            prefix="/ws",
            status="ok"
        )
        
        logger.info("✅ Conversation Service (REST + WebSocket) importé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'importation du Conversation Service: {e}")
        logger.error(traceback.format_exc())
        service_registry.mark_service_failed("conversation_service", e)

    logger.info("📦 Fin de l'importation de TOUS les services")

    # ======== INCLUSION DES ROUTERS ========

    logger.info("🔗 Inclusion des routers...")
    # Inclure tous les routers disponibles
    for service_info in service_registry.get_available_routers():
        try:
            app.include_router(
                service_info["router"],
                prefix=service_info["prefix"],
                tags=[service_info["name"]]
            )
            logger.info(f"✅ Router {service_info['name']} inclus avec préfixe {service_info['prefix']}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'inclusion du router {service_info['name']}: {e}")
            logger.error(traceback.format_exc())
            # Marquer le service comme échoué si l'inclusion du router échoue
            service_registry.mark_service_failed(service_info["name"], e)

    # ======== MIDDLEWARE DE MÉTRIQUES ========

    @app.middleware("http")
    async def add_metrics_and_logging(request, call_next):
        """Middleware pour les métriques et le logging des requêtes."""
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
        response.headers["X-Powered-By"] = "Harena Finance Platform"
        return response

    # ======== ENDPOINTS DE BASE ========

    @app.get("/", tags=["health"])
    async def root():
        """Point d'entrée racine pour vérifier que l'application est en ligne."""
        service_statuses = service_registry.get_service_status()
        active_services = [name for name, status in service_statuses.items() if status == "ok"]
        failed_services = [name for name, status in service_statuses.items() if status == "failed"]
        disabled_services = [name for name, status in service_statuses.items() if status == "disabled"]
        
        # Calculer les capacités disponibles
        capabilities = {
            "user_management": "user_service" in active_services,
            "bank_sync": any(s in active_services for s in ["sync_service", "transactions_service"]),
            "data_enrichment": "enrichment_service" in active_services,
            "smart_search": "search_service" in active_services,
            "ai_assistant": "conversation_service" in active_services,
            "websocket_chat": "conversation_websocket" in active_services,
            "webhooks": "webhooks_service" in active_services,
            "accounts_management": "accounts_service" in active_services,
            "stocks_tracking": "stocks_service" in active_services,
            "financial_insights": "insights_service" in active_services
        }
        
        return {
            "status": "running",
            "application": "Harena Finance API (Complete)",
            "version": "1.0.0",
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "services": {
                "active": active_services,
                "failed": failed_services,
                "disabled": disabled_services,
                "total": len(service_statuses)
            },
            "capabilities": capabilities,
            "features": {
                "user_management": capabilities["user_management"],
                "bank_sync": capabilities["bank_sync"],
                "data_enrichment": capabilities["data_enrichment"],
                "smart_search": capabilities["smart_search"],
                "ai_assistant": capabilities["ai_assistant"],
                "real_time_chat": capabilities["websocket_chat"]
            },
            "api_endpoints": {
                "documentation": {
                    "swagger": "/docs",
                    "redoc": "/redoc"
                },
                "health": "/health",
                "users": "/api/v1/users",
                "transactions": "/api/v1/transactions",
                "search": "/api/v1/search",
                "chat": "/api/v1/conversation",
                "websocket": "/ws"
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health", tags=["health"])
    async def health_check():
        """Vérification détaillée de l'état de santé de tous les services."""
        
        # Vérifier la connexion à la base de données
        db_status = "unknown"
        db_error = None
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                db_version = result.fetchone()[0] if result else "unknown"
            db_status = "connected"
        except Exception as e:
            db_status = "error"
            db_error = str(e)[:200]
        
        # Vérification des services externes
        external_services = {
            "bridge_api": {
                "status": "configured" if (os.environ.get("BRIDGE_CLIENT_ID") and os.environ.get("BRIDGE_CLIENT_SECRET")) else "not_configured",
                "required_for": ["bank synchronization", "transaction fetching"]
            },
            "openai_api": {
                "status": "configured" if os.environ.get("OPENAI_API_KEY") else "not_configured",
                "required_for": ["embeddings generation", "semantic search"]
            },
            "deepseek_api": {
                "status": "configured" if os.environ.get("DEEPSEEK_API_KEY") else "not_configured",
                "required_for": ["AI conversation", "intent detection"]
            },
            "qdrant": {
                "status": "configured" if os.environ.get("QDRANT_URL") else "not_configured",
                "required_for": ["vector storage", "semantic search"]
            },
            "elasticsearch": {
                "status": "configured" if (os.environ.get("SEARCHBOX_URL") or os.environ.get("BONSAI_URL")) else "not_configured",
                "required_for": ["lexical search", "full-text search"]
            },
            "cohere": {
                "status": "configured" if os.environ.get("COHERE_KEY") else "not_configured",
                "required_for": ["search result reranking"]
            }
        }
        
        # État général de l'application
        service_statuses = service_registry.get_service_status()
        service_errors = service_registry.get_service_errors()
        failed_services = [name for name, status in service_statuses.items() if status == "failed"]
        active_services = [name for name, status in service_statuses.items() if status == "ok"]
        
        if db_status == "error":
            overall_status = "critical"
        elif len(failed_services) > len(active_services):
            overall_status = "critical"
        elif failed_services:
            overall_status = "degraded"
        elif len(active_services) >= 3:  # Au moins les services de base
            overall_status = "healthy"
        else:
            overall_status = "limited"
        
        # Calculer l'uptime
        uptime = time.time() - startup_time if startup_time else 0
        
        # Capacités système
        system_capabilities = {
            "can_sync_banks": (
                external_services["bridge_api"]["status"] == "configured" and 
                any(s in active_services for s in ["sync_service", "transactions_service"])
            ),
            "can_search_semantic": (
                external_services["openai_api"]["status"] == "configured" and 
                external_services["qdrant"]["status"] == "configured" and
                "search_service" in active_services
            ),
            "can_search_lexical": (
                external_services["elasticsearch"]["status"] == "configured" and
                "search_service" in active_services
            ),
            "can_chat_ai": (
                external_services["deepseek_api"]["status"] == "configured" and
                "conversation_service" in active_services
            ),
            "can_rerank_results": (
                external_services["cohere"]["status"] == "configured" and
                "search_service" in active_services
            ),
            "can_enrich_data": (
                external_services["openai_api"]["status"] == "configured" and
                external_services["qdrant"]["status"] == "configured" and
                "enrichment_service" in active_services
            ),
            "can_track_stocks": "stocks_service" in active_services,
            "can_manage_accounts": "accounts_service" in active_services,
            "can_receive_webhooks": "webhooks_service" in active_services,
            "has_real_time_chat": "conversation_websocket" in active_services
        }
        
        response_data = {
            "status": overall_status,
            "database": {
                "status": db_status,
                "error": db_error,
                "version": db_version if db_status == "connected" else None
            },
            "services": {
                "statuses": service_statuses,
                "errors": service_errors,
                "active_count": len(active_services),
                "failed_count": len(failed_services),
                "total_count": len(service_statuses)
            },
            "external_services": external_services,
            "system": {
                "environment": os.environ.get("ENVIRONMENT", "production"),
                "uptime_seconds": uptime,
                "uptime_human": str(timedelta(seconds=int(uptime))),
                "python_version": sys.version.split()[0],
                "platform": "Heroku"
            },
            "capabilities": system_capabilities,
            "performance": {
                "avg_response_time": "unknown",  # Pourrait être calculé avec des métriques
                "requests_per_minute": "unknown",
                "error_rate": "unknown"
            },
            "configuration": {
                "cors_origins": len(ALLOWED_ORIGINS),
                "api_docs_enabled": True,
                "debug_mode": os.environ.get("DEBUG", "False").lower() == "true"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return response_data

    @app.get("/services", tags=["health"])
    async def services_status():
        """Détail du statut de chaque service."""
        service_statuses = service_registry.get_service_status()
        service_errors = service_registry.get_service_errors()
        
        services_detail = {}
        for name, status in service_statuses.items():
            services_detail[name] = {
                "status": status,
                "description": _get_service_description(name),
                "error": service_errors.get(name),
                "prefix": service_registry.services[name].get("prefix"),
                "active": status == "ok"
            }
        
        return {
            "services": services_detail,
            "summary": {
                "total": len(service_statuses),
                "active": len([s for s in service_statuses.values() if s == "ok"]),
                "failed": len([s for s in service_statuses.values() if s == "failed"]),
                "disabled": len([s for s in service_statuses.values() if s == "disabled"])
            },
            "timestamp": datetime.now().isoformat()
        }

    def _get_service_description(service_name: str) -> str:
        """Retourne une description pour chaque service."""
        descriptions = {
            "user_service": "Gestion des utilisateurs et authentification",
            "sync_service": "Synchronisation des données bancaires",
            "transactions_service": "Gestion des transactions financières",
            "webhooks_service": "Traitement des webhooks Bridge API",
            "accounts_service": "Gestion des comptes bancaires",
            "items_service": "Gestion des connexions bancaires",
            "stocks_service": "Gestion des actions et investissements",
            "categories_service": "Catégorisation des transactions",
            "insights_service": "Analyses et insights financiers",
            "enrichment_service": "Enrichissement et vectorisation des données",
            "search_service": "Recherche hybride dans les transactions",
            "conversation_service": "Assistant IA conversationnel",
            "conversation_websocket": "WebSocket pour chat temps réel"
        }
        return descriptions.get(service_name, "Service Harena")

    # ======== ENDPOINTS DE DÉVELOPPEMENT ========

    @app.get("/debug/info", tags=["debug"])
    async def debug_info(request: Request):
        """Informations de debug (en mode développement uniquement)."""
        if os.environ.get("ENVIRONMENT", "production").lower() == "production":
            raise HTTPException(status_code=404, detail="Not found")
        
        return {
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client": request.client.host if request.client else None
            },
            "environment": dict(os.environ),
            "python_path": sys.path,
            "services": service_registry.services,
            "timestamp": datetime.now().isoformat()
        }

    # ======== GESTIONNAIRE D'EXCEPTIONS ========

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Gestionnaire global d'exceptions pour toute l'application."""
        logger.error(f"Exception non gérée sur {request.method} {request.url.path}: {str(exc)}", exc_info=True)
        
        # En production, ne pas exposer les détails de l'erreur
        is_production = os.environ.get("ENVIRONMENT", "production").lower() == "production"
        
        if is_production:
            error_detail = "Une erreur interne est survenue. Veuillez contacter le support."
            error_id = f"ERR-{int(time.time())}"
        else:
            error_detail = str(exc)
            error_id = None
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "detail": error_detail,
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
        """Gestionnaire pour les erreurs 404."""
        return JSONResponse(
            status_code=404,
            content={
                "status": "not_found",
                "message": f"Endpoint {request.url.path} not found",
                "available_endpoints": {
                    "health": "/health",
                    "services": "/services", 
                    "docs": "/docs",
                    "redoc": "/redoc",
                    "api": {
                        "users": "/api/v1/users",
                        "sync": "/api/v1/sync",
                        "transactions": "/api/v1/transactions",
                        "accounts": "/api/v1/accounts",
                        "search": "/api/v1/search",
                        "conversation": "/api/v1/conversation",
                        "enrichment": "/api/v1/enrich"
                    },
                    "websocket": "/ws"
                },
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.exception_handler(500)
    async def internal_server_error_handler(request: Request, exc: HTTPException):
        """Gestionnaire pour les erreurs 500."""
        logger.error(f"500 error on {request.method} {request.url.path}: {exc.detail}")
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "internal_error",
                "message": "Service temporarily unavailable",
                "detail": "Please try again later or contact support",
                "timestamp": datetime.now().isoformat()
            }
        )

    # ======== ENDPOINTS DE MONITORING ========

    @app.get("/metrics", tags=["monitoring"])
    async def metrics_endpoint():
        """Endpoint pour les métriques de monitoring (format simple)."""
        service_statuses = service_registry.get_service_status()
        active_count = len([s for s in service_statuses.values() if s == "ok"])
        failed_count = len([s for s in service_statuses.values() if s == "failed"])
        
        # Métriques basiques
        uptime = time.time() - startup_time if startup_time else 0
        
        metrics = {
            "harena_services_active": active_count,
            "harena_services_failed": failed_count,
            "harena_services_total": len(service_statuses),
            "harena_uptime_seconds": uptime,
            "harena_database_connected": 1 if _check_database_connection() else 0,
            "harena_external_services_configured": _count_configured_external_services(),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics

    def _check_database_connection() -> bool:
        """Vérifie rapidement la connexion à la base de données."""
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False

    def _count_configured_external_services() -> int:
        """Compte le nombre de services externes configurés."""
        external_vars = [
            "BRIDGE_CLIENT_ID", "BRIDGE_CLIENT_SECRET",
            "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
            "QDRANT_URL", "COHERE_KEY",
            "SEARCHBOX_URL", "BONSAI_URL"
        ]
        return len([var for var in external_vars if os.environ.get(var)])

    # ======== ENDPOINT DE PERFORMANCE ========

    @app.get("/performance", tags=["monitoring"])
    async def performance_info():
        """Informations de performance de l'application."""
        service_statuses = service_registry.get_service_status()
        
        # Calculer des métriques de base
        total_services = len(service_statuses)
        active_services = len([s for s in service_statuses.values() if s == "ok"])
        health_ratio = active_services / total_services if total_services > 0 else 0
        
        uptime = time.time() - startup_time if startup_time else 0
        
        return {
            "health_ratio": health_ratio,
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
                "connected": _check_database_connection(),
                "connection_pool": "SQLAlchemy default"
            },
            "external_dependencies": {
                "configured_count": _count_configured_external_services(),
                "bridge_api": bool(os.environ.get("BRIDGE_CLIENT_ID")),
                "ai_services": bool(os.environ.get("DEEPSEEK_API_KEY")),
                "vector_storage": bool(os.environ.get("QDRANT_URL")),
                "search_engine": bool(os.environ.get("SEARCHBOX_URL") or os.environ.get("BONSAI_URL"))
            },
            "timestamp": datetime.now().isoformat()
        }

    # ======== ENDPOINTS POUR LE STATUT GLOBAL ========

    @app.get("/status", tags=["health"])
    async def application_status():
        """Statut global de l'application (version condensée du health check)."""
        service_statuses = service_registry.get_service_status()
        active_services = [name for name, status in service_statuses.items() if status == "ok"]
        failed_services = [name for name, status in service_statuses.items() if status == "failed"]
        
        # Déterminer le statut global
        if not _check_database_connection():
            status = "critical"
        elif len(failed_services) > len(active_services):
            status = "critical"
        elif failed_services:
            status = "degraded"
        elif len(active_services) >= 5:  # Si au moins 5 services fonctionnent
            status = "healthy"
        else:
            status = "limited"
        
        return {
            "status": status,
            "services": {
                "active": len(active_services),
                "failed": len(failed_services),
                "total": len(service_statuses)
            },
            "core_features": {
                "authentication": "user_service" in active_services,
                "data_sync": any(s in active_services for s in ["sync_service", "transactions_service"]),
                "search": "search_service" in active_services,
                "ai_chat": "conversation_service" in active_services,
                "data_enrichment": "enrichment_service" in active_services
            },
            "database_connected": _check_database_connection(),
            "uptime_hours": round((time.time() - startup_time) / 3600, 2) if startup_time else 0,
            "timestamp": datetime.now().isoformat()
        }

    # ======== GESTION GRACIEUSE DES SIGNAUX ========

    import signal
    
    def signal_handler(signum, frame):
        """Gestionnaire pour les signaux système."""
        logger.info(f"Signal {signum} reçu, arrêt gracieux en cours...")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # ======== TÂCHES DE FOND ========

    async def periodic_health_check():
        """Tâche de vérification périodique de la santé des services."""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Vérifier la base de données
                db_ok = _check_database_connection()
                if not db_ok:
                    logger.warning("❌ Perte de connexion à la base de données détectée")
                
                # Compter les services actifs
                service_statuses = service_registry.get_service_status()
                active_count = len([s for s in service_statuses.values() if s == "ok"])
                total_count = len(service_statuses)
                
                logger.info(f"🔍 Health check: {active_count}/{total_count} services actifs, DB: {'✅' if db_ok else '❌'}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans periodic_health_check: {e}")

    # Démarrer la tâche de fond
    @app.on_event("startup")
    async def start_background_tasks():
        """Démarre les tâches de fond."""
        if os.environ.get("ENABLE_HEALTH_CHECK", "true").lower() == "true":
            asyncio.create_task(periodic_health_check())
            logger.info("✅ Tâche de vérification de santé périodique démarrée")

    logger.info("✅ Application heroku_app.py COMPLÈTE entièrement initialisée")
    logger.info(f"📊 Services configurés: {len(service_registry.services)}")
    logger.info(f"🔗 Routeurs inclus: {len(service_registry.get_available_routers())}")

except Exception as e:
    logger.error(f"❌ ERREUR FATALE lors de l'importation de heroku_app.py: {e}")
    logger.error(traceback.format_exc())
    raise

# Point d'entrée pour le serveur gunicorn configuré dans Procfile
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Démarrage autonome de l'application Harena COMPLÈTE sur port {port}")
    logger.info("🎯 TOUS LES SERVICES SONT ACTIVÉS")
    
    uvicorn.run(
        "heroku_app:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )