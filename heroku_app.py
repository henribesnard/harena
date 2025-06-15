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
            self.failed_services = {}
            
        def register_service(self, name: str, router=None, prefix: str = None, status: str = "pending", 
                           app=None, health_check=None):
            """Enregistre un service dans le registre."""
            self.services[name] = {
                "router": router,
                "prefix": prefix,
                "status": status,
                "app": app,
                "error": None,
                "registered_at": datetime.now()
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
            
            self.failed_services[name] = {
                "error": str(error),
                "failed_at": datetime.now()
            }
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

        def get_active_routers(self) -> List[Dict[str, Any]]:
            """Retourne la liste des routeurs actifs à enregistrer."""
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

    # ======== FONCTIONS UTILITAIRES ========

    def _check_database_connection() -> bool:
        """Vérifie la connexion à la base de données."""
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception:
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

        # 🔧 INITIALISATION DES SERVICES CRITIQUES
        logger.info("🔧 Initialisation des services critiques...")
        
        # Initialisation du service d'embeddings pour enrichment_service
        if os.environ.get("OPENAI_API_KEY"):
            try:
                logger.info("🔧 Initialisation de l'EmbeddingService...")
                from enrichment_service.core.embeddings import embedding_service
                await embedding_service.initialize()
                logger.info("✅ EmbeddingService initialisé avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'initialisation de l'EmbeddingService: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ OPENAI_API_KEY non définie, EmbeddingService non initialisé")

        # Initialisation du service d'embeddings pour search_service
        if os.environ.get("OPENAI_API_KEY"):
            try:
                logger.info("🔧 Initialisation de l'EmbeddingService pour search...")
                from search_service.core.embeddings import embedding_service as search_embedding_service
                await search_embedding_service.initialize()
                logger.info("✅ Search EmbeddingService initialisé avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'initialisation du Search EmbeddingService: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ OPENAI_API_KEY non définie, Search EmbeddingService non initialisé")

        # Initialisation du stockage Qdrant pour enrichment_service
        if os.environ.get("QDRANT_URL"):
            try:
                logger.info("🔧 Initialisation du QdrantStorage...")
                from enrichment_service.storage.qdrant import QdrantStorage
                qdrant_storage = QdrantStorage()
                await qdrant_storage.initialize()
                
                # Créer et injecter le transaction processor
                from enrichment_service.core.processor import TransactionProcessor
                transaction_processor = TransactionProcessor(qdrant_storage)
                
                # Injecter dans les routes
                import enrichment_service.api.routes as enrichment_routes
                enrichment_routes.qdrant_storage = qdrant_storage
                enrichment_routes.transaction_processor = transaction_processor
                
                logger.info("✅ QdrantStorage et TransactionProcessor initialisés avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'initialisation de Qdrant: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ QDRANT_URL non définie, QdrantStorage non initialisé")

        logger.info("✅ Initialisation des services critiques terminée")

    async def shutdown():
        """Fonction de nettoyage lors de l'arrêt de l'application"""
        logger.info("⏹️ Application Harena complète en arrêt sur Heroku...")
        
        # Nettoyage des services d'embeddings
        try:
            from enrichment_service.core.embeddings import embedding_service
            await embedding_service.close()
            logger.info("✅ EmbeddingService fermé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture de l'EmbeddingService: {e}")
        
        try:
            from search_service.core.embeddings import embedding_service as search_embedding_service
            await search_embedding_service.close()
            logger.info("✅ Search EmbeddingService fermé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture du Search EmbeddingService: {e}")

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

    # 4. SEARCH SERVICE
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

    # 5. CONVERSATION SERVICE
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
        uptime = time.time() - startup_time if startup_time else 0
        
        return {
            "message": "Harena Finance API - Tous services activés",
            "status": "online",
            "version": "1.0.0-complete",
            "uptime_seconds": round(uptime, 2),
            "services_count": len([s for s in service_registry.services.values() if s["status"] == "ok"]),
            "timestamp": datetime.now().isoformat(),
            "environment": os.environ.get("ENVIRONMENT", "production")
        }

    @app.get("/health", tags=["health"])
    async def health_check():
        """Endpoint de vérification de santé détaillé."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Vérification des services critiques
        services_status = {}
        for name, info in service_registry.services.items():
            services_status[name] = {
                "status": info["status"],
                "registered_at": info["registered_at"].isoformat()
            }
        
        # Vérification de la base de données
        db_status = "unknown"
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                db_status = "connected"
        except Exception:
            db_status = "disconnected"
        
        health_data = {
            "status": "healthy",
            "version": "1.0.0-complete",
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.now().isoformat(),
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "database": {
                "status": db_status,
                "url_configured": bool(os.environ.get("DATABASE_URL"))
            },
            "services": services_status,
            "external_apis": {
                "bridge_configured": bool(os.environ.get("BRIDGE_CLIENT_ID")),
                "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
                "deepseek_configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
                "qdrant_configured": bool(os.environ.get("QDRANT_URL")),
                "cohere_configured": bool(os.environ.get("COHERE_KEY"))
            },
            "failed_services": service_registry.failed_services
        }
        
        # Déterminer le status global
        if db_status == "disconnected":
            health_data["status"] = "degraded"
        elif len(service_registry.failed_services) > 0:
            health_data["status"] = "degraded"
        
        return health_data

    @app.get("/services", tags=["admin"])
    async def list_services():
        """Liste tous les services enregistrés et leur statut."""
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
        
        services_info = {}
        for name, info in service_registry.services.items():
            services_info[name] = {
                "status": info["status"],
                "prefix": info["prefix"],
                "description": descriptions.get(name, "Service Harena"),
                "registered_at": info["registered_at"].isoformat()
            }
        
        return {
            "services": services_info,
            "active_count": len([s for s in service_registry.services.values() if s["status"] == "ok"]),
            "failed_count": len(service_registry.failed_services),
            "failed_services": service_registry.failed_services,
            "timestamp": datetime.now().isoformat()
        }

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
        uptime = time.time() - startup_time if startup_time else 0
        
        # Compter les services actifs vs total
        total_services = len(service_registry.services)
        active_services = len([s for s in service_registry.services.values() if s["status"] == "ok"])
        failed_services = len([s for s in service_registry.services.values() if s["status"] == "failed"])
        
        # Déterminer le statut global
        if failed_services == 0:
            global_status = "operational"
        elif active_services > failed_services:
            global_status = "degraded"
        else:
            global_status = "major_outage"
        
        # Vérification DB rapide
        db_connected = _check_database_connection()
        if not db_connected:
            global_status = "major_outage"
        
        return {
            "status": global_status,
            "uptime_hours": round(uptime / 3600, 2),
            "services": {
                "active": active_services,
                "failed": failed_services,
                "total": total_services
            },
            "database_connected": db_connected,
            "external_services_configured": _count_configured_external_services(),
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/metrics", tags=["monitoring"])
    async def application_metrics():
        """Métriques détaillées pour monitoring externe."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Métriques des services
        service_metrics = {}
        for name, info in service_registry.services.items():
            service_metrics[name] = {
                "status": 1 if info["status"] == "ok" else 0,
                "error_count": 1 if info["status"] == "failed" else 0,
                "uptime_seconds": uptime  # Tous les services ont le même uptime pour l'instant
            }
        
        # Métriques système
        system_metrics = {
            "app_uptime_seconds": uptime,
            "total_services": len(service_registry.services),
            "healthy_services": len([s for s in service_registry.services.values() if s["status"] == "ok"]),
            "failed_services": len([s for s in service_registry.services.values() if s["status"] == "failed"]),
            "database_connected": 1 if _check_database_connection() else 0,
            "external_apis_configured": _count_configured_external_services()
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "services": service_metrics,
            "system": system_metrics,
            "environment": os.environ.get("ENVIRONMENT", "production")
        }

    def get_service_description(service_name: str) -> str:
        """Retourne la description d'un service."""
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

    @app.get("/debug/services", tags=["debug"])
    async def debug_services():
        """Debug détaillé des services (développement uniquement)."""
        if os.environ.get("ENVIRONMENT", "production").lower() == "production":
            raise HTTPException(status_code=404, detail="Not found")
        
        debug_info = {}
        for name, info in service_registry.services.items():
            debug_info[name] = {
                "status": info["status"],
                "prefix": info["prefix"],
                "has_router": info["router"] is not None,
                "router_type": type(info["router"]).__name__ if info["router"] else None,
                "error": info.get("error"),
                "registered_at": info["registered_at"].isoformat() if info.get("registered_at") else None
            }
        
        return {
            "services": debug_info,
            "registry_state": {
                "total_services": len(service_registry.services),
                "failed_services": len(service_registry.failed_services),
                "service_apps_count": len(service_registry.service_apps)
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/debug/routes", tags=["debug"])
    async def debug_routes():
        """Liste toutes les routes enregistrées (développement uniquement)."""
        if os.environ.get("ENVIRONMENT", "production").lower() == "production":
            raise HTTPException(status_code=404, detail="Not found")
        
        routes_info = []
        for route in app.routes:
            route_info = {
                "path": route.path,
                "methods": list(route.methods) if hasattr(route, 'methods') else [],
                "name": route.name if hasattr(route, 'name') else None,
                "tags": route.tags if hasattr(route, 'tags') else []
            }
            routes_info.append(route_info)
        
        return {
            "total_routes": len(routes_info),
            "routes": routes_info,
            "timestamp": datetime.now().isoformat()
        }

    # ======== ENDPOINTS D'ADMINISTRATION ========

    @app.post("/admin/restart-service/{service_name}", tags=["admin"])
    async def restart_service(service_name: str):
        """Redémarre un service spécifique (admin uniquement)."""
        # Note: En production, cet endpoint devrait être protégé par authentification
        if service_name not in service_registry.services:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        # Pour l'instant, retourner un message d'information
        # Dans une implémentation complète, on pourrait réimporter le module
        return {
            "message": f"Service restart requested for {service_name}",
            "note": "Service restart functionality not implemented in this version",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/admin/logs/{service_name}", tags=["admin"])
    async def get_service_logs(service_name: str, lines: int = 100):
        """Récupère les logs d'un service spécifique (admin uniquement)."""
        if service_name not in service_registry.services:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        # Note: Implémentation basique - en production, on lirait les vrais logs
        return {
            "service": service_name,
            "lines_requested": lines,
            "logs": [
                f"[INFO] Service {service_name} operational",
                f"[DEBUG] Last status check: {datetime.now().isoformat()}"
            ],
            "note": "Log retrieval functionality would be implemented with proper log aggregation",
            "timestamp": datetime.now().isoformat()
        }

    # ======== ENDPOINTS DE MONITORING AVANCÉ ========

    @app.get("/monitoring/database", tags=["monitoring"])
    async def database_monitoring():
        """Informations détaillées sur l'état de la base de données."""
        try:
            from db_service.session import engine
            from sqlalchemy import text
            
            # Test de connexion de base
            start_time = time.time()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            connection_time = time.time() - start_time
            
            # Informations sur le pool de connexions
            pool_info = {
                "size": engine.pool.size(),
                "checked_in": engine.pool.checkedin(),
                "checked_out": engine.pool.checkedout(),
                "overflow": engine.pool.overflow()
            }
            
            return {
                "status": "connected",
                "connection_time_ms": round(connection_time * 1000, 2),
                "pool": pool_info,
                "url_configured": bool(os.environ.get("DATABASE_URL")),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "url_configured": bool(os.environ.get("DATABASE_URL")),
                "timestamp": datetime.now().isoformat()
            }

    @app.get("/monitoring/external-apis", tags=["monitoring"])
    async def external_apis_monitoring():
        """État des APIs externes configurées."""
        apis_status = {}
        
        # Bridge API
        if os.environ.get("BRIDGE_CLIENT_ID") and os.environ.get("BRIDGE_CLIENT_SECRET"):
            apis_status["bridge"] = {
                "configured": True,
                "client_id_set": bool(os.environ.get("BRIDGE_CLIENT_ID")),
                "client_secret_set": bool(os.environ.get("BRIDGE_CLIENT_SECRET")),
                "webhook_secret_set": bool(os.environ.get("BRIDGE_WEBHOOK_SECRET"))
            }
        else:
            apis_status["bridge"] = {"configured": False}
        
        # OpenAI API
        apis_status["openai"] = {
            "configured": bool(os.environ.get("OPENAI_API_KEY")),
            "api_key_set": bool(os.environ.get("OPENAI_API_KEY"))
        }
        
        # DeepSeek API
        apis_status["deepseek"] = {
            "configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "api_key_set": bool(os.environ.get("DEEPSEEK_API_KEY"))
        }
        
        # Qdrant
        apis_status["qdrant"] = {
            "configured": bool(os.environ.get("QDRANT_URL")),
            "url_set": bool(os.environ.get("QDRANT_URL")),
            "api_key_set": bool(os.environ.get("QDRANT_API_KEY"))
        }
        
        # Cohere
        apis_status["cohere"] = {
            "configured": bool(os.environ.get("COHERE_KEY")),
            "api_key_set": bool(os.environ.get("COHERE_KEY"))
        }
        
        # Elasticsearch (Searchbox/Bonsai)
        searchbox_url = os.environ.get("SEARCHBOX_URL")
        bonsai_url = os.environ.get("BONSAI_URL")
        apis_status["elasticsearch"] = {
            "configured": bool(searchbox_url or bonsai_url),
            "searchbox_set": bool(searchbox_url),
            "bonsai_set": bool(bonsai_url),
            "active_provider": "searchbox" if searchbox_url else "bonsai" if bonsai_url else None
        }
        
        total_configured = len([api for api in apis_status.values() if api.get("configured", False)])
        
        return {
            "total_configured": total_configured,
            "apis": apis_status,
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
            error_type = "internal_error"
        else:
            error_detail = str(exc)
            error_type = type(exc).__name__
        
        return JSONResponse(
            status_code=500,
            content={
                "error": error_type,
                "detail": error_detail,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path,
                "method": request.method
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Gestionnaire pour les exceptions HTTP."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "detail": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path,
                "method": request.method
            }
        )

    # ======== ENDPOINTS DE COMPATIBILITÉ ========

    @app.get("/ping", tags=["health"])
    async def ping():
        """Endpoint simple pour les checks de santé externes."""
        return {"status": "pong", "timestamp": datetime.now().isoformat()}

    @app.get("/version", tags=["info"])
    async def version_info():
        """Informations de version de l'application."""
        return {
            "version": "1.0.0-complete",
            "build": "heroku-production",
            "python_version": sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "platform": "Heroku",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/robots.txt", include_in_schema=False)
    async def robots_txt():
        """Fichier robots.txt pour les crawlers."""
        return JSONResponse(
            content="User-agent: *\nDisallow: /",
            media_type="text/plain"
        )

    # ======== FINALISATION ========

    logger.info("✅ Application Harena complète configurée et prête pour Heroku")

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