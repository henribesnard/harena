"""
Application Harena complète pour déploiement Heroku.

Module optimisé pour le déploiement sur Heroku, avec gestion de tous les services :
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
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("harena")

# Correction de l'URL de base de données pour Heroku
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    logger.info("DATABASE_URL corrigé pour SQLAlchemy 1.4+")

# Définir l'environnement global
os.environ["ENVIRONMENT"] = os.getenv("ENVIRONMENT", "production")

# S'assurer que tous les modules sont accessibles
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    logger.info(f"Ajout du répertoire courant au sys.path: {current_dir}")

logger.info(f"Python path: {sys.path}")

# ======== DÉFINITION DU REGISTRE DE SERVICES ========

class ServiceRegistry:
    """Classe pour gérer tous les services disponibles et leurs routeurs."""
    
    def __init__(self):
        self.services = {}
        self.service_apps = {}  # Pour les applications complètes
        
    def register_service(self, name: str, router=None, prefix: str = None, status: str = "pending", app=None):
        """Enregistre un service dans le registre."""
        self.services[name] = {
            "router": router,
            "prefix": prefix,
            "status": status,
            "app": app
        }
        if app:
            self.service_apps[name] = app
        logger.info(f"Service {name} enregistré avec statut {status}")
        
    def get_service_status(self) -> Dict[str, str]:
        """Retourne le statut de tous les services."""
        return {name: info["status"] for name, info in self.services.items()}
    
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

# Variables globales
startup_time = None

# ======== FONCTION DU CYCLE DE VIE ========

async def startup():
    """Fonction d'initialisation de l'application"""
    global startup_time
    logger.info("Application Harena complète en démarrage sur Heroku...")
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
        logger.error(f"Variables d'environnement critiques manquantes: {', '.join(missing_required)}")
        raise RuntimeError(f"Missing required environment variables: {missing_required}")
    
    missing_optional = [var for var in optional_env_vars.keys() if not os.environ.get(var)]
    if missing_optional:
        logger.warning("Variables d'environnement optionnelles manquantes:")
        for var in missing_optional:
            logger.warning(f"  - {var}: {optional_env_vars[var]} ne fonctionnera pas")
    
    # Test de la connexion base de données
    try:
        from db_service.session import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Connexion à la base de données établie avec succès")
    except Exception as db_error:
        logger.error(f"Erreur de connexion à la base de données: {db_error}")
        raise RuntimeError(f"Database connection failed: {db_error}")

async def shutdown():
    """Fonction de nettoyage lors de l'arrêt de l'application"""
    logger.info("Application Harena complète en arrêt sur Heroku...")

# ======== GESTIONNAIRE DE CYCLE DE VIE ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    await startup()
    yield
    await shutdown()

# ======== CRÉATION DE L'APPLICATION ========

app = FastAPI(
    title="Harena Finance API (Production)",
    description="API complète pour les services financiers Harena - Déploiement Heroku",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configuration CORS sécurisée pour production
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "https://app.harena.finance").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ======== IMPORTATION DES SERVICES ========

# 1. USER SERVICE
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
except ImportError as e:
    logger.error(f"❌ Erreur lors de l'importation du User Service: {e}")
    service_registry.register_service("user_service", status="failed")

# 2. SYNC SERVICE - Import de tous les endpoints
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
    
except ImportError as e:
    logger.error(f"❌ Erreur lors de l'importation du Sync Service: {e}")
    logger.error(traceback.format_exc())
    service_registry.register_service("sync_service", status="failed")

# 3. ENRICHMENT SERVICE
try:
    from enrichment_service.api.routes import router as enrichment_router
    service_registry.register_service(
        "enrichment_service",
        router=enrichment_router,
        prefix="/api/v1/enrich",
        status="ok"
    )
    logger.info("✅ Enrichment Service importé avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur lors de l'importation du Enrichment Service: {e}")
    logger.error(traceback.format_exc())
    service_registry.register_service("enrichment_service", status="failed")

# 4. SEARCH SERVICE
try:
    from search_service.api.routes import router as search_router
    service_registry.register_service(
        "search_service",
        router=search_router,
        prefix="/api/v1/search",
        status="ok"
    )
    logger.info("✅ Search Service importé avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur lors de l'importation du Search Service: {e}")
    logger.error(traceback.format_exc())
    service_registry.register_service("search_service", status="failed")

# 5. CONVERSATION SERVICE
try:
    from conversation_service.api.routes import router as conversation_router
    from conversation_service.api.websocket import websocket_router
    
    service_registry.register_service(
        "conversation_service",
        router=conversation_router,
        prefix="/api/v1/conversation",
        status="ok"
    )
    
    service_registry.register_service(
        "conversation_websocket",
        router=websocket_router,
        prefix="/ws",
        status="ok"
    )
    
    logger.info("✅ Conversation Service importé avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur lors de l'importation du Conversation Service: {e}")
    logger.error(traceback.format_exc())
    service_registry.register_service("conversation_service", status="failed")

# ======== INCLUSION DES ROUTERS ========

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

# ======== MIDDLEWARE DE MÉTRIQUES ========

@app.middleware("http")
async def add_metrics_and_logging(request, call_next):
    """Middleware pour les métriques et le logging des requêtes."""
    start_time = time.time()
    
    # Logger les requêtes importantes
    if request.url.path not in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
        logger.info(f"🔄 {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Logger les requêtes lentes
    if process_time > 2.0:
        logger.warning(f"🐌 Requête lente: {request.method} {request.url.path} - {process_time:.2f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Service-Version"] = "1.0.0"
    return response

# ======== ENDPOINTS DE BASE ========

@app.get("/", tags=["health"])
async def root():
    """Point d'entrée racine pour vérifier que l'application est en ligne."""
    service_statuses = service_registry.get_service_status()
    active_services = [name for name, status in service_statuses.items() if status == "ok"]
    failed_services = [name for name, status in service_statuses.items() if status == "failed"]
    
    return {
        "status": "running",
        "application": "Harena Finance API (Complete)",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "services": {
            "active": active_services,
            "failed": failed_services,
            "total": len(service_statuses)
        },
        "features": {
            "user_management": "user_service" in active_services,
            "bank_sync": "sync_service" in active_services,
            "data_enrichment": "enrichment_service" in active_services,
            "smart_search": "search_service" in active_services,
            "ai_assistant": "conversation_service" in active_services
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Vérification détaillée de l'état de santé de tous les services."""
    
    # Vérifier la connexion à la base de données
    db_status = "unknown"
    try:
        from db_service.session import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:100]}"
    
    # Vérification des services externes
    external_services = {
        "bridge_api": "configured" if os.environ.get("BRIDGE_CLIENT_ID") else "not_configured",
        "openai_api": "configured" if os.environ.get("OPENAI_API_KEY") else "not_configured",
        "deepseek_api": "configured" if os.environ.get("DEEPSEEK_API_KEY") else "not_configured",
        "qdrant": "configured" if os.environ.get("QDRANT_URL") else "not_configured",
        "elasticsearch": "configured" if (os.environ.get("SEARCHBOX_URL") or os.environ.get("BONSAI_URL")) else "not_configured",
        "cohere": "configured" if os.environ.get("COHERE_KEY") else "not_configured"
    }
    
    # État général de l'application
    service_statuses = service_registry.get_service_status()
    failed_services = [name for name, status in service_statuses.items() if status == "failed"]
    
    if db_status.startswith("error"):
        overall_status = "critical"
    elif failed_services:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    # Calculer l'uptime
    uptime = time.time() - startup_time if startup_time else 0
    
    return {
        "status": overall_status,
        "database": db_status,
        "services": service_statuses,
        "external_services": external_services,
        "system": {
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "uptime_seconds": uptime,
            "uptime_human": str(timedelta(seconds=int(uptime))),
            "memory_usage": "unknown",  # Heroku ne permet pas facilement d'accéder à ces infos
            "python_version": sys.version.split()[0]
        },
        "capabilities": {
            "can_sync_banks": external_services["bridge_api"] == "configured",
            "can_search_semantic": external_services["openai_api"] == "configured" and external_services["qdrant"] == "configured",
            "can_search_lexical": external_services["elasticsearch"] == "configured",
            "can_chat_ai": external_services["deepseek_api"] == "configured",
            "can_rerank_results": external_services["cohere"] == "configured"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status", tags=["health"])
async def service_status():
    """État détaillé de chaque service."""
    service_statuses = service_registry.get_service_status()
    
    detailed_status = {}
    for service_name, status in service_statuses.items():
        service_info = service_registry.services.get(service_name, {})
        detailed_status[service_name] = {
            "status": status,
            "has_router": service_info.get("router") is not None,
            "prefix": service_info.get("prefix"),
            "description": _get_service_description(service_name)
        }
    
    return {
        "services": detailed_status,
        "summary": {
            "total": len(service_statuses),
            "active": len([s for s in service_statuses.values() if s == "ok"]),
            "failed": len([s for s in service_statuses.values() if s == "failed"]),
            "pending": len([s for s in service_statuses.values() if s == "pending"])
        }
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

@app.get("/features", tags=["info"])
async def list_features():
    """Liste toutes les fonctionnalités disponibles selon les services actifs."""
    service_statuses = service_registry.get_service_status()
    active_services = [name for name, status in service_statuses.items() if status == "ok"]
    
    features = {
        "authentication": {
            "available": "user_service" in active_services,
            "endpoints": ["/api/v1/users/login", "/api/v1/users/register"] if "user_service" in active_services else []
        },
        "bank_synchronization": {
            "available": "sync_service" in active_services,
            "endpoints": ["/api/v1/sync/", "/webhooks/bridge"] if "sync_service" in active_services else []
        },
        "transaction_management": {
            "available": "transactions_service" in active_services,
            "endpoints": ["/api/v1/transactions/"] if "transactions_service" in active_services else []
        },
        "data_enrichment": {
            "available": "enrichment_service" in active_services,
            "endpoints": ["/api/v1/enrich/"] if "enrichment_service" in active_services else []
        },
        "smart_search": {
            "available": "search_service" in active_services,
            "endpoints": ["/api/v1/search/"] if "search_service" in active_services else []
        },
        "ai_conversation": {
            "available": "conversation_service" in active_services,
            "endpoints": ["/api/v1/conversation/chat", "/ws/chat/"] if "conversation_service" in active_services else []
        }
    }
    
    return {
        "features": features,
        "total_features": len(features),
        "available_features": len([f for f in features.values() if f["available"]])
    }

@app.get("/debug", tags=["debug"])
async def debug_info():
    """Informations de debug détaillées (limitées en production)."""
    is_production = os.environ.get("ENVIRONMENT", "production").lower() == "production"
    
    basic_info = {
        "status": "debug enabled" if not is_production else "debug limited in production",
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "services": service_registry.get_service_status(),
        "uptime_seconds": time.time() - startup_time if startup_time else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    if is_production:
        return basic_info
    else:
        # Version détaillée pour dev/staging
        try:
            basic_info.update({
                "python_version": sys.version,
                "python_path": sys.path[:3],  # Limiter pour éviter la verbosité
                "environment_vars": {
                    key: "***" if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower() 
                    else os.environ.get(key, "")
                    for key in [
                        "DATABASE_URL", "BRIDGE_CLIENT_ID", "OPENAI_API_KEY", 
                        "DEEPSEEK_API_KEY", "QDRANT_URL", "COHERE_KEY"
                    ]
                },
                "routes_count": len(app.routes),
                "middleware_count": len(app.user_middleware)
            })
        except Exception as e:
            basic_info["debug_error"] = str(e)
        
        return basic_info

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
            "available_endpoints": [
                "/health", "/status", "/features", "/docs", "/redoc"
            ],
            "timestamp": datetime.now().isoformat()
        }
    )

# ======== ENDPOINTS DE MONITORING ========

@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Métriques de base pour le monitoring (format simple)."""
    service_statuses = service_registry.get_service_status()
    
    return {
        "harena_services_total": len(service_statuses),
        "harena_services_active": len([s for s in service_statuses.values() if s == "ok"]),
        "harena_services_failed": len([s for s in service_statuses.values() if s == "failed"]),
        "harena_uptime_seconds": time.time() - startup_time if startup_time else 0,
        "timestamp": time.time()
    }

# Point d'entrée pour le serveur gunicorn configuré dans Procfile
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Démarrage autonome de l'application Harena complète sur port {port}")
    logger.info(f"📊 Services configurés: {len(service_registry.get_service_status())}")
    
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=port)