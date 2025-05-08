"""
Application Harena pour développement local.

Module optimisé pour le développement et le débogage en environnement local,
avec des fonctionnalités supplémentaires pour faciliter le développement.
"""

import logging
import os
import sys
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configuration du logging pour développement
logging.basicConfig(
    level=logging.DEBUG,  # Niveau DEBUG pour plus de détails en local
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("harena_local")

# Définir l'environnement global
os.environ["ENVIRONMENT"] = "development"

# S'assurer que tous les modules sont accessibles
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    logger.info(f"Ajout du répertoire courant au sys.path: {current_dir}")

logger.info(f"Python path: {sys.path}")

# ======== DÉFINITION DES SERVICES ========

class ServiceRegistry:
    """Classe pour gérer les services disponibles et leurs routeurs."""
    
    def __init__(self):
        self.services = {}
        
    def register(self, name: str, router=None, prefix: str = None, status: str = "pending"):
        """Enregistre un service dans le registre."""
        self.services[name] = {
            "router": router,
            "prefix": prefix,
            "status": status
        }
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

# Création du registre de services
service_registry = ServiceRegistry()

# ======== FONCTION DU CYCLE DE VIE ========

async def startup():
    """Fonction d'initialisation de l'application"""
    logger.info("Application Harena en démarrage en mode développement...")
    
    # Vérification des variables d'environnement critiques
    required_env_vars = ["DATABASE_URL", "BRIDGE_CLIENT_ID", "BRIDGE_CLIENT_SECRET"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Variables d'environnement manquantes: {', '.join(missing_vars)}")
        logger.warning("Pour faciliter le développement, des valeurs par défaut seront utilisées si possible.")
    
    # Test de la connexion base de données
    try:
        from user_service.db.session import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Connexion à la base de données établie avec succès")
    except Exception as db_error:
        logger.error(f"Erreur de connexion à la base de données: {db_error}")
        logger.info("Vérifiez que votre fichier .env contient DATABASE_URL=postgresql://user:password@localhost:5432/harena")


async def shutdown():
    """Fonction de nettoyage lors de l'arrêt de l'application"""
    logger.info("Application Harena en arrêt...")
    # Aucune fermeture spécifique nécessaire

# ======== GESTIONNAIRE DE CYCLE DE VIE ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application.
    Utilise le nouveau modèle recommandé par FastAPI.
    """
    # Code d'initialisation (avant le yield)
    await startup()
    
    yield  # L'application s'exécute ici
    
    # Code de nettoyage (après le yield)
    await shutdown()

# ======== CRÉATION DE L'APPLICATION ========

# Création de l'application FastAPI pour développement local
app = FastAPI(
    title="Harena Finance API (Local Development)",
    description="API pour les services financiers Harena - Développement local",
    version="1.0.0-dev",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    debug=True,  # Active le mode debug pour plus d'informations
    lifespan=lifespan  # Utilisation du nouveau gestionnaire de cycle de vie
)

# Configuration CORS permissive pour développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En développement, on accepte toutes les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== IMPORTATION DES SERVICES ET ENDPOINTS ========

# User Service - Essai d'importation avec gestion d'erreur détaillée
try:
    from user_service.api.endpoints import users as users_router
    from config_service.config import settings as user_settings
    service_registry.register(
        "user_service", 
        router=users_router.router, 
        prefix=user_settings.API_V1_STR + "/users",
        status="ok"
    )
    logger.info(f"User Service importé avec succès, préfixe: {user_settings.API_V1_STR}/users")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du User Service: {e}")
    logger.error(traceback.format_exc())
    service_registry.register("user_service", status="failed")
    logger.info("Pour activer le User Service, assurez-vous que les modules user_service et config_service sont disponibles")

# Sync Service - Utiliser l'API router centralisé pour enregistrer tous les endpoints
try:
    from sync_service.api.router import register_routers, setup_middleware
    from sync_service.main import create_app as create_sync_app
    
    # Créer l'application Sync Service
    sync_app = create_sync_app()
    service_registry.register("sync_service_app", router=None, status="ok")
    
    # Enregistrer les endpoints d'API Sync Service
    try:
        from sync_service.api.endpoints.sync import router as sync_router
        service_registry.register(
            "sync_service", 
            router=sync_router,
            prefix="/api/v1/sync",
            status="ok"
        )
        logger.info("Sync Service importé avec succès, préfixe: /api/v1/sync")
    except ImportError as e:
        logger.error(f"Erreur lors de l'importation du Sync Router: {e}")
        logger.error(traceback.format_exc())
        service_registry.register("sync_service", status="failed")
        
    # Sync Service - Transactions
    try:
        from sync_service.api.endpoints.transactions import router as transactions_router
        service_registry.register(
            "transactions_service", 
            router=transactions_router,
            prefix="/api/v1/transactions",
            status="ok"
        )
        logger.info("Transactions Service importé avec succès, préfixe: /api/v1/transactions")
    except ImportError as e:
        logger.error(f"Erreur lors de l'importation du Transactions Router: {e}")
        service_registry.register("transactions_service", status="failed")
    
    # Sync Service - Webhooks
    try:
        from sync_service.api.endpoints.webhooks import router as webhooks_router
        service_registry.register(
            "webhooks_service", 
            router=webhooks_router,
            prefix="/webhooks",  # Important! Le préfixe doit correspondre à celui utilisé dans le router
            status="ok"
        )
        logger.info("Webhooks Service importé avec succès, préfixe: /webhooks")
    except ImportError as e:
        logger.error(f"Erreur lors de l'importation du Webhooks Router: {e}")
        service_registry.register("webhooks_service", status="failed")
    
    # Sync Service - Comptes
    try:
        from sync_service.api.endpoints.accounts import router as accounts_router
        service_registry.register(
            "accounts_service", 
            router=accounts_router,
            prefix="/api/v1/accounts",
            status="ok"
        )
        logger.info("Accounts Service importé avec succès, préfixe: /api/v1/accounts")
    except ImportError as e:
        logger.error(f"Erreur lors de l'importation du Accounts Router: {e}")
        service_registry.register("accounts_service", status="failed")
    
    # Sync Service - Items
    try:
        from sync_service.api.endpoints.items import router as items_router
        service_registry.register(
            "items_service", 
            router=items_router,
            prefix="/api/v1/items",
            status="ok"
        )
        logger.info("Items Service importé avec succès, préfixe: /api/v1/items")
    except ImportError as e:
        logger.error(f"Erreur lors de l'importation du Items Router: {e}")
        service_registry.register("items_service", status="failed")
        
    # Sync Service - Stocks
    try:
        from sync_service.api.endpoints.stocks import router as stocks_router
        service_registry.register(
            "stocks_service", 
            router=stocks_router,
            prefix="/api/v1/stocks",
            status="ok"
        )
        logger.info("Stocks Service importé avec succès, préfixe: /api/v1/stocks")
    except ImportError as e:
        logger.error(f"Erreur lors de l'importation du Stocks Router: {e}")
        service_registry.register("stocks_service", status="failed")
    
    # Sync Service - Categories
    try:
        from sync_service.api.endpoints.categories import router as categories_router
        service_registry.register(
            "categories_service", 
            router=categories_router,
            prefix="/api/v1/categories",
            status="ok"
        )
        logger.info("Categories Service importé avec succès, préfixe: /api/v1/categories")
    except ImportError as e:
        logger.error(f"Erreur lors de l'importation du Categories Router: {e}")
        service_registry.register("categories_service", status="failed")
    
    # Sync Service - Insights
    try:
        from sync_service.api.endpoints.insights import router as insights_router
        service_registry.register(
            "insights_service", 
            router=insights_router,
            prefix="/api/v1/insights",
            status="ok"
        )
        logger.info("Insights Service importé avec succès, préfixe: /api/v1/insights")
    except ImportError as e:
        logger.error(f"Erreur lors de l'importation du Insights Router: {e}")
        service_registry.register("insights_service", status="failed")
        
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Sync Service: {e}")
    logger.error(traceback.format_exc())
    service_registry.register("sync_service_app", status="failed")
    logger.info("Pour activer le Sync Service, assurez-vous que le module sync_service est disponible")

# ======== INCLUSION DES ROUTERS ========

# Inclure tous les routers disponibles
for service_info in service_registry.get_available_routers():
    try:
        # Utiliser include_router au lieu de mount pour les routers FastAPI
        app.include_router(
            service_info["router"],
            prefix=service_info["prefix"],
            tags=[service_info["name"]]
        )
        logger.info(f"Router {service_info['name']} inclus avec préfixe {service_info['prefix']}")
    except Exception as e:
        logger.error(f"Erreur lors de l'inclusion du router {service_info['name']}: {e}")
        logger.error(traceback.format_exc())

# ======== ENDPOINTS DE BASE ========

@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    """
    active_services = [name for name, info in service_registry.services.items() if info["status"] == "ok"]
    
    return {
        "status": "ok",
        "application": "Harena Finance API (Local Development)",
        "version": "1.0.0-dev",
        "environment": "development",
        "services": active_services,
        "documentation": {
            "main": "/docs",
        }
    }

@app.get("/health", tags=["health"])
async def health_check():
    """
    Vérification de l'état de santé de tous les services.
    """
    # Vérifier la connexion à la base de données
    db_status = "unknown"
    try:
        from user_service.db.session import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Vérification des services externes
    bridge_status = "configured" if os.environ.get("BRIDGE_CLIENT_ID") else "not_configured"
    
    # État général de l'application
    overall_status = "ok"
    service_statuses = service_registry.get_service_status()
    
    if "failed" in service_statuses.values() or db_status.startswith("error"):
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "services": service_statuses,
        "database": db_status,
        "bridge_api": bridge_status,
        "environment": "development",
        "timestamp": str(datetime.now())
    }

@app.get("/debug", tags=["debug"])
async def debug_info():
    """
    Endpoint pour le débogage - fournit des informations détaillées sur l'environnement.
    """
    # En mode développement, afficher toutes les informations utiles
    env_vars = {
        k: ("***HIDDEN***" if k in ["DATABASE_URL", "BRIDGE_CLIENT_SECRET", "SECRET_KEY"] 
            else ("SET" if v else "NOT SET"))
        for k, v in os.environ.items() 
        if k.isupper()
    }
    
    return {
        "status": "debug enabled",
        "environment": "development",
        "python_version": sys.version,
        "services": service_registry.get_service_status(),
        "environment_variables": env_vars,
        "routes": [
            {"path": route.path, "name": route.name, "methods": route.methods if hasattr(route, "methods") else None} 
            for route in app.routes
        ],
        "timestamp": str(datetime.now())
    }

@app.get("/test-webhook", tags=["debug"])
async def test_webhook(background_tasks: BackgroundTasks):
    """
    Endpoint pour tester manuellement le traitement d'un webhook Bridge.
    
    Returns:
        Dict: Résultat du test
    """
    try:
        from sync_service.webhook_handler.processor import process_webhook
        
        # Créer un payload de test (similaire à celui envoyé par Bridge)
        test_payload = {
            "type": "TEST_EVENT",
            "content": {
                "message": "This is a test webhook from the local app",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Obtenir une session de base de données
        from user_service.db.session import get_db
        db = next(get_db())
        
        # Traiter le webhook
        webhook_event = await process_webhook(db, test_payload)
        
        return {
            "status": "success",
            "message": "Webhook test processed successfully",
            "webhook_id": webhook_event.id if webhook_event else None,
            "webhook_type": "TEST_EVENT"
        }
    except Exception as e:
        logger.error(f"Erreur lors du test du webhook: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error processing test webhook: {str(e)}",
            "traceback": traceback.format_exc()
        }

# ======== GESTIONNAIRE D'EXCEPTIONS ========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions pour toute l'application.
    Fournit des détails complets en mode développement.
    """
    logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
    
    # En développement, on fournit tous les détails de l'erreur
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc),
            "traceback": traceback.format_exc(),
            "request_url": str(request.url),
            "request_method": request.method
        }
    )

# Point d'entrée pour le serveur de développement
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    
    logger.info(f"Démarrage de l'application Harena en mode développement sur port {port}")
    uvicorn.run("local_app:app", host="0.0.0.0", port=port, reload=True)