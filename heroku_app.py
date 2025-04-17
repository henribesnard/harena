# heroku_app.py
"""
Application Harena pour déploiement Heroku.

Module optimisé pour le déploiement sur Heroku, avec gestion adaptée des variables d'environnement
et des dépendances pour assurer un démarrage fiable.
"""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any, List

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

# ======== GESTIONNAIRE DU CYCLE DE VIE ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application.
    Initialise les ressources au démarrage et les libère à l'arrêt.
    """
    # Initialization
    logger.info("Application Harena en démarrage sur Heroku...")
    
    # Vérification des variables d'environnement critiques
    required_env_vars = ["DATABASE_URL", "BRIDGE_CLIENT_ID", "BRIDGE_CLIENT_SECRET"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Variables d'environnement critiques manquantes: {', '.join(missing_vars)}")
        # On continue quand même, mais certaines fonctionnalités ne marcheront pas
    
    # Test de la connexion base de données
    try:
        from user_service.db.session import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Connexion à la base de données établie avec succès")
    except Exception as db_error:
        logger.error(f"Erreur de connexion à la base de données: {db_error}")
    
    yield  # L'application s'exécute ici
    
    # Cleanup
    logger.info("Application Harena en arrêt sur Heroku...")

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
        
    def get_service_status(self) -> Dict[str, str]:
        """Retourne le statut de tous les services."""
        return {name: info["status"] for name, info in self.services.items()}
    
    def get_available_routers(self) -> List[Dict[str, Any]]:
        """Retourne les routeurs disponibles avec leurs préfixes."""
        return [
            {"name": name, "router": info["router"], "prefix": info["prefix"]}
            for name, info in self.services.items()
            if info["status"] == "ok" and info["router"] is not None
        ]

# Création du registre de services
service_registry = ServiceRegistry()

# ======== IMPORTATION DES SERVICES ========

# Préfixes API
API_V1_PREFIX = "/api/v1"

# Service utilisateur
try:
    from user_service.api.endpoints.users import router as users_router
    service_registry.register(
        "user_service", 
        router=users_router,
        prefix=f"{API_V1_PREFIX}/users",
        status="ok"
    )
    logger.info("User Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du router User Service: {e}")
    service_registry.register("user_service", status="failed")

# Service de synchronisation
try:
    from sync_service.api.endpoints.sync import router as sync_router
    service_registry.register(
        "sync_service", 
        router=sync_router,
        prefix=f"{API_V1_PREFIX}/sync",
        status="ok"
    )
    
    # Importer également le router des webhooks si disponible
    try:
        from sync_service.api.endpoints.webhooks import router as webhooks_router
        service_registry.register(
            "webhooks_service", 
            router=webhooks_router,
            prefix="/webhooks",
            status="ok"
        )
    except ImportError as webhook_e:
        logger.warning(f"Router Webhooks non disponible: {webhook_e}")
    
    logger.info("Sync Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du router Sync Service: {e}")
    service_registry.register("sync_service", status="failed")

# ======== CRÉATION DE L'APPLICATION ========

# Création de l'application FastAPI pour Heroku
app = FastAPI(
    title="Harena Finance API (Heroku)",
    description="API pour les services financiers Harena - Déploiement Heroku",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configuration CORS sécurisée pour production
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "https://app.harena.finance").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ======== INCLUSION DES ROUTERS ========

# Inclure tous les routers disponibles avec leurs préfixes
for service_info in service_registry.get_available_routers():
    app.include_router(
        service_info["router"],
        prefix=service_info["prefix"],
        tags=[service_info["name"]]
    )
    logger.info(f"Router {service_info['name']} inclus avec préfixe {service_info['prefix']}")

# ======== ENDPOINTS DE BASE ========

@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    """
    return {
        "status": "ok",
        "application": "Harena Finance API (Heroku)",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "services": service_registry.get_service_status(),
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
    
    # Vérifier l'état du stockage vectoriel si disponible
    vector_status = "unknown"
    try:
        from sync_service.services.vector_storage import VectorStorageService
        vector_service = VectorStorageService()
        if vector_service.client:
            vector_status = "connected"
        else:
            vector_status = "client_not_initialized"
    except ImportError:
        vector_status = "module_not_available"
    except Exception as e:
        vector_status = f"error: {str(e)}"
    
    # Vérification du service Bridge
    bridge_status = "configured" if os.environ.get("BRIDGE_CLIENT_ID") else "not_configured"
    
    return {
        "status": "ok" if all(status == "ok" for status in service_registry.get_service_status().values()) else "degraded",
        "services": service_registry.get_service_status(),
        "database": db_status,
        "vector_storage": vector_status,
        "bridge_api": bridge_status,
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "timestamp": str(datetime.now())
    }

@app.get("/debug", tags=["debug"])
async def debug_info():
    """
    Endpoint pour le débogage - fournit des informations détaillées sur l'environnement.
    """
    # Ne pas exposer d'informations sensibles en production
    is_production = os.environ.get("ENVIRONMENT", "production").lower() == "production"
    
    if is_production:
        return {
            "status": "debug limited in production",
            "timestamp": str(datetime.now()),
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "services": service_registry.get_service_status()
        }
    else:
        # Version plus détaillée pour dev/staging
        return {
            "status": "debug enabled",
            "environment": os.environ.get("ENVIRONMENT", "unknown"),
            "python_version": sys.version,
            "services": service_registry.get_service_status(),
            "database_config": {
                "url_type": type(os.environ.get("DATABASE_URL", "")).__name__,
                "url_length": len(os.environ.get("DATABASE_URL", "")),
                "has_bridge_config": bool(os.environ.get("BRIDGE_CLIENT_ID", ""))
            },
            "timestamp": str(datetime.now())
        }

# ======== GESTIONNAIRE D'EXCEPTIONS ========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions pour toute l'application.
    """
    logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
    
    # En production, ne pas exposer les détails de l'erreur
    is_production = os.environ.get("ENVIRONMENT", "production").lower() == "production"
    error_detail = "Une erreur interne est survenue. Contactez l'administrateur." if is_production else str(exc)
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": error_detail
        }
    )

# Point d'entrée pour le serveur gunicorn configuré dans Procfile
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Démarrage autonome de l'application Harena sur port {port}")
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=port)