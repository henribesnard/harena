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
from fastapi.responses import JSONResponse, RedirectResponse
from datetime import datetime
from typing import Dict, Any

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

# ======== GESTION DU CYCLE DE VIE DE L'APPLICATION ========

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

# Registre des services
service_registry = {}

# ======== INITIALISATION ET MONTAGE DES SERVICES ========

# Service utilisateur
try:
    from user_service.main import create_app as create_user_app
    user_app = create_user_app()
    from user_service.core.config import settings as user_settings
    service_registry["user_service"] = {
        "app": user_app,
        "status": "ok",
        "api_prefix": user_settings.API_V1_STR
    }
    
    # Monter l'app utilisateur uniquement sur /user
    # Ne pas monter sur API_V1_STR pour éviter le double préfixe
    app.mount("/user", user_app)
    logger.info(f"User Service monté sur /user")
    
except ImportError as e:
    logger.error(f"Erreur lors de l'initialisation du User Service: {e}")
    service_registry["user_service"] = {"status": "failed", "error": str(e)}

# Service de synchronisation
try:
    from sync_service.main import create_app as create_sync_app
    sync_app = create_sync_app()
    service_registry["sync_service"] = {
        "app": sync_app,
        "status": "ok",
        "api_prefix": "/api/v1/sync"
    }
    
    # Monter l'app sync uniquement sur /sync
    # Ne pas monter sur /api/v1/sync pour éviter le double préfixe
    app.mount("/sync", sync_app)
    logger.info("Sync Service monté sur /sync")
    
except ImportError as e:
    logger.error(f"Erreur lors de l'initialisation du Sync Service: {e}")
    service_registry["sync_service"] = {"status": "failed", "error": str(e)}

# ======== REDIRECTIONS API ========

# Créer des redirections pour les routes API populaires
@app.get("/api/v1/{service}/{path:path}")
@app.post("/api/v1/{service}/{path:path}")
@app.put("/api/v1/{service}/{path:path}")
@app.delete("/api/v1/{service}/{path:path}")
async def api_redirect(request: Request, service: str, path: str):
    """Redirection intelligente vers les services montés pour les requêtes API."""
    
    # Mapper le service à l'application appropriée
    target_app = None
    if service == "users":
        # Pour user_service
        if "user_service" in service_registry and service_registry["user_service"]["status"] == "ok":
            target_app = user_app
            # Rediriger vers /users/{path} dans l'app user
            new_path = f"/users/{path}"
    elif service == "sync":
        # Pour sync_service
        if "sync_service" in service_registry and service_registry["sync_service"]["status"] == "ok":
            target_app = sync_app
            # Rediriger vers /{path} dans l'app sync car son préfixe inclut déjà sync
            new_path = f"/{path}"
    
    if target_app:
        # Modifier le chemin de la requête pour l'application cible
        request.scope["path"] = new_path
        logger.debug(f"Redirection API: /api/v1/{service}/{path} -> {new_path}")
        return await target_app.handle_request(request)
    else:
        logger.warning(f"Tentative d'accès à un service non disponible: {service}")
        raise HTTPException(status_code=404, detail=f"Service {service} not found or not available")

# ======== REDIRECTIONS DOCUMENTATION ========

@app.get("/api-docs", include_in_schema=False)
async def api_docs_redirect():
    """Redirection vers la documentation principale des utilisateurs."""
    return RedirectResponse(url="/user/docs")

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
        "services": {name: info["status"] for name, info in service_registry.items()},
        "documentation": {
            "main": "/docs",
            "user_service": "/user/docs",
            "sync_service": "/sync/docs"
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
    
    return {
        "status": "ok" if all(info["status"] == "ok" for info in service_registry.values()) else "degraded",
        "services": {name: info["status"] for name, info in service_registry.items()},
        "database": db_status,
        "vector_storage": vector_status,
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
            "services": {name: info["status"] for name, info in service_registry.items()}
        }
    else:
        # Version plus détaillée pour dev/staging
        service_details = {}
        for name, info in service_registry.items():
            if info["status"] == "ok":
                service_details[name] = {
                    "status": info["status"],
                    "api_prefix": info.get("api_prefix", "unknown")
                }
            else:
                service_details[name] = info
                
        return {
            "status": "debug enabled",
            "environment": os.environ.get("ENVIRONMENT", "unknown"),
            "python_version": sys.version,
            "services": service_details,
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