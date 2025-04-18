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
    
    # Initialisation des services de stockage
    try:
        # Initialisation des services de recherche si disponibles
        from search_service.storage.elasticsearch import init_elasticsearch
        from search_service.storage.qdrant import init_qdrant
        
        # Initialisation asynchrone des clients de stockage
        es_client_future = init_elasticsearch()
        qdrant_client_future = init_qdrant()
        
        # Attendre l'initialisation des services de recherche
        import asyncio
        es_client, qdrant_client = await asyncio.gather(
            es_client_future, qdrant_client_future, 
            return_exceptions=True
        )
        
        if isinstance(es_client, Exception):
            logger.error(f"Erreur d'initialisation d'Elasticsearch: {es_client}")
        elif es_client:
            logger.info("Service Elasticsearch initialisé avec succès")
        
        if isinstance(qdrant_client, Exception):
            logger.error(f"Erreur d'initialisation de Qdrant: {qdrant_client}")
        elif qdrant_client:
            logger.info("Service Qdrant initialisé avec succès")
            
    except ImportError:
        logger.warning("Services de recherche non disponibles. Certaines fonctionnalités seront limitées.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des services de stockage: {e}")
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("Application Harena en arrêt sur Heroku...")
    
    # Fermeture des connexions
    try:
        from search_service.storage.elasticsearch import close_es_client
        await close_es_client()
        logger.info("Connexions Elasticsearch fermées")
    except (ImportError, Exception):
        pass

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
    from user_service.api.endpoints import users
    service_registry.register(
        "user_service", 
        router=users.router,
        prefix=f"{API_V1_PREFIX}/users",
        status="ok"
    )
    logger.info("User Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du router User Service: {e}")
    service_registry.register("user_service", status="failed")

# Service de synchronisation
try:
    from sync_service.api.endpoints import sync
    service_registry.register(
        "sync_service", 
        router=sync.router,
        prefix=f"{API_V1_PREFIX}/sync",
        status="ok"
    )
    
    # Importer également le router des webhooks si disponible
    try:
        from sync_service.api.endpoints import webhooks
        service_registry.register(
            "webhooks_service", 
            router=webhooks.router,
            prefix="/webhooks",
            status="ok"
        )
    except ImportError as webhook_e:
        logger.warning(f"Router Webhooks non disponible: {webhook_e}")
    
    logger.info("Sync Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du router Sync Service: {e}")
    service_registry.register("sync_service", status="failed")

# Service de recherche
try:
    from search_service.api.endpoints import search, health as search_health
    service_registry.register(
        "search_service",
        router=search.router,
        prefix=f"{API_V1_PREFIX}/search",
        status="ok"
    )
    
    # Ajouter également le endpoint de santé du service de recherche
    service_registry.register(
        "search_health", 
        router=search_health.router,
        prefix=f"{API_V1_PREFIX}/search/health",
        status="ok"
    )
    
    logger.info("Search Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du Search Service: {e}")
    service_registry.register("search_service", status="failed")

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
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "https://app.harena.finance").split(",")

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
        from search_service.storage.qdrant import get_qdrant_client
        qdrant_client = await get_qdrant_client()
        if qdrant_client:
            vector_status = "connected"
        else:
            vector_status = "client_not_initialized"
    except ImportError:
        vector_status = "module_not_available"
    except Exception as e:
        vector_status = f"error: {str(e)}"
    
    # Vérifier l'état d'Elasticsearch si disponible
    es_status = "unknown"
    try:
        from search_service.storage.elasticsearch import get_es_client
        es_client = await get_es_client()
        if es_client:
            es_status = "connected"
        else:
            es_status = "client_not_initialized"
    except ImportError:
        es_status = "module_not_available"
    except Exception as e:
        es_status = f"error: {str(e)}"
    
    # Vérification des services externes
    bridge_status = "configured" if os.environ.get("BRIDGE_CLIENT_ID") else "not_configured"
    deepseek_status = "configured" if os.environ.get("DEEPSEEK_API_KEY") else "not_configured"
    
    # État général de l'application
    overall_status = "ok"
    service_statuses = service_registry.get_service_status()
    
    if "failed" in service_statuses.values() or db_status.startswith("error"):
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "services": service_statuses,
        "database": db_status,
        "vector_storage": vector_status,
        "elasticsearch": es_status,
        "bridge_api": bridge_status,
        "deepseek_api": deepseek_status,
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
                "has_bridge_config": bool(os.environ.get("BRIDGE_CLIENT_ID", "")),
                "has_deepseek_config": bool(os.environ.get("DEEPSEEK_API_KEY", "")),
                "has_qdrant_config": bool(os.environ.get("QDRANT_URL", ""))
            },
            "memory_usage": {
                "process": get_process_memory_usage()
            },
            "timestamp": str(datetime.now())
        }

def get_process_memory_usage():
    """Obtient l'utilisation mémoire du processus actuel."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024)
        }
    except ImportError:
        return {"error": "psutil not installed"}
    except Exception as e:
        return {"error": str(e)}

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