"""
Application Harena pour déploiement Heroku.

Module optimisé pour le déploiement sur Heroku, avec gestion adaptée des variables d'environnement
et des dépendances pour assurer un démarrage fiable.
"""

import logging
import os
import sys
import traceback
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
    logger.info(f"Ajout du répertoire courant au sys.path: {current_dir}")

logger.info(f"Python path: {sys.path}")

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
        logger.info("Tentative d'initialisation des services de stockage")
        
        # Initialisation du moteur de recherche unifié
        try:
            from search_service.storage.unified_engine import get_unified_engine
            unified_engine = get_unified_engine()
            logger.info(f"Moteur de recherche unifié initialisé avec {unified_engine.primary_engine_type} comme moteur principal")
        except ImportError as engine_import_err:
            logger.error(f"Erreur lors de l'importation du moteur de recherche unifié: {engine_import_err}")
        
        # Initialisation de Qdrant (toujours utilisé pour la recherche vectorielle)
        try:
            from search_service.storage.qdrant import init_qdrant
            logger.info("Module search_service.storage.qdrant importé avec succès")
        except ImportError as qdrant_import_err:
            logger.error(f"Erreur lors de l'importation de Qdrant: {qdrant_import_err}")
            
        # Initialisation asynchrone des clients de stockage
        import asyncio
            
        # Initialiser Qdrant
        qdrant_client = None
        try:
            if 'init_qdrant' in locals():
                logger.info("Démarrage de l'initialisation de Qdrant")
                qdrant_client_future = init_qdrant()
                qdrant_client = await qdrant_client_future
                if qdrant_client:
                    logger.info("Service Qdrant initialisé avec succès")
                else:
                    logger.warning("Initialisation de Qdrant terminée mais client None")
        except Exception as qdrant_init_err:
            logger.error(f"Erreur lors de l'initialisation de Qdrant: {qdrant_init_err}")
            
    except ImportError as import_error:
        logger.warning(f"Services de recherche non disponibles: {import_error}")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des services de stockage: {e}")
        logger.error(traceback.format_exc())
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("Application Harena en arrêt sur Heroku...")
    
    # Aucune fermeture spécifique nécessaire pour les nouveaux moteurs de recherche
    # car ils n'ont pas de connexions persistantes à fermer

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

# ======== IMPORTATION DES SERVICES ========

# Préfixes API
API_V1_PREFIX = "/api/v1"

# Service utilisateur
try:
    logger.info("Tentative d'importation du module user_service.api.endpoints.users")
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
except Exception as e:
    logger.error(f"Erreur inattendue lors de l'importation du User Service: {e}")
    logger.error(traceback.format_exc())
    service_registry.register("user_service", status="failed")

# Service de synchronisation
try:
    logger.info("Tentative d'importation du module sync_service.api.endpoints.sync")
    from sync_service.api.endpoints import sync
    service_registry.register(
        "sync_service", 
        router=sync.router,
        prefix=f"{API_V1_PREFIX}/sync",
        status="ok"
    )
    
    # Importer également le router des webhooks si disponible
    try:
        logger.info("Tentative d'importation du module sync_service.api.endpoints.webhooks")
        from sync_service.api.endpoints import webhooks
        service_registry.register(
            "webhooks_service", 
            router=webhooks.router,
            prefix="/webhooks",
            status="ok"
        )
        logger.info("Webhooks Service importé avec succès")
    except ImportError as webhook_e:
        logger.warning(f"Router Webhooks non disponible: {webhook_e}")
    except Exception as webhook_e:
        logger.error(f"Erreur inattendue lors de l'importation du Webhooks Service: {webhook_e}")
        logger.error(traceback.format_exc())
    
    logger.info("Sync Service importé avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du router Sync Service: {e}")
    service_registry.register("sync_service", status="failed")
except Exception as e:
    logger.error(f"Erreur inattendue lors de l'importation du Sync Service: {e}")
    logger.error(traceback.format_exc())
    service_registry.register("sync_service", status="failed")

# Service de recherche
try:
    # Vérifier si les modules existent
    search_module_path = Path("search_service/api/endpoints/search.py")
    health_module_path = Path("search_service/api/endpoints/health.py")
    
    if search_module_path.exists():
        logger.info(f"Fichier search.py trouvé: {search_module_path.absolute()}")
    else:
        logger.error(f"Fichier search.py manquant: {search_module_path.absolute()}")
    
    if health_module_path.exists():
        logger.info(f"Fichier health.py trouvé: {health_module_path.absolute()}")
    else:
        logger.error(f"Fichier health.py manquant: {health_module_path.absolute()}")
    
    # Importer les modules
    logger.info("Tentative d'importation du module search_service.api.endpoints.search")
    try:
        from search_service.api.endpoints import search
        logger.info("Module search importé avec succès")
        
        if not hasattr(search, 'router'):
            logger.error("Le module search n'a pas d'attribut 'router'")
        else:
            logger.info("Router search trouvé")
    except ImportError as search_e:
        logger.error(f"Erreur lors de l'importation du module search: {search_e}")
        raise
    
    logger.info("Tentative d'importation du module search_service.api.endpoints.health")
    try:
        from search_service.api.endpoints import health as search_health
        logger.info("Module health importé avec succès")
        
        if not hasattr(search_health, 'router'):
            logger.error("Le module health n'a pas d'attribut 'router'")
        else:
            logger.info("Router health trouvé")
    except ImportError as health_e:
        logger.error(f"Erreur lors de l'importation du module health: {health_e}")
        raise
    
    # Enregistrer les services
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
    logger.error(traceback.format_exc())
    service_registry.register("search_service", status="failed")
except Exception as e:
    logger.error(f"Erreur inattendue lors de l'importation du Search Service: {e}")
    logger.error(traceback.format_exc())
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
available_routers = service_registry.get_available_routers()
logger.info(f"Inclusion de {len(available_routers)} routers dans l'application")

for service_info in available_routers:
    try:
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
    
    # Vérifier l'état des moteurs de recherche
    search_engine_status = "unknown"
    try:
        from search_service.storage.unified_engine import get_unified_engine
        engine = get_unified_engine()
        stats = engine.get_stats()
        search_engine_status = {
            "status": "ok",
            "primary_engine": stats["primary_engine"],
            "engines_available": list(stats["engines"].keys())
        }
    except ImportError:
        search_engine_status = "module_not_available"
    except Exception as e:
        search_engine_status = f"error: {str(e)}"
    
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
        "search_engine": search_engine_status,
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