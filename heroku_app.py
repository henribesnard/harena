"""
Application Harena complète pour le déploiement Heroku.
Intégration optimisée des services bancaires avec gestion améliorée des dépendances.
"""

import logging
import os
import sys
import importlib
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ======== CONFIGURATION INITIALE ========

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("harena")
logger.info("Initialisation de l'application Harena avec niveau de log DEBUG")

# Définir la variable d'environnement pour propager aux autres modules
os.environ["LOG_LEVEL"] = "DEBUG"

# Correction de l'URL de base de données pour Heroku
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    logger.info("DATABASE_URL corrigé pour SQLAlchemy 1.4+")

# S'assurer que tous les modules sont accessibles
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    logger.info(f"Ajout du répertoire courant {current_dir} au PYTHONPATH")

# Cache des modules importés
imported_modules = {}

# Suivi des services intégrés
service_status = {
    "user_service": False,
    "sync_service": False
}

# ======== FONCTIONS UTILITAIRES ========

def safe_import(module_name: str, required: bool = False) -> Any:
    """Importe un module avec gestion d'erreur détaillée."""
    if module_name in imported_modules:
        return imported_modules[module_name]
        
    try:
        logger.info(f"Importation du module: {module_name}")
        module = importlib.import_module(module_name)
        imported_modules[module_name] = module
        logger.info(f"Module {module_name} importé avec succès")
        return module
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erreur lors de l'importation de {module_name}: {str(e)}")
        logger.debug(f"Traceback complet: {error_trace}")
        
        if required:
            logger.critical(f"Module requis {module_name} n'a pas pu être importé. Application peut être instable.")
        
        imported_modules[module_name] = None
        return None

def reload_module(module_name: str) -> Any:
    """Recharge un module en le supprimant du cache si nécessaire."""
    if module_name in sys.modules:
        logger.debug(f"Suppression du module {module_name} du cache sys.modules")
        del sys.modules[module_name]
    if module_name in imported_modules:
        del imported_modules[module_name]
    return safe_import(module_name)

def init_database() -> bool:
    """Initialise la connexion à la base de données et vérifie si elle fonctionne."""
    try:
        # Importer les modules nécessaires
        db_session = safe_import("user_service.db.session", required=True)
        if not db_session:
            logger.error("Échec de l'initialisation de la base de données: module de session introuvable")
            return False
            
        # Extraire les références importantes
        engine = getattr(db_session, "engine", None)
        get_db = getattr(db_session, "get_db", None)
        
        if not engine or not get_db:
            logger.error("Échec de l'initialisation de la base de données: engine ou get_db introuvable")
            return False
            
        # Vérifier la connexion
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        logger.info("Connexion à la base de données établie avec succès")
        
        # Créer les tables si demandé
        if os.environ.get("CREATE_TABLES", "false").lower() == "true":
            base_module = safe_import("user_service.models.base", required=True)
            if base_module and hasattr(base_module, "Base"):
                Base = base_module.Base
                Base.metadata.create_all(bind=engine)
                logger.info("Tables créées avec succès")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la base de données: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_core_modules() -> bool:
    """Charge les modules de base nécessaires au fonctionnement de l'application."""
    try:
        # Liste des modules centraux à charger
        core_modules = [
            "user_service.core.config",
            "user_service.models.base",
            "user_service.models.user",
            "sync_service.models.sync"
        ]
        
        success = True
        for module_name in core_modules:
            module = safe_import(module_name, required=True)
            if not module:
                logger.error(f"Échec du chargement du module central: {module_name}")
                success = False
                
        return success
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modules centraux: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def create_required_objects() -> bool:
    """Crée et initialise les objets partagés nécessaires."""
    try:
        # Configurer les objets partagés si nécessaire
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la création des objets partagés: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# ======== INITIALISATION DES SERVICES ========

def init_user_service(app: FastAPI) -> bool:
    """Initialise et monte le service utilisateur."""
    try:
        # Charger les dépendances du service utilisateur
        deps_module = safe_import("user_service.api.deps")
        if not deps_module:
            logger.error("Échec de l'initialisation du service utilisateur: module deps introuvable")
            return False
            
        # Charger les endpoints utilisateur
        users_module = safe_import("user_service.api.endpoints.users")
        if not users_module or not hasattr(users_module, "router"):
            logger.error("Échec de l'initialisation du service utilisateur: router introuvable")
            return False
            
        # Monter les routes
        app.include_router(
            users_module.router,
            prefix="/api/v1/users",
            tags=["users"]
        )
        
        logger.info("Service utilisateur initialisé et monté avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du service utilisateur: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def init_sync_service(app: FastAPI) -> bool:
    """Initialise et monte le service de synchronisation."""
    try:
        # Charger les services requis
        required_services = [
            "sync_service.services.webhook_handler",
            "sync_service.services.sync_manager",
            "sync_service.services.transaction_sync",
            "sync_service.services.vector_storage", 
            "sync_service.services.embedding_service"
        ]
        
        for service_name in required_services:
            service = safe_import(service_name, required=True)
            if not service:
                logger.error(f"Échec de l'initialisation du service de synchronisation: module {service_name} introuvable")
                return False
                
        # Charger les endpoints
        sync_router_module = safe_import("sync_service.api.endpoints.sync")
        webhooks_router_module = safe_import("sync_service.api.endpoints.webhooks")
        
        if not sync_router_module or not hasattr(sync_router_module, "router"):
            logger.error("Échec de l'initialisation du service de synchronisation: sync router introuvable")
            return False
            
        if not webhooks_router_module or not hasattr(webhooks_router_module, "router"):
            logger.error("Échec de l'initialisation du service de synchronisation: webhooks router introuvable")
            return False
        
        # Monter les routes
        app.include_router(
            sync_router_module.router,
            prefix="/api/v1/sync",
            tags=["synchronization"]
        )
        
        app.include_router(
            webhooks_router_module.router,
            prefix="/webhooks",
            tags=["webhooks"]
        )
        
        logger.info("Service de synchronisation initialisé et monté avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du service de synchronisation: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# ======== GESTIONNAIRE DE CYCLE DE VIE ========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    # ===== DÉMARRAGE =====
    logger.info("Démarrage de l'application Harena...")
    
    # Initialiser la base de données
    db_ok = init_database()
    if not db_ok:
        logger.error("Problèmes lors de l'initialisation de la base de données")
    
    # Charger les modules centraux
    core_ok = load_core_modules()
    if not core_ok:
        logger.warning("Problèmes lors du chargement des modules centraux")
    
    # Créer les objets partagés
    objects_ok = create_required_objects()
    if not objects_ok:
        logger.warning("Problèmes lors de la création des objets partagés")
    
    yield  # L'application s'exécute ici
    
    # ===== ARRÊT =====
    logger.info("Arrêt de l'application Harena...")
    
    # Effectuer le nettoyage nécessaire
    try:
        # Fermeture des connexions BD, etc.
        pass
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {str(e)}")

# ======== CRÉATION DE L'APPLICATION ========

app = FastAPI(
    title="Harena Finance API",
    description="API pour les services financiers Harena - Version Optimisée",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== INITIALISATION ET MONTAGE DES SERVICES ========

# Service utilisateur
service_status["user_service"] = init_user_service(app)

# Service de synchronisation - dépend du service utilisateur
if service_status["user_service"]:
    service_status["sync_service"] = init_sync_service(app)
else:
    logger.error("Impossible d'initialiser le service de synchronisation: service utilisateur non disponible")

# Journaliser le statut des services
active_services = sum(1 for status in service_status.values() if status)
logger.info(f"{active_services}/{len(service_status)} services actifs")
for name, status in service_status.items():
    logger.info(f"Service {name}: {'ACTIF' if status else 'INACTIF'}")

# ======== ENDPOINTS DE BASE ========

@app.get("/", tags=["health"])
async def root():
    """Point d'entrée racine pour vérifier que l'application est en ligne."""
    active_services = [name for name, status in service_status.items() if status]
    
    return {
        "status": "ok",
        "application": "Harena Finance API",
        "version": "1.1.0",
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "active_services": active_services
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Vérification de l'état de santé de tous les services."""
    # Vérifier la connexion à la base de données
    db_status = "unknown"
    
    try:
        db_session = imported_modules.get("user_service.db.session")
        if db_session and hasattr(db_session, "engine"):
            engine = db_session.engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_status = "connected"
        else:
            db_status = "engine not initialized"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Vérifier l'état du stockage vectoriel
    vector_status = "unknown"
    try:
        # Essayer d'initialiser le service vectoriel pour vérifier sa disponibilité
        vector_module = imported_modules.get("sync_service.services.vector_storage")
        if vector_module:
            vector_service = vector_module.VectorStorageService()
            vector_status = "available"
    except Exception as e:
        vector_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "services": service_status,
        "database": db_status,
        "vector_storage": vector_status,
        "version": "1.1.0",
        "timestamp": str(datetime.now())
    }

@app.post("/services/restart/{service_name}", tags=["admin"])
async def restart_service(service_name: str):
    """Redémarre un service spécifique."""
    if service_name not in service_status:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    
    result = {"service": service_name, "status": "unknown"}
    
    try:
        if service_name == "user_service":
            service_status["user_service"] = init_user_service(app)
            result["status"] = "restarted" if service_status["user_service"] else "failed"
            
        elif service_name == "sync_service":
            service_status["sync_service"] = init_sync_service(app)
            result["status"] = "restarted" if service_status["sync_service"] else "failed"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"Erreur lors du redémarrage du service {service_name}: {str(e)}")
        logger.error(traceback.format_exc())
    
    return result

@app.get("/debug", tags=["debug"])
async def debug_info():
    """Endpoint de débogage pour vérifier les configurations."""
    import pkg_resources
    
    # Récupérer les routeurs FastAPI
    routes_info = []
    for route in app.routes:
        methods = getattr(route, "methods", set())
        methods_str = list(methods) if methods else []
            
        route_info = {
            "path": getattr(route, "path", "unknown"),
            "name": getattr(route, "name", "unnamed"),
            "methods": methods_str,
        }
        routes_info.append(route_info)
    
    # Trier les routes par chemin
    routes_info.sort(key=lambda x: x["path"])
    
    # Collecter les informations sur les packages installés
    installed_packages = [
        {"name": pkg.key, "version": pkg.version}
        for pkg in pkg_resources.working_set
    ]
    
    # Collecter les variables d'environnement (sans les secrets)
    safe_env_vars = {}
    for key, value in os.environ.items():
        if any(secret_key in key.lower() for secret_key in ["key", "secret", "password", "token"]):
            safe_env_vars[key] = "[REDACTED]"
        else:
            safe_env_vars[key] = value
    
    # Vérifier les tables de la base de données si possible
    db_tables = []
    try:
        db_session = imported_modules.get("user_service.db.session")
        if db_session and hasattr(db_session, "engine"):
            engine = db_session.engine
            from sqlalchemy import inspect
            inspector = inspect(engine)
            db_tables = inspector.get_table_names()
    except Exception as e:
        db_tables = [f"Error: {str(e)}"]
    
    return {
        "python_version": sys.version,
        "service_status": service_status,
        "routes_count": len(routes_info),
        "routes": routes_info[:20],  # Limiter le nombre de routes affichées
        "database_tables": db_tables,
        "modules_loaded": len(imported_modules),
        "env_vars_count": len(safe_env_vars),
        "installed_packages_count": len(installed_packages)
    }

@app.get("/debug-modules", tags=["debug"])
async def debug_modules():
    """Liste des modules chargés pour diagnostiquer les problèmes d'importation."""
    modules_to_check = [
        "user_service.core.config",
        "user_service.models.base",
        "user_service.models.user",
        "user_service.db.session",
        "user_service.api.deps",
        "user_service.api.endpoints.users",
        "sync_service.models.sync",
        "sync_service.services.webhook_handler",
        "sync_service.services.sync_manager",
        "sync_service.services.transaction_sync",
        "sync_service.services.vector_storage",
        "sync_service.services.embedding_service",
        "sync_service.api.endpoints.sync",
        "sync_service.api.endpoints.webhooks"
    ]
    
    results = {}
    
    for module_name in modules_to_check:
        if module_name in imported_modules:
            module = imported_modules[module_name]
            if module:
                results[module_name] = {
                    "status": "loaded",
                    "path": getattr(module, "__file__", "unknown")
                }
            else:
                results[module_name] = {
                    "status": "import_failed"
                }
        elif module_name in sys.modules:
            module = sys.modules[module_name]
            results[module_name] = {
                "status": "loaded_by_system",
                "path": getattr(module, "__file__", "unknown")
            }
        else:
            results[module_name] = {
                "status": "not_loaded"
            }
    
    return results

# ======== GESTIONNAIRE D'EXCEPTIONS ========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestionnaire global d'exceptions pour toute l'application."""
    logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
    
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    error_detail = str(exc) if debug_mode else "Une erreur interne est survenue."
    
    if debug_mode:
        error_detail += "\n\n" + traceback.format_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Une erreur interne est survenue",
            "detail": error_detail,
            "path": request.url.path
        }
    )

# Point d'entrée pour démarrer directement l'application
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=port)