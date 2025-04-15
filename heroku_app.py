"""
Application Harena complète pour le déploiement Heroku.
Intégration optimisée de tous les services bancaires.
"""

import logging
import os
import sys
import importlib
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("harena")

# Vérifier et configurer les variables d'environnement
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    # Heroku utilise "postgres://" mais SQLAlchemy 1.4+ requiert "postgresql://"
    os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    logger.info("DATABASE_URL corrigé pour SQLAlchemy 1.4+")

# Gestion des dépendances circulaires
os.environ["DEFERRED_RELATIONSHIP_LOADING"] = "True"

# S'assurer que tous les modules sont accessibles
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    logger.info(f"Ajout du répertoire courant {current_dir} au PYTHONPATH")

# Suivi des services intégrés
service_status = {
    "user_service": False,
    "sync_service": False,
    "transaction_vector_service": False,
    "conversation_service": False
}

# Fonction pour importer un module avec gestion d'erreur
def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        logger.error(f"Erreur lors de l'importation de {module_name}: {str(e)}")
        return None

# Fonction pour réinitialiser un module s'il est déjà chargé
def reload_module(module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return safe_import(module_name)

# Gestionnaire de cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    logger.info("Démarrage de l'application Harena...")
    
    # Résumé des services
    active_count = sum(1 for status in service_status.values() if status)
    logger.info(f"{active_count}/{len(service_status)} services actifs")
    
    for service, status in service_status.items():
        status_txt = "ACTIF" if status else "INACTIF"
        logger.info(f"Service {service}: {status_txt}")
    
    yield  # Ici l'application s'exécute
    
    # Code exécuté à l'arrêt
    logger.info("Arrêt de l'application Harena...")

# Création de l'application FastAPI avec lifespan
app = FastAPI(
    title="Harena Finance API",
    description="API pour les services financiers Harena",
    version="1.0.0",
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

# ======== IMPORTATION DES MODULES ET INITIALISATION ========

# Préparation des configurations et modèles de base
try:
    # Forcer la suppression des modules problématiques pour les réimporter proprement
    modules_to_reset = [
        "user_service.models.user",
        "user_service.db.session",
        "sync_service.models.sync"
    ]
    
    for module_name in modules_to_reset:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # Importer la configuration utilisateur en premier
    user_settings_module = safe_import("user_service.core.config")
    if user_settings_module:
        user_settings = user_settings_module.settings
        logger.info("Configuration utilisateur chargée")
    else:
        logger.error("Impossible de charger la configuration utilisateur")
    
    # Importer les modèles de base
    base_module = safe_import("user_service.models.base")
    if base_module:
        Base = base_module.Base
        logger.info("Modèle de base chargé")
    else:
        logger.error("Impossible de charger le modèle de base")
    
    # Création de la session de base de données
    session_module = safe_import("user_service.db.session")
    if session_module:
        engine = session_module.engine
        get_db = session_module.get_db
        logger.info("Session de base de données initialisée")
    else:
        logger.error("Impossible d'initialiser la session de base de données")
    
    # Vérifier la connexion BD
    try:
        if engine:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Connexion à la base de données établie")
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données: {str(e)}")
    
    # Initialiser les tables si demandé
    if os.environ.get("CREATE_TABLES", "false").lower() == "true":
        try:
            if Base and engine:
                Base.metadata.create_all(bind=engine)
                logger.info("Tables créées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création des tables: {str(e)}")

except Exception as e:
    logger.error(f"Erreur lors de l'initialisation générale: {str(e)}")

# ======== IMPORTATION ET MONTAGE DES SERVICES ========

# Import et montage des services transaction_vector et conversation en premier
# car ils ont moins de dépendances vers les autres services

# ======== TRANSACTION VECTOR SERVICE ========
try:
    # Importer les composants nécessaires
    transactions_router_module = safe_import("transaction_vector_service.api.endpoints.transactions")
    
    if transactions_router_module:
        # Inclure les routes
        app.include_router(
            transactions_router_module.router,
            prefix="/api/v1/transactions",
            tags=["transactions"]
        )
        
        # Initialiser les services
        try:
            dependencies_module = safe_import("transaction_vector_service.api.dependencies")
            if dependencies_module:
                dependencies_module.initialize_services()
                logger.info("Services Transaction Vector initialisés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des services Transaction Vector: {str(e)}")
        
        service_status["transaction_vector_service"] = True
        logger.info("Transaction Vector Service intégré avec succès")
    else:
        logger.error("Module de routage Transaction Vector introuvable")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Transaction Vector Service: {str(e)}")

# ======== CONVERSATION SERVICE ========
try:
    # Importer les composants nécessaires
    conversation_router_module = safe_import("conversation_service.api.endpoints")
    
    if conversation_router_module:
        # Inclure les routes
        app.include_router(
            conversation_router_module.router,
            prefix="/api/v1/conversations",
            tags=["conversations"]
        )
        
        service_status["conversation_service"] = True
        logger.info("Conversation Service intégré avec succès")
    else:
        logger.error("Module de routage Conversation introuvable")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Conversation Service: {str(e)}")

# ======== USER SERVICE ========
try:
    # Réimporter le module modèle utilisateur pour garantir qu'il est bien chargé
    user_model_module = reload_module("user_service.models.user")
    
    if user_model_module:
        logger.info("Modèles utilisateur chargés correctement")
        
        # Vérifier les dépendances d'authentification
        deps_module = safe_import("user_service.api.deps")
        
        if deps_module:
            logger.info("Dépendances utilisateur chargées correctement")
            
            # Charger les endpoints
            users_endpoints_module = safe_import("user_service.api.endpoints.users")
            
            if users_endpoints_module and hasattr(users_endpoints_module, 'router'):
                # Inclure les routes utilisateur
                app.include_router(
                    users_endpoints_module.router, 
                    prefix="/api/v1/users", 
                    tags=["users"]
                )
                service_status["user_service"] = True
                logger.info("User Service intégré avec succès")
            else:
                logger.error("Module endpoints utilisateur invalide")
        else:
            logger.error("Impossible de charger les dépendances utilisateur")
    else:
        logger.error("Impossible de charger les modèles utilisateur")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du User Service: {str(e)}")
    traceback.print_exc()

# ======== SYNC SERVICE ========
try:
    # Charger les modules de synchronisation
    sync_router_module = safe_import("sync_service.api.endpoints.sync")
    webhooks_router_module = safe_import("sync_service.api.endpoints.webhooks")
    
    if sync_router_module and webhooks_router_module:
        # Inclure les routes
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
        
        service_status["sync_service"] = True
        logger.info("Sync Service intégré avec succès")
    else:
        if not sync_router_module:
            logger.error("Module de routage Sync introuvable")
        if not webhooks_router_module:
            logger.error("Module de routage Webhooks introuvable")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Sync Service: {str(e)}")

# ======== ENDPOINTS DE BASE ========

@app.get("/", tags=["health"])
async def root():
    """
    Point d'entrée racine pour vérifier que l'application est en ligne.
    """
    # Collecter les services actifs
    active_services = [name for name, status in service_status.items() if status]
    
    return {
        "status": "ok",
        "application": "Harena Finance API",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "active_services": active_services
    }

@app.get("/health", tags=["health"])
async def health_check():
    """
    Vérification de l'état de santé de tous les services.
    """
    # Vérifier la connexion à la base de données
    db_status = "unknown"
    
    try:
        if 'engine' in globals():
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            db_status = "connected"
        else:
            db_status = "engine not initialized"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "services": service_status,
        "database": db_status,
        "version": "1.0.0",
        "timestamp": str(datetime.now())
    }

@app.get("/debug-modules", tags=["debug"])
async def debug_modules():
    """
    Diagnostic détaillé des modules chargés.
    """
    modules_to_check = [
        "user_service",
        "user_service.core.config",
        "user_service.models.base",
        "user_service.models.user",
        "user_service.db.session",
        "user_service.services.users",
        "user_service.services.bridge",
        "user_service.api.deps",
        "user_service.api.endpoints.users",
        "sync_service",
        "sync_service.models.sync",
        "sync_service.api.endpoints.sync",
        "sync_service.api.endpoints.webhooks",
        "transaction_vector_service",
        "transaction_vector_service.api.endpoints.transactions",
        "conversation_service",
        "conversation_service.api.endpoints"
    ]
    
    results = {}
    
    for module_name in modules_to_check:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            results[module_name] = {
                "loaded": True,
                "path": getattr(module, "__file__", "unknown")
            }
        else:
            results[module_name] = {
                "loaded": False
            }
    
    return results

@app.get("/debug", tags=["debug"])
async def debug_info():
    """
    Endpoint de débogage pour vérifier les configurations.
    """
    import pkg_resources
    
    # Récupérer les routeurs FastAPI
    routes_info = []
    for route in app.routes:
        methods = getattr(route, "methods", set())
        if methods:
            methods_str = list(methods)
        else:
            methods_str = []
            
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
        if 'engine' in globals():
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
        "env_vars_count": len(safe_env_vars),
        "installed_packages_count": len(installed_packages),
        "python_path": sys.path
    }

@app.get("/debug-test-user-service", tags=["debug"])
async def debug_test_user_service():
    """
    Test spécifique du user_service.
    """
    results = {}
    
    # Tester l'accès à la base de données
    try:
        user_db_session = safe_import("user_service.db.session")
        if user_db_session and hasattr(user_db_session, 'engine'):
            with user_db_session.engine.connect() as conn:
                conn.execute("SELECT 1")
            results["db_connection"] = "success"
        else:
            results["db_connection"] = "module not properly loaded"
    except Exception as e:
        results["db_connection"] = f"error: {str(e)}"
    
    # Tester l'accès aux modèles
    try:
        user_models = safe_import("user_service.models.user")
        if user_models:
            model_names = [name for name in dir(user_models) if not name.startswith("_")]
            results["user_models"] = model_names
        else:
            results["user_models"] = "module not loaded"
    except Exception as e:
        results["user_models"] = f"error: {str(e)}"
    
    # Tester le routeur
    try:
        user_endpoints = safe_import("user_service.api.endpoints.users")
        if user_endpoints and hasattr(user_endpoints, 'router'):
            route_paths = [route.path for route in user_endpoints.router.routes]
            results["router_routes"] = route_paths
        else:
            results["router_routes"] = "router not available"
    except Exception as e:
        results["router_routes"] = f"error: {str(e)}"
    
    return results

# ======== GESTIONNAIRE D'EXCEPTIONS ========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions.
    """
    logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
    
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    error_detail = str(exc) if debug_mode else "Contactez l'administrateur pour plus d'informations."
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Une erreur interne est survenue",
            "detail": error_detail
        }
    )