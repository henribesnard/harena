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
from datetime import datetime
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,  # Augmenter le niveau de détail des logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("harena")
logger.info("Initialisation de l'application Harena avec niveau de log DEBUG")

# Définir la variable d'environnement pour propager aux autres modules
os.environ["LOG_LEVEL"] = "DEBUG"

# Vérifier et configurer les variables d'environnement
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    # Heroku utilise "postgres://" mais SQLAlchemy 1.4+ requiert "postgresql://"
    os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    logger.info("DATABASE_URL corrigé pour SQLAlchemy 1.4+")

# Désactiver la gestion des dépendances circulaires
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

# Fonction pour importer un module avec gestion d'erreur détaillée
def safe_import(module_name, required=False):
    try:
        logger.info(f"Tentative d'importation du module: {module_name}")
        module = importlib.import_module(module_name)
        logger.info(f"Module {module_name} importé avec succès")
        return module
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erreur lors de l'importation de {module_name}: {str(e)}")
        logger.debug(f"Traceback complet: {error_trace}")
        
        if required:
            logger.critical(f"Module requis {module_name} n'a pas pu être importé. Application peut être instable.")
        
        return None

# Fonction pour réinitialiser un module s'il est déjà chargé
def reload_module(module_name):
    if module_name in sys.modules:
        logger.debug(f"Suppression du module {module_name} du cache sys.modules")
        del sys.modules[module_name]
    return safe_import(module_name)

# Gestionnaire de cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    logger.info("Démarrage de l'application Harena...")
    
    try:
        # Initialiser les composants critiques
        logger.info("Initialisation des composants de l'application")
        
        # Ici vous pourriez ajouter du code pour initialiser des ressources
        # comme une connexion à une base de données, configuration de cache, etc.
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des composants: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Résumé des services
    active_count = sum(1 for status in service_status.values() if status)
    logger.info(f"{active_count}/{len(service_status)} services actifs")
    
    for service, status in service_status.items():
        status_txt = "ACTIF" if status else "INACTIF"
        logger.info(f"Service {service}: {status_txt}")
    
    yield  # Ici l'application s'exécute
    
    # Code exécuté à l'arrêt
    logger.info("Arrêt de l'application Harena...")
    
    # Fermeture propre des ressources
    logger.info("Fermeture des ressources de l'application")

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

# ======== IMPORTATION DES MODULES DE BASE ========

# Préparation des configurations et modèles de base
try:
    # Liste des modules de base à charger en premier
    base_modules = [
        "user_service.core.config",
        "user_service.models.base",
        "user_service.db.session"
    ]
    
    # Charger les modules de base dans l'ordre
    loaded_modules = {}
    for module_name in base_modules:
        loaded_modules[module_name] = safe_import(module_name, required=True)
    
    # Extraire les références importantes
    if loaded_modules["user_service.core.config"]:
        user_settings = loaded_modules["user_service.core.config"].settings
        logger.info("Configuration utilisateur chargée")
    
    if loaded_modules["user_service.models.base"]:
        Base = loaded_modules["user_service.models.base"].Base
        logger.info("Modèle de base chargé")
    
    if loaded_modules["user_service.db.session"]:
        engine = loaded_modules["user_service.db.session"].engine
        get_db = loaded_modules["user_service.db.session"].get_db
        logger.info("Session de base de données initialisée")
    
    # Vérifier la connexion BD
    try:
        if 'engine' in locals():
            with engine.connect() as conn:
                # Utiliser text() pour encapsuler la requête SQL
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            logger.info("Connexion à la base de données établie")
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données: {str(e)}")
    
    # Initialiser les tables si demandé
    if os.environ.get("CREATE_TABLES", "false").lower() == "true":
        try:
            if 'Base' in locals() and 'engine' in locals():
                Base.metadata.create_all(bind=engine)
                logger.info("Tables créées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création des tables: {str(e)}")

except Exception as e:
    logger.error(f"Erreur lors de l'initialisation des modules de base: {str(e)}")
    logger.error(traceback.format_exc())

# ======== IMPORTATION ET MONTAGE DES SERVICES ========

# ======== USER SERVICE ========
try:
    # Réimporter le module modèle utilisateur
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
    logger.error(traceback.format_exc())

# ======== SYNC SERVICE ========
try:
    # Charger les modèles de synchronisation d'abord
    sync_model_module = safe_import("sync_service.models.sync")
    
    if sync_model_module:
        logger.info("Modèles de synchronisation chargés correctement")
        
        # Charger les services de gestion de synchronisation
        webhook_handler_module = safe_import("sync_service.services.webhook_handler")
        sync_manager_module = safe_import("sync_service.services.sync_manager")
        transaction_sync_module = safe_import("sync_service.services.transaction_sync")
        
        if webhook_handler_module and sync_manager_module and transaction_sync_module:
            logger.info("Services de synchronisation chargés correctement")
            
            # Charger les modules de routage
            sync_router_module = safe_import("sync_service.api.endpoints.sync")
            webhooks_router_module = safe_import("sync_service.api.endpoints.webhooks")
            
            # Monter les routeurs si disponibles
            if sync_router_module and hasattr(sync_router_module, 'router'):
                app.include_router(
                    sync_router_module.router, 
                    prefix="/api/v1/sync", 
                    tags=["synchronization"]
                )
                logger.info("Routes de synchronisation montées avec succès")
            else:
                logger.error("Module de routage sync introuvable ou invalide")
                
            if webhooks_router_module and hasattr(webhooks_router_module, 'router'):
                app.include_router(
                    webhooks_router_module.router,
                    prefix="/webhooks",
                    tags=["webhooks"]
                )
                logger.info("Routes de webhooks montées avec succès")
                
                # Service considéré actif si les deux routeurs sont montés
                if sync_router_module and hasattr(sync_router_module, 'router'):
                    service_status["sync_service"] = True
                    logger.info("Sync Service intégré avec succès")
            else:
                logger.error("Module de routage webhooks introuvable ou invalide")
        else:
            missing_modules = []
            if not webhook_handler_module:
                missing_modules.append("webhook_handler")
            if not sync_manager_module:
                missing_modules.append("sync_manager")
            if not transaction_sync_module:
                missing_modules.append("transaction_sync")
            
            logger.error(f"Modules de service manquants: {', '.join(missing_modules)}")
    else:
        logger.error("Impossible de charger les modèles de synchronisation")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Sync Service: {str(e)}")
    logger.error(traceback.format_exc())

# ======== TRANSACTION VECTOR SERVICE ========
try:
    # Importer d'abord les modules de configuration
    vector_config_modules = [
        "transaction_vector_service.config.logging_config",
        "transaction_vector_service.config.settings",
        "transaction_vector_service.config.constants"
    ]
    
    vector_configs = {}
    for module_name in vector_config_modules:
        vector_configs[module_name] = safe_import(module_name)
    
    # Vérifier si les configurations sont chargées correctement
    if all(vector_configs.values()):
        logger.info("Configurations Vector Service chargées correctement")
        
        # Charger les services essentiels dans l'ordre
        vector_services = [
            "transaction_vector_service.services.embedding_service",
            "transaction_vector_service.services.qdrant_client",
            "transaction_vector_service.services.category_service",
            "transaction_vector_service.services.merchant_service",
            "transaction_vector_service.services.transaction_service",
            "transaction_vector_service.services.sync_service"
        ]
        
        vector_service_modules = {}
        for module_name in vector_services:
            vector_service_modules[module_name] = safe_import(module_name)
        
        # Vérifier si les services essentiels sont chargés
        if all(vector_service_modules.values()):
            logger.info("Services Vector chargés correctement")
            
            # Charger les dépendances API
            vector_api_deps = safe_import("transaction_vector_service.api.dependencies")
            
            if vector_api_deps:
                logger.info("Dépendances API Vector chargées correctement")
                
                # Charger les endpoints
                vector_transactions_endpoint = safe_import("transaction_vector_service.api.endpoints.transactions")
                
                if vector_transactions_endpoint and hasattr(vector_transactions_endpoint, 'router'):
                    # Inclure les routes
                    app.include_router(
                        vector_transactions_endpoint.router,
                        prefix="/api/v1/transactions",
                        tags=["transactions"]
                    )
                    
                    # Initialiser les services
                    try:
                        vector_api_deps.initialize_services()
                        service_status["transaction_vector_service"] = True
                        logger.info("Transaction Vector Service intégré avec succès")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'initialisation des services Vector: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.error("Module d'endpoints transactions introuvable ou invalide")
            else:
                logger.error("Impossible de charger les dépendances API Vector")
        else:
            missing_services = [name for name, module in vector_service_modules.items() if not module]
            logger.error(f"Services Vector manquants: {', '.join(missing_services)}")
    else:
        missing_configs = [name for name, module in vector_configs.items() if not module]
        logger.error(f"Configurations Vector manquantes: {', '.join(missing_configs)}")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Transaction Vector Service: {str(e)}")
    logger.error(traceback.format_exc())

# ======== CONVERSATION SERVICE ========
try:
    # Charger le routeur de conversation
    conversation_router_module = safe_import("conversation_service.api.router")
    
    if conversation_router_module and hasattr(conversation_router_module, 'router'):
        # Inclure les routes
        app.include_router(
            conversation_router_module.router,
            prefix="/api/v1/conversations",
            tags=["conversations"]
        )
        
        service_status["conversation_service"] = True
        logger.info("Conversation Service intégré avec succès")
    else:
        logger.error("Module de routage Conversation introuvable ou invalide")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Conversation Service: {str(e)}")
    logger.error(traceback.format_exc())

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
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
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

@app.get("/debug-init", tags=["debug"])
async def debug_init():
    """
    Re-initialisation des services en échec
    """
    results = {}
    
    # Réinitialiser le service Vector si nécessaire
    if not service_status["transaction_vector_service"]:
        logger.info("Tentative de réinitialisation du Transaction Vector Service...")
        try:
            # Réimporter et initialiser les modules clés
            vector_modules = [
                "transaction_vector_service.services.sync_service",
                "transaction_vector_service.services.qdrant_client",
                "transaction_vector_service.services.transaction_service"
            ]
            
            for module_name in vector_modules:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    logger.debug(f"Module {module_name} supprimé du cache")
            
            # Réimporter le module de dépendances
            deps_module = reload_module("transaction_vector_service.api.dependencies")
            
            if deps_module:
                # Réinitialiser les services
                deps_module.initialize_services()
                service_status["transaction_vector_service"] = True
                results["transaction_vector_service"] = "reinitialized"
                logger.info("Transaction Vector Service réinitialisé avec succès")
            else:
                results["transaction_vector_service"] = "failed to reload dependencies"
        except Exception as e:
            results["transaction_vector_service"] = f"error: {str(e)}"
            logger.error(f"Erreur lors de la réinitialisation du Vector Service: {str(e)}")
    else:
        results["transaction_vector_service"] = "already active"
    
    # Réinitialiser le service Sync si nécessaire
    if not service_status["sync_service"]:
        logger.info("Tentative de réinitialisation du Sync Service...")
        try:
            # Réimporter les modules clés
            sync_modules = [
                "sync_service.services.webhook_handler",
                "sync_service.services.sync_manager",
                "sync_service.services.transaction_sync"
            ]
            
            reloaded = True
            for module_name in sync_modules:
                module = reload_module(module_name)
                if not module:
                    reloaded = False
                    results["sync_service"] = f"failed to reload {module_name}"
            
            if reloaded:
                # Réimporter les routeurs
                sync_router = reload_module("sync_service.api.endpoints.sync")
                webhooks_router = reload_module("sync_service.api.endpoints.webhooks")
                
                if sync_router and webhooks_router:
                    # Monter les routeurs
                    try:
                        app.include_router(
                            sync_router.router, 
                            prefix="/api/v1/sync", 
                            tags=["synchronization"]
                        )
                        
                        app.include_router(
                            webhooks_router.router,
                            prefix="/webhooks",
                            tags=["webhooks"]
                        )
                        
                        service_status["sync_service"] = True
                        results["sync_service"] = "reinitialized"
                        logger.info("Sync Service réinitialisé avec succès")
                    except Exception as e:
                        results["sync_service"] = f"router mounting error: {str(e)}"
                else:
                    results["sync_service"] = "failed to reload routers"
        except Exception as e:
            results["sync_service"] = f"error: {str(e)}"
            logger.error(f"Erreur lors de la réinitialisation du Sync Service: {str(e)}")
    else:
        results["sync_service"] = "already active"
    
    return {
        "status": "completed",
        "results": results,
        "services_status": service_status
    }

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
        "modules_count": len(sys.modules),
        "env_vars_count": len(safe_env_vars),
        "installed_packages_count": len(installed_packages),
        "python_path": sys.path
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
        "sync_service.services.webhook_handler",
        "sync_service.services.sync_manager",
        "sync_service.services.transaction_sync",
        "sync_service.api.endpoints.sync",
        "sync_service.api.endpoints.webhooks",
        "transaction_vector_service",
        "transaction_vector_service.config.logging_config",
        "transaction_vector_service.config.settings",
        "transaction_vector_service.services.embedding_service",
        "transaction_vector_service.services.qdrant_client",
        "transaction_vector_service.services.sync_service",
        "transaction_vector_service.api.dependencies",
        "transaction_vector_service.api.endpoints.transactions",
        "conversation_service",
        "conversation_service.api.router"
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

# ======== GESTIONNAIRE D'EXCEPTIONS ========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Gestionnaire global d'exceptions pour toute l'application.
    Capture et formate les erreurs non gérées.
    """
    logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
    
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    error_detail = str(exc) if debug_mode else "Contactez l'administrateur pour plus d'informations."
    
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