"""
Application Harena complète pour le déploiement Heroku.
Intégration de tous les services via inclusion directe des routeurs.
"""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
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

# Gestionnaire de cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    logger.info("Démarrage de l'application Harena...")
    
    # Vérifier les variables d'environnement essentielles
    required_vars = [
        "DATABASE_URL",
        "SECRET_KEY",
        "BRIDGE_CLIENT_ID",
        "BRIDGE_CLIENT_SECRET"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.warning(f"Variables d'environnement manquantes: {', '.join(missing_vars)}")
    
    # Initialiser la base de données si nécessaire
    if os.environ.get("CREATE_TABLES", "false").lower() == "true":
        try:
            from user_service.db.session import engine, Base
            logger.info("Création des tables de base de données...")
            Base.metadata.create_all(bind=engine)
            logger.info("Tables créées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création des tables: {str(e)}")
    
    # Initialiser les services Transaction Vector si disponibles
    try:
        from transaction_vector_service.api.dependencies import initialize_services
        initialize_services()
        logger.info("Services Transaction Vector initialisés avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des services Transaction Vector: {str(e)}")
    
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

# ======== INITIALISATION DE LA BASE DE DONNÉES ========
try:
    # Importer et configurer la session de base de données
    from user_service.db.session import get_db, engine, Base
    from user_service.core.config import settings as user_settings
    logger.info("Session de base de données initialisée avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation de la base de données: {str(e)}")

# ======== USER SERVICE ========
try:
    # Importer les dépendances et routes
    from user_service.api.endpoints import users as users_endpoints
    
    # Vérifier que les dépendances d'authentification sont accessibles
    from user_service.api.deps import get_current_user, get_current_active_user
    
    # Inclure les routes utilisateur
    app.include_router(
        users_endpoints.router, 
        prefix=f"{user_settings.API_V1_STR}/users", 
        tags=["users"]
    )
    
    # Test spécifique pour s'assurer que les modèles sont accessibles
    from user_service.models.user import User, BridgeConnection, UserPreference
    logger.info("Modèles User Service chargés avec succès")
    
    service_status["user_service"] = True
    logger.info("User Service intégré avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du User Service: {str(e)}")

# ======== SYNC SERVICE ========
try:
    # Importer les routes de synchronisation avec gestion d'erreur explicite
    try:
        from sync_service.api.endpoints import sync as sync_router
        logger.info("Module sync_service.api.endpoints.sync importé avec succès")
    except Exception as e:
        logger.error(f"Erreur d'importation de sync_router: {str(e)}")
        raise
    
    try:
        from sync_service.api.endpoints import webhooks as webhooks_router
        logger.info("Module sync_service.api.endpoints.webhooks importé avec succès")
    except Exception as e:
        logger.error(f"Erreur d'importation de webhooks_router: {str(e)}")
        raise
    
    # Vérifier l'accessibilité des modèles
    from sync_service.models.sync import SyncItem, SyncAccount, WebhookEvent
    logger.info("Modèles Sync Service chargés avec succès")
    
    # Vérifier l'accessibilité des services
    from sync_service.services import sync_manager, webhook_handler, transaction_sync
    logger.info("Services Sync chargés avec succès")
    
    # Inclure les routes
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
    logger.info("Sync Service intégré avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Sync Service: {str(e)}")

# ======== TRANSACTION VECTOR SERVICE ========
try:
    # Importer les composants nécessaires
    from transaction_vector_service.api.endpoints.transactions import router as transactions_router
    from transaction_vector_service.config.settings import settings as transaction_settings
    
    # Vérifier l'accessibilité des dépendances
    from transaction_vector_service.api.dependencies import (
        get_transaction_service,
        get_current_user as tvs_get_current_user,
        get_rate_limiter
    )
    
    # Inclure les routes
    app.include_router(
        transactions_router,
        prefix="/api/v1/transactions",
        tags=["transactions"]
    )
    
    service_status["transaction_vector_service"] = True
    logger.info("Transaction Vector Service intégré avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Transaction Vector Service: {str(e)}")

# ======== CONVERSATION SERVICE ========
try:
    # Importer les composants nécessaires
    from conversation_service.api.endpoints import router as conversation_router
    from conversation_service.config.settings import settings as conversation_settings
    
    # Vérifier l'accessibilité des dépendances clés
    from conversation_service.services.conversation_manager import ConversationManager
    from conversation_service.llm.llm_service import LLMService
    
    # Inclure les routes
    app.include_router(
        conversation_router,
        prefix="/api/v1/conversations",
        tags=["conversations"]
    )
    
    service_status["conversation_service"] = True
    logger.info("Conversation Service intégré avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'intégration du Conversation Service: {str(e)}")

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
        from user_service.db.session import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "services": service_status,
        "database": db_status,
        "version": "1.0.0",
        "timestamp": str(datetime.now())
    }

@app.get("/debug", tags=["debug"])
async def debug_info():
    """
    Endpoint de débogage pour vérifier les configurations.
    """
    import sys
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
    
    return {
        "python_version": sys.version,
        "service_status": service_status,
        "routes_count": len(routes_info),
        "routes": routes_info[:20],  # Limiter le nombre de routes affichées
        "env_vars": safe_env_vars,
        "installed_packages": sorted(installed_packages, key=lambda x: x["name"]),
        "python_path": sys.path
    }

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