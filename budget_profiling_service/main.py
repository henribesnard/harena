"""
Point d'entrée principal Budget Profiling Service
Service de profilage utilisateur et recommandations budgétaires intelligentes
"""
import logging
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from dotenv import load_dotenv

# Charger le fichier .env au démarrage
load_dotenv()

# Configuration path pour imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Imports locaux
from config_service.config import settings
from budget_profiling_service.api.routes.budget_profile import router as budget_router
from budget_profiling_service.api.middleware.auth_middleware import JWTAuthMiddleware

# Configuration logging
logging.basicConfig(
    level=getattr(logging, os.getenv('BUDGET_PROFILING_LOG_LEVEL', 'INFO'), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("budget_profiling_service")


class BudgetProfilingServiceLoader:
    """
    Service loader pour le service de profilage budgétaire
    """

    def __init__(self):
        self.service_healthy = False
        self.initialization_error = None
        self.service_start_time = datetime.now(timezone.utc)

        # Configuration service
        self.service_config = {
            "version": "1.0.0",
            "features": [
                "transaction_analysis",
                "fixed_charges_detection",
                "budget_profiling",
                "recommendations",
                "savings_goals",
                "seasonal_patterns"
            ],
            "jwt_compatible": True,
        }

        logger.info("BudgetProfilingServiceLoader initialisé")

    def initialize_service(self, app: FastAPI) -> bool:
        """
        Initialise le service de profilage budgétaire
        """
        try:
            logger.info("🚀 Initialisation Budget Profiling Service")

            # Vérification configuration service
            if not getattr(settings, 'BUDGET_PROFILING_ENABLED', True):
                logger.info("⚠️ Budget Profiling Service désactivé par configuration")
                return False

            # Validation configuration base de données
            db_validation = self._validate_database_configuration()
            if not db_validation:
                logger.error("❌ Configuration base de données invalide")
                return False

            logger.info("✅ Budget Profiling Service initialisé avec succès")
            self.service_healthy = True
            return True

        except Exception as e:
            logger.error(f"❌ Erreur initialisation service: {e}", exc_info=True)
            self.initialization_error = str(e)
            self.service_healthy = False
            return False

    def _validate_database_configuration(self) -> bool:
        """
        Valide la configuration de la base de données
        """
        try:
            required_vars = ['DATABASE_URL']
            missing_vars = [var for var in required_vars if not getattr(settings, var, None)]

            if missing_vars:
                logger.error(f"❌ Variables manquantes: {missing_vars}")
                return False

            logger.info("✅ Configuration base de données validée")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur validation DB: {e}")
            return False


# Instance globale du loader
service_loader = BudgetProfilingServiceLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application
    """
    # Startup
    logger.info("🚀 Démarrage Budget Profiling Service...")

    try:
        initialization_success = service_loader.initialize_service(app)

        if initialization_success:
            logger.info("✅ Service démarré avec succès")
        else:
            logger.error("❌ Échec initialisation service")

    except Exception as e:
        logger.error(f"❌ Erreur critique au démarrage: {e}", exc_info=True)

    yield

    # Shutdown
    logger.info("🛑 Arrêt Budget Profiling Service...")


# Création de l'application FastAPI
app = FastAPI(
    title="Budget Profiling Service",
    description="Service de profilage utilisateur et recommandations budgétaires intelligentes",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
logger.info(f"🌍 Environment: {ENVIRONMENT}")

if ENVIRONMENT in ['dev', 'development', 'testing']:
    # Mode développement: autoriser toutes les origines
    logger.info("🔓 CORS: Mode développement - Autorisation de toutes les origines")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Mode production: origines restreintes (géré par Nginx)
    allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173').split(',')
    logger.info(f"🔒 CORS: Mode production - Origines autorisées: {allowed_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )


@app.get("/health")
def health_check():
    """
    Endpoint de health check
    """
    uptime = (datetime.now(timezone.utc) - service_loader.service_start_time).total_seconds()

    health_status = {
        "status": "healthy" if service_loader.service_healthy else "unhealthy",
        "service": "budget_profiling",
        "version": service_loader.service_config["version"],
        "uptime_seconds": uptime,
        "features": service_loader.service_config["features"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if service_loader.initialization_error:
        health_status["initialization_error"] = service_loader.initialization_error

    status_code = 200 if service_loader.service_healthy else 503
    return JSONResponse(content=health_status, status_code=status_code)


# Ajouter le middleware d'authentification JWT
app.add_middleware(JWTAuthMiddleware)

# Inclure les routes
app.include_router(budget_router)


@app.get("/")
def root():
    """
    Endpoint racine
    """
    return {
        "service": "Budget Profiling Service",
        "version": service_loader.service_config["version"],
        "status": "running",
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BUDGET_PROFILING_PORT", 3006))
    host = os.getenv("BUDGET_PROFILING_HOST", "0.0.0.0")

    logger.info(f"🚀 Démarrage serveur sur {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") in ["dev", "development"],
        log_level="info"
    )
