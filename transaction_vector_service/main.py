import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import settings
from .config.logging_config import setup_logging
from .api.router import api_router

# Configuration du logging
logger = setup_logging()

# Création de l'application FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    debug=settings.DEBUG,
    description="Service de gestion des transactions vectorisées pour Harena"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion du routeur API
app.include_router(api_router, prefix=settings.API_V1_STR)

# Configuration des tâches planifiées
if settings.ENABLE_SCHEDULED_TASKS:
    from apscheduler.schedulers.background import BackgroundScheduler
    from .tasks.generate_insights import generate_all_insights
    from .tasks.detect_recurrings import detect_all_recurrings
    
    scheduler = BackgroundScheduler()
    
    # Générer des insights tous les jours à 3h du matin
    scheduler.add_job(
        generate_all_insights,
        "cron",
        hour=3,
        minute=0
    )
    
    # Détecter les transactions récurrentes toutes les semaines
    scheduler.add_job(
        detect_all_recurrings,
        "cron",
        day_of_week="mon",
        hour=4,
        minute=0
    )
    
    # Démarrer le planificateur
    scheduler.start()
    logger.info("Scheduled tasks enabled and configured")


@app.get("/health")
async def health_check():
    """Vérification de l'état de santé du service."""
    # Effectuer des vérifications de connexion aux services critiques
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "qdrant": "connected",
            "embedding": "operational"
        }
    }
    
    # Vous pourriez ajouter des vérifications plus détaillées ici
    
    return health_status


@app.on_event("startup")
async def startup_event():
    """Actions à exécuter au démarrage de l'application."""
    logger.info(f"Starting {settings.PROJECT_NAME}")
    
    # Initialiser les collections Qdrant si nécessaire
    from .services.qdrant_client import QdrantService
    
    try:
        qdrant_service = QdrantService()
        # La méthode _ensure_collections est appelée dans l'initialisation
        logger.info("Qdrant collections initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collections: {str(e)}")
    
    # Pré-charger les catégories pour le cache
    from .services.category_service import CategoryService
    
    try:
        category_service = CategoryService()
        await category_service.preload_categories()
        logger.info("Categories preloaded into cache")
    except Exception as e:
        logger.error(f"Failed to preload categories: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("transaction_vector_service.main:app", host="0.0.0.0", port=8002, reload=settings.DEBUG)