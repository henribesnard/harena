"""
Metric Service - FastAPI Application
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Optional
import os
from datetime import datetime, timezone

from metric_service.api.routes import trends, health, patterns, expenses, income, coverage
from metric_service.core.cache import cache_manager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan pour initialisation et nettoyage
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation et nettoyage de l'application"""
    logger.info("🚀 Démarrage du Metric Service")

    # Mode strict pour validation de configuration (défaut: False)
    STRICT_CONFIG_CHECK = os.getenv("STRICT_CONFIG_CHECK", "false").lower() == "true"

    # Vérification des configurations critiques
    config_issues = []

    # Vérifier Redis (optionnel mais recommandé)
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        config_issues.append("REDIS_URL non définie - cache désactivé")
        logger.warning("⚠️ REDIS_URL non définie - le service fonctionnera sans cache")

    if config_issues and STRICT_CONFIG_CHECK:
        error_msg = f"Configuration critique manquante: {', '.join(config_issues)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Initialiser Redis
    try:
        await cache_manager.connect()
        logger.info("✅ Cache Redis connecté")
    except Exception as e:
        logger.warning(f"⚠️ Échec connexion Redis: {e}")
        if STRICT_CONFIG_CHECK:
            raise

    logger.info("✅ Configuration validée avec succès")

    yield

    # Nettoyage
    logger.info("🛑 Arrêt du Metric Service")
    await cache_manager.disconnect()

# Application FastAPI
app = FastAPI(
    title="Harena Metric Service",
    description="Service de métriques financières avec prévisions Prophet",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Activée en développement local, désactivée en prod (gérée par Nginx)
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
if ENVIRONMENT == "dev":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5174",  # Frontend Docker
            "http://localhost:5173",  # Frontend Vite direct
            "http://localhost:3000",  # Autre port dev
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routes - 5 Métriques Essentielles (Specs conformes)
app.include_router(expenses.router, prefix="/api/v1/metrics/expenses", tags=["Métriques Dépenses"])
app.include_router(income.router, prefix="/api/v1/metrics/income", tags=["Métriques Revenus"])
app.include_router(coverage.router, prefix="/api/v1/metrics/coverage", tags=["Taux de Couverture"])

# Anciennes routes (à deprecier)
app.include_router(trends.router, prefix="/api/v1/metrics/trends", tags=["Trends (deprecated)"])
app.include_router(health.router, prefix="/api/v1/metrics/health", tags=["Health (deprecated)"])
app.include_router(patterns.router, prefix="/api/v1/metrics/patterns", tags=["Patterns (deprecated)"])

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "metric_service",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check détaillé avec vérification DB et cache"""
    from db_service.health import check_database_health
    from fastapi import status
    from fastapi.responses import JSONResponse

    # Vérifier la base de données
    db_healthy, db_message = check_database_health()

    # Vérifier le cache
    cache_status = await cache_manager.ping()

    # Le service est healthy si DB ET cache sont OK
    overall_healthy = db_healthy and cache_status

    # Préparer la réponse
    health_status = {
        "service": "metric_service",
        "status": "healthy" if overall_healthy else "unhealthy",
        "version": "1.0.0",
        "database": {
            "healthy": db_healthy,
            "message": db_message
        },
        "cache": {
            "connected": cache_status
        }
    }

    # Retourner 503 si un composant n'est pas accessible
    if not overall_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )

    return health_status

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("METRIC_SERVICE_PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
