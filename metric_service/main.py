"""
Metric Service - FastAPI Application
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Optional
import os

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

    # Initialiser Redis
    await cache_manager.connect()

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

# CORS - Désactivée car gérée par Nginx
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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
    """Health check détaillé"""
    cache_status = await cache_manager.ping()

    return {
        "service": "metric_service",
        "status": "healthy" if cache_status else "degraded",
        "cache": "connected" if cache_status else "disconnected",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("METRIC_SERVICE_PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
