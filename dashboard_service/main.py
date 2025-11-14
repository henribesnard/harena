"""
Entry point for the Harena dashboard service.
Provides advanced aggregations for the user dashboard, inspired by the budget_profiling_service layout.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

# Ensure relative imports work when running `uvicorn main:app`
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from config_service.config import settings  # noqa: E402
from dashboard_service.api.routes.dashboard import router as dashboard_router  # noqa: E402
from dashboard_service.api.middleware.auth_middleware import JWTAuthMiddleware  # noqa: E402


logging.basicConfig(
  level=getattr(logging, os.getenv("DASHBOARD_SERVICE_LOG_LEVEL", "INFO"), logging.INFO),
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  stream=sys.stdout,
  force=True,
)
logger = logging.getLogger("dashboard_service")


class DashboardServiceLoader:
  def __init__(self) -> None:
    self.service_healthy = False
    self.initialization_error = None
    self.start_time = datetime.now(timezone.utc)
    self.service_config = {
      "version": "0.2.0",
      "features": [
        "advanced_aggregations",
        "cash_flow_series",
        "category_breakdown",
        "goal_tracking",
      ],
    }

  def initialize(self) -> bool:
    logger.info("🚀 Initialisation Dashboard Service")
    try:
      if not getattr(settings, "SECRET_KEY", None):
        raise RuntimeError("SECRET_KEY manquant pour valider les tokens JWT.")
      self.service_healthy = True
      return True
    except Exception as exc:
      self.initialization_error = str(exc)
      self.service_healthy = False
      logger.exception("Échec d'initialisation du dashboard_service: %s", exc)
      return False


service_loader = DashboardServiceLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
  service_loader.initialize()
  yield
  logger.info("🛑 Arrêt dashboard_service")


app = FastAPI(
  title="Harena Dashboard Service",
  description="Agrégations avancées pour le tableau de bord utilisateur.",
  version=service_loader.service_config["version"],
  lifespan=lifespan,
)


ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
if ENVIRONMENT in {"development", "dev", "local"}:
  logger.info("CORS ouvert (environnement développement)")
  app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )
else:
  allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
  logger.info("CORS restreint aux origines: %s", allowed_origins)
  app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
  )


app.add_middleware(JWTAuthMiddleware)
app.include_router(dashboard_router)


@app.get("/health")
def health_check():
  uptime = (datetime.now(timezone.utc) - service_loader.start_time).total_seconds()
  status = "healthy" if service_loader.service_healthy else "unhealthy"
  payload = {
    "status": status,
    "service": "dashboard_service",
    "version": service_loader.service_config["version"],
    "uptime_seconds": uptime,
    "features": service_loader.service_config["features"],
    "timestamp": datetime.now(timezone.utc).isoformat(),
  }
  if service_loader.initialization_error:
    payload["initialization_error"] = service_loader.initialization_error
  return JSONResponse(payload, status_code=200 if service_loader.service_healthy else 503)


@app.get("/")
def root():
  return {
    "service": "dashboard_service",
    "version": service_loader.service_config["version"],
    "status": "running",
    "documentation": "/docs",
  }


if __name__ == "__main__":
  import uvicorn

  host = os.getenv("DASHBOARD_SERVICE_HOST", "0.0.0.0")
  port = int(os.getenv("DASHBOARD_SERVICE_PORT", "3009"))
  uvicorn.run("main:app", host=host, port=port, reload=ENVIRONMENT in {"dev", "development", "local"})
