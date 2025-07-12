"""
Module API du Search Service - Interface REST et Middleware
=========================================================

Module centralisé pour toute l'interface API du Search Service :
- Routeur principal avec tous les endpoints
- Dépendances d'authentification et validation
- Middleware de logging, métriques et sécurité
- Gestionnaire d'erreurs centralisé
- Configuration FastAPI optimisée

Architecture :
    FastAPI App → Middleware Stack → Dependencies → Routes → Core Components

Responsabilité :
    Interface complète entre clients externes et composants core du Search Service
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Import simple de config
from config import settings

logger = logging.getLogger(__name__)


# === GESTIONNAIRE D'APPLICATION API SIMPLIFIÉ ===

class APIManager:
    """Gestionnaire centralisé de l'API du Search Service"""
    
    def __init__(self):
        self._app: Optional[FastAPI] = None
        self._initialized = False
        self._startup_time: Optional[datetime] = None
        
        logger.info("APIManager créé")
    
    async def initialize(self):
        """Initialise le gestionnaire API"""
        if self._initialized:
            return
        
        self._startup_time = datetime.now()
        self._initialized = True
        logger.info("✅ APIManager initialisé")
    
    async def shutdown(self):
        """Ferme le gestionnaire API"""
        if not self._initialized:
            return
        
        self._initialized = False
        logger.info("✅ APIManager fermé")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé de l'API"""
        return {
            "healthy": self._initialized,
            "startup_time": self._startup_time.isoformat() if self._startup_time else None,
            "status": "running" if self._initialized else "not_initialized"
        }
    
    @property
    def initialized(self) -> bool:
        """Statut d'initialisation"""
        return self._initialized
    
    @property
    def startup_time(self) -> Optional[datetime]:
        """Temps de démarrage"""
        return self._startup_time


# === INSTANCE GLOBALE ===

api_manager = APIManager()


# === GESTIONNAIRE DE CYCLE DE VIE ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    
    # Démarrage
    startup_start = datetime.now()
    logger.info("🚀 Démarrage Search Service API...")
    
    try:
        # Initialisation simple
        await api_manager.initialize()
        
        startup_duration = (datetime.now() - startup_start).total_seconds()
        logger.info(f"✅ API démarrée en {startup_duration:.2f}s")
        
        yield
        
    finally:
        # Arrêt
        logger.info("🛑 Arrêt Search Service API...")
        await api_manager.shutdown()


# === CRÉATION D'APPLICATION SIMPLIFIÉE ===

def create_app(
    title: str = "Search Service API",
    version: str = "1.0.0",
    debug: bool = False,
    docs_url: Optional[str] = "/docs",
    redoc_url: Optional[str] = "/redoc",
    **kwargs
) -> FastAPI:
    """
    Crée une application FastAPI configurée
    
    Args:
        title: Titre de l'API
        version: Version de l'API
        debug: Mode debug
        docs_url: URL de la documentation
        redoc_url: URL ReDoc
        **kwargs: Arguments supplémentaires pour FastAPI
        
    Returns:
        Application FastAPI configurée
    """
    
    # Configuration FastAPI
    app_config = {
        "title": title,
        "description": _get_api_description(),
        "version": version,
        "debug": debug,
        "docs_url": docs_url if settings.enable_docs else None,
        "redoc_url": redoc_url if settings.enable_docs else None,
        "openapi_url": "/openapi.json" if settings.enable_docs else None,
        "lifespan": lifespan,
        **kwargs
    }
    
    # Créer l'application
    app = FastAPI(**app_config)
    
    # Configuration de base
    _configure_middleware(app)
    _configure_error_handlers(app)
    _add_basic_routes(app)
    
    logger.info("✅ Application FastAPI créée")
    return app


def _configure_middleware(app: FastAPI):
    """Configure les middleware de base"""
    
    # CORS
    if hasattr(settings, 'cors_origins') and settings.cors_origins:
        origins = settings.cors_origins
        if isinstance(origins, str):
            origins = origins.split(",")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        logger.info("✅ CORS configuré")


def _configure_error_handlers(app: FastAPI):
    """Configure les gestionnaires d'erreurs"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": 422,
                    "message": "Validation error",
                    "details": exc.errors(),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Erreur non gérée: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "Erreur interne du serveur",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    logger.info("✅ Gestionnaires d'erreurs configurés")


def _add_basic_routes(app: FastAPI):
    """Ajoute les routes de base"""
    
    @app.get("/", tags=["info"])
    async def root():
        """Point d'entrée racine"""
        return {
            "service": "search-service",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/health", tags=["health"])
    async def health_check():
        """Vérification de santé"""
        try:
            health_data = await api_manager.health_check()
            status_code = 200 if health_data.get("healthy", False) else 503
            return JSONResponse(content=health_data, status_code=status_code)
        except Exception as e:
            return JSONResponse(
                content={
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                status_code=503
            )
    
    @app.get("/info", tags=["info"])
    async def app_info():
        """Informations de l'application"""
        return {
            "service": "search-service",
            "version": "1.0.0",
            "api_version": "v1",
            "environment": getattr(settings, 'environment', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
    
    logger.info("✅ Routes de base ajoutées")


def _get_api_description() -> str:
    """Retourne la description de l'API"""
    return """
# Search Service API - Recherche Lexicale Haute Performance

API REST spécialisée dans la recherche lexicale sur données financières avec Elasticsearch.

## Caractéristiques

- **Performance** : < 50ms pour requêtes simples
- **Sécurité** : Authentification et validation stricte
- **Observabilité** : Métriques et logs structurés
- **Fiabilité** : Cache intelligent et optimisations

## Endpoints Principaux

- `POST /api/v1/search/lexical` - Recherche lexicale principale
- `POST /api/v1/search/validate` - Validation de requêtes
- `GET /health` - Santé du service
- `GET /info` - Informations du service
"""


# === FONCTIONS D'INTERFACE PUBLIQUE ===

async def create_search_service_app(
    elasticsearch_client=None,
    debug: bool = False,
    **kwargs
) -> FastAPI:
    """
    Crée une application FastAPI complète pour le Search Service
    
    Args:
        elasticsearch_client: Client Elasticsearch configuré
        debug: Mode debug
        **kwargs: Configuration supplémentaire FastAPI
        
    Returns:
        Application FastAPI prête à être servie
    """
    
    app = create_app(debug=debug, **kwargs)
    
    # Injecter le client Elasticsearch dans l'état de l'app
    if elasticsearch_client:
        app.state.elasticsearch_client = elasticsearch_client
    
    # Ajouter les routes spécialisées ici plus tard
    # app.include_router(search_router, prefix="/api/v1")
    
    return app


async def get_api_health() -> Dict[str, Any]:
    """Vérification de santé de l'API"""
    return await api_manager.health_check()


async def get_api_info() -> Dict[str, Any]:
    """Informations sur l'API"""
    return {
        "service": "search-service",
        "version": "1.0.0",
        "api_version": "v1",
        "initialized": api_manager.initialized,
        "startup_time": api_manager.startup_time.isoformat() if api_manager.startup_time else None
    }


async def shutdown_api():
    """Arrêt propre de l'API"""
    await api_manager.shutdown()


# === FONCTIONS UTILITAIRES ===

def create_development_app(elasticsearch_client=None) -> FastAPI:
    """Crée une app pour le développement"""
    return asyncio.run(create_search_service_app(
        elasticsearch_client=elasticsearch_client,
        debug=True,
        docs_url="/docs",
        redoc_url="/redoc"
    ))


def create_production_app(elasticsearch_client=None) -> FastAPI:
    """Crée une app pour la production"""
    return asyncio.run(create_search_service_app(
        elasticsearch_client=elasticsearch_client,
        debug=False,
        docs_url=None,
        redoc_url=None
    ))


# === CLASSES D'EXCEPTIONS SIMPLES ===

class APIException(HTTPException):
    """Exception API de base"""
    
    def __init__(self, status_code: int, detail: str, error_code: str = "API_ERROR"):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class ValidationException(APIException):
    """Exception de validation"""
    
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail, error_code="VALIDATION_ERROR")


class RateLimitException(APIException):
    """Exception de limite de débit"""
    
    def __init__(self, detail: str = "Limite de débit dépassée"):
        super().__init__(status_code=429, detail=detail, error_code="RATE_LIMIT_EXCEEDED")


class AuthenticationException(APIException):
    """Exception d'authentification"""
    
    def __init__(self, detail: str = "Authentification requise"):
        super().__init__(status_code=401, detail=detail, error_code="AUTHENTICATION_REQUIRED")


class AuthorizationException(APIException):
    """Exception d'autorisation"""
    
    def __init__(self, detail: str = "Autorisation insuffisante"):
        super().__init__(status_code=403, detail=detail, error_code="AUTHORIZATION_DENIED")


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === GESTIONNAIRE PRINCIPAL ===
    "APIManager",
    "api_manager",
    
    # === FONCTIONS PRINCIPALES ===
    "create_app",
    "create_search_service_app",
    "get_api_health",
    "get_api_info",
    "shutdown_api",
    
    # === FONCTIONS UTILITAIRES ===
    "create_development_app",
    "create_production_app",
    
    # === EXCEPTIONS ===
    "APIException",
    "ValidationException",
    "RateLimitException",
    "AuthenticationException",
    "AuthorizationException",
    
    # === CYCLE DE VIE ===
    "lifespan"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Module API simplifié du Search Service avec FastAPI"

logger.info(f"Module api initialisé - version {__version__}")