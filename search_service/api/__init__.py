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

# Imports des composants API
from .routes import (
    router, admin_router, 
    initialize_routes, shutdown_routes,
    customize_openapi_schema, custom_http_exception_handler
)
from .dependencies import (
    # Gestionnaires principaux
    auth_manager, rate_limiter,
    
    # Dépendances
    get_authenticated_user, validate_rate_limit, validate_search_request,
    check_service_health,
    
    # Factories de dépendances
    create_search_dependencies, create_validation_dependencies,
    create_metrics_dependencies, create_admin_dependencies,
    
    # Exceptions
    APIException, ValidationException, RateLimitException,
    AuthenticationException, AuthorizationException,
    
    # Fonctions d'initialisation
    initialize_dependencies, shutdown_dependencies
)
from .middleware import (
    # Classes de middleware
    StructuredLoggingMiddleware, MetricsMiddleware, SecurityMiddleware,
    ErrorHandlingMiddleware, CompressionMiddleware, GlobalRateLimitMiddleware,
    
    # Factory et utilitaires
    create_middleware_stack, initialize_middleware, get_request_context
)

# Imports des autres modules
from core import initialize_core_components, shutdown_core_components, get_core_health
from utils import initialize_utils, shutdown_utils, get_utils_health
from config import settings


logger = logging.getLogger(__name__)


# === GESTIONNAIRE D'APPLICATION API ===

class APIManager:
    """Gestionnaire centralisé de l'API du Search Service"""
    
    def __init__(self):
        self._app: Optional[FastAPI] = None
        self._initialized = False
        self._startup_time: Optional[datetime] = None
        
        logger.info("APIManager créé")
    
    async def create_app(self, **kwargs) -> FastAPI:
        """
        Crée et configure l'application FastAPI complète
        
        Args:
            **kwargs: Configuration supplémentaire pour FastAPI
            
        Returns:
            Application FastAPI configurée et prête
        """
        
        if self._app is not None:
            logger.warning("Application FastAPI déjà créée")
            return self._app
        
        logger.info("Création de l'application FastAPI...")
        
        # Configuration FastAPI
        app_config = {
            "title": "Search Service API",
            "description": self._get_api_description(),
            "version": "1.0.0",
            "docs_url": "/docs" if settings.enable_docs else None,
            "redoc_url": "/redoc" if settings.enable_docs else None,
            "openapi_url": "/openapi.json" if settings.enable_docs else None,
            "lifespan": self._lifespan_handler,
            **kwargs
        }
        
        # Créer l'application
        self._app = FastAPI(**app_config)
        
        # Configurer l'application
        await self._configure_app()
        
        logger.info("✅ Application FastAPI créée et configurée")
        return self._app
    
    @asynccontextmanager
    async def _lifespan_handler(self, app: FastAPI):
        """Gestionnaire de cycle de vie de l'application"""
        
        # Démarrage
        try:
            await self._startup()
            yield
        finally:
            # Arrêt
            await self._shutdown()
    
    async def _startup(self):
        """Séquence de démarrage de l'API"""
        
        self._startup_time = datetime.now()
        logger.info("🚀 Démarrage du Search Service API...")
        
        try:
            # 1. Initialiser les utilitaires
            logger.info("1️⃣ Initialisation des utilitaires...")
            utils_result = await initialize_utils()
            if utils_result.get("status") != "success":
                raise RuntimeError(f"Utils initialization failed: {utils_result}")
            
            # 2. Initialiser les composants core
            logger.info("2️⃣ Initialisation des composants core...")
            
            # Client Elasticsearch (sera injecté depuis main.py)
            elasticsearch_client = getattr(self._app.state, 'elasticsearch_client', None)
            if not elasticsearch_client:
                raise RuntimeError("Elasticsearch client not provided")
            
            core_result = await initialize_core_components(
                elasticsearch_client=elasticsearch_client
            )
            if core_result.get("status") != "success":
                raise RuntimeError(f"Core initialization failed: {core_result}")
            
            # 3. Initialiser les dépendances API
            logger.info("3️⃣ Initialisation des dépendances API...")
            await initialize_dependencies()
            
            # 4. Initialiser les middleware
            logger.info("4️⃣ Initialisation des middleware...")
            await initialize_middleware()
            
            # 5. Initialiser les routes
            logger.info("5️⃣ Initialisation des routes...")
            await initialize_routes()
            
            # 6. Vérification de santé finale
            logger.info("6️⃣ Vérification de santé finale...")
            health_status = await self._perform_startup_health_check()
            
            if health_status["overall_status"] != "healthy":
                logger.warning(f"Service started with warnings: {health_status}")
            
            # Enregistrer l'état de démarrage
            self._app.state.startup_time = self._startup_time
            self._app.state.startup_health = health_status
            self._initialized = True
            
            startup_duration = (datetime.now() - self._startup_time).total_seconds()
            logger.info(f"✅ Search Service API démarré avec succès en {startup_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage: {e}")
            raise
    
    async def _shutdown(self):
        """Séquence d'arrêt de l'API"""
        
        logger.info("🛑 Arrêt du Search Service API...")
        
        try:
            # Arrêter dans l'ordre inverse du démarrage
            
            # 1. Arrêter les routes
            logger.info("1️⃣ Arrêt des routes...")
            await shutdown_routes()
            
            # 2. Arrêter les dépendances
            logger.info("2️⃣ Arrêt des dépendances...")
            await shutdown_dependencies()
            
            # 3. Arrêter les composants core
            logger.info("3️⃣ Arrêt des composants core...")
            await shutdown_core_components()
            
            # 4. Arrêter les utilitaires
            logger.info("4️⃣ Arrêt des utilitaires...")
            await shutdown_utils()
            
            self._initialized = False
            
            if self._startup_time:
                uptime = (datetime.now() - self._startup_time).total_seconds()
                logger.info(f"✅ Search Service API arrêté après {uptime:.1f}s d'uptime")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'arrêt: {e}")
    
    async def _configure_app(self):
        """Configure l'application FastAPI"""
        
        # 1. Ajouter les middleware dans l'ordre correct
        await self._add_middleware()
        
        # 2. Configurer les gestionnaires d'erreurs
        self._configure_error_handlers()
        
        # 3. Ajouter les routes
        self._add_routes()
        
        # 4. Personnaliser le schéma OpenAPI
        self._customize_openapi()
        
        # 5. Ajouter les events handlers
        self._add_event_handlers()
    
    async def _add_middleware(self):
        """Ajoute les middleware dans le bon ordre"""
        
        # Middleware CORS (en premier)
        if settings.cors_origins:
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=settings.cors_origins.split(","),
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE"],
                allow_headers=["*"],
            )
        
        # Middleware de trusted hosts
        if hasattr(settings, 'trusted_hosts') and settings.trusted_hosts:
            self._app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=settings.trusted_hosts.split(",")
            )
        
        # Ajouter la pile de middleware personnalisés
        middleware_stack = create_middleware_stack()
        
        for middleware_class in reversed(middleware_stack):  # Ordre inverse pour ASGI
            self._app.add_middleware(middleware_class)
        
        logger.info(f"✅ {len(middleware_stack) + 1} middleware ajoutés")
    
    def _configure_error_handlers(self):
        """Configure les gestionnaires d'erreurs globaux"""
        
        # Gestionnaire pour les exceptions HTTP personnalisées
        @self._app.exception_handler(APIException)
        async def api_exception_handler(request: Request, exc: APIException):
            correlation_id = getattr(request.state, 'correlation_id', 'unknown')
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.status_code,
                        "message": exc.detail,
                        "error_code": exc.error_code,
                        "correlation_id": correlation_id,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )
        
        # Gestionnaire pour les erreurs de validation Pydantic
        @self._app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            correlation_id = getattr(request.state, 'correlation_id', 'unknown')
            
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": 422,
                        "message": "Request validation failed",
                        "error_code": "VALIDATION_ERROR",
                        "correlation_id": correlation_id,
                        "timestamp": datetime.now().isoformat(),
                        "details": exc.errors()
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )
        
        # Gestionnaire pour les exceptions HTTP Starlette
        @self._app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(request: Request, exc: StarletteHTTPException):
            correlation_id = getattr(request.state, 'correlation_id', 'unknown')
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.status_code,
                        "message": exc.detail,
                        "correlation_id": correlation_id,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )
        
        logger.info("✅ Gestionnaires d'erreurs configurés")
    
    def _add_routes(self):
        """Ajoute les routes à l'application"""
        
        # Router principal
        self._app.include_router(router)
        
        # Endpoint de santé au niveau racine (sans authentification)
        @self._app.get("/health", tags=["health"])
        async def root_health_check():
            """Endpoint de santé simple au niveau racine"""
            
            try:
                # Vérification rapide sans authentification
                if not self._initialized:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "status": "starting",
                            "message": "Service is initializing",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                
                return JSONResponse(
                    content={
                        "status": "healthy",
                        "service": "search-service",
                        "version": "1.0.0",
                        "timestamp": datetime.now().isoformat(),
                        "uptime_seconds": (
                            datetime.now() - self._startup_time
                        ).total_seconds() if self._startup_time else 0
                    }
                )
                
            except Exception as e:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        # Endpoint de version
        @self._app.get("/version", tags=["info"])
        async def get_version():
            """Informations de version du service"""
            
            return {
                "service": "search-service",
                "version": "1.0.0",
                "api_version": "v1",
                "build_time": getattr(settings, 'build_time', 'unknown'),
                "commit_hash": getattr(settings, 'commit_hash', 'unknown'),
                "environment": getattr(settings, 'environment', 'unknown')
            }
        
        logger.info("✅ Routes ajoutées à l'application")
    
    def _customize_openapi(self):
        """Personnalise le schéma OpenAPI"""
        
        if settings.enable_docs:
            custom_schema = customize_openapi_schema()
            
            def custom_openapi():
                if self._app.openapi_schema:
                    return self._app.openapi_schema
                
                openapi_schema = self._app.openapi()
                openapi_schema.update(custom_schema)
                self._app.openapi_schema = openapi_schema
                return self._app.openapi_schema
            
            self._app.openapi = custom_openapi
            
            logger.info("✅ Schéma OpenAPI personnalisé")
    
    def _add_event_handlers(self):
        """Ajoute les gestionnaires d'événements"""
        
        # Middleware pour ajouter des informations d'état
        @self._app.middleware("http")
        async def add_state_info(request: Request, call_next):
            # Ajouter des informations globales à l'état de la requête
            request.state.api_manager = self
            request.state.app_initialized = self._initialized
            request.state.startup_time = self._startup_time
            
            response = await call_next(request)
            return response
        
        logger.info("✅ Gestionnaires d'événements ajoutés")
    
    async def _perform_startup_health_check(self) -> Dict[str, Any]:
        """Effectue une vérification de santé au démarrage"""
        
        try:
            health_checks = await asyncio.gather(
                get_core_health(),
                get_utils_health(),
                return_exceptions=True
            )
            
            core_health, utils_health = health_checks
            
            # Analyser les résultats
            overall_healthy = True
            
            if isinstance(core_health, Exception):
                core_health = {"status": "error", "error": str(core_health)}
                overall_healthy = False
            elif core_health.get("system_status") != "healthy":
                overall_healthy = False
            
            if isinstance(utils_health, Exception):
                utils_health = {"status": "error", "error": str(utils_health)}
                overall_healthy = False
            elif utils_health.get("system_status") != "healthy":
                overall_healthy = False
            
            return {
                "overall_status": "healthy" if overall_healthy else "degraded",
                "components": {
                    "core": core_health,
                    "utils": utils_health
                },
                "startup_time": self._startup_time.isoformat() if self._startup_time else None
            }
            
        except Exception as e:
            logger.error(f"Startup health check failed: {e}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "startup_time": self._startup_time.isoformat() if self._startup_time else None
            }
    
    def _get_api_description(self) -> str:
        """Retourne la description de l'API"""
        
        return """
# Search Service API - Recherche Lexicale Haute Performance

API REST spécialisée dans la recherche lexicale sur données financières avec Elasticsearch.

## Caractéristiques

- **Performance** : < 50ms pour requêtes simples, < 200ms pour complexes
- **Sécurité** : Authentification multi-mode, isolation utilisateur stricte
- **Observabilité** : Métriques détaillées, tracing distribué, logs structurés
- **Fiabilité** : Cache intelligent, optimisations automatiques, fallbacks

## Endpoints Principaux

- `POST /api/v1/search/lexical` - Recherche lexicale principale
- `POST /api/v1/search/validate` - Validation de requêtes
- `GET /api/v1/health` - Santé détaillée du service
- `GET /api/v1/metrics` - Export de métriques

## Authentification

Trois modes supportés :
1. **Bearer Token** : `Authorization: Bearer <token>`
2. **API Key** : Headers `X-User-Id` + `X-API-Key`
3. **Mode dev** : Header `X-User-Id` uniquement (développement)

## Rate Limiting

Limites par tier : Standard (100/min), Premium (500/min), Enterprise (2000/min)
        """
    
    @property
    def app(self) -> Optional[FastAPI]:
        """Accès à l'application FastAPI"""
        return self._app
    
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


# === FONCTIONS D'INTERFACE PUBLIQUE ===

async def create_search_service_app(elasticsearch_client=None, **kwargs) -> FastAPI:
    """
    Crée une application FastAPI complète pour le Search Service
    
    Args:
        elasticsearch_client: Client Elasticsearch configuré
        **kwargs: Configuration supplémentaire FastAPI
        
    Returns:
        Application FastAPI prête à être servie
    """
    
    app = await api_manager.create_app(**kwargs)
    
    # Injecter le client Elasticsearch dans l'état de l'app
    if elasticsearch_client:
        app.state.elasticsearch_client = elasticsearch_client
    
    return app


async def get_api_health() -> Dict[str, Any]:
    """Vérification de santé de l'API"""
    
    if not api_manager.initialized:
        return {
            "status": "not_initialized",
            "message": "API not yet initialized"
        }
    
    try:
        return await api_manager._perform_startup_health_check()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def get_api_info() -> Dict[str, Any]:
    """Informations sur l'API"""
    
    return {
        "service": "search-service",
        "version": "1.0.0",
        "api_version": "v1",
        "initialized": api_manager.initialized,
        "startup_time": api_manager.startup_time.isoformat() if api_manager.startup_time else None,
        "components": {
            "routes": "loaded",
            "dependencies": "configured",
            "middleware": "active",
            "error_handlers": "configured"
        }
    }


async def shutdown_api():
    """Arrêt propre de l'API"""
    
    if api_manager.initialized:
        await api_manager._shutdown()


# === FONCTIONS UTILITAIRES ===

def create_development_app(elasticsearch_client=None) -> FastAPI:
    """Crée une app pour le développement avec configurations simplifiées"""
    
    import asyncio
    
    async def _create():
        return await create_search_service_app(
            elasticsearch_client=elasticsearch_client,
            debug=True,
            docs_url="/docs",
            redoc_url="/redoc"
        )
    
    return asyncio.run(_create())


def create_production_app(elasticsearch_client=None) -> FastAPI:
    """Crée une app pour la production avec sécurité renforcée"""
    
    import asyncio
    
    async def _create():
        return await create_search_service_app(
            elasticsearch_client=elasticsearch_client,
            debug=False,
            docs_url=None,  # Pas de docs en production
            redoc_url=None
        )
    
    return asyncio.run(_create())


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === GESTIONNAIRE PRINCIPAL ===
    "APIManager",
    "api_manager",
    
    # === FONCTIONS PRINCIPALES ===
    "create_search_service_app",
    "get_api_health",
    "get_api_info",
    "shutdown_api",
    
    # === FONCTIONS UTILITAIRES ===
    "create_development_app",
    "create_production_app",
    
    # === COMPOSANTS API (re-exports) ===
    # Routes
    "router",
    "admin_router",
    
    # Dependencies
    "auth_manager",
    "rate_limiter",
    "get_authenticated_user",
    "validate_rate_limit",
    "validate_search_request",
    "check_service_health",
    
    # Middleware
    "StructuredLoggingMiddleware",
    "MetricsMiddleware",
    "SecurityMiddleware",
    "ErrorHandlingMiddleware",
    "CompressionMiddleware",
    "GlobalRateLimitMiddleware",
    
    # Exceptions
    "APIException",
    "ValidationException",
    "RateLimitException",
    "AuthenticationException",
    "AuthorizationException",
    
    # Utilitaires
    "get_request_context"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Module API complet du Search Service avec FastAPI"

logger.info(f"Module api initialisé - version {__version__}")