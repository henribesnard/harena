"""
Middleware FastAPI pour le Search Service.

Ce module contient les middlewares pour :
- Logging des requêtes et réponses
- Métriques de performance
- Gestion des erreurs
- Tracing des requêtes
- Monitoring de santé
"""

import time
import json
import uuid
from typing import Callable, Dict, Any, Optional
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from fastapi import Request, Response, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from ..utils.metrics import MetricsCollector
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION MIDDLEWARE ====================

def setup_middleware(app: FastAPI) -> None:
    """
    Configure tous les middlewares pour l'application FastAPI.
    
    Ordre important : du plus externe au plus interne.
    """
    settings = get_settings()
    
    # 1. CORS - Plus externe
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"]
    )
    
    # 2. GZip - Compression
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000  # Compresse si > 1KB
    )
    
    # 3. Middlewares personnalisés
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Tous les middlewares ont été configurés")

# ==================== MIDDLEWARE DE SÉCURITÉ ====================

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware pour les en-têtes de sécurité."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Traitement de la requête
        response = await call_next(request)
        
        # Ajout des en-têtes de sécurité
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # En-têtes API
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Service"] = "search-service"
        
        return response

# ==================== MIDDLEWARE DE TRAÇAGE ====================

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware pour le traçage des requêtes."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Génération ou récupération du request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Ajout du request ID au contexte
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Traitement de la requête
        response = await call_next(request)
        
        # Ajout du request ID à la réponse
        response.headers["X-Request-ID"] = request_id
        
        # Calcul du temps de traitement
        processing_time = time.time() - request.state.start_time
        response.headers["X-Response-Time"] = f"{processing_time:.3f}s"
        
        return response

# ==================== MIDDLEWARE DE MÉTRIQUES ====================

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware pour la collecte de métriques."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = MetricsCollector.get_instance()
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Enregistrement du début de la requête
        start_time = time.time()
        method = request.method
        path = request.url.path
        
        # Métriques de début
        self.metrics.increment_counter(
            "http_requests_total",
            {"method": method, "path": path}
        )
        
        self.metrics.increment_counter("http_requests_in_progress")
        
        try:
            # Traitement de la requête
            response = await call_next(request)
            
            # Métriques de fin
            duration = time.time() - start_time
            status_code = response.status_code
            
            # Histogram du temps de réponse
            self.metrics.record_histogram(
                "http_request_duration_seconds",
                duration,
                {"method": method, "path": path, "status": str(status_code)}
            )
            
            # Compteur par statut
            self.metrics.increment_counter(
                "http_responses_total",
                {"method": method, "path": path, "status": str(status_code)}
            )
            
            # Métriques spécifiques aux erreurs
            if status_code >= 400:
                self.metrics.increment_counter(
                    "http_errors_total",
                    {"method": method, "path": path, "status": str(status_code)}
                )
            
            return response
            
        except Exception as e:
            # Métriques d'erreur
            duration = time.time() - start_time
            
            self.metrics.increment_counter(
                "http_errors_total",
                {"method": method, "path": path, "status": "500", "error": type(e).__name__}
            )
            
            self.metrics.record_histogram(
                "http_request_duration_seconds",
                duration,
                {"method": method, "path": path, "status": "500"}
            )
            
            raise
            
        finally:
            # Décrémentation des requêtes en cours
            self.metrics.decrement_counter("http_requests_in_progress")

# ==================== MIDDLEWARE DE LOGGING ====================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour le logging des requêtes et réponses."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Informations de la requête
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Lecture du body pour le logging (si configuré)
        body = None
        if self.settings.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            body = await self._read_body(request)
        
        # Log de la requête entrante
        logger.info(
            "Requête entrante",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "content_length": request.headers.get("content-length"),
                "body_preview": body[:500] if body else None
            }
        )
        
        try:
            # Traitement de la requête
            response = await call_next(request)
            
            # Calcul du temps de traitement
            processing_time = time.time() - start_time
            
            # Log de la réponse
            logger.info(
                "Réponse sortante",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_time": f"{processing_time:.3f}s",
                    "response_size": response.headers.get("content-length")
                }
            )
            
            return response
            
        except Exception as e:
            # Log des erreurs
            processing_time = time.time() - start_time
            
            logger.error(
                "Erreur de traitement",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time": f"{processing_time:.3f}s"
                },
                exc_info=True
            )
            
            raise
    
    async def _read_body(self, request: Request) -> Optional[str]:
        """Lit le body de la requête en toute sécurité."""
        try:
            body = await request.body()
            if body:
                return body.decode('utf-8')
        except Exception as e:
            logger.warning(f"Impossible de lire le body: {e}")
        return None

# ==================== MIDDLEWARE DE GESTION D'ERREURS ====================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware pour la gestion centralisée des erreurs."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await call_next(request)
            
        except Exception as e:
            return await self._handle_error(request, e)
    
    async def _handle_error(self, request: Request, error: Exception) -> Response:
        """Gère les erreurs non capturées."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log de l'erreur
        logger.error(
            f"Erreur non gérée: {error}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__
            },
            exc_info=True
        )
        
        # Réponse d'erreur standardisée
        error_response = {
            "error": {
                "type": "internal_server_error",
                "message": "Une erreur interne est survenue",
                "code": 500
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(
            status_code=500,
            content=error_response,
            headers={"X-Request-ID": request_id}
        )

# ==================== MIDDLEWARE DE HEALTH CHECK ====================

class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware pour les health checks rapides."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Health check rapide pour les load balancers
        if request.url.path in ["/health", "/ping", "/status"]:
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "timestamp": datetime.utcnow().isoformat()}
            )
        
        return await call_next(request)

# ==================== UTILITAIRES ====================

def get_client_ip(request: Request) -> str:
    """Récupère l'adresse IP réelle du client."""
    # Vérification des en-têtes de proxy
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # IP directe
    return request.client.host if request.client else "unknown"

def is_health_check_path(path: str) -> bool:
    """Vérifie si le chemin est un health check."""
    health_paths = ["/health", "/ping", "/status", "/metrics", "/ready", "/live"]
    return path.lower() in health_paths

@asynccontextmanager
async def request_context(request: Request):
    """Context manager pour le contexte de requête."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    start_time = time.time()
    
    try:
        yield {
            "request_id": request_id,
            "start_time": start_time,
            "method": request.method,
            "path": request.url.path,
            "client_ip": get_client_ip(request)
        }
    finally:
        processing_time = time.time() - start_time
        logger.debug(f"Request {request_id} completed in {processing_time:.3f}s")