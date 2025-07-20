"""
Middleware FastAPI pour performance, logging et error handling
Optimisé pour architecture hybride et contraintes Heroku
"""

import time
import uuid
import json
from typing import Callable, Dict, Any
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..config import settings
from ..utils.logging import get_logger, set_request_context, clear_request_context
from ..utils.performance import PerformanceMonitor, ResourceThrottler
from ..utils.metrics import RequestMetrics

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware logging structuré avec contexte requête
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_metrics = RequestMetrics()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Génération ID requête unique
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Configuration contexte logging
        user_id = None
        if hasattr(request.state, 'user') and request.state.user:
            user_id = request.state.user.get('id')
        
        set_request_context(request_id, user_id)
        
        # Ajout métadonnées request
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        try:
            # Log requête entrante
            logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={
                    "extra_data": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "query_params": str(request.query_params),
                        "user_agent": request.headers.get("user-agent", ""),
                        "client_ip": request.client.host if request.client else "unknown"
                    }
                }
            )
            
            # Traitement requête
            response = await call_next(request)
            
            # Calcul métriques
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Enregistrement métriques
            await self.request_metrics.record_endpoint_call(
                endpoint=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms
            )
            
            # Log requête terminée
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "extra_data": {
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                        "response_size": response.headers.get("content-length", 0)
                    }
                }
            )
            
            # Ajout headers response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(duration_ms)
            
            return response
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log erreur
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "extra_data": {
                        "request_id": request_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms
                    }
                },
                exc_info=True
            )
            
            # Enregistrement erreur dans métriques
            await self.request_metrics.record_endpoint_call(
                endpoint=request.url.path,
                status_code=500,
                duration_ms=duration_ms
            )
            
            raise
        
        finally:
            # Nettoyage contexte
            clear_request_context()


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware monitoring performance avec throttling intelligent
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.performance_monitor = PerformanceMonitor()
        self.throttler = ResourceThrottler()
        self.request_count = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        self.request_count += 1
        
        # Vérification throttling tous les 100 requêtes
        if self.request_count % 100 == 0:
            throttle_result = await self.throttler.check_and_throttle()
            
            if throttle_result.get("throttling_activated"):
                logger.warning(
                    "Resource throttling activated",
                    extra={"extra_data": throttle_result}
                )
        
        # Monitoring mémoire si nécessaire
        if self.request_count % 50 == 0:
            memory_check = await self.performance_monitor.check_memory_pressure()
            
            if memory_check["status"] != "normal":
                logger.warning(
                    f"Memory pressure detected: {memory_check['status']}",
                    extra={"extra_data": memory_check}
                )
                
                # GC préventif si pression mémoire
                if memory_check["should_gc"]:
                    gc_result = await self.performance_monitor.trigger_garbage_collection()
                    logger.info(
                        f"Preventive GC triggered, freed: {gc_result['memory_freed_mb']:.2f}MB"
                    )
        
        # Ajout métriques performance à la requête
        request.state.performance_monitor = self.performance_monitor
        
        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware gestion d'erreurs centralisée avec réponses structurées
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
            
        except HTTPException as e:
            # HTTPException FastAPI - passthrough avec logging
            logger.warning(
                f"HTTP Exception: {e.status_code} - {e.detail}",
                extra={
                    "extra_data": {
                        "status_code": e.status_code,
                        "detail": e.detail,
                        "path": request.url.path
                    }
                }
            )
            raise
            
        except ValueError as e:
            # Erreurs validation - 400 Bad Request
            logger.warning(f"Validation error: {str(e)}")
            
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "validation_error",
                        "message": "Invalid request data",
                        "details": str(e)
                    },
                    "request_id": getattr(request.state, 'request_id', 'unknown'),
                    "timestamp": time.time()
                }
            )
            
        except ConnectionError as e:
            # Erreurs connexion externe (Redis, DeepSeek) - 503 Service Unavailable
            logger.error(f"Connection error: {str(e)}")
            
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "service_unavailable",
                        "message": "External service temporarily unavailable",
                        "details": "Please try again in a few moments"
                    },
                    "request_id": getattr(request.state, 'request_id', 'unknown'),
                    "timestamp": time.time()
                }
            )
            
        except TimeoutError as e:
            # Timeouts - 504 Gateway Timeout
            logger.error(f"Timeout error: {str(e)}")
            
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "type": "timeout_error",
                        "message": "Request timeout",
                        "details": "The request took too long to process"
                    },
                    "request_id": getattr(request.state, 'request_id', 'unknown'),
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            # Erreurs inattendues - 500 Internal Server Error
            logger.error(
                f"Unexpected error: {str(e)}",
                exc_info=True,
                extra={
                    "extra_data": {
                        "error_type": type(e).__name__,
                        "path": request.url.path,
                        "method": request.method
                    }
                }
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "type": "internal_error",
                        "message": "An unexpected error occurred",
                        "details": "Please contact support if the problem persists" if settings.is_production() else str(e)
                    },
                    "request_id": getattr(request.state, 'request_id', 'unknown'),
                    "timestamp": time.time()
                }
            )


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware sécurité avec rate limiting et validation headers
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.rate_limit_store: Dict[str, Dict[str, Any]] = {}
        self.max_requests_per_minute = settings.RATE_LIMIT_PER_MINUTE
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Rate limiting par IP
        client_ip = request.client.host if request.client else "unknown"
        
        if not await self._check_rate_limit(client_ip):
            logger.warning(
                f"Rate limit exceeded for IP: {client_ip}",
                extra={"extra_data": {"client_ip": client_ip, "path": request.url.path}}
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "rate_limit_exceeded",
                        "message": f"Too many requests. Limit: {self.max_requests_per_minute}/minute",
                        "retry_after": 60
                    },
                    "timestamp": time.time()
                },
                headers={"Retry-After": "60"}
            )
        
        # Validation taille requête
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "type": "payload_too_large",
                        "message": f"Request too large. Max size: {settings.MAX_REQUEST_SIZE} bytes",
                        "current_size": int(content_length)
                    },
                    "timestamp": time.time()
                }
            )
        
        # Validation headers sécurité
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains" if settings.is_production() else None
        }
        
        response = await call_next(request)
        
        # Ajout headers sécurité
        for header, value in security_headers.items():
            if value:
                response.headers[header] = value
        
        return response
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Vérification rate limiting par IP"""
        current_time = time.time()
        window_start = current_time - 60  # Fenêtre 1 minute
        
        # Nettoyage ancien état
        if client_ip in self.rate_limit_store:
            self.rate_limit_store[client_ip]["requests"] = [
                req_time for req_time in self.rate_limit_store[client_ip]["requests"]
                if req_time > window_start
            ]
        else:
            self.rate_limit_store[client_ip] = {"requests": []}
        
        # Vérification limite
        current_requests = len(self.rate_limit_store[client_ip]["requests"])
        
        if current_requests >= self.max_requests_per_minute:
            return False
        
        # Enregistrement requête
        self.rate_limit_store[client_ip]["requests"].append(current_time)
        
        return True


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Middleware CORS personnalisé avec configuration dynamique
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.allowed_origins = settings.CORS_ORIGINS
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("origin")
        
        # Gestion preflight OPTIONS
        if request.method == "OPTIONS":
            response = Response()
            response.status_code = 200
        else:
            response = await call_next(request)
        
        # Configuration headers CORS
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Max-Age"] = "86400"  # 24h
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Vérification origine autorisée"""
        if "*" in self.allowed_origins:
            return True
        
        return origin in self.allowed_origins


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware compression intelligent pour optimiser bande passante Heroku
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.min_size = 1000  # 1KB minimum
        self.compressible_types = {
            "application/json",
            "text/plain", 
            "text/html",
            "text/css",
            "application/javascript"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Vérification si compression supportée
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        # Vérification type contenu
        content_type = response.headers.get("content-type", "").split(";")[0]
        if content_type not in self.compressible_types:
            return response
        
        # Vérification taille minimale
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.min_size:
            return response
        
        # Note: FastAPI GZipMiddleware gère déjà la compression
        # Ce middleware peut ajouter de la logique métier spécifique
        
        return response
