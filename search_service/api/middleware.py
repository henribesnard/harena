"""
Middleware FastAPI pour le Search Service
========================================

Middleware spécialisés pour l'API REST du Search Service :
- Logging structuré des requêtes et réponses
- Métriques automatiques de performance API
- Tracing distribué avec correlation IDs
- Gestion d'erreurs centralisée avec reporting
- Headers de sécurité et cache
- Monitoring santé en temps réel
- Limitation de taille de requête

Architecture :
    Request → Security → Logging → Metrics → Rate Limiting → Business Logic → Response
"""

import logging
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import traceback
import gzip

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.status import (
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_408_REQUEST_TIMEOUT
)
import psutil

from search_service.utils.metrics import (
    metrics_collector, api_metrics, performance_profiler

)
from search_service.api.dependencies import APIException
from search_service.config import settings


logger = logging.getLogger(__name__)


# === MIDDLEWARE DE LOGGING STRUCTURÉ ===

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour le logging structuré des requêtes API"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.excluded_paths = {"/health", "/metrics", "/docs", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Générer un correlation ID unique pour cette requête
        correlation_id = str(uuid.uuid4())[:8]
        request.state.correlation_id = correlation_id
        
        # Informations de base de la requête
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        # Préparer le contexte de logging
        log_context = {
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown"),
            "content_length": request.headers.get("content-length", 0)
        }
        
        # Logger le début de la requête (sauf pour les endpoints exclus)
        if request.url.path not in self.excluded_paths:
            logger.info("Request started", extra=log_context)
        
        try:
            # Traiter la requête
            response = await call_next(request)
            
            # Calculer les métriques de performance
            duration_ms = (time.time() - start_time) * 1000
            
            # Enrichir le contexte avec la réponse
            response_context = {
                **log_context,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "response_size": response.headers.get("content-length", 0)
            }
            
            # Ajouter le correlation ID aux headers de réponse
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            # Logger la fin de la requête
            if request.url.path not in self.excluded_paths:
                if response.status_code >= 400:
                    logger.warning("Request completed with error", extra=response_context)
                else:
                    logger.info("Request completed successfully", extra=response_context)
            
            return response
            
        except Exception as e:
            # Calculer la durée même en cas d'erreur
            duration_ms = (time.time() - start_time) * 1000
            
            # Contexte d'erreur
            error_context = {
                **log_context,
                "duration_ms": round(duration_ms, 2),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            
            # Logger l'erreur
            logger.error("Request failed with exception", extra=error_context)
            
            # Re-lever l'exception pour que FastAPI puisse la gérer
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP réelle du client en tenant compte des proxies"""
        
        # Vérifier les headers de proxy dans l'ordre de priorité
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback sur l'IP de connexion directe
        return request.client.host if request.client else "unknown"


# === MIDDLEWARE DE MÉTRIQUES AUTOMATIQUES ===

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware pour la collecte automatique de métriques API"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.system_metrics_interval = 30  # secondes
        self._last_system_metrics = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Collecter les métriques système périodiquement
        await self._collect_system_metrics()
        
        # Métriques de requête
        start_time = time.time()
        endpoint = self._normalize_endpoint(request.url.path)
        
        # Extraire les informations utilisateur si disponibles
        user_id = getattr(request.state, 'user_id', None)
        auth_method = getattr(request.state, 'auth_method', 'unknown')
        
        try:
            # Traiter la requête avec profiling
            with performance_profiler.profile_operation(
                f"api_request_{endpoint}",
                {"method": request.method, "endpoint": endpoint}
            ):
                response = await call_next(request)
            
            # Calculer les métriques
            duration_ms = (time.time() - start_time) * 1000
            
            # Enregistrer les métriques de succès
            self._record_request_metrics(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_id=user_id,
                auth_method=auth_method,
                success=True
            )
            
            return response
            
        except Exception as e:
            # Calculer la durée même en cas d'erreur
            duration_ms = (time.time() - start_time) * 1000
            
            # Enregistrer les métriques d'erreur
            self._record_request_metrics(
                method=request.method,
                endpoint=endpoint,
                status_code=500,  # Erreur serveur par défaut
                duration_ms=duration_ms,
                user_id=user_id,
                auth_method=auth_method,
                success=False,
                error_type=type(e).__name__
            )
            
            raise
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalise le chemin pour regrouper les métriques"""
        
        # Remplacer les IDs par des placeholders
        import re
        
        # Remplacer les UUIDs
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        
        # Remplacer les IDs numériques
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Nettoyer les paramètres de query
        if '?' in path:
            path = path.split('?')[0]
        
        return path
    
    def _record_request_metrics(self, method: str, endpoint: str, status_code: int,
                               duration_ms: float, user_id: Optional[int],
                               auth_method: str, success: bool,
                               error_type: Optional[str] = None):
        """Enregistre les métriques d'une requête"""
        
        tags = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
            "auth_method": auth_method,
            "success": str(success)
        }
        
        if user_id:
            tags["user_id"] = str(user_id)
        
        if error_type:
            tags["error_type"] = error_type
        
        # Métriques de base
        metrics_collector.record("api_request_duration_ms", duration_ms, tags)
        metrics_collector.increment("api_request_count", tags=tags)
        
        # Métriques de succès/erreur
        if success:
            metrics_collector.increment("api_request_success_count", tags=tags)
        else:
            metrics_collector.increment("api_error_count", tags=tags)
        
        # Métriques spécialisées par endpoint
        if endpoint.startswith("/search"):
            api_metrics.record_search_api_call(
                endpoint=endpoint,
                duration_ms=duration_ms,
                success=success,
                user_id=user_id
            )
        elif endpoint.startswith("/validate"):
            api_metrics.record_validation_api_call(
                duration_ms=duration_ms,
                success=success,
                user_id=user_id
            )
    
    async def _collect_system_metrics(self):
        """Collecte les métriques système périodiquement"""
        
        now = time.time()
        if now - self._last_system_metrics < self.system_metrics_interval:
            return
        
        try:
            # Métriques processus
            process = psutil.Process()
            
            # Mémoire
            memory_info = process.memory_info()
            metrics_collector.set_gauge("api_memory_usage_bytes", memory_info.rss)
            
            # CPU
            cpu_percent = process.cpu_percent()
            metrics_collector.set_gauge("api_cpu_usage_percent", cpu_percent)
            
            # Connexions réseau
            connections = len(process.connections())
            metrics_collector.set_gauge("api_network_connections", connections)
            
            # Métriques globales système si disponibles
            try:
                system_memory = psutil.virtual_memory()
                metrics_collector.set_gauge("system_memory_usage_percent", system_memory.percent)
                
                system_cpu = psutil.cpu_percent()
                metrics_collector.set_gauge("system_cpu_usage_percent", system_cpu)
            except:
                pass  # Ignorer si pas de permissions
            
            self._last_system_metrics = now
            
        except Exception as e:
            logger.warning(f"Erreur collecte métriques système: {e}")


# === MIDDLEWARE DE SÉCURITÉ ===

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware pour les headers de sécurité et protections"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.max_request_size = getattr(settings, 'max_request_size_bytes', 10 * 1024 * 1024)  # 10MB
        self.request_timeout = getattr(settings, 'request_timeout_seconds', 30)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Vérifier la taille de la requête
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request entity too large",
                    "max_size_bytes": self.max_request_size,
                    "received_size_bytes": int(content_length)
                }
            )
        
        try:
            # Appliquer un timeout global
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.request_timeout
            )
            
            # Ajouter les headers de sécurité
            self._add_security_headers(response)
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout: {request.method} {request.url.path} "
                f"exceeded {self.request_timeout}s"
            )
            return JSONResponse(
                status_code=HTTP_408_REQUEST_TIMEOUT,
                content={
                    "error": "Request timeout",
                    "timeout_seconds": self.request_timeout
                }
            )
    
    def _add_security_headers(self, response: Response):
        """Ajoute les headers de sécurité standard"""
        
        # Headers de sécurité de contenu
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Headers API
        response.headers["X-API-Version"] = "1.0"
        response.headers["X-Service"] = "search-service"
        
        # Headers de cache selon le type de contenu
        if hasattr(response, 'headers') and response.headers.get("content-type", "").startswith("application/json"):
            # Pas de cache pour les réponses JSON dynamiques
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"


# === MIDDLEWARE DE GESTION D'ERREURS ===

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware pour la gestion centralisée d'erreurs"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
            
        except APIException as e:
            # Les APIException sont déjà gérées, les laisser passer
            raise
            
        except HTTPException as e:
            # Enregistrer l'erreur HTTP dans les métriques
            self._record_error_metrics(request, e.status_code, "HTTPException")
            raise
            
        except Exception as e:
            # Erreur inattendue - la capturer et formater
            correlation_id = getattr(request.state, 'correlation_id', 'unknown')
            
            # Logger l'erreur complète
            logger.error(
                f"Unhandled exception in request {correlation_id}",
                exc_info=True,
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            # Enregistrer dans les métriques
            self._record_error_metrics(request, 500, type(e).__name__)
            
            # Retourner une réponse d'erreur standardisée
            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now().isoformat(),
                    "message": "An unexpected error occurred" if not settings.debug_mode else str(e)
                },
                headers={"X-Correlation-ID": correlation_id}
            )
    
    def _record_error_metrics(self, request: Request, status_code: int, error_type: str):
        """Enregistre les métriques d'erreur"""
        
        endpoint = request.url.path
        user_id = getattr(request.state, 'user_id', None)
        
        tags = {
            "endpoint": endpoint,
            "status_code": str(status_code),
            "error_type": error_type,
            "method": request.method
        }
        
        if user_id:
            tags["user_id"] = str(user_id)
        
        metrics_collector.increment("api_error_count", tags=tags)
        
        # Incrémenter les compteurs d'erreur spécifiques
        if status_code >= 500:
            metrics_collector.increment("api_server_error_count", tags=tags)
        elif status_code >= 400:
            metrics_collector.increment("api_client_error_count", tags=tags)


# === MIDDLEWARE DE COMPRESSION ===

class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware pour la compression automatique des réponses"""
    
    def __init__(self, app: ASGIApp, minimum_size: int = 1024):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compressible_types = {
            "application/json",
            "text/plain",
            "text/html",
            "text/css",
            "text/javascript",
            "application/javascript"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Vérifier si le client accepte la compression
        accept_encoding = request.headers.get("accept-encoding", "")
        supports_gzip = "gzip" in accept_encoding.lower()
        
        if not supports_gzip:
            return await call_next(request)
        
        response = await call_next(request)
        
        # Vérifier si la réponse doit être compressée
        if self._should_compress(response):
            return self._compress_response(response)
        
        return response
    
    def _should_compress(self, response: Response) -> bool:
        """Détermine si la réponse doit être compressée"""
        
        # Vérifier si déjà compressée
        if response.headers.get("content-encoding"):
            return False
        
        # Vérifier le type de contenu
        content_type = response.headers.get("content-type", "").split(";")[0]
        if content_type not in self.compressible_types:
            return False
        
        # Vérifier la taille minimum
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return False
        
        return True
    
    def _compress_response(self, response: Response) -> Response:
        """Compresse le contenu de la réponse"""
        
        try:
            # Récupérer le contenu
            if hasattr(response, 'body'):
                content = response.body
            else:
                # Pour les réponses streaming, ne pas compresser
                return response
            
            # Compresser avec gzip
            compressed_content = gzip.compress(content)
            
            # Vérifier que la compression est bénéfique
            if len(compressed_content) >= len(content):
                return response
            
            # Créer une nouvelle réponse avec le contenu compressé
            compressed_response = Response(
                content=compressed_content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
            
            # Mettre à jour les headers
            compressed_response.headers["content-encoding"] = "gzip"
            compressed_response.headers["content-length"] = str(len(compressed_content))
            
            # Enregistrer les métriques de compression
            compression_ratio = len(content) / len(compressed_content)
            metrics_collector.record("api_compression_ratio", compression_ratio)
            metrics_collector.increment("api_compression_count")
            
            return compressed_response
            
        except Exception as e:
            logger.warning(f"Erreur lors de la compression: {e}")
            return response


# === MIDDLEWARE DE RATE LIMITING GLOBAL ===

class GlobalRateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting global pour protéger le service"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.global_limit = getattr(settings, 'global_rate_limit_per_minute', 1000)
        self.request_counts = {}
        self.window_size = 60  # 1 minute
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Rate limiting global simple (en mémoire)
        current_minute = int(time.time() // self.window_size)
        
        if current_minute not in self.request_counts:
            # Nettoyer les anciennes fenêtres
            old_minutes = [m for m in self.request_counts.keys() if m < current_minute - 1]
            for old_minute in old_minutes:
                del self.request_counts[old_minute]
            
            self.request_counts[current_minute] = 0
        
        # Vérifier la limite
        if self.request_counts[current_minute] >= self.global_limit:
            logger.warning(
                f"Global rate limit exceeded: {self.request_counts[current_minute]}/{self.global_limit}"
            )
            
            metrics_collector.increment("api_global_rate_limit_exceeded")
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Global rate limit exceeded",
                    "limit": self.global_limit,
                    "window_seconds": self.window_size,
                    "retry_after": self.window_size
                },
                headers={"Retry-After": str(self.window_size)}
            )
        
        # Incrémenter le compteur
        self.request_counts[current_minute] += 1
        
        return await call_next(request)


# === FACTORY DE MIDDLEWARE ===

def create_middleware_stack() -> List[BaseHTTPMiddleware]:
    """Crée la pile de middleware dans le bon ordre"""
    
    middleware_stack = []
    
    # 1. Rate limiting global (première protection)
    if getattr(settings, 'enable_global_rate_limiting', True):
        middleware_stack.append(GlobalRateLimitMiddleware)
    
    # 2. Sécurité (validation taille, timeout, headers)
    middleware_stack.append(SecurityMiddleware)
    
    # 3. Gestion d'erreurs (capture toutes les erreurs)
    middleware_stack.append(ErrorHandlingMiddleware)
    
    # 4. Logging structuré (trace toutes les requêtes)
    middleware_stack.append(StructuredLoggingMiddleware)
    
    # 5. Métriques (collecte automatique)
    if getattr(settings, 'enable_metrics_middleware', True):
        middleware_stack.append(MetricsMiddleware)
    
    # 6. Compression (optimisation finale)
    if getattr(settings, 'enable_compression', True):
        middleware_stack.append(CompressionMiddleware)
    
    return middleware_stack


# === FONCTIONS UTILITAIRES ===

async def initialize_middleware():
    """Initialise les middleware (à appeler au démarrage)"""
    
    logger.info("Initialisation des middleware API...")
    
    try:
        # Initialiser les systèmes de métriques si pas déjà fait
        if not hasattr(metrics_collector, '_initialized'):
            from utils.metrics import initialize_metrics_system
            initialize_metrics_system()
        
        logger.info("✅ Middleware API initialisés")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation middleware: {e}")
        raise


def get_request_context(request: Request) -> Dict[str, Any]:
    """Extrait le contexte d'une requête pour logging/debugging"""
    
    return {
        "correlation_id": getattr(request.state, 'correlation_id', None),
        "user_id": getattr(request.state, 'user_id', None),
        "auth_method": getattr(request.state, 'auth_method', None),
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else None
    }


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === MIDDLEWARE CLASSES ===
    "StructuredLoggingMiddleware",
    "MetricsMiddleware",
    "SecurityMiddleware",
    "ErrorHandlingMiddleware",
    "CompressionMiddleware",
    "GlobalRateLimitMiddleware",
    
    # === FACTORY ET UTILITAIRES ===
    "create_middleware_stack",
    "initialize_middleware",
    "get_request_context"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Middleware FastAPI spécialisés pour le Search Service"

logger.info(f"Module api.middleware chargé - version {__version__}")