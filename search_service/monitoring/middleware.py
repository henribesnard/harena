"""
Middleware amélioré pour le service de recherche avec logging et métriques détaillés.
"""
import logging
import time
import json
from typing import Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("search_service.middleware")
access_logger = logging.getLogger("search_service.access")


class SearchMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware pour capturer les métriques et logs de recherche."""
    
    def __init__(self, app, monitor=None):
        super().__init__(app)
        self.monitor = monitor
        self.request_count = 0
    
    async def dispatch(self, request: Request, call_next):
        """Traite chaque requête avec logging et métriques détaillés."""
        self.request_count += 1
        request_id = f"req_{self.request_count}_{int(time.time()*1000)}"
        
        # Démarrer le timing
        start_time = time.time()
        
        # Extraire les informations de la requête
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log de début de requête
        logger.info(f"🌐 [{request_id}] {method} {path} from {client_ip}")
        
        # Log des paramètres de requête si présents
        if query_params:
            logger.debug(f"🔍 [{request_id}] Query params: {query_params}")
        
        # Variables pour capturer les détails de la réponse
        response = None
        status_code = 500
        response_size = 0
        error_type = None
        search_type = None
        
        try:
            # Capturer le body pour les requêtes POST (recherches)
            request_body = None
            if method == "POST" and path == "/search":
                request_body = await self._get_request_body(request)
                search_type = self._extract_search_type(request_body)
                
                logger.info(f"🔍 [{request_id}] Type de recherche: {search_type}")
                
                if request_body and logger.isEnabledFor(logging.DEBUG):
                    # Log sécurisé du body (sans données sensibles)
                    safe_body = self._sanitize_request_body(request_body)
                    logger.debug(f"🔍 [{request_id}] Request body: {safe_body}")
            
            # Exécuter la requête
            response = await call_next(request)
            status_code = response.status_code
            
            # Calculer le temps de traitement
            processing_time = time.time() - start_time
            processing_time_ms = processing_time * 1000
            
            # Estimer la taille de la réponse
            response_size = self._estimate_response_size(response)
            
            # Déterminer le succès
            is_success = 200 <= status_code < 400
            
            # Log de fin de requête
            status_icon = "✅" if is_success else "❌"
            logger.info(
                f"{status_icon} [{request_id}] {status_code} completed in {processing_time_ms:.2f}ms"
            )
            
            # Log d'accès structuré
            access_logger.info(
                f"{client_ip} - [{request_id}] \"{method} {path}\" "
                f"{status_code} {response_size} {processing_time_ms:.3f}ms"
            )
            
            # Enregistrer les métriques si c'est une recherche
            if self.monitor and path == "/search" and method == "POST":
                self.monitor.record_search(
                    search_type=search_type or "unknown",
                    success=is_success,
                    response_time_ms=processing_time_ms,
                    error_type=error_type
                )
            
            # Log des performances si lent
            if processing_time_ms > 1000:  # Plus de 1 seconde
                logger.warning(
                    f"🐌 [{request_id}] Requête lente: {processing_time_ms:.2f}ms "
                    f"pour {method} {path}"
                )
            
            # Log détaillé des résultats de recherche en mode debug
            if (path == "/search" and is_success and 
                logger.isEnabledFor(logging.DEBUG)):
                await self._log_search_results(request_id, response)
            
        except Exception as e:
            # Calculer le temps même en cas d'erreur
            processing_time = time.time() - start_time
            processing_time_ms = processing_time * 1000
            
            error_type = type(e).__name__
            
            logger.error(
                f"💥 [{request_id}] Exception après {processing_time_ms:.2f}ms: "
                f"{error_type}: {str(e)}"
            )
            logger.error(f"📍 [{request_id}] Détails", exc_info=True)
            
            # Enregistrer l'erreur dans les métriques
            if self.monitor and path == "/search" and method == "POST":
                self.monitor.record_search(
                    search_type=search_type or "unknown",
                    success=False,
                    response_time_ms=processing_time_ms,
                    error_type=error_type
                )
            
            # Créer une réponse d'erreur
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "type": error_type
                }
            )
            status_code = 500
        
        # Log final de métriques
        self._log_final_metrics(request_id, method, path, status_code, 
                               processing_time_ms, response_size)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extrait l'IP du client en tenant compte des proxies."""
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # IP directe
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def _get_request_body(self, request: Request) -> Dict[str, Any]:
        """Récupère et parse le body de la requête."""
        try:
            body = await request.body()
            if body:
                return json.loads(body.decode())
            return {}
        except Exception as e:
            logger.warning(f"⚠️ Impossible de parser le body: {e}")
            return {}
    
    def _extract_search_type(self, request_body: Dict[str, Any]) -> str:
        """Extrait le type de recherche du body."""
        if not request_body:
            return "unknown"
        
        search_type = request_body.get("search_type", "hybrid")
        return search_type
    
    def _sanitize_request_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie le body pour les logs (supprime les données sensibles)."""
        if not body:
            return {}
        
        # Créer une copie pour ne pas modifier l'original
        safe_body = {}
        
        # Champs autorisés pour les logs
        allowed_fields = {
            "query", "search_type", "limit", "filters", 
            "include_highlights", "include_explanations"
        }
        
        for key, value in body.items():
            if key in allowed_fields:
                safe_body[key] = value
        
        return safe_body
    
    def _estimate_response_size(self, response: Response) -> int:
        """Estime la taille de la réponse."""
        try:
            # Pour JSONResponse, on peut estimer via le content
            if hasattr(response, 'body'):
                return len(response.body)
            
            # Estimation par défaut
            return 0
        except:
            return 0
    
    async def _log_search_results(self, request_id: str, response: Response):
        """Log détaillé des résultats de recherche."""
        try:
            if hasattr(response, 'body') and response.body:
                # Parse le JSON de réponse
                response_data = json.loads(response.body.decode())
                
                results = response_data.get("results", [])
                total_found = response_data.get("total_found", 0)
                search_time = response_data.get("search_time_ms", 0)
                
                logger.debug(
                    f"🔍 [{request_id}] Résultats: {len(results)}/{total_found} "
                    f"en {search_time}ms"
                )
                
                # Log des scores des top résultats
                for i, result in enumerate(results[:3]):
                    score = result.get("score", 0)
                    source = result.get("source", "unknown")
                    logger.debug(f"🔍 [{request_id}] #{i+1}: {source} (score: {score:.3f})")
                
        except Exception as e:
            logger.debug(f"⚠️ [{request_id}] Impossible de parser les résultats: {e}")
    
    def _log_final_metrics(self, request_id: str, method: str, path: str, 
                          status_code: int, processing_time_ms: float, 
                          response_size: int):
        """Log final des métriques de la requête."""
        # Métriques structurées pour monitoring externe
        metrics_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "processing_time_ms": processing_time_ms,
            "response_size_bytes": response_size,
            "timestamp": time.time()
        }
        
        # Log en format metrics (pour parsing par des outils externes)
        metrics_logger = logging.getLogger("search_service.metrics.requests")
        metrics_logger.info(
            f"request.completed,"
            f"method={method},"
            f"path={path.replace('/', '_')},"
            f"status={status_code},"
            f"time={processing_time_ms:.3f},"
            f"size={response_size}"
        )


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware spécifique pour les health checks."""
    
    def __init__(self, app, monitor=None):
        super().__init__(app)
        self.monitor = monitor
    
    async def dispatch(self, request: Request, call_next):
        """Traite les health checks avec logging minimal."""
        path = request.url.path
        
        # Health checks simples sans logging détaillé
        if path in ["/health", "/", "/ping"]:
            start_time = time.time()
            
            try:
                response = await call_next(request)
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Log minimal pour les health checks
                if processing_time_ms > 100:  # Log seulement si lent
                    logger.debug(f"🩺 Health check lent: {processing_time_ms:.2f}ms")
                
                return response
                
            except Exception as e:
                logger.error(f"❌ Health check failed: {e}")
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "error": str(e)}
                )
        
        # Autres requêtes passent au middleware suivant
        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware pour la gestion centralisée des erreurs."""
    
    async def dispatch(self, request: Request, call_next):
        """Gère les erreurs de manière centralisée."""
        try:
            return await call_next(request)
            
        except Exception as e:
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            # Log détaillé de l'erreur
            logger.error(f"💥 [{request_id}] Unhandled exception: {type(e).__name__}")
            logger.error(f"📍 [{request_id}] Path: {request.method} {request.url.path}")
            logger.error(f"📍 [{request_id}] Error: {str(e)}", exc_info=True)
            
            # Déterminer le type d'erreur pour la réponse
            if "timeout" in str(e).lower():
                status_code = 504
                error_type = "timeout"
            elif "connection" in str(e).lower():
                status_code = 503
                error_type = "service_unavailable"
            else:
                status_code = 500
                error_type = "internal_error"
            
            # Réponse d'erreur structurée
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "type": error_type,
                        "message": "An error occurred while processing your request",
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                }
            )


def setup_middleware(app, monitor=None):
    """Configure tous les middlewares pour l'application."""
    logger.info("🔧 Configuration des middlewares...")
    
    # Ordre important : les middlewares sont appliqués en ordre inverse
    
    # 1. Gestion d'erreurs (en dernier, pour capturer toutes les erreurs)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # 2. Health checks (avant les métriques pour éviter le spam)
    app.add_middleware(HealthCheckMiddleware, monitor=monitor)
    
    # 3. Métriques et logging (en premier, pour tout capturer)
    app.add_middleware(SearchMetricsMiddleware, monitor=monitor)
    
    logger.info("✅ Middlewares configurés")


# Configuration du logging pour les métriques
def setup_metrics_logging():
    """Configure les loggers spécialisés pour les métriques."""
    
    # Logger pour les métriques de requests
    requests_logger = logging.getLogger("search_service.metrics.requests")
    requests_handler = logging.StreamHandler()
    requests_handler.setFormatter(
        logging.Formatter('%(asctime)s - METRICS - %(message)s')
    )
    requests_logger.addHandler(requests_handler)
    requests_logger.setLevel(logging.INFO)
    
    # Logger pour l'accès (format Apache-like)
    access_logger = logging.getLogger("search_service.access")
    access_handler = logging.StreamHandler()
    access_handler.setFormatter(
        logging.Formatter('%(asctime)s - ACCESS - %(message)s')
    )
    access_logger.addHandler(access_handler)
    access_logger.setLevel(logging.INFO)
    
    logger.info("📊 Logging des métriques configuré")