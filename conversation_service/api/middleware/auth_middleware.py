"""
Middleware d'authentification JWT optimisé avec sécurité avancée
"""
import logging
import time
from jose import jwt, JWTError
from jose.exceptions import ExpiredSignatureError, JWSSignatureError, JWSError
from typing import Optional, Dict, Any, Set
from datetime import datetime, timezone
from dataclasses import dataclass
from fastapi import Request, HTTPException
from fastapi.security.utils import get_authorization_scheme_param
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from config_service.config import settings
from conversation_service.utils.metrics_collector import metrics_collector

# Configuration du logger
logger = logging.getLogger("conversation_service.auth")


@dataclass
class AuthenticationResult:
    """Résultat d'authentification avec métadonnées"""
    success: bool
    user_id: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    token_payload: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None


class SecurityHeaders:
    """Headers de sécurité recommandés"""
    
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY", 
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    @classmethod
    def add_security_headers(cls, response) -> None:
        """Ajoute les headers de sécurité à la réponse"""
        for header, value in cls.SECURITY_HEADERS.items():
            response.headers[header] = value


class JWTValidator:
    """Validateur JWT avec cache et sécurité avancée"""
    
    def __init__(self):
        self.algorithm = getattr(settings, 'JWT_ALGORITHM', 'HS256')
        # Use the unified SECRET_KEY from settings for JWT verification
        self.secret_key = settings.SECRET_KEY

        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.blacklisted_tokens: Set[str] = set()
        
        # Statistiques
        self.cache_hits = 0
        self.cache_misses = 0
        self.validation_count = 0
        
        logger.info(f"JWTValidator initialisé - Algorithme: {self.algorithm}")
    
    def validate_token(self, token: str) -> AuthenticationResult:
        """
        Validation JWT avec cache et vérifications de sécurité
        
        Args:
            token: Token JWT à valider
            
        Returns:
            AuthenticationResult: Résultat de la validation
        """
        start_time = time.time()
        self.validation_count += 1
        
        try:
            # Vérification blacklist
            if token in self.blacklisted_tokens:
                metrics_collector.increment_counter("auth.validation.blacklisted")
                return AuthenticationResult(
                    success=False,
                    error_code="TOKEN_BLACKLISTED",
                    error_message="Token révoqué",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Vérification cache
            cached_result = self._get_cached_validation(token)
            if cached_result:
                self.cache_hits += 1
                metrics_collector.increment_counter("auth.validation.cache_hit")
                return AuthenticationResult(
                    success=True,
                    user_id=cached_result["user_id"],
                    token_payload=cached_result["payload"],
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            self.cache_misses += 1
            metrics_collector.increment_counter("auth.validation.cache_miss")
            
            # Décodage JWT
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": False,  # Pas d'audience pour Phase 1
                    "require": ["exp", "sub"]
                }
            )
            
            # Validation payload
            validation_error = self._validate_payload(payload)
            if validation_error:
                return AuthenticationResult(
                    success=False,
                    error_code="PAYLOAD_INVALID",
                    error_message=validation_error,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Extract user identifier: prefer the standard "sub" claim
            raw_user_id = payload.get("sub")
            if raw_user_id is None:
                # Fallback to legacy "user_id" if provided
                raw_user_id = payload.get("user_id")

            user_id = int(raw_user_id)
            
            # Vérifications de sécurité supplémentaires
            security_check = self._perform_security_checks(payload, token)
            if not security_check["valid"]:
                return AuthenticationResult(
                    success=False,
                    error_code="SECURITY_CHECK_FAILED",
                    error_message=security_check["reason"],
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Mise en cache
            self._cache_validation_result(token, user_id, payload)
            
            metrics_collector.increment_counter("auth.validation.success")
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                token_payload=payload,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
        except ExpiredSignatureError:
            metrics_collector.increment_counter("auth.validation.expired")
            return AuthenticationResult(
                success=False,
                error_code="TOKEN_EXPIRED",
                error_message="Token expiré",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except JWSSignatureError:
            metrics_collector.increment_counter("auth.validation.invalid_signature")
            return AuthenticationResult(
                success=False,
                error_code="INVALID_SIGNATURE",
                error_message="Signature token invalide",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        except JWSError:
            metrics_collector.increment_counter("auth.validation.decode_error")
            return AuthenticationResult(
                success=False,
                error_code="DECODE_ERROR",
                error_message="Format token invalide",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        except JWTError as e:
            metrics_collector.increment_counter("auth.validation.invalid_token")
            return AuthenticationResult(
                success=False,
                error_code="INVALID_TOKEN",
                error_message=f"Token invalide: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            metrics_collector.increment_counter("auth.validation.unexpected_error")
            logger.error(f"Erreur inattendue validation JWT: {str(e)}", exc_info=True)
            return AuthenticationResult(
                success=False,
                error_code="VALIDATION_ERROR",
                error_message="Erreur validation token",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _get_cached_validation(self, token: str) -> Optional[Dict[str, Any]]:
        """Récupération depuis cache avec vérification TTL"""
        if token not in self.token_cache:
            return None
        
        cached_entry = self.token_cache[token]
        cache_age = time.time() - cached_entry["cached_at"]
        
        if cache_age > self.cache_ttl:
            del self.token_cache[token]
            return None
        
        return cached_entry
    
    def _cache_validation_result(self, token: str, user_id: int, payload: Dict[str, Any]) -> None:
        """Mise en cache du résultat de validation"""
        # Limitation taille cache (LRU simple)
        if len(self.token_cache) > 1000:
            # Suppression des 100 plus anciens
            sorted_cache = sorted(
                self.token_cache.items(),
                key=lambda x: x[1]["cached_at"]
            )
            for old_token, _ in sorted_cache[:100]:
                del self.token_cache[old_token]
        
        self.token_cache[token] = {
            "user_id": user_id,
            "payload": payload,
            "cached_at": time.time()
        }
    
    def _validate_payload(self, payload: Dict[str, Any]) -> Optional[str]:
        """Validation contenu payload JWT"""
        # Vérification identifiant utilisateur (claim sub obligatoire)
        if "sub" not in payload:
            return "sub manquant dans le token"

        try:
            user_id = int(payload["sub"])
            if user_id <= 0:
                return "sub invalide (doit être > 0)"
        except (ValueError, TypeError):
            return "sub doit être un entier valide"
        
        # Vérification timestamps
        current_time = time.time()
        
        # iat (issued at) est optionnel mais ne doit pas être dans le futur s'il est présent
        if "iat" in payload:
            iat = payload["iat"]
            if iat > current_time + 300:  # 5 minutes de tolérance
                return "Token émis dans le futur"
        
        # Vérification exp (expiration)
        if "exp" in payload:
            exp = payload["exp"]
            if exp <= current_time:
                return "Token expiré selon payload"
        
        # Vérification durée de vie raisonnable
        if "iat" in payload and "exp" in payload:
            token_lifetime = payload["exp"] - payload["iat"]
            max_lifetime = 30 * 24 * 3600  # 30 jours maximum
            if token_lifetime > max_lifetime:
                return f"Durée de vie token excessive: {token_lifetime/3600:.1f}h"
        
        return None
    
    def _perform_security_checks(self, payload: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Vérifications de sécurité avancées"""
        
        # Vérification longueur token (protection contre tokens malformés)
        if len(token) > 2048:  # JWT typique < 1KB
            return {"valid": False, "reason": "Token trop long"}
        
        if len(token) < 50:  # JWT minimal
            return {"valid": False, "reason": "Token trop court"}
        
        # Vérification structure payload
        suspicious_keys = ["admin", "root", "system", "privilege", "role", "scope"]
        for key in suspicious_keys:
            if key in payload and payload[key]:
                logger.warning(f"Token avec clé suspecte: {key}")
        
        # Vérification identifiant utilisateur raisonnable
        raw_user_id = payload.get("sub")
        if raw_user_id is None:
            raw_user_id = payload.get("user_id")
        try:
            user_id = int(raw_user_id) if raw_user_id is not None else 0
        except (TypeError, ValueError):
            user_id = 0
        if user_id > 1000000:  # Ajustable selon la base utilisateur
            logger.warning(f"User ID inhabituel: {user_id}")
        
        return {"valid": True}
    
    def blacklist_token(self, token: str) -> None:
        """Ajoute un token à la blacklist"""
        self.blacklisted_tokens.add(token)
        if token in self.token_cache:
            del self.token_cache[token]
        
        # Limitation taille blacklist
        if len(self.blacklisted_tokens) > 5000:
            # Garder seulement les 2500 plus récents (approximation)
            tokens_list = list(self.blacklisted_tokens)
            self.blacklisted_tokens = set(tokens_list[-2500:])
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Statistiques de validation"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / max(total_requests, 1)) * 100
        
        return {
            "validation_count": self.validation_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cache_size": len(self.token_cache),
            "blacklist_size": len(self.blacklisted_tokens)
        }


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware authentification JWT optimisé avec sécurité renforcée"""
    
    def __init__(self, app):
        super().__init__(app)
        self.jwt_validator = JWTValidator()
        
        # Paths exclus de l'authentification
        self.excluded_paths = {
            "/health", "/health/live", "/health/ready",
            "/docs", "/openapi.json", "/redoc",
            "/metrics"
        }
        
        # Paths nécessitant authentification
        self.protected_patterns = ["/api/v1/conversation"]
        
        # Statistiques middleware
        self.requests_processed = 0
        self.auth_successes = 0
        self.auth_failures = 0
        
        logger.info("JWT Auth Middleware initialisé avec sécurité avancée")
    
    async def dispatch(self, request: Request, call_next):
        """Traitement authentification pour chaque requête"""
        
        self.requests_processed += 1
        start_time = time.time()
        
        try:
            # Vérification si path nécessite authentification
            if not self._requires_authentication(request.url.path):
                response = await call_next(request)
                SecurityHeaders.add_security_headers(response)
                return response
            
            # Log requête protégée
            client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            logger.debug(f"Auth required for {request.method} {request.url.path} from {client_ip}")
            
            # Authentification
            auth_result = await self._authenticate_request(request)
            
            if not auth_result.success:
                self.auth_failures += 1
                processing_time = (time.time() - start_time) * 1000
                
                # Log échec authentification
                logger.warning(
                    f"Auth failed - Code: {auth_result.error_code}, "
                    f"IP: {client_ip}, "
                    f"Path: {request.url.path}, "
                    f"Time: {processing_time:.1f}ms"
                )
                
                # Métriques
                metrics_collector.increment_counter(f"auth.failed.{auth_result.error_code.lower()}")
                metrics_collector.record_histogram("auth.processing_time", processing_time)
                
                # Réponse d'erreur sécurisée
                error_response = self._create_auth_error_response(auth_result)
                SecurityHeaders.add_security_headers(error_response)
                return error_response
            
            # Authentification réussie
            self.auth_successes += 1
            request.state.user_id = auth_result.user_id
            request.state.authenticated = True
            request.state.token_payload = auth_result.token_payload
            
            # Métriques succès
            processing_time = (time.time() - start_time) * 1000
            metrics_collector.increment_counter("auth.success")
            metrics_collector.record_histogram("auth.processing_time", processing_time)
            
            # Log succès (niveau debug)
            logger.debug(f"Auth success - User: {auth_result.user_id}, Time: {processing_time:.1f}ms")
            
            # Traitement requête authentifiée
            response = await call_next(request)
            
            # Headers de sécurité et informations utilisateur
            SecurityHeaders.add_security_headers(response)
            response.headers["X-User-ID"] = str(auth_result.user_id)
            
            return response
            
        except Exception as e:
            # Erreur inattendue dans le middleware
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Erreur middleware auth: {str(e)}", exc_info=True)
            
            metrics_collector.increment_counter("auth.middleware.error")
            
            # Réponse d'erreur générique
            error_response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Erreur interne d'authentification",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            SecurityHeaders.add_security_headers(error_response)
            return error_response
    
    def _requires_authentication(self, path: str) -> bool:
        """Détermine si le path nécessite une authentification"""
        # Vérification paths exclus
        if path in self.excluded_paths:
            return False
        
        # Vérification patterns exclus (ex: paths statiques)
        excluded_patterns = ["/static/", "/favicon.ico", "/.well-known/"]
        if any(pattern in path for pattern in excluded_patterns):
            return False
        
        # Vérification paths protégés
        return any(pattern in path for pattern in self.protected_patterns)
    
    async def _authenticate_request(self, request: Request) -> AuthenticationResult:
        """Authentification complète de la requête"""
        
        try:
            # Extraction token depuis header Authorization
            authorization = request.headers.get("Authorization")
            if not authorization:
                return AuthenticationResult(
                    success=False,
                    error_code="MISSING_AUTHORIZATION",
                    error_message="Header Authorization manquant"
                )
            
            # Parsing scheme et token
            scheme, token = get_authorization_scheme_param(authorization)
            if scheme.lower() != "bearer":
                return AuthenticationResult(
                    success=False,
                    error_code="INVALID_SCHEME",
                    error_message="Schema d'authentification invalide. Utilisez 'Bearer <token>'"
                )
            
            if not token:
                return AuthenticationResult(
                    success=False,
                    error_code="MISSING_TOKEN",
                    error_message="Token JWT manquant"
                )
            
            # Validation token
            validation_result = self.jwt_validator.validate_token(token)
            
            # Log temps de traitement si élevé
            if validation_result.processing_time_ms and validation_result.processing_time_ms > 100:
                logger.warning(f"Validation JWT lente: {validation_result.processing_time_ms:.1f}ms")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Erreur authentification requête: {str(e)}")
            return AuthenticationResult(
                success=False,
                error_code="AUTHENTICATION_ERROR", 
                error_message="Erreur lors de l'authentification"
            )
    
    def _create_auth_error_response(self, auth_result: AuthenticationResult) -> JSONResponse:
        """Crée une réponse d'erreur d'authentification sécurisée"""
        
        # Mapping codes d'erreur vers codes HTTP
        status_code_mapping = {
            "MISSING_AUTHORIZATION": 401,
            "INVALID_SCHEME": 401,
            "MISSING_TOKEN": 401,
            "TOKEN_EXPIRED": 401,
            "INVALID_SIGNATURE": 401,
            "DECODE_ERROR": 400,
            "INVALID_TOKEN": 401,
            "PAYLOAD_INVALID": 401,
            "TOKEN_BLACKLISTED": 401,
            "SECURITY_CHECK_FAILED": 401,
            "VALIDATION_ERROR": 401,
            "AUTHENTICATION_ERROR": 500
        }
        
        status_code = status_code_mapping.get(auth_result.error_code, 401)
        
        # Message d'erreur sécurisé (évite la fuite d'information)
        secure_messages = {
            "TOKEN_EXPIRED": "Token expiré",
            "INVALID_SIGNATURE": "Token invalide",
            "DECODE_ERROR": "Format de token invalide",
            "MISSING_AUTHORIZATION": "Authentification requise",
            "INVALID_SCHEME": "Type d'authentification invalide"
        }
        
        public_message = secure_messages.get(
            auth_result.error_code,
            "Authentification échouée"
        )
        
        response_content = {
            "detail": public_message,
            "error_code": auth_result.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Headers WWW-Authenticate pour conformité HTTP
        response = JSONResponse(
            status_code=status_code,
            content=response_content
        )
        
        if status_code == 401:
            response.headers["WWW-Authenticate"] = 'Bearer realm="Harena API"'
        
        return response
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Statistiques du middleware"""
        success_rate = (self.auth_successes / max(self.requests_processed, 1)) * 100
        
        return {
            "requests_processed": self.requests_processed,
            "auth_successes": self.auth_successes,
            "auth_failures": self.auth_failures,
            "success_rate_percent": round(success_rate, 2),
            "jwt_validator_stats": self.jwt_validator.get_validation_stats()
        }


# Fonctions helper pour utilisation dans les dépendances
async def get_current_user_id(request: Request) -> int:
    """
    Fonction helper pour récupérer user_id depuis request state
    
    Args:
        request: Requête FastAPI
        
    Returns:
        int: ID utilisateur authentifié
        
    Raises:
        HTTPException: Si utilisateur non authentifié
    """
    if not hasattr(request.state, 'user_id') or not request.state.user_id:
        metrics_collector.increment_counter("auth.helper.not_authenticated")
        raise HTTPException(
            status_code=401,
            detail="Utilisateur non authentifié"
        )
    
    return request.state.user_id


async def get_current_token_payload(request: Request) -> Dict[str, Any]:
    """Récupère le payload du token JWT depuis request state"""
    if not hasattr(request.state, 'token_payload') or not request.state.token_payload:
        raise HTTPException(
            status_code=401,
            detail="Payload token non disponible"
        )
    
    return request.state.token_payload


async def verify_user_id_match(request: Request, path_user_id: int) -> None:
    """
    Vérification que user_id du path correspond au token avec logging sécurité
    
    Args:
        request: Requête FastAPI
        path_user_id: User ID depuis l'URL
        
    Raises:
        HTTPException: Si user_id ne correspond pas
    """
    token_user_id = await get_current_user_id(request)
    
    if path_user_id != token_user_id:
        # Log sécurité sans données sensibles
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        
        logger.warning(
            f"User ID mismatch detected - Path: {path_user_id}, "
            f"Token: {token_user_id}, "
            f"IP: {client_ip}, "
            f"URL: {request.url.path}"
        )
        
        metrics_collector.increment_counter("auth.user_id_mismatch")
        
        raise HTTPException(
            status_code=403,
            detail={
                "error": "user_id dans l'URL ne correspond pas au token",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


def require_admin_role(request: Request) -> None:
    """
    Vérification rôle admin (pour futures extensions)
    
    Note: Non utilisé en Phase 1 mais préparé pour phases suivantes
    """
    # Récupération payload token
    if not hasattr(request.state, 'token_payload'):
        raise HTTPException(status_code=401, detail="Token payload indisponible")
    
    payload = request.state.token_payload
    
    # Vérification rôle admin (sera implémenté dans phases futures)
    admin_role = payload.get("role") == "admin"
    admin_scope = "admin" in payload.get("scopes", [])
    
    if not (admin_role or admin_scope):
        metrics_collector.increment_counter("auth.insufficient_privileges")
        raise HTTPException(
            status_code=403,
            detail="Privilèges administrateur requis"
        )