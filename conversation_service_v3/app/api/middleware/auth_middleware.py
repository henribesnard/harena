"""
Middleware d'authentification JWT pour conversation_service_v3
Compatible avec user_service et conversation_service
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

from ...config.settings import settings

# Configuration du logger
logger = logging.getLogger("conversation_service_v3.auth")


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
    """Validateur JWT compatible user_service"""

    def __init__(self):
        self.algorithm = getattr(settings, 'JWT_ALGORITHM', 'HS256')
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
        Validation JWT compatible user_service

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
                logger.debug(f"Cache hit pour user {cached_result['user_id']}")
                return AuthenticationResult(
                    success=True,
                    user_id=cached_result["user_id"],
                    token_payload=cached_result["payload"],
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            self.cache_misses += 1

            # Décodage JWT
            try:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                    options={
                        "verify_signature": True,
                        "verify_exp": True,
                        "verify_iat": False,
                        "verify_aud": False,
                        "require": ["exp", "sub"]
                    }
                )
            except Exception as decode_error:
                logger.error(f"Erreur décodage JWT: {str(decode_error)}")
                raise decode_error

            # Validation payload
            validation_error = self._validate_payload(payload)
            if validation_error:
                logger.warning(f"Payload invalide: {validation_error}")
                return AuthenticationResult(
                    success=False,
                    error_code="PAYLOAD_INVALID",
                    error_message=validation_error,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Extract user identifier
            raw_user_id = payload.get("sub")
            if raw_user_id is None:
                raw_user_id = payload.get("user_id")

            try:
                user_id = int(raw_user_id)
            except (ValueError, TypeError):
                logger.error(f"Impossible de convertir user_id: {raw_user_id}")
                return AuthenticationResult(
                    success=False,
                    error_code="INVALID_USER_ID",
                    error_message=f"User ID invalide: {raw_user_id}",
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Vérifications de sécurité
            security_check = self._perform_security_checks(payload, token)
            if not security_check["valid"]:
                logger.warning(f"Échec vérification sécurité: {security_check['reason']}")
                return AuthenticationResult(
                    success=False,
                    error_code="SECURITY_CHECK_FAILED",
                    error_message=security_check["reason"],
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Mise en cache
            self._cache_validation_result(token, user_id, payload)

            logger.debug(f"Token validé avec succès pour user_id: {user_id}")
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                token_payload=payload,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        except ExpiredSignatureError:
            logger.debug("Token expiré")
            return AuthenticationResult(
                success=False,
                error_code="TOKEN_EXPIRED",
                error_message="Token expiré",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except JWSSignatureError:
            logger.error("Signature token invalide")
            return AuthenticationResult(
                success=False,
                error_code="INVALID_SIGNATURE",
                error_message="Signature token invalide",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except JWSError:
            logger.error("Format token invalide")
            return AuthenticationResult(
                success=False,
                error_code="DECODE_ERROR",
                error_message="Format token invalide",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except JWTError as e:
            logger.error(f"Erreur JWT: {str(e)}")
            return AuthenticationResult(
                success=False,
                error_code="INVALID_TOKEN",
                error_message=f"Token invalide: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
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
        """Validation payload compatible avec user_service"""
        # Vérification claim sub (obligatoire)
        if "sub" not in payload:
            return "Claim 'sub' manquant dans le token"

        try:
            raw_sub = payload["sub"]
            if isinstance(raw_sub, str):
                user_id = int(raw_sub)
            else:
                user_id = int(raw_sub)

            if user_id <= 0:
                return "Subject invalide (doit être > 0)"
        except (ValueError, TypeError):
            return f"Subject invalide: {raw_sub}"

        # Vérification timestamps
        current_time = time.time()

        # exp (expiration) - critique
        if "exp" in payload:
            try:
                exp = int(payload["exp"])
                if exp <= current_time - 60:  # 1 minute de grâce
                    return "Token expiré"
            except (ValueError, TypeError):
                return "Timestamp 'exp' invalide"

        return None

    def _perform_security_checks(self, payload: Dict[str, Any], token: str) -> Dict[str, Any]:
        """Vérifications de sécurité"""
        # Vérification longueur token
        if len(token) > 4096:
            return {"valid": False, "reason": "Token trop long"}

        if len(token) < 50:
            return {"valid": False, "reason": "Token trop court"}

        # Vérification user_id
        raw_user_id = payload.get("sub") or payload.get("user_id")
        try:
            user_id = int(raw_user_id) if raw_user_id is not None else 0
        except (TypeError, ValueError):
            return {"valid": False, "reason": "User ID invalide dans le payload"}

        if user_id <= 0:
            return {"valid": False, "reason": "User ID doit être positif"}

        return {"valid": True}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware authentification JWT compatible user_service"""

    def __init__(self, app):
        super().__init__(app)
        self.jwt_validator = JWTValidator()

        # Paths exclus de l'authentification
        self.excluded_paths = {
            "/", "/health", "/docs", "/openapi.json", "/redoc",
            "/api/v3/conversation/health",
            "/api/v3/conversation/status",
            "/api/v3/conversation/metrics"
        }

        # Patterns exclus
        self.excluded_patterns = ["/static/", "/.well-known/"]

        # Paths nécessitant authentification
        self.protected_patterns = ["/api/v3/conversation/"]

        # Statistiques
        self.requests_processed = 0
        self.auth_successes = 0
        self.auth_failures = 0

        logger.info("JWT Auth Middleware initialisé")

    async def dispatch(self, request: Request, call_next):
        """Traitement authentification pour chaque requête"""
        self.requests_processed += 1
        start_time = time.time()
        path = request.url.path

        try:
            # Allow OPTIONS requests (CORS preflight)
            if request.method == "OPTIONS":
                logger.debug(f"OPTIONS request for {path} - skipping authentication")
                response = await call_next(request)
                SecurityHeaders.add_security_headers(response)
                return response

            # Vérification si path nécessite authentification
            if not self._requires_authentication(path):
                logger.debug(f"Path public: {path}")
                response = await call_next(request)
                SecurityHeaders.add_security_headers(response)
                return response

            # Log requête protégée
            client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            logger.debug(f"Auth requise pour {request.method} {path} depuis {client_ip}")

            # Authentification
            auth_result = await self._authenticate_request(request)

            if not auth_result.success:
                self.auth_failures += 1
                processing_time = (time.time() - start_time) * 1000

                logger.warning(
                    f"Auth échouée - Code: {auth_result.error_code}, "
                    f"IP: {client_ip}, Path: {path}, "
                    f"Time: {processing_time:.1f}ms"
                )

                # Réponse d'erreur sécurisée
                error_response = self._create_auth_error_response(auth_result)
                SecurityHeaders.add_security_headers(error_response)
                return error_response

            # Authentification réussie
            self.auth_successes += 1
            request.state.user_id = auth_result.user_id
            request.state.authenticated = True
            request.state.token_payload = auth_result.token_payload

            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Auth réussie - User: {auth_result.user_id}, Time: {processing_time:.1f}ms")

            # Traitement requête authentifiée
            response = await call_next(request)

            # Headers de sécurité
            SecurityHeaders.add_security_headers(response)
            response.headers["X-User-ID"] = str(auth_result.user_id)

            return response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Erreur middleware auth: {str(e)}", exc_info=True)

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
        """Détermine si un path nécessite authentification"""
        # 1. Paths publics explicites
        if path in self.excluded_paths:
            return False

        # 2. Patterns publics
        for pattern in self.excluded_patterns:
            if pattern in path:
                return False

        # 3. Vérification patterns protégés avec exceptions
        for pattern in self.protected_patterns:
            if pattern in path:
                # Exception pour les endpoints de monitoring
                if any(monitor in path for monitor in ["health", "metrics", "status"]):
                    return False
                return True

        # Par défaut, pas d'authentification requise
        return False

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
                    error_message="Schéma d'authentification invalide. Utilisez 'Bearer <token>'"
                )

            if not token:
                return AuthenticationResult(
                    success=False,
                    error_code="MISSING_TOKEN",
                    error_message="Token JWT manquant"
                )

            # Validation token
            validation_result = self.jwt_validator.validate_token(token)

            return validation_result

        except Exception as e:
            logger.error(f"Erreur authentification requête: {str(e)}")
            return AuthenticationResult(
                success=False,
                error_code="AUTHENTICATION_ERROR",
                error_message="Erreur lors de l'authentification"
            )

    def _create_auth_error_response(self, auth_result: AuthenticationResult) -> JSONResponse:
        """Crée une réponse d'erreur d'authentification user-friendly"""
        # Mapping codes d'erreur vers codes HTTP
        status_code_mapping = {
            "MISSING_AUTHORIZATION": 401,
            "INVALID_SCHEME": 401,
            "MISSING_TOKEN": 401,
            "TOKEN_EXPIRED": 401,
            "INVALID_SIGNATURE": 401,
            "DECODE_ERROR": 401,
            "INVALID_TOKEN": 401,
            "PAYLOAD_INVALID": 401,
            "TOKEN_BLACKLISTED": 401,
            "SECURITY_CHECK_FAILED": 401,
            "VALIDATION_ERROR": 401,
            "INVALID_USER_ID": 401,
            "AUTHENTICATION_ERROR": 500
        }

        # Messages d'erreur français user-friendly
        user_friendly_messages = {
            "MISSING_AUTHORIZATION": "Authentification requise",
            "INVALID_SCHEME": "Schéma d'authentification invalide",
            "MISSING_TOKEN": "Token d'authentification manquant",
            "TOKEN_EXPIRED": "Votre session a expiré. Veuillez vous reconnecter.",
            "INVALID_SIGNATURE": "Token invalide",
            "DECODE_ERROR": "Token malformé",
            "INVALID_TOKEN": "Token invalide",
            "PAYLOAD_INVALID": "Token invalide",
            "TOKEN_BLACKLISTED": "Token révoqué",
            "SECURITY_CHECK_FAILED": "Token invalide",
            "VALIDATION_ERROR": "Erreur d'authentification",
            "INVALID_USER_ID": "Token invalide",
            "AUTHENTICATION_ERROR": "Erreur interne d'authentification"
        }

        status_code = status_code_mapping.get(auth_result.error_code, 401)
        error_message = user_friendly_messages.get(
            auth_result.error_code,
            "Authentification requise"
        )

        response_data = {
            "detail": error_message,
            "error_code": auth_result.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        response = JSONResponse(
            status_code=status_code,
            content=response_data
        )

        # Headers WWW-Authenticate pour conformité HTTP
        if status_code == 401:
            response.headers["WWW-Authenticate"] = 'Bearer realm="Harena API"'

        return response
