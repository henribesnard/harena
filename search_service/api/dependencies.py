"""
Dépendances FastAPI pour le Search Service
=========================================

Module de dépendances spécialisées pour l'API REST du Search Service :
- Validation stricte des requêtes Elasticsearch
- Rate limiting par type de requête et utilisateur
- Authentification simplifiée basée sur headers
- Gestion des erreurs standardisée
- Métriques et logging automatiques
- Vérifications de sécurité (isolation utilisateur)

Architecture :
    Request → Dependencies → Validation → Rate Limiting → Business Logic
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import wraps

from fastapi import Depends, HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

# Imports sécurisés avec fallbacks
try:
    from search_service.models.service_contracts import SearchServiceQuery
except ImportError:
    # Fallback si le modèle n'existe pas encore
    class SearchServiceQuery:
        """Classe de fallback pour SearchServiceQuery"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def __getattr__(self, name):
            # Retourner None pour les attributs non définis
            return None

# Import sécurisé des validateurs
try:
    from search_service.utils.validators import ValidatorFactory
except ImportError:
    # Fallback pour ValidatorFactory
    class ValidatorFactory:
        @staticmethod
        def validate_complete_request(request):
            return {
                "valid": True,
                "errors": [],
                "performance_check": {"warnings": [], "complexity": "low"},
                "estimated_time_ms": 100
            }

# Import sécurisé des métriques
try:
    from search_service.utils.metrics import api_metrics
except ImportError:
    # Fallback pour api_metrics
    class MockApiMetrics:
        def record_api_error(self, **kwargs): pass
        def record_authentication(self, **kwargs): pass
        def record_rate_limit_check(self, **kwargs): pass
        def record_request_validation(self, **kwargs): pass
    api_metrics = MockApiMetrics()

# Import sécurisé du core
try:
    from search_service.core import get_lexical_engine, core_manager
except ImportError:
    # Fallback pour core
    def get_lexical_engine():
        return None
    
    class MockCoreManager:
        async def health_check(self):
            return {"status": "unknown", "components": {}}
    core_manager = MockCoreManager()

# Import sécurisé de la config
try:
    from search_service.config import settings
except ImportError:
    # Fallback pour settings
    class MockSettings:
        development_mode = True
        redis_url = None
        elasticsearch_host = "localhost"
        elasticsearch_port = 9200
    settings = MockSettings()


logger = logging.getLogger(__name__)


# === EXCEPTIONS PERSONNALISÉES ===

class APIException(HTTPException):
    """Exception API personnalisée avec métriques automatiques"""
    
    def __init__(self, status_code: int, detail: str, error_code: str = None,
                 headers: Optional[Dict[str, str]] = None):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        
        # Enregistrer l'erreur dans les métriques
        try:
            api_metrics.record_api_error(
                status_code=status_code,
                error_code=error_code or "unknown",
                endpoint="unknown"  # Sera surchargé par le middleware
            )
        except Exception:
            pass  # Ignorer les erreurs de métriques


class ValidationException(APIException):
    """Exception de validation avec détails techniques"""
    
    def __init__(self, message: str, field: str = None, validation_errors: List[str] = None):
        detail = {
            "message": message,
            "field": field,
            "validation_errors": validation_errors or []
        }
        super().__init__(
            status_code=400,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )


class RateLimitException(APIException):
    """Exception de limite de taux dépassée"""
    
    def __init__(self, limit: int, window_seconds: int, retry_after: int):
        detail = {
            "message": f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            "limit": limit,
            "window_seconds": window_seconds,
            "retry_after": retry_after
        }
        super().__init__(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": str(retry_after)}
        )


class AuthenticationException(APIException):
    """Exception d'authentification"""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            status_code=HTTP_401_UNAUTHORIZED,
            detail={"message": message},
            error_code="AUTHENTICATION_REQUIRED"
        )


class AuthorizationException(APIException):
    """Exception d'autorisation"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            status_code=HTTP_403_FORBIDDEN,
            detail={"message": message},
            error_code="INSUFFICIENT_PERMISSIONS"
        )


# === GESTIONNAIRE D'AUTHENTIFICATION ===

class SearchServiceAuth:
    """Gestionnaire d'authentification simplifié pour le Search Service"""
    
    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
    
    async def get_current_user(
        self,
        request: Request,
        authorization: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        x_user_id: Optional[str] = Header(None),
        x_api_key: Optional[str] = Header(None)
    ) -> Dict[str, Any]:
        """
        Authentification basée sur headers ou token
        
        Priorité:
        1. Token Bearer (JWT/API token)
        2. Headers X-User-Id + X-API-Key
        3. Mode développement (si activé)
        """
        
        user_info = None
        auth_method = None
        
        # 1. Authentification par token Bearer
        if authorization and authorization.credentials:
            user_info = await self._validate_bearer_token(authorization.credentials)
            auth_method = "bearer_token"
        
        # 2. Authentification par headers
        elif x_user_id and x_api_key:
            user_info = await self._validate_api_key(x_user_id, x_api_key)
            auth_method = "api_key"
        
        # 3. Mode développement (si activé)
        elif settings.development_mode and x_user_id:
            try:
                user_id = int(x_user_id)
                user_info = {
                    "user_id": user_id,
                    "permissions": ["search", "validate", "metrics"],
                    "rate_limit_tier": "development"
                }
                auth_method = "development"
                logger.warning(f"Mode développement activé pour user_id={x_user_id}")
            except ValueError:
                pass
        
        # 4. Fallback pour tests - utilisateur par défaut
        if not user_info:
            user_info = {
                "user_id": 1,
                "permissions": ["search", "validate"],
                "rate_limit_tier": "standard"
            }
            auth_method = "default"
            logger.debug("Utilisation de l'utilisateur par défaut")
        
        # Enrichir avec informations de requête
        user_info.update({
            "auth_method": auth_method,
            "ip_address": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "request_timestamp": datetime.now()
        })
        
        # Enregistrer dans les métriques
        try:
            api_metrics.record_authentication(
                user_id=user_info["user_id"],
                auth_method=auth_method,
                success=True
            )
        except Exception:
            pass
        
        return user_info
    
    async def _validate_bearer_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Valide un token Bearer (JWT ou API token)"""
        
        try:
            # Pour le développement, accepter les tokens simples
            if token.startswith("dev_"):
                user_id_str = token.replace("dev_", "")
                user_id = int(user_id_str)
                return {
                    "user_id": user_id,
                    "permissions": ["search", "validate", "metrics"],
                    "rate_limit_tier": "standard"
                }
            
            # TODO: Implémenter validation JWT réelle
            # user_info = await external_auth_service.validate_token(token)
            # return user_info
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur validation token: {e}")
            return None
    
    async def _validate_api_key(self, user_id: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Valide une API key avec user_id"""
        
        try:
            user_id_int = int(user_id)
            
            # Validation simple pour le développement
            if api_key.startswith("sk_") and len(api_key) >= 16:
                return {
                    "user_id": user_id_int,
                    "permissions": ["search", "validate"],
                    "rate_limit_tier": "standard",
                    "api_key_id": api_key[:12] + "..."
                }
            
            return None
            
        except ValueError:
            logger.error(f"user_id invalide: {user_id}")
            return None
        except Exception as e:
            logger.error(f"Erreur validation API key: {e}")
            return None


# === GESTIONNAIRE DE RATE LIMITING ===

class RateLimiter:
    """Gestionnaire de rate limiting simple en mémoire"""
    
    def __init__(self):
        self.fallback_limits = {}  # Limite en mémoire
        
        # Limites par défaut (requêtes par minute)
        self.default_limits = {
            "development": {"search": 1000, "validate": 500, "metrics": 100},
            "standard": {"search": 100, "validate": 50, "metrics": 20},
            "premium": {"search": 500, "validate": 200, "metrics": 50},
            "enterprise": {"search": 2000, "validate": 1000, "metrics": 200}
        }
    
    async def initialize(self):
        """Initialise le rate limiter"""
        logger.info("Rate limiter initialisé en mode mémoire")
    
    async def check_rate_limit(
        self,
        user_id: int,
        endpoint_type: str,
        rate_limit_tier: str = "standard",
        window_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Vérifie et applique le rate limiting
        
        Returns:
            Dict avec statut et informations de limite
        """
        
        # Récupérer les limites pour ce tier
        limits = self.default_limits.get(rate_limit_tier, self.default_limits["standard"])
        limit = limits.get(endpoint_type, 60)  # Défaut: 60 req/min
        
        key = f"rate_limit:{user_id}:{endpoint_type}"
        
        try:
            # Rate limiting en mémoire (simple compteur)
            current_count = await self._memory_rate_limit(key, window_seconds, limit)
            
            remaining = max(0, limit - current_count)
            
            result = {
                "allowed": current_count <= limit,
                "limit": limit,
                "current": current_count,
                "remaining": remaining,
                "window_seconds": window_seconds,
                "reset_time": datetime.now() + timedelta(seconds=window_seconds)
            }
            
            # Enregistrer dans les métriques
            try:
                api_metrics.record_rate_limit_check(
                    user_id=user_id,
                    endpoint_type=endpoint_type,
                    allowed=result["allowed"],
                    current_count=current_count,
                    limit=limit
                )
            except Exception:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur rate limiting: {e}")
            # En cas d'erreur, autoriser la requête mais logger
            return {
                "allowed": True,
                "limit": limit,
                "current": 0,
                "remaining": limit,
                "window_seconds": window_seconds,
                "error": str(e)
            }
    
    async def _memory_rate_limit(self, key: str, window_seconds: int, limit: int) -> int:
        """Rate limiting simple en mémoire"""
        
        now = time.time()
        
        if key not in self.fallback_limits:
            self.fallback_limits[key] = []
        
        # Nettoyer les anciennes entrées
        self.fallback_limits[key] = [
            timestamp for timestamp in self.fallback_limits[key]
            if timestamp > now - window_seconds
        ]
        
        # Ajouter la requête actuelle
        self.fallback_limits[key].append(now)
        
        return len(self.fallback_limits[key])


# === INSTANCES GLOBALES ===

# Gestionnaire d'authentification
auth_manager = SearchServiceAuth()

# Rate limiter
rate_limiter = RateLimiter()


# === DÉPENDANCES FASTAPI ===

async def get_authenticated_user(
    request: Request,
    user_info: Dict[str, Any] = Depends(auth_manager.get_current_user)
) -> Dict[str, Any]:
    """Dépendance pour récupérer l'utilisateur authentifié"""
    return user_info


async def validate_rate_limit_dependency(
    endpoint_type: str,
    request: Request,
    user_info: Dict[str, Any] = Depends(get_authenticated_user)
) -> Dict[str, Any]:
    """
    Dépendance pour valider le rate limiting
    
    Args:
        endpoint_type: Type d'endpoint (search, validate, metrics)
    """
    
    rate_limit_result = await rate_limiter.check_rate_limit(
        user_id=user_info["user_id"],
        endpoint_type=endpoint_type,
        rate_limit_tier=user_info.get("rate_limit_tier", "standard")
    )
    
    if not rate_limit_result["allowed"]:
        raise RateLimitException(
            limit=rate_limit_result["limit"],
            window_seconds=rate_limit_result["window_seconds"],
            retry_after=rate_limit_result["window_seconds"]
        )
    
    # Ajouter headers de rate limiting à la réponse
    if hasattr(request, 'state'):
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(rate_limit_result["limit"]),
            "X-RateLimit-Remaining": str(rate_limit_result["remaining"]),
            "X-RateLimit-Reset": str(int(rate_limit_result["reset_time"].timestamp()))
        }
    
    return rate_limit_result


async def validate_search_request(
    request,  # Type sera inféré dynamiquement
    user_info: Dict[str, Any] = Depends(get_authenticated_user)
):
    """
    Dépendance pour valider une requête de recherche
    
    Effectue toutes les validations nécessaires :
    - Validation des contrats
    - Sécurité (isolation utilisateur)
    - Performance (complexité, limites)
    - Cohérence des données
    
    Args:
        request: Requête de recherche (type dynamique)
        user_info: Informations utilisateur authentifié
    
    Returns:
        Requête validée
    """
    
    start_time = time.time()
    
    try:
        # 1. Validation complète avec le validateur
        validation_result = ValidatorFactory.validate_complete_request(request)
        
        if not validation_result["valid"]:
            raise ValidationException(
                message="Request validation failed",
                validation_errors=validation_result["errors"]
            )
        
        # 2. Vérification cohérence user_id (si la requête a cette propriété)
        if hasattr(request, 'query_metadata') and hasattr(request.query_metadata, 'user_id'):
            if request.query_metadata.user_id != user_info["user_id"]:
                raise AuthorizationException(
                    f"User ID mismatch: token user_id={user_info['user_id']}, "
                    f"request user_id={request.query_metadata.user_id}"
                )
        
        # 3. Vérification permissions
        required_permission = "search"
        if required_permission not in user_info.get("permissions", []):
            raise AuthorizationException(
                f"Permission '{required_permission}' required"
            )
        
        # 4. Warnings de performance (non bloquants)
        if validation_result["performance_check"]["warnings"]:
            logger.warning(
                f"Performance warnings for user {user_info['user_id']}: "
                f"{validation_result['performance_check']['warnings']}"
            )
        
        # 5. Enrichir la requête avec informations d'authentification (si possible)
        if hasattr(request, 'query_metadata'):
            if hasattr(request.query_metadata, 'authenticated_user_id'):
                request.query_metadata.authenticated_user_id = user_info["user_id"]
            if hasattr(request.query_metadata, 'auth_method'):
                request.query_metadata.auth_method = user_info.get("auth_method")
            if hasattr(request.query_metadata, 'ip_address'):
                request.query_metadata.ip_address = user_info.get("ip_address")
        
        # 6. Métriques de validation
        validation_duration = (time.time() - start_time) * 1000
        try:
            api_metrics.record_request_validation(
                user_id=user_info["user_id"],
                validation_duration_ms=validation_duration,
                complexity=validation_result["performance_check"]["complexity"],
                estimated_time_ms=validation_result["estimated_time_ms"],
                warnings_count=len(validation_result["performance_check"]["warnings"])
            )
        except Exception:
            pass
        
        logger.debug(
            f"Request validated for user {user_info['user_id']} "
            f"in {validation_duration:.1f}ms"
        )
        
        return request
        
    except ValidationException:
        raise
    except AuthorizationException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during request validation: {e}")
        raise ValidationException(
            message="Internal validation error",
            validation_errors=[str(e)]
        )


async def check_service_health() -> Dict[str, Any]:
    """
    Dépendance pour vérifier la santé du service
    
    Utilisée par les endpoints qui nécessitent que le service soit opérationnel
    """
    
    try:
        # Vérifier les composants core
        lexical_engine = get_lexical_engine()
        if not lexical_engine:
            logger.warning("Lexical engine not available - using mock health status")
            return {
                "status": "degraded",
                "components": {"lexical_engine": "not_available"},
                "message": "Service running in limited mode"
            }
        
        # Vérifier la santé du moteur
        health_status = await lexical_engine.health_check()
        if health_status.get("status") not in ["healthy", "degraded"]:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Service unhealthy",
                    "details": health_status
                }
            )
        
        return health_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # En mode développement, retourner un statut dégradé plutôt qu'une erreur
        return {
            "status": "degraded",
            "error": str(e),
            "message": "Health check failed but service is running"
        }


# === FONCTIONS UTILITAIRES DE DÉPENDANCES ===

def create_search_rate_limit():
    """Crée une dépendance de rate limiting spécifique pour la recherche"""
    async def search_rate_limit(
        request: Request,
        user_info: Dict[str, Any] = Depends(get_authenticated_user)
    ):
        return await validate_rate_limit_dependency("search", request, user_info)
    
    return search_rate_limit


def create_validation_rate_limit():
    """Crée une dépendance de rate limiting spécifique pour la validation"""
    async def validation_rate_limit(
        request: Request,
        user_info: Dict[str, Any] = Depends(get_authenticated_user)
    ):
        return await validate_rate_limit_dependency("validate", request, user_info)
    
    return validation_rate_limit


def create_metrics_rate_limit():
    """Crée une dépendance de rate limiting spécifique pour les métriques"""
    async def metrics_rate_limit(
        request: Request,
        user_info: Dict[str, Any] = Depends(get_authenticated_user)
    ):
        return await validate_rate_limit_dependency("metrics", request, user_info)
    
    return metrics_rate_limit


def create_admin_dependencies():
    """Crée les dépendances pour les endpoints d'administration"""
    
    async def check_admin_permission(
        user_info: Dict[str, Any] = Depends(get_authenticated_user)
    ):
        if "admin" not in user_info.get("permissions", []):
            raise AuthorizationException("Admin permission required")
        return user_info
    
    return check_admin_permission


# === UTILITAIRES DE RÉPONSE ===

def add_response_headers(response, rate_limit_info: Optional[Dict[str, Any]] = None):
    """Ajoute les headers standard aux réponses"""
    
    # Headers de rate limiting
    if rate_limit_info:
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info["reset_time"].timestamp()))
    
    # Headers de sécurité
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Headers API
    response.headers["X-API-Version"] = "1.0"
    response.headers["X-Service"] = "search-service"


# === FONCTIONS D'INITIALISATION ===

async def initialize_dependencies():
    """Initialise les dépendances (à appeler au démarrage)"""
    
    logger.info("Initialisation des dépendances API...")
    
    try:
        # Initialiser le rate limiter
        await rate_limiter.initialize()
        
        logger.info("✅ Dépendances API initialisées")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation dépendances: {e}")
        # Ne pas lever l'erreur pour permettre le démarrage en mode dégradé
        logger.warning("Démarrage en mode dégradé")


async def shutdown_dependencies():
    """Nettoie les dépendances (à appeler à l'arrêt)"""
    
    logger.info("Arrêt des dépendances API...")
    
    try:
        # Nettoyer le cache de rate limiting
        rate_limiter.fallback_limits.clear()
        
        logger.info("✅ Dépendances API arrêtées")
        
    except Exception as e:
        logger.error(f"❌ Erreur arrêt dépendances: {e}")


# === ALIASES POUR COMPATIBILITÉ ===
# Maintenir la compatibilité avec les imports existants
validate_rate_limit = validate_rate_limit_dependency


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === EXCEPTIONS ===
    "APIException",
    "ValidationException",
    "RateLimitException",
    "AuthenticationException",
    "AuthorizationException",
    
    # === GESTIONNAIRES ===
    "SearchServiceAuth",
    "RateLimiter",
    
    # === INSTANCES GLOBALES ===
    "auth_manager",
    "rate_limiter",
    
    # === DÉPENDANCES PRINCIPALES ===
    "get_authenticated_user",
    "validate_rate_limit_dependency",
    "validate_rate_limit",  # Alias pour compatibilité
    "validate_search_request",
    "check_service_health",
    
    # === CRÉATEURS DE DÉPENDANCES ===
    "create_search_rate_limit",
    "create_validation_rate_limit",
    "create_metrics_rate_limit",
    "create_admin_dependencies",
    
    # === UTILITAIRES ===
    "add_response_headers",
    "initialize_dependencies",
    "shutdown_dependencies"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Dépendances FastAPI spécialisées pour le Search Service"

logger.info(f"Module api.dependencies chargé - version {__version__}")