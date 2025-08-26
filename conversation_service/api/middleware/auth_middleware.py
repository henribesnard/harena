"""
Middleware d'authentification JWT pour conversation service
"""
import logging
import jwt
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
from fastapi.security.utils import get_authorization_scheme_param
from starlette.middleware.base import BaseHTTPMiddleware
from config_service.config import settings

# Configuration du logger
logger = logging.getLogger("conversation_service.auth")

class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware authentification JWT pour toutes les routes conversation"""
    
    def __init__(self, app):
        super().__init__(app)
        self.excluded_paths = [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc"
        ]
        
        logger.info("JWT Auth Middleware initialisé")
    
    async def dispatch(self, request: Request, call_next):
        """Traitement authentification pour chaque requête"""
        
        # Skip authentification pour certaines routes
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        # Vérification authentification pour routes conversation
        if self._requires_auth(request.url.path):
            try:
                user_id = await self._authenticate_request(request)
                request.state.user_id = user_id
                request.state.authenticated = True
                
            except HTTPException as e:
                logger.warning(f"Auth failed for {request.url.path}: {e.detail}")
                raise e
            except Exception as e:
                logger.error(f"Auth error for {request.url.path}: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal auth error")
        
        return await call_next(request)
    
    def _is_excluded_path(self, path: str) -> bool:
        """Vérification si le path est exclu de l'auth"""
        return any(excluded in path for excluded in self.excluded_paths)
    
    def _requires_auth(self, path: str) -> bool:
        """Vérification si le path nécessite une authentification"""
        conversation_paths = ["/api/v1/conversation"]
        return any(conv_path in path for conv_path in conversation_paths)
    
    async def _authenticate_request(self, request: Request) -> int:
        """Authentification et extraction user_id depuis JWT"""
        
        # Extraction token depuis header Authorization
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(
                status_code=401, 
                detail="Header Authorization manquant"
            )
        
        scheme, token = get_authorization_scheme_param(authorization)
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Schema d'authentification invalide. Utilisez 'Bearer <token>'"
            )
        
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Token JWT manquant"
            )
        
        # Décodage et validation JWT
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            user_id = payload.get("user_id")
            if user_id is None:
                raise HTTPException(
                    status_code=401,
                    detail="Token invalide: user_id manquant"
                )
            
            # Validation user_id est un entier
            try:
                user_id = int(user_id)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=401,
                    detail="Token invalide: user_id doit être un entier"
                )
            
            return user_id
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token expiré"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Token JWT invalide"
            )
        except Exception as e:
            logger.error(f"Erreur décodage JWT: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Erreur validation token"
            )

async def get_current_user_id(request: Request) -> int:
    """Fonction helper pour récupérer user_id depuis request state"""
    if not hasattr(request.state, 'user_id'):
        raise HTTPException(
            status_code=401,
            detail="Utilisateur non authentifié"
        )
    
    return request.state.user_id

async def verify_user_id_match(request: Request, path_user_id: int) -> None:
    """Vérification que user_id du path correspond au token"""
    token_user_id = await get_current_user_id(request)
    
    if path_user_id != token_user_id:
        logger.warning(
            f"User ID mismatch: path={path_user_id}, token={token_user_id}, "
            f"IP={request.client.host if request.client else 'unknown'}"
        )
        raise HTTPException(
            status_code=403,
            detail="user_id dans l'URL ne correspond pas au token"
        )