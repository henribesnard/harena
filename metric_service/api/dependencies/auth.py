"""
Authentification pour metric_service
Compatible avec le système JWT de user_service
"""
import logging
from typing import Optional
from fastapi import Request, HTTPException, Header
from jose import jwt, JWTError
from config_service.config import settings

logger = logging.getLogger(__name__)


async def get_current_user_id(
    request: Request,
    authorization: Optional[str] = Header(None)
) -> int:
    """
    Récupère le user_id depuis le token JWT

    Args:
        request: Requête FastAPI
        authorization: Header Authorization (format: "Bearer <token>")

    Returns:
        int: ID de l'utilisateur authentifié

    Raises:
        HTTPException 401: Si le token est manquant ou invalide
    """
    # Vérifier si déjà dans request.state (mis par un middleware)
    if hasattr(request.state, 'user_id') and request.state.user_id:
        return request.state.user_id

    # Sinon, extraire du header Authorization
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Token d'authentification manquant"
        )

    # Extraire le token (format: "Bearer <token>")
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Type d'authentification invalide. Utilisez 'Bearer <token>'"
            )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Format du header Authorization invalide"
        )

    # Décoder le JWT
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[getattr(settings, 'JWT_ALGORITHM', 'HS256')],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": False,
                "verify_aud": False,
                "require": ["exp", "sub"]
            }
        )

        # Extraire user_id
        raw_user_id = payload.get("sub")
        if raw_user_id is None:
            raw_user_id = payload.get("user_id")

        if raw_user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Token invalide: 'sub' manquant"
            )

        user_id = int(raw_user_id)

        # Stocker dans request.state pour réutilisation
        request.state.user_id = user_id

        logger.debug(f"Utilisateur authentifié: {user_id}")
        return user_id

    except JWTError as e:
        logger.warning(f"Erreur validation JWT: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Token invalide: {str(e)}"
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Erreur conversion user_id: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Token invalide: format user_id incorrect"
        )


async def get_optional_user_id(
    request: Request,
    authorization: Optional[str] = Header(None)
) -> Optional[int]:
    """
    Récupère le user_id si présent, sinon retourne None
    Utile pour les endpoints publics avec fonctionnalités optionnelles pour utilisateurs connectés

    Args:
        request: Requête FastAPI
        authorization: Header Authorization (optionnel)

    Returns:
        Optional[int]: ID utilisateur ou None
    """
    try:
        return await get_current_user_id(request, authorization)
    except HTTPException:
        return None
