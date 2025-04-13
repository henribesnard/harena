"""
Dépendances de l'API pour FastAPI.

Ce module fournit des fonctions de dépendance qui peuvent être
injectées dans les routes FastAPI pour l'authentification,
l'accès à la base de données, et d'autres fonctionnalités partagées.
"""

from fastapi import Depends, HTTPException, Header, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, Generator

from ..config.settings import settings
from ..db.session import get_db
from ..llm.llm_service import LLMService
from ..services.conversation_manager import ConversationManager
from ..services.intent_classifier import IntentClassifier
from ..services.query_builder import QueryBuilder
from ..services.response_generator import ResponseGenerator

# Configuration OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/token")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> Dict[str, Any]:
    """
    Valide le token JWT et retourne les informations utilisateur.
    
    Dépendance FastAPI pour authentifier l'utilisateur.
    
    Args:
        token: Token d'authentification JWT
        
    Returns:
        Dict contenant les informations utilisateur
    
    Raises:
        HTTPException: Si le token est invalide
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants invalides",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # NOTE: Dans un environnement de production, vérifiez correctement le JWT
        # avec une clé secrète appropriée et un algorithme sécurisé
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"]
        )
        
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        return {"user_id": user_id}
    except JWTError:
        raise credentials_exception


async def get_llm_service() -> LLMService:
    """
    Fournit une instance du service LLM.
    
    Returns:
        LLMService: Instance du service LLM
    """
    return LLMService()


async def get_conversation_manager(
    db: Session = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service),
) -> ConversationManager:
    """
    Fournit une instance du gestionnaire de conversation.
    
    Args:
        db: Session de base de données
        llm_service: Service LLM
        
    Returns:
        ConversationManager: Instance du gestionnaire de conversation
    """
    intent_classifier = IntentClassifier(llm_service)
    query_builder = QueryBuilder()
    response_generator = ResponseGenerator(llm_service)
    
    return ConversationManager(
        db=db,
        llm_service=llm_service,
        intent_classifier=intent_classifier,
        query_builder=query_builder,
        response_generator=response_generator
    )


async def rate_limiter(request: Request):
    """
    Limite le taux de requêtes par IP.
    
    Args:
        request: Requête HTTP
        
    Raises:
        HTTPException: Si la limite de requêtes est dépassée
    """
    if not settings.RATE_LIMIT_ENABLED:
        return
    
    # NOTE: Dans un environnement de production, utilisez un système
    # de rate-limiting distribué comme Redis pour éviter les problèmes
    # avec plusieurs instances du service
    
    # Exemple simple de rate-limiting en mémoire
    client_ip = request.client.host
    
    # Implementer le mécanisme de rate-limiting ici
    # Par souci de simplicité, nous n'implémentons pas
    # le mécanisme complet dans cet exemple