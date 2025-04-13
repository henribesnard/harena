# transaction_vector_service/api/dependencies.py
"""
Dependency functions for FastAPI.

This module provides dependency injections for routes.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from typing import Dict, Any, Optional
import time

from ..services.transaction_service import TransactionService
from ..services.embedding_service import EmbeddingService
from ..services.qdrant_client import QdrantService
from ..services.merchant_service import MerchantService
from ..services.category_service import CategoryService
from ..search.hybrid_search import HybridSearch
from ..config.settings import settings
from ..config.constants import API_RATE_LIMIT, API_RATE_LIMIT_PERIOD

# Configurer OAuth2 pour correspondre au service utilisateur
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/v1/users/auth/login")

# Simple rate limiting en mémoire
rate_limit_store = {}

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Valide le token JWT et retourne les informations utilisateur.
    
    Args:
        token: Token JWT depuis l'en-tête d'autorisation
        
    Returns:
        Dictionnaire d'informations utilisateur
    
    Raises:
        HTTPException: Si le token est invalide ou expiré
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Décode le token JWT
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # Convertir en entier car c'est ce qui est utilisé dans user_service
        return {"user_id": int(user_id)}
    except JWTError:
        raise credentials_exception


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Vérifie que l'utilisateur est actif.
    
    Dans une implémentation complète, cette fonction vérifierait
    le statut de l'utilisateur dans la base de données.
    
    Args:
        current_user: Information de l'utilisateur courant
        
    Returns:
        Information de l'utilisateur si actif
        
    Raises:
        HTTPException: Si l'utilisateur est inactif
    """
    # Dans une implémentation complète, on vérifierait si l'utilisateur est actif
    # Pour l'instant, on suppose que tous les utilisateurs sont actifs
    return current_user


async def get_rate_limiter():
    """
    Dépendance pour limiter les requêtes et éviter les abus.
    
    Returns:
        Fonction de limitation de débit
    """
    async def rate_limit(request: Request):
        # Récupérer l'IP client
        client_ip = request.client.host
        
        # Obtenir le timestamp actuel
        current_time = time.time()
        
        # Créer/mettre à jour l'entrée pour ce client
        if client_ip not in rate_limit_store:
            rate_limit_store[client_ip] = {
                "count": 1,
                "reset_at": current_time + API_RATE_LIMIT_PERIOD
            }
        else:
            # Vérifier si la période a été réinitialisée
            if current_time > rate_limit_store[client_ip]["reset_at"]:
                # Réinitialiser le compteur
                rate_limit_store[client_ip] = {
                    "count": 1,
                    "reset_at": current_time + API_RATE_LIMIT_PERIOD
                }
            else:
                # Incrémenter le compteur
                rate_limit_store[client_ip]["count"] += 1
                
                # Vérifier si la limite est dépassée
                if rate_limit_store[client_ip]["count"] > API_RATE_LIMIT:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded. Try again later."
                    )
    
    return rate_limit


def get_transaction_service() -> TransactionService:
    """
    Dépendance pour l'injection du service de transaction.
    
    Returns:
        Instance initialisée de TransactionService
    """
    # Initialiser les services
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    merchant_service = MerchantService(embedding_service, qdrant_service)
    category_service = CategoryService()
    search_service = HybridSearch()
    
    # Créer et retourner le service de transaction
    return TransactionService(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        merchant_service=merchant_service,
        category_service=category_service,
        search_service=search_service
    )