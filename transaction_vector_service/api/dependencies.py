"""
Dependency functions for FastAPI.

This module provides dependency injections for routes.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from typing import Dict, Any, Optional, Callable
import time

from ..config.settings import settings
from ..config.constants import API_RATE_LIMIT, API_RATE_LIMIT_PERIOD
from ..models.interfaces import (
    TransactionServiceInterface,
    EmbeddingServiceInterface,
    QdrantServiceInterface,
    MerchantServiceInterface,
    CategoryServiceInterface,
    SearchServiceInterface
)
from ..services.embedding_service import EmbeddingService
from ..services.qdrant_client import QdrantService
from ..services.merchant_service import MerchantService
from ..services.category_service import CategoryService
from ..services.transaction_service import TransactionService
from ..search.hybrid_search import HybridSearch
from ..search.bm25_search import BM25Search
from ..search.vector_search import VectorSearch
from ..search.cross_encoder import CrossEncoderRanker

# Configurer OAuth2 pour correspondre au service utilisateur
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/v1/users/auth/login")

# Simple rate limiting en mémoire
rate_limit_store = {}

# Cache global des services pour réutiliser les instances
_service_cache = {}


def get_embedding_service() -> EmbeddingServiceInterface:
    """
    Obtient une instance du service d'embedding.
    
    Returns:
        Instance du service d'embedding
    """
    if 'embedding_service' not in _service_cache:
        _service_cache['embedding_service'] = EmbeddingService()
    return _service_cache['embedding_service']


def get_qdrant_service() -> QdrantServiceInterface:
    """
    Obtient une instance du service Qdrant.
    
    Returns:
        Instance du service Qdrant
    """
    if 'qdrant_service' not in _service_cache:
        _service_cache['qdrant_service'] = QdrantService()
    return _service_cache['qdrant_service']


def get_category_service() -> CategoryServiceInterface:
    """
    Obtient une instance du service de catégories.
    
    Returns:
        Instance du service de catégories
    """
    if 'category_service' not in _service_cache:
        _service_cache['category_service'] = CategoryService()
    return _service_cache['category_service']


def get_merchant_service() -> MerchantServiceInterface:
    """
    Obtient une instance du service de marchands.
    
    Returns:
        Instance du service de marchands
    """
    if 'merchant_service' not in _service_cache:
        embedding_service = get_embedding_service()
        qdrant_service = get_qdrant_service()
        _service_cache['merchant_service'] = MerchantService(
            embedding_service=embedding_service,
            qdrant_service=qdrant_service
        )
    return _service_cache['merchant_service']


def get_bm25_search():
    """
    Obtient une instance du service de recherche BM25.
    
    Returns:
        Instance du service de recherche BM25
    """
    if 'bm25_search' not in _service_cache:
        _service_cache['bm25_search'] = BM25Search()
    return _service_cache['bm25_search']


def get_vector_search():
    """
    Obtient une instance du service de recherche vectorielle.
    
    Returns:
        Instance du service de recherche vectorielle
    """
    if 'vector_search' not in _service_cache:
        embedding_service = get_embedding_service()
        qdrant_service = get_qdrant_service()
        _service_cache['vector_search'] = VectorSearch(
            embedding_service=embedding_service,
            qdrant_service=qdrant_service
        )
    return _service_cache['vector_search']


def get_cross_encoder():
    """
    Obtient une instance du service de cross-encoder.
    
    Returns:
        Instance du service de cross-encoder
    """
    if 'cross_encoder' not in _service_cache:
        _service_cache['cross_encoder'] = CrossEncoderRanker()
    return _service_cache['cross_encoder']


def get_search_service() -> SearchServiceInterface:
    """
    Obtient une instance du service de recherche hybride.
    
    Returns:
        Instance du service de recherche hybride
    """
    if 'search_service' not in _service_cache:
        hybrid_search = HybridSearch()
        
        # Injecter les composants de recherche
        bm25_search = get_bm25_search()
        vector_search = get_vector_search()
        cross_encoder = get_cross_encoder()
        
        hybrid_search.set_search_components(bm25_search, vector_search, cross_encoder)
        _service_cache['search_service'] = hybrid_search
    
    return _service_cache['search_service']


def get_transaction_service() -> TransactionServiceInterface:
    """
    Dépendance pour l'injection du service de transaction.
    
    Returns:
        Instance initialisée de TransactionService
    """
    if 'transaction_service' not in _service_cache:
        # Initialiser les services nécessaires
        embedding_service = get_embedding_service()
        qdrant_service = get_qdrant_service()
        merchant_service = get_merchant_service()
        category_service = get_category_service()
        search_service = get_search_service()
        
        # Créer et retourner le service de transaction
        transaction_service = TransactionService(
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            merchant_service=merchant_service,
            category_service=category_service,
            search_service=search_service
        )
        
        _service_cache['transaction_service'] = transaction_service
    
    return _service_cache['transaction_service']


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


# Fonction pour initialiser tous les services au démarrage
def initialize_services():
    """
    Initialise tous les services nécessaires.
    Cette fonction est appelée lors du démarrage de l'application.
    """
    get_embedding_service()
    get_qdrant_service()
    get_category_service()
    get_merchant_service()
    get_bm25_search()
    get_vector_search()
    get_cross_encoder()
    get_search_service()
    get_transaction_service()