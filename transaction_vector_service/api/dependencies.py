"""
Dependency functions for FastAPI.

This module provides dependency injections for routes.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from typing import Dict, Any, Optional, Callable
import time
import traceback

from ..config.settings import settings
from ..config.constants import API_RATE_LIMIT, API_RATE_LIMIT_PERIOD
from ..config.logging_config import get_logger

from ..services.embedding_service import EmbeddingService
from ..services.qdrant_client import QdrantService
from ..services.merchant_service import MerchantService
from ..services.category_service import CategoryService
from ..services.transaction_service import TransactionService
from ..search.hybrid_search import HybridSearch
from ..search.bm25_search import BM25Search
from ..search.vector_search import VectorSearch
from ..search.cross_encoder import CrossEncoderRanker

# Configurer le logger
logger = get_logger(__name__)

# Configurer OAuth2 pour correspondre au service utilisateur
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/v1/users/auth/login")

# Simple rate limiting en mémoire
rate_limit_store = {}

# Cache global des services pour réutiliser les instances
_service_cache = {}


def get_embedding_service():
    """
    Obtient une instance du service d'embedding.
    
    Returns:
        Instance du service d'embedding
    """
    if 'embedding_service' not in _service_cache:
        try:
            _service_cache['embedding_service'] = EmbeddingService()
            logger.info("Service d'embedding initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service d'embedding: {str(e)}")
            raise
    return _service_cache['embedding_service']


def get_qdrant_service():
    """
    Obtient une instance du service Qdrant.
    
    Returns:
        Instance du service Qdrant
    """
    if 'qdrant_service' not in _service_cache:
        try:
            _service_cache['qdrant_service'] = QdrantService()
            logger.info("Service Qdrant initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service Qdrant: {str(e)}")
            raise
    return _service_cache['qdrant_service']


def get_category_service():
    """
    Obtient une instance du service de catégories.
    
    Returns:
        Instance du service de catégories
    """
    if 'category_service' not in _service_cache:
        try:
            _service_cache['category_service'] = CategoryService()
            logger.info("Service de catégories initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service de catégories: {str(e)}")
            raise
    return _service_cache['category_service']


def get_merchant_service():
    """
    Obtient une instance du service de marchands.
    
    Returns:
        Instance du service de marchands
    """
    if 'merchant_service' not in _service_cache:
        try:
            embedding_service = get_embedding_service()
            qdrant_service = get_qdrant_service()
            _service_cache['merchant_service'] = MerchantService(
                embedding_service=embedding_service,
                qdrant_service=qdrant_service
            )
            logger.info("Service de marchands initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service de marchands: {str(e)}")
            raise
    return _service_cache['merchant_service']


def get_bm25_search():
    """
    Obtient une instance du service de recherche BM25.
    
    Returns:
        Instance du service de recherche BM25
    """
    if 'bm25_search' not in _service_cache:
        try:
            # Créer d'abord l'instance sans transaction_service
            _service_cache['bm25_search'] = BM25Search()
            logger.info("Service BM25Search initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service BM25Search: {str(e)}")
            raise
    return _service_cache['bm25_search']


def get_vector_search():
    """
    Obtient une instance du service de recherche vectorielle.
    
    Returns:
        Instance du service de recherche vectorielle
    """
    if 'vector_search' not in _service_cache:
        try:
            embedding_service = get_embedding_service()
            qdrant_service = get_qdrant_service()
            # Créer l'instance sans transaction_service
            _service_cache['vector_search'] = VectorSearch(
                embedding_service=embedding_service,
                qdrant_service=qdrant_service
            )
            logger.info("Service VectorSearch initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service VectorSearch: {str(e)}")
            raise
    return _service_cache['vector_search']


def get_cross_encoder():
    """
    Obtient une instance du service de cross-encoder.
    
    Returns:
        Instance du service de cross-encoder
    """
    if 'cross_encoder' not in _service_cache:
        try:
            _service_cache['cross_encoder'] = CrossEncoderRanker()
            logger.info("Service CrossEncoderRanker initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service CrossEncoderRanker: {str(e)}")
            raise
    return _service_cache['cross_encoder']


def get_transaction_service():
    """
    Obtient une instance du service de transaction.
    
    Returns:
        Instance du service de transaction
    """
    if 'transaction_service' not in _service_cache:
        try:
            # Initialiser les services nécessaires
            embedding_service = get_embedding_service()
            qdrant_service = get_qdrant_service()
            merchant_service = get_merchant_service()
            category_service = get_category_service()
            
            # Créer le service de transaction
            transaction_service = TransactionService(
                embedding_service=embedding_service,
                qdrant_service=qdrant_service,
                merchant_service=merchant_service,
                category_service=category_service,
                search_service=None  # Sera défini plus tard
            )
            
            _service_cache['transaction_service'] = transaction_service
            logger.info("Service de transactions initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service de transactions: {str(e)}")
            raise
    
    return _service_cache['transaction_service']


def get_search_service():
    """
    Obtient une instance du service de recherche hybride.
    
    Returns:
        Instance du service de recherche hybride
    """
    if 'search_service' not in _service_cache:
        try:
            hybrid_search = HybridSearch()
            logger.info("HybridSearch créé")
            
            # Injecter les composants de recherche
            bm25_search = get_bm25_search()
            vector_search = get_vector_search()
            cross_encoder = get_cross_encoder()
            
            hybrid_search.set_search_components(bm25_search, vector_search, cross_encoder)
            logger.info("Composants de recherche injectés dans HybridSearch")
            
            _service_cache['search_service'] = hybrid_search
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du service de recherche hybride: {str(e)}")
            raise
    
    return _service_cache['search_service']


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
    try:
        logger.info("Début de l'initialisation des services...")
        
        # Obtenir d'abord les services de base
        embedding_service = get_embedding_service()
        qdrant_service = get_qdrant_service()
        category_service = get_category_service() 
        merchant_service = get_merchant_service()
        
        # Obtenir le service de transaction
        transaction_service = get_transaction_service()
        
        # Obtenir les services de recherche
        bm25_search = get_bm25_search()
        vector_search = get_vector_search()
        cross_encoder = get_cross_encoder()
        
        # Injecter le service de transaction dans BM25Search
        if hasattr(bm25_search, 'set_transaction_service'):
            bm25_search.set_transaction_service(transaction_service)
            logger.info("Service de transaction injecté dans BM25Search")
        
        # Injecter le service de transaction dans VectorSearch si nécessaire
        if hasattr(vector_search, 'set_transaction_service'):
            vector_search.set_transaction_service(transaction_service)
            logger.info("Service de transaction injecté dans VectorSearch")
        
        # Obtenir le service de recherche hybride
        search_service = get_search_service()
        
        # Injecter le service de recherche dans le service de transaction
        if hasattr(transaction_service, 'set_search_service'):
            transaction_service.set_search_service(search_service)
            logger.info("Service de recherche injecté dans TransactionService")
        
        logger.info("Services initialisés avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur critique lors de l'initialisation des services: {str(e)}")
        logger.error(traceback.format_exc())
        return False