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


def get_transaction_service() -> TransactionServiceInterface:
    """
    Obtient une instance du service de transaction.
    
    Returns:
        Instance du service de transaction
    """
    if 'transaction_service' not in _service_cache:
        # Initialiser les services nécessaires
        embedding_service = get_embedding_service()
        qdrant_service = get_qdrant_service()
        merchant_service = get_merchant_service()
        category_service = get_category_service()
        
        # Créer le service de transaction sans search_service pour éviter la dépendance circulaire
        transaction_service = TransactionService(
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            merchant_service=merchant_service,
            category_service=category_service,
            search_service=None  # Sera défini plus tard
        )
        
        _service_cache['transaction_service'] = transaction_service
    
    return _service_cache['transaction_service']


def get_bm25_search():
    """
    Obtient une instance du service de recherche BM25.
    
    Returns:
        Instance du service de recherche BM25
    """
    if 'bm25_search' not in _service_cache:
        # Créer l'instance sans transaction_service
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
        transaction_service = get_transaction_service()
        _service_cache['vector_search'] = VectorSearch(
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            transaction_service=transaction_service
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
        
        # Obtenir les composants de recherche
        bm25_search = get_bm25_search()
        vector_search = get_vector_search()
        cross_encoder = get_cross_encoder()
        
        # Injecter les composants de recherche
        hybrid_search.set_search_components(bm25_search, vector_search, cross_encoder)
        _service_cache['search_service'] = hybrid_search
    
    return _service_cache['search_service']


# Fonction pour initialiser tous les services au démarrage
def initialize_services():
    """
    Initialise tous les services nécessaires.
    Cette fonction est appelée lors du démarrage de l'application.
    """
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
    bm25_search.set_transaction_service(transaction_service)
    
    # Obtenir le service de recherche hybride
    search_service = get_search_service()
    
    # Injecter le service de recherche dans le service de transaction
    if hasattr(transaction_service, 'set_search_service'):
        transaction_service.set_search_service(search_service)