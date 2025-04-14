"""
Interfaces for service components.

This module defines abstract interfaces for the main services in the
Transaction Vector Service, allowing implementation details to be decoupled
from interface definitions to avoid circular imports.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import date, datetime
from uuid import UUID


class EmbeddingServiceInterface(ABC):
    """Interface for embedding generation services."""
    
    @abstractmethod
    async def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Get the embedding vector for a text string.
        
        Args:
            text: Text to embed
            use_cache: Whether to use the cache
            
        Returns:
            Embedding vector as a list of floats
        """
        pass
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use the cache
            
        Returns:
            List of embedding vectors
        """
        pass


class QdrantServiceInterface(ABC):
    """Interface for Qdrant vector database services."""
    
    @abstractmethod
    async def upsert_transaction(
        self, 
        transaction_id: Union[str, UUID], 
        embedding: List[float], 
        payload: Dict[str, Any]
    ) -> bool:
        """
        Insert or update a transaction in the vector database.
        
        Args:
            transaction_id: Unique identifier for the transaction
            embedding: Vector embedding for the transaction
            payload: Additional data to store with the transaction
            
        Returns:
            True if the operation was successful
        """
        pass
    
    @abstractmethod
    async def search_similar_transactions(
        self, 
        embedding: List[float], 
        user_id: Optional[int] = None, 
        limit: int = 10, 
        score_threshold: float = 0.75,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for transactions similar to a given embedding.
        
        Args:
            embedding: Vector embedding to search against
            user_id: Optional user ID to filter results
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Additional filter conditions
            
        Returns:
            List of similar transactions with their similarity scores
        """
        pass
    
    @abstractmethod
    async def filter_transactions(
        self,
        user_id: int,
        filter_conditions: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Filter transactions based on conditions.
        
        Args:
            user_id: User ID to filter by
            filter_conditions: Dictionary of filter conditions
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of transactions, total count)
        """
        pass


class MerchantServiceInterface(ABC):
    """Interface for merchant detection and management services."""
    
    @abstractmethod
    async def normalize_merchant_name(self, description: str) -> Optional[str]:
        """
        Extract and normalize merchant name from a transaction description.
        
        Args:
            description: Transaction description
            
        Returns:
            Normalized merchant name or None if no match
        """
        pass
    
    @abstractmethod
    async def find_merchant_by_description(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Find a merchant based on transaction description using vector similarity.
        
        Args:
            description: Transaction description
            
        Returns:
            Merchant data if found, None otherwise
        """
        pass


class CategoryServiceInterface(ABC):
    """Interface for category management services."""
    
    @abstractmethod
    async def get_category(self, category_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a category by ID.
        
        Args:
            category_id: Category ID to retrieve
            
        Returns:
            Category data or None if not found
        """
        pass
    
    @abstractmethod
    async def predict_category(self, description: str, amount: float) -> Optional[int]:
        """
        Predict the most likely category for a transaction based on description and amount.
        
        Args:
            description: Transaction description
            amount: Transaction amount
            
        Returns:
            Predicted category ID or None
        """
        pass


class SearchServiceInterface(ABC):
    """Interface for search services."""
    
    @abstractmethod
    async def search(
        self, 
        user_id: int,
        query: str,
        search_params: Any,
        top_k_initial: int = 100,
        top_k_final: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Execute a search on transactions.
        
        Args:
            user_id: User ID to search for
            query: Search query
            search_params: Additional search parameters
            top_k_initial: Number of initial results to retrieve
            top_k_final: Number of final results to return
            
        Returns:
            Tuple of (search results, total count)
        """
        pass


class TransactionServiceInterface(ABC):
    """Interface for transaction management services."""
    
    @abstractmethod
    async def process_transaction(self, transaction_data: Any) -> Optional[Dict[str, Any]]:
        """
        Process and store a new transaction.
        
        Args:
            transaction_data: Transaction data to process
            
        Returns:
            Processed transaction object or None if processing failed
        """
        pass
    
    @abstractmethod
    async def get_transaction(self, transaction_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """
        Get a transaction by ID.
        
        Args:
            transaction_id: Transaction ID to retrieve
            
        Returns:
            Transaction data or None if not found
        """
        pass
    
    @abstractmethod
    async def update_transaction(self, transaction_id: Union[str, UUID], updates: Dict[str, Any]) -> bool:
        """
        Update a transaction's information.
        
        Args:
            transaction_id: Transaction ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful
        """
        pass
    
    @abstractmethod
    async def search_transactions(
        self,
        user_id: int,
        search_params: Any
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search for transactions based on various criteria.
        
        Args:
            user_id: User ID to search transactions for
            search_params: Search parameters
            
        Returns:
            Tuple of (list of transactions, total count)
        """
        pass
    
    @abstractmethod
    async def find_similar_transactions(
        self,
        embedding: Optional[List[float]] = None,
        description: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find transactions similar to the given embedding or description.
        
        Args:
            embedding: Optional pre-computed embedding to use
            description: Optional description to find similar transactions for
            user_id: Optional user ID to filter results
            limit: Maximum number of results to return
            
        Returns:
            List of similar transactions with similarity scores
        """
        pass