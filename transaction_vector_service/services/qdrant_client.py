"""
Interface with Qdrant vector database.

This module provides a client for interacting with the Qdrant vector database,
used for storing and querying vector embeddings of transactions and merchants.
"""

import httpx
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config.settings import settings
from ..config.constants import TRANSACTION_COLLECTION, MERCHANT_COLLECTION
from ..config.logging_config import get_logger

logger = get_logger(__name__)


class QdrantService:
    """Service for interacting with the Qdrant vector database."""

    def __init__(self):
        """Initialize the Qdrant service."""
        self.url = settings.QDRANT_URL
        self.api_key = settings.QDRANT_API_KEY
        self.vector_size = settings.EMBEDDING_DIMENSION
        
        # Initialize the Qdrant client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30.0  # seconds
        )
        
        # Initialize collections
        self._ensure_collections()
        
        logger.info("Qdrant service initialized")

    def _ensure_collections(self):
        """Ensure that required collections exist and have proper configuration."""
        try:
            # Ensure transactions collection
            self._create_collection_if_not_exists(
                collection_name=TRANSACTION_COLLECTION,
                vector_size=self.vector_size,
                with_payload=True
            )
            
            # Ensure merchants collection
            self._create_collection_if_not_exists(
                collection_name=MERCHANT_COLLECTION,
                vector_size=self.vector_size,
                with_payload=True
            )
            
            logger.info("Qdrant collections initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collections: {str(e)}")
            raise

    def _create_collection_if_not_exists(
        self, 
        collection_name: str, 
        vector_size: int, 
        with_payload: bool = True
    ):
        """
        Create a collection if it doesn't already exist.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors to store
            with_payload: Whether to store payload alongside vectors
        """
        try:
            # Check if collection exists
            collections_list = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections_list]
            
            if collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                    optimizers_config=qmodels.OptimizersConfigDiff(
                        indexing_threshold=10000,  # Index after this many vectors
                    )
                )
                
                # Create payload indexes for common search fields
                if collection_name == TRANSACTION_COLLECTION:
                    # Add indexes for transaction fields
                    indexes = [
                        ("user_id", "keyword"),
                        ("account_id", "keyword"),
                        ("transaction_date", "datetime"),
                        ("category_id", "keyword"),
                        ("merchant_id", "keyword"),
                        ("amount", "float"),
                        ("operation_type", "keyword"),
                        ("is_recurring", "bool"),
                    ]
                    
                    for field_name, field_type in indexes:
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name=field_name,
                            field_schema=field_type
                        )
                
                elif collection_name == MERCHANT_COLLECTION:
                    # Add indexes for merchant fields
                    indexes = [
                        ("name", "text"),
                        ("normalized_name", "keyword"),
                        ("category_id", "keyword"),
                    ]
                    
                    for field_name, field_type in indexes:
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name=field_name,
                            field_schema=field_type
                        )
                
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection already exists: {collection_name}")
        
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {str(e)}")
            raise

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
        try:
            # Convert UUID to string if needed
            if isinstance(transaction_id, UUID):
                transaction_id = str(transaction_id)
                
            # Ensure payload is JSON-serializable
            payload = json.loads(json.dumps(payload, default=str))
            
            # Upsert the vector
            self.client.upsert(
                collection_name=TRANSACTION_COLLECTION,
                points=[
                    qmodels.PointStruct(
                        id=transaction_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            return True
        except Exception as e:
            logger.error(f"Error upserting transaction {transaction_id}: {str(e)}")
            return False

    async def upsert_merchant(
        self, 
        merchant_id: Union[str, UUID], 
        embedding: List[float], 
        payload: Dict[str, Any]
    ) -> bool:
        """
        Insert or update a merchant in the vector database.
        
        Args:
            merchant_id: Unique identifier for the merchant
            embedding: Vector embedding for the merchant
            payload: Additional data to store with the merchant
            
        Returns:
            True if the operation was successful
        """
        try:
            # Convert UUID to string if needed
            if isinstance(merchant_id, UUID):
                merchant_id = str(merchant_id)
                
            # Ensure payload is JSON-serializable
            payload = json.loads(json.dumps(payload, default=str))
            
            # Upsert the vector
            self.client.upsert(
                collection_name=MERCHANT_COLLECTION,
                points=[
                    qmodels.PointStruct(
                        id=merchant_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            return True
        except Exception as e:
            logger.error(f"Error upserting merchant {merchant_id}: {str(e)}")
            return False

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
        try:
            # Build the filter
            filter_dict = {}
            
            if user_id is not None:
                filter_dict["user_id"] = user_id
                
            # Add any additional filter conditions
            if filter_conditions:
                filter_dict.update(filter_conditions)
                
            # Convert filter dict to Qdrant filter
            qdrant_filter = None
            if filter_dict:
                # Renommage de la variable pour éviter les confusions
                field_conditions = []
                for field, value in filter_dict.items():
                    field_conditions.append(qmodels.FieldCondition(
                        key=field,
                        match=qmodels.MatchValue(value=value)
                    ))
                qdrant_filter = qmodels.Filter(must=field_conditions)
            
            # Perform the search
            search_results = self.client.search(
                collection_name=TRANSACTION_COLLECTION,
                query_vector=embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
                filter=qdrant_filter
            )
            
            # Process the results
            results = []
            for result in search_results:
                # Combine payload with score and ID
                item = result.payload.copy() if result.payload else {}
                item["id"] = result.id
                item["score"] = result.score
                results.append(item)
                
            return results
        except Exception as e:
            logger.error(f"Error searching similar transactions: {str(e)}")
            return []

    async def search_merchants_by_pattern(
        self, 
        embedding: List[float], 
        limit: int = 5, 
        score_threshold: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Search for merchants matching a pattern embedding.
        
        Args:
            embedding: Vector embedding of the pattern to match
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching merchants with their similarity scores
        """
        try:
            # Perform the search
            search_results = self.client.search(
                collection_name=MERCHANT_COLLECTION,
                query_vector=embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Process the results
            results = []
            for result in search_results:
                # Combine payload with score and ID
                item = result.payload.copy() if result.payload else {}
                item["id"] = result.id
                item["score"] = result.score
                results.append(item)
                
            return results
        except Exception as e:
            logger.error(f"Error searching merchants by pattern: {str(e)}")
            return []

    async def get_transaction(self, transaction_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """
        Get a transaction by ID.
        
        Args:
            transaction_id: ID of the transaction to retrieve
            
        Returns:
            Transaction data if found, None otherwise
        """
        try:
            # Convert UUID to string if needed
            if isinstance(transaction_id, UUID):
                transaction_id = str(transaction_id)
                
            # Get the point
            result = self.client.retrieve(
                collection_name=TRANSACTION_COLLECTION,
                ids=[transaction_id],
                with_payload=True,
                with_vectors=True
            )
            
            if result and len(result) > 0:
                # Combine payload with vector and ID
                point = result[0]
                data = point.payload.copy() if point.payload else {}
                data["id"] = point.id
                data["embedding"] = point.vector
                return data
            
            return None
        except Exception as e:
            logger.error(f"Error getting transaction {transaction_id}: {str(e)}")
            return None

    async def delete_transaction(self, transaction_id: Union[str, UUID]) -> bool:
        """
        Delete a transaction by ID.
        
        Args:
            transaction_id: ID of the transaction to delete
            
        Returns:
            True if the deletion was successful
        """
        try:
            # Convert UUID to string if needed
            if isinstance(transaction_id, UUID):
                transaction_id = str(transaction_id)
                
            # Delete the point
            self.client.delete(
                collection_name=TRANSACTION_COLLECTION,
                points_selector=qmodels.PointIdsList(
                    points=[transaction_id]
                )
            )
            
            return True
        except Exception as e:
            logger.error(f"Error deleting transaction {transaction_id}: {str(e)}")
            return False

    async def delete_merchant(self, merchant_id: Union[str, UUID]) -> bool:
        """
        Delete a merchant by ID.
        
        Args:
            merchant_id: ID of the merchant to delete
            
        Returns:
            True if the deletion was successful
        """
        try:
            # Convert UUID to string if needed
            if isinstance(merchant_id, UUID):
                merchant_id = str(merchant_id)
                
            # Delete the point
            self.client.delete(
                collection_name=MERCHANT_COLLECTION,
                points_selector=qmodels.PointIdsList(
                    points=[merchant_id]
                )
            )
            
            return True
        except Exception as e:
            logger.error(f"Error deleting merchant {merchant_id}: {str(e)}")
            return False

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
        try:
            # Prepare filter conditions
            must_conditions = [
                qmodels.FieldCondition(
                    key="user_id",
                    match=qmodels.MatchValue(value=user_id)
                )
            ]
            
            # Add other filter conditions
            for key, value in filter_conditions.items():
                if key == "amount_range" and isinstance(value, dict):
                    # Handle range filters
                    if "min_amount" in value and value["min_amount"] is not None:
                        must_conditions.append(qmodels.FieldCondition(
                            key="amount",
                            range=qmodels.Range(
                                gte=value["min_amount"]
                            )
                        ))
                    if "max_amount" in value and value["max_amount"] is not None:
                        must_conditions.append(qmodels.FieldCondition(
                            key="amount",
                            range=qmodels.Range(
                                lte=value["max_amount"]
                            )
                        ))
                elif key == "date_range" and isinstance(value, dict):
                    # Handle date range filters
                    if "start_date" in value and value["start_date"] is not None:
                        must_conditions.append(qmodels.FieldCondition(
                            key="transaction_date",
                            range=qmodels.Range(
                                gte=value["start_date"].isoformat()
                            )
                        ))
                    if "end_date" in value and value["end_date"] is not None:
                        must_conditions.append(qmodels.FieldCondition(
                            key="transaction_date",
                            range=qmodels.Range(
                                lte=value["end_date"].isoformat()
                            )
                        ))
                elif key == "categories" and isinstance(value, list) and value:
                    # Handle category filters
                    must_conditions.append(qmodels.FieldCondition(
                        key="category_id",
                        match=qmodels.MatchAny(any=value)
                    ))
                elif key == "operation_types" and isinstance(value, list) and value:
                    # Handle operation type filters
                    must_conditions.append(qmodels.FieldCondition(
                        key="operation_type",
                        match=qmodels.MatchAny(any=value)
                    ))
                elif key == "merchant_ids" and isinstance(value, list) and value:
                    # Handle merchant filters
                    must_conditions.append(qmodels.FieldCondition(
                        key="merchant_id",
                        match=qmodels.MatchAny(any=value)
                    ))
                elif key == "account_ids" and isinstance(value, list) and value:
                    # Handle account filters
                    must_conditions.append(qmodels.FieldCondition(
                        key="account_id",
                        match=qmodels.MatchAny(any=value)
                    ))
                elif key == "is_recurring" and isinstance(value, bool):
                    # Handle recurring filter
                    must_conditions.append(qmodels.FieldCondition(
                        key="is_recurring",
                        match=qmodels.MatchValue(value=value)
                    ))
                elif key not in ["amount_range", "date_range", "categories", "operation_types", 
                                "merchant_ids", "account_ids", "is_recurring"]:
                    # Handle simple equality filters
                    must_conditions.append(qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=value)
                    ))
            
            # Create the filter
            qdrant_filter = qmodels.Filter(must=must_conditions)
            
            # Get total count first
            count_response = self.client.count(
                collection_name=TRANSACTION_COLLECTION,
                count_filter=qdrant_filter
            )
            total_count = count_response.count
            
            # Perform the search
            scroll_response = self.client.scroll(
                collection_name=TRANSACTION_COLLECTION,
                filter=qdrant_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            # Process the results
            results = []
            for point in scroll_response[0]:
                item = point.payload.copy() if point.payload else {}
                item["id"] = point.id
                results.append(item)
                
            return results, total_count
        except Exception as e:
            logger.error(f"Error filtering transactions: {str(e)}")
            return [], 0

    async def get_transaction_stats(
        self,
        user_id: int,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get statistical summary of transactions.
        
        Args:
            user_id: User ID to filter by
            filter_conditions: Optional additional filter conditions
            
        Returns:
            Dictionary of statistics
        """
        try:
            # Prepare filter conditions
            must_conditions = [
                qmodels.FieldCondition(
                    key="user_id",
                    match=qmodels.MatchValue(value=user_id)
                )
            ]
            
            # Add other filter conditions
            if filter_conditions:
                for key, value in filter_conditions.items():
                    if key not in ["user_id"]:
                        must_conditions.append(qmodels.FieldCondition(
                            key=key,
                            match=qmodels.MatchValue(value=value)
                        ))
            
            # Create the filter
            qdrant_filter = qmodels.Filter(must=must_conditions)
            
            # Get the data
            scroll_response = self.client.scroll(
                collection_name=TRANSACTION_COLLECTION,
                filter=qdrant_filter,
                limit=10000,  # Use a large limit to get all data
                with_payload=True,
                with_vectors=False
            )
            
            # Process the results for statistics
            transactions = []
            for point in scroll_response[0]:
                if point.payload:
                    transactions.append(point.payload)
            
            # Calculate statistics
            stats = {
                "total_count": len(transactions),
                "total_amount": 0.0,
                "average_amount": 0.0,
                "min_amount": float('inf') if transactions else 0.0,
                "max_amount": float('-inf') if transactions else 0.0,
                "by_category": {},
                "by_month": {}
            }
            
            if transactions:
                # Basic stats
                amounts = [t.get("amount", 0.0) for t in transactions]
                stats["total_amount"] = sum(amounts)
                stats["average_amount"] = stats["total_amount"] / len(transactions)
                stats["min_amount"] = min(amounts)
                stats["max_amount"] = max(amounts)
                
                # By category
                categories = {}
                for t in transactions:
                    category_id = t.get("category_id")
                    if category_id:
                        if category_id not in categories:
                            categories[category_id] = {
                                "count": 0,
                                "total": 0.0
                            }
                        categories[category_id]["count"] += 1
                        categories[category_id]["total"] += t.get("amount", 0.0)
                
                stats["by_category"] = categories
                
                # By month
                months = {}
                for t in transactions:
                    date_str = t.get("transaction_date")
                    if date_str:
                        # Extract year-month
                        if isinstance(date_str, str) and len(date_str) >= 7:
                            month = date_str[:7]  # YYYY-MM
                            if month not in months:
                                months[month] = {
                                    "count": 0,
                                    "total": 0.0
                                }
                            months[month]["count"] += 1
                            months[month]["total"] += t.get("amount", 0.0)
                
                stats["by_month"] = months
            
            return stats
        except Exception as e:
            logger.error(f"Error getting transaction stats: {str(e)}")
            return {
                "total_count": 0,
                "total_amount": 0.0,
                "average_amount": 0.0,
                "min_amount": 0.0,
                "max_amount": 0.0,
                "by_category": {},
                "by_month": {}
            }

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Statistics about the collection
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name=collection_name)
            
            # Build stats
            stats = {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name,
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats for {collection_name}: {str(e)}")
            return {}