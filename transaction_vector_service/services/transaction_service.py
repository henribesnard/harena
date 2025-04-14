"""
Service for managing financial transactions.

This module provides functionality for processing, enriching, storing, and
retrieving financial transactions.
"""

import logging
import asyncio
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING
from datetime import datetime, date
from uuid import UUID, uuid4

from ..config.logging_config import get_logger
from ..models.interfaces import (
    TransactionServiceInterface,
    EmbeddingServiceInterface,
    QdrantServiceInterface,
    MerchantServiceInterface,
    CategoryServiceInterface,
    SearchServiceInterface
)
from ..common.types import SIMILARITY_THRESHOLD

# Import types only for type checking
if TYPE_CHECKING:
    from ..models.transaction import Transaction, TransactionCreate, TransactionVector, TransactionSearch, TransactionRead

logger = get_logger(__name__)


class TransactionService(TransactionServiceInterface):
    """Service for managing financial transactions."""

    def __init__(
        self, 
        embedding_service: EmbeddingServiceInterface,
        qdrant_service: QdrantServiceInterface,
        merchant_service: MerchantServiceInterface,
        category_service: CategoryServiceInterface,
        search_service: Optional[SearchServiceInterface] = None
    ):
        """
        Initialize the transaction service.
        
        Args:
            embedding_service: Embedding service instance
            qdrant_service: Qdrant service instance
            merchant_service: Merchant service instance
            category_service: Category service instance
            search_service: Optional search service instance
        """
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.merchant_service = merchant_service
        self.category_service = category_service
        self.search_service = search_service
        
        logger.info("Transaction service initialized")

    async def process_transaction(self, transaction_data: Any) -> Optional[Dict[str, Any]]:
        """
        Process and store a new transaction.
        
        Args:
            transaction_data: Transaction data to process
            
        Returns:
            Processed transaction object or None if processing failed
        """
        try:
            # Generate a UUID for the transaction
            transaction_id = uuid4()
            
            # Import the actual model class here to avoid circular imports
            from ..models.transaction import Transaction
            
            # Create the transaction model
            transaction = Transaction(
                id=transaction_id,
                bridge_transaction_id=transaction_data.bridge_transaction_id,
                account_id=transaction_data.account_id,
                user_id=transaction_data.user_id,
                amount=transaction_data.amount,
                currency_code=transaction_data.currency_code,
                description=transaction_data.description,
                clean_description=transaction_data.clean_description or transaction_data.description,
                transaction_date=transaction_data.transaction_date,
                value_date=transaction_data.value_date,
                booking_date=transaction_data.booking_date,
                category_id=transaction_data.category_id,
                operation_type=transaction_data.operation_type,
                is_future=transaction_data.is_future,
                is_deleted=transaction_data.is_deleted
            )
            
            # Enrich the transaction
            await self._enrich_transaction(transaction)
            
            # Generate embedding
            description_for_embedding = transaction.clean_description or transaction.description
            description_embedding = await self.embedding_service.get_embedding(description_for_embedding)
            
            # Import the vector model here
            from ..models.transaction import TransactionVector
            
            # Create transaction vector
            transaction_vector = TransactionVector(
                **transaction.dict(),
                description_embedding=description_embedding,
                vector_updated_at=datetime.now()
            )
            
            # Find similar transactions
            similar_transactions = await self.find_similar_transactions(
                embedding=description_embedding,
                user_id=transaction.user_id
            )
            
            if similar_transactions:
                transaction_vector.similar_transactions = [t["id"] for t in similar_transactions]
                if similar_transactions[0]["score"] > SIMILARITY_THRESHOLD:
                    transaction_vector.similarity_score = similar_transactions[0]["score"]
            
            # Prepare transaction payload for Qdrant
            transaction_payload = transaction.dict()
            transaction_payload.pop("id", None)  # ID is stored separately
            
            # Add additional fields for search and filtering
            transaction_payload["description_normalized"] = description_for_embedding.lower()
            transaction_payload["month_year"] = transaction.transaction_date.strftime("%Y-%m")
            transaction_payload["is_recurring"] = False  # Will be updated by recurring detection
            transaction_payload["fingerprint"] = self._generate_transaction_fingerprint(transaction)
            
            # Store in Qdrant
            success = await self.qdrant_service.upsert_transaction(
                transaction_id=transaction_id,
                embedding=description_embedding,
                payload=transaction_payload
            )
            
            if success:
                # Update merchant transaction count if applicable
                if transaction.merchant_id:
                    await self.merchant_service.add_transaction_to_merchant(
                        merchant_id=transaction.merchant_id,
                        transaction_id=transaction_id
                    )
                
                return transaction.dict()
            else:
                logger.error(f"Failed to store transaction in Qdrant: {transaction_id}")
                return None
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            return None

    async def _enrich_transaction(self, transaction: Any) -> None:
        """
        Enrich a transaction with additional information.
        
        Args:
            transaction: Transaction to enrich
        """
        try:
            # Clean description if not already cleaned
            if not transaction.clean_description:
                from ..utils.text_processors import clean_transaction_description
                transaction.clean_description = clean_transaction_description(transaction.description)
            
            # Find merchant
            merchant = await self.merchant_service.find_merchant_by_description(
                transaction.clean_description or transaction.description
            )
            
            if merchant:
                # Set merchant information
                transaction.normalized_merchant = merchant.get("name")
                merchant_id_str = merchant.get("id")
                if merchant_id_str:
                    try:
                        transaction.merchant_id = UUID(merchant_id_str)
                    except ValueError:
                        pass
                
                # Set category if not already set and merchant has a category
                if transaction.category_id is None and merchant.get("category_id") is not None:
                    transaction.category_id = merchant.get("category_id")
            else:
                # Try to normalize merchant name without vector search
                normalized_name = await self.merchant_service.normalize_merchant_name(
                    transaction.clean_description or transaction.description
                )
                if normalized_name:
                    transaction.normalized_merchant = normalized_name
            
            # Predict category if not set
            if transaction.category_id is None:
                predicted_category = await self.category_service.predict_category(
                    description=transaction.clean_description or transaction.description,
                    amount=transaction.amount
                )
                if predicted_category is not None:
                    transaction.category_id = predicted_category
            
            # Set parent category if we have a category
            if transaction.category_id is not None:
                category = await self.category_service.get_category(transaction.category_id)
                if category and category.get("parent_id") is not None:
                    transaction.parent_category_id = category.get("parent_id")
            
            # Extract geo information if available
            # This would use regex patterns or NLP to extract location info
            # For now, just a placeholder
            geo_info = self._extract_geo_info(transaction.description)
            if geo_info:
                transaction.geo_info = geo_info
        except Exception as e:
            logger.error(f"Error enriching transaction: {str(e)}")

    def _extract_geo_info(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Extract geographical information from a transaction description.
        
        Args:
            description: Transaction description
            
        Returns:
            Dictionary of geographical information or None if not found
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP and pattern matching
        geo_info = None
        
        # Example pattern: detect country codes
        country_match = re.search(r'\b([A-Z]{2})\b', description)
        if country_match:
            geo_info = {
                "country_code": country_match.group(1)
            }
        
        return geo_info

    def _generate_transaction_fingerprint(self, transaction: Any) -> str:
        """
        Generate a fingerprint for a transaction to identify duplicates.
        
        Args:
            transaction: Transaction to generate fingerprint for
            
        Returns:
            Fingerprint string
        """
        # Include key transaction properties in the fingerprint
        # This is a simplified implementation - real systems would use more robust methods
        components = [
            str(transaction.account_id),
            str(abs(transaction.amount)),  # Absolute value to match reversed transactions
            transaction.clean_description or transaction.description,
            transaction.transaction_date.isoformat()
        ]
        
        fingerprint = "_".join(components)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    async def get_transaction(self, transaction_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """
        Get a transaction by ID.
        
        Args:
            transaction_id: Transaction ID to retrieve
            
        Returns:
            Transaction data or None if not found
        """
        try:
            # Get from Qdrant
            return await self.qdrant_service.get_transaction(transaction_id)
        except Exception as e:
            logger.error(f"Error getting transaction {transaction_id}: {str(e)}")
            return None

    async def update_transaction(self, transaction_id: Union[str, UUID], updates: Dict[str, Any]) -> bool:
        """
        Update a transaction's information.
        
        Args:
            transaction_id: Transaction ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful
        """
        try:
            # Get current transaction data
            transaction = await self.get_transaction(transaction_id)
            if not transaction:
                logger.error(f"Transaction not found for update: {transaction_id}")
                return False
            
            # Update the fields
            for key, value in updates.items():
                if key in transaction and key not in ["id", "created_at", "bridge_transaction_id"]:
                    transaction[key] = value
            
            # Update timestamps
            transaction["updated_at"] = datetime.now().isoformat()
            
            # If description was updated, we need to regenerate the embedding
            if "description" in updates or "clean_description" in updates:
                description_for_embedding = transaction.get("clean_description") or transaction.get("description", "")
                description_embedding = await self.embedding_service.get_embedding(description_for_embedding)
                # Update fingerprint if description changed
                transaction["fingerprint"] = self._generate_transaction_fingerprint_from_dict(transaction)
            else:
                description_embedding = transaction.get("embedding", [])
            
            # Store updated transaction
            success = await self.qdrant_service.upsert_transaction(
                transaction_id=transaction_id,
                embedding=description_embedding,
                payload=transaction
            )
            
            return success
        except Exception as e:
            logger.error(f"Error updating transaction {transaction_id}: {str(e)}")
            return False

    def _generate_transaction_fingerprint_from_dict(self, transaction: Dict[str, Any]) -> str:
        """
        Generate a fingerprint for a transaction dictionary.
        
        Args:
            transaction: Transaction dictionary to generate fingerprint for
            
        Returns:
            Fingerprint string
        """
        # Include key transaction properties in the fingerprint
        components = [
            str(transaction.get("account_id", "")),
            str(abs(transaction.get("amount", 0))),
            transaction.get("clean_description", "") or transaction.get("description", ""),
            transaction.get("transaction_date", "")
        ]
        
        fingerprint = "_".join(components)
        return hashlib.md5(fingerprint.encode()).hexdigest()

    async def delete_transaction(self, transaction_id: Union[str, UUID]) -> bool:
        """
        Delete a transaction.
        
        Args:
            transaction_id: Transaction ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            return await self.qdrant_service.delete_transaction(transaction_id)
        except Exception as e:
            logger.error(f"Error deleting transaction {transaction_id}: {str(e)}")
            return False

    async def mark_transaction_as_deleted(self, transaction_id: Union[str, UUID]) -> bool:
        """
        Mark a transaction as deleted without actually removing it.
        
        Args:
            transaction_id: Transaction ID to mark as deleted
            
        Returns:
            True if update was successful
        """
        try:
            return await self.update_transaction(transaction_id, {"is_deleted": True})
        except Exception as e:
            logger.error(f"Error marking transaction as deleted {transaction_id}: {str(e)}")
            return False

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
        try:
            # Get embedding if not provided
            if embedding is None:
                if description:
                    embedding = await self.embedding_service.get_embedding(description)
                else:
                    raise ValueError("Either embedding or description must be provided")
            
            # Set up filter conditions
            filter_conditions = {}
            if user_id is not None:
                filter_conditions["user_id"] = user_id
            
            # Search for similar transactions
            similar_transactions = await self.qdrant_service.search_similar_transactions(
                embedding=embedding,
                user_id=user_id,
                limit=limit,
                score_threshold=0.6,  # Lower threshold to get more results
                filter_conditions=filter_conditions
            )
            
            return similar_transactions
        except Exception as e:
            logger.error(f"Error finding similar transactions: {str(e)}")
            return []

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
        try:
            # Si une requête textuelle est présente, utiliser la recherche hybride
            if search_params.query and self.search_service:
                return await self.search_service.search(
                    user_id=user_id,
                    query=search_params.query,
                    search_params=search_params,
                    top_k_initial=min(100, search_params.limit * 3),
                    top_k_final=search_params.limit
                )
            
            # Pas de requête textuelle, utiliser le filtrage standard
            filter_conditions = {"user_id": user_id}
            
            # Handle date range
            if search_params.start_date or search_params.end_date:
                date_range = {}
                if search_params.start_date:
                    date_range["start_date"] = search_params.start_date
                if search_params.end_date:
                    date_range["end_date"] = search_params.end_date
                filter_conditions["date_range"] = date_range
            
            # Handle amount range
            if search_params.min_amount is not None or search_params.max_amount is not None:
                amount_range = {}
                if search_params.min_amount is not None:
                    amount_range["min_amount"] = search_params.min_amount
                if search_params.max_amount is not None:
                    amount_range["max_amount"] = search_params.max_amount
                filter_conditions["amount_range"] = amount_range
            
            # Handle categories
            if search_params.categories:
                filter_conditions["categories"] = search_params.categories
            
            # Handle merchant names
            if search_params.merchant_names:
                # Convert merchant names to IDs if possible
                merchant_ids = []
                for name in search_params.merchant_names:
                    merchant = await self.merchant_service.find_merchant_by_description(name)
                    if merchant and "id" in merchant:
                        merchant_ids.append(merchant["id"])
                
                if merchant_ids:
                    filter_conditions["merchant_ids"] = merchant_ids
            
            # Handle operation types
            if search_params.operation_types:
                filter_conditions["operation_types"] = search_params.operation_types
            
            # Handle account IDs
            if search_params.account_ids:
                filter_conditions["account_ids"] = search_params.account_ids
            
            # Handle inclusion flags
            if not search_params.include_future:
                filter_conditions["is_future"] = False
            
            if not search_params.include_deleted:
                filter_conditions["is_deleted"] = False
            
            # Execute the search
            return await self.qdrant_service.filter_transactions(
                user_id=user_id,
                filter_conditions=filter_conditions,
                limit=search_params.limit,
                offset=search_params.offset
            )
            
        except Exception as e:
            logger.error(f"Error searching transactions: {str(e)}")
            return [], 0

    async def get_transaction_stats(
        self,
        user_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        account_id: Optional[int] = None,
        category_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get statistical summary of transactions.
        
        Args:
            user_id: User ID to get stats for
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            account_id: Optional account ID for filtering
            category_id: Optional category ID for filtering
            
        Returns:
            Dictionary of transaction statistics
        """
        try:
            # Prepare filter conditions
            filter_conditions = {}
            
            if start_date or end_date:
                date_range = {}
                if start_date:
                    date_range["start_date"] = start_date
                if end_date:
                    date_range["end_date"] = end_date
                filter_conditions["date_range"] = date_range
            
            if account_id:
                filter_conditions["account_id"] = account_id
                
            if category_id:
                filter_conditions["category_id"] = category_id
            
            # Get stats from Qdrant
            return await self.qdrant_service.get_transaction_stats(
                user_id=user_id,
                filter_conditions=filter_conditions
            )
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

    async def process_transactions_batch(self, transactions: List[Any]) -> List[Optional[Dict[str, Any]]]:
        """
        Process and store a batch of transactions.
        
        Args:
            transactions: List of transaction data to process
            
        Returns:
            List of processed transactions or None for failed ones
        """
        tasks = [self.process_transaction(tx) for tx in transactions]
        return await asyncio.gather(*tasks)

    async def explain_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate an explanation of search results.
        
        Args:
            query: Search query
            results: Search results
            
        Returns:
            Explanation data
        """
        explanation = {
            "query": query,
            "result_count": len(results),
            "search_method": "hybrid",
            "matching_terms": [],
            "top_categories": [],
            "date_range": {},
            "merchant_distribution": {},
            "relevance_factors": {
                "lexical_match": "Correspondance des termes exacts de la requête",
                "semantic_similarity": "Similarité sémantique avec les transactions",
                "contextual_relevance": "Pertinence contextuelle globale"
            }
        }
        
        # Extraire les termes de la requête
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Analyser les résultats
        if results:
            # Trouver les termes correspondants
            all_terms = []
            for result in results[:5]:  # Analyser les 5 premiers résultats
                desc = result.get("clean_description", "") or result.get("description", "")
                if desc:
                    desc_terms = set(re.findall(r'\b\w+\b', desc.lower()))
                    matching = desc_terms.intersection(query_terms)
                    all_terms.extend(matching)
            
            # Compter les occurrences
            term_counts = {}
            for term in all_terms:
                if term in term_counts:
                    term_counts[term] += 1
                else:
                    term_counts[term] = 1
            
            # Trier par fréquence
            explanation["matching_terms"] = sorted(
                term_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # 5 termes les plus fréquents
            
            # Analyser les catégories
            categories = {}
            for result in results:
                cat_id = result.get("category_id")
                cat_name = result.get("category_name") or "Inconnue"
                if cat_id:
                    key = f"{cat_id}:{cat_name}"
                    if key in categories:
                        categories[key] += 1
                    else:
                        categories[key] = 1
            
            # Top 3 catégories
            explanation["top_categories"] = sorted(
                categories.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

# Période couverte
            dates = [
                result.get("transaction_date")
                for result in results
                if "transaction_date" in result
            ]
            if dates:
                explanation["date_range"] = {
                    "min": min(dates),
                    "max": max(dates)
                }
            
            # Distribution des marchands
            merchants = {}
            for result in results:
                merchant = result.get("normalized_merchant") or "Inconnu"
                if merchant in merchants:
                    merchants[merchant] += 1
                else:
                    merchants[merchant] = 1
            
            explanation["merchant_distribution"] = sorted(
                merchants.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 marchands
        
        return explanation