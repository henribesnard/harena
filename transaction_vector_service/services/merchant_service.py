"""
Service for managing merchant data.

This module provides functionality for detecting, normalizing, and managing
merchant information from transaction descriptions.
"""

import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime
from uuid import UUID, uuid4

from ..config.logging_config import get_logger
from ..common.types import SIMILARITY_THRESHOLD
from ..config.constants import MERCHANT_CACHE_TTL 
from ..models.interfaces import MerchantServiceInterface, EmbeddingServiceInterface, QdrantServiceInterface

logger = get_logger(__name__)


class MerchantService(MerchantServiceInterface):
    """Service for managing merchant data and detection."""

    def __init__(self, embedding_service: EmbeddingServiceInterface, qdrant_service: QdrantServiceInterface):
        """
        Initialize the merchant service.
        
        Args:
            embedding_service: Embedding service instance
            qdrant_service: Qdrant service instance
        """
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self._merchant_cache: Dict[str, Dict[str, Any]] = {}
        self._last_cache_update: Optional[datetime] = None
        
        # Common patterns to clean from transaction descriptions
        self._common_patterns = [
            r'PAIEMENT PAR CARTE',
            r'CB \d{2}/\d{2} ',
            r'CARTE \d{2}/\d{2} ',
            r'ACHAT (DU|LE) \d{2}/\d{2}',
            r'FACTURE \d+',
            r'TRANSFERT DE ',
            r'VIR(EMENT)? (DE|POUR) ',
            r'PRELEVEMENT (DE|POUR) ',
            r'\d{2}/\d{2}/\d{2,4}',  # Dates
            r'REF: \w+',  # Reference numbers
            r'ID: \w+',   # ID numbers
        ]
        
        # Common prefixes to remove
        self._common_prefixes = [
            'PAIEMENT ',
            'ACHAT ',
            'FACTURE ',
            'REGLEMENT ',
        ]
        
        # Common merchant name patterns
        self._merchant_patterns = [
            # Large retailers
            {'pattern': r'(?:carrefour|carref\s)', 'name': 'Carrefour'},
            {'pattern': r'(?:auchan)', 'name': 'Auchan'},
            {'pattern': r'(?:leclerc|e\.leclerc)', 'name': 'E.Leclerc'},
            {'pattern': r'(?:lidl)', 'name': 'Lidl'},
            {'pattern': r'(?:intermarche|intermarché)', 'name': 'Intermarché'},
            {'pattern': r'(?:monoprix)', 'name': 'Monoprix'},
            {'pattern': r'(?:franprix)', 'name': 'Franprix'},
            {'pattern': r'(?:casino\s)', 'name': 'Casino'},
            
            # Fast food chains
            {'pattern': r'(?:mcdo|mcdonald)', 'name': 'McDonald\'s'},
            {'pattern': r'(?:burger\s*king)', 'name': 'Burger King'},
            {'pattern': r'(?:kfc)', 'name': 'KFC'},
            {'pattern': r'(?:subway)', 'name': 'Subway'},
            
            # Tech companies
            {'pattern': r'(?:amazon|amzn)', 'name': 'Amazon'},
            {'pattern': r'(?:netflix)', 'name': 'Netflix'},
            {'pattern': r'(?:spotify)', 'name': 'Spotify'},
            {'pattern': r'(?:apple\s|itunes)', 'name': 'Apple'},
            {'pattern': r'(?:google\s|googleplay)', 'name': 'Google'},
            
            # Transportation
            {'pattern': r'(?:uber|uber\s*eats)', 'name': 'Uber'},
            {'pattern': r'(?:sncf)', 'name': 'SNCF'},
            {'pattern': r'(?:ratp)', 'name': 'RATP'},
            {'pattern': r'(?:flixbus)', 'name': 'FlixBus'},
            
            # Telecom
            {'pattern': r'(?:orange\s|orange\b)', 'name': 'Orange'},
            {'pattern': r'(?:sfr)', 'name': 'SFR'},
            {'pattern': r'(?:free\s|free\b)', 'name': 'Free'},
            {'pattern': r'(?:bouygues)', 'name': 'Bouygues Telecom'},
        ]
        
        logger.info("Merchant service initialized")

    async def normalize_merchant_name(self, description: str) -> Optional[str]:
        """
        Extract and normalize merchant name from a transaction description.
        
        Args:
            description: Transaction description
            
        Returns:
            Normalized merchant name or None if no match
        """
        if not description:
            return None
        
        # Convert to lowercase
        description_lower = description.lower().strip()
        
        # Remove common patterns
        cleaned_desc = description_lower
        for pattern in self._common_patterns:
            cleaned_desc = re.sub(pattern, '', cleaned_desc, flags=re.IGNORECASE)
        
        # Remove common prefixes
        for prefix in self._common_prefixes:
            if cleaned_desc.startswith(prefix.lower()):
                cleaned_desc = cleaned_desc[len(prefix):].strip()
        
        # Check for known merchant patterns
        for merchant_pattern in self._merchant_patterns:
            if re.search(merchant_pattern['pattern'], cleaned_desc, re.IGNORECASE):
                return merchant_pattern['name']
        
        # If no pattern match, try heuristics to extract the name
        # Split and take first 2-3 words as merchant name
        words = cleaned_desc.split()
        if len(words) >= 3:
            return ' '.join(words[:2]).strip()
        elif len(words) > 0:
            return words[0].strip()
        
        return None

    async def find_merchant_by_description(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Find a merchant based on transaction description using vector similarity.
        
        Args:
            description: Transaction description
            
        Returns:
            Merchant data if found, None otherwise
        """
        if not description:
            return None
            
        # First try rule-based normalization
        normalized_name = await self.normalize_merchant_name(description)
        
        # If we have a normalized name, check if it's in our local cache first
        if normalized_name and normalized_name in self._merchant_cache:
            return self._merchant_cache[normalized_name]
        
        # If not found in cache or no normalized name, use vector similarity
        try:
            # Generate embedding for the description
            description_embedding = await self.embedding_service.get_embedding(description)
            
            # Search for similar merchants
            similar_merchants = await self.qdrant_service.search_merchants_by_pattern(
                embedding=description_embedding,
                limit=1,
                score_threshold=SIMILARITY_THRESHOLD
            )
            
            if similar_merchants and len(similar_merchants) > 0:
                merchant = similar_merchants[0]
                
                # Add to cache if not already there
                if normalized_name and normalized_name not in self._merchant_cache:
                    self._merchant_cache[normalized_name] = merchant
                    
                return merchant
        except Exception as e:
            logger.error(f"Error finding merchant by description: {str(e)}")
        
        return None

    async def add_transaction_to_merchant(self, merchant_id: Union[str, UUID], transaction_id: Union[str, UUID]) -> bool:
        """
        Associate a transaction with a merchant and update stats.
        
        Args:
            merchant_id: Merchant ID
            transaction_id: Transaction ID to associate
            
        Returns:
            True if association was successful
        """
        try:
            # Get current merchant data
            merchant = await self.get_merchant(merchant_id)
            if not merchant:
                logger.error(f"Merchant not found: {merchant_id}")
                return False
            
            # Update transaction count
            transaction_count = merchant.get("transaction_count", 0) + 1
            
            # Store transaction ID if tracking individual transactions
            transaction_ids = merchant.get("transaction_ids", [])
            if isinstance(transaction_id, UUID):
                transaction_id = str(transaction_id)
            
            if transaction_id not in transaction_ids:
                transaction_ids.append(transaction_id)
            
            # Update merchant
            updates = {
                "transaction_count": transaction_count,
                "transaction_ids": transaction_ids,
                "updated_at": datetime.now().isoformat()
            }
            
            return await self.update_merchant(merchant_id, updates)
        except Exception as e:
            logger.error(f"Error adding transaction to merchant: {str(e)}")
            return False

    async def get_merchant(self, merchant_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """
        Get a merchant by ID.
        
        Args:
            merchant_id: Merchant ID to retrieve
            
        Returns:
            Merchant data or None if not found
        """
        try:
            # Get from Qdrant
            return await self.qdrant_service.get_merchant(merchant_id)
        except Exception as e:
            logger.error(f"Error getting merchant {merchant_id}: {str(e)}")
            return None

    async def update_merchant(self, merchant_id: Union[str, UUID], updates: Dict[str, Any]) -> bool:
        """
        Update a merchant's information.
        
        Args:
            merchant_id: Merchant ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful
        """
        try:
            # Get current merchant data
            merchant = await self.get_merchant(merchant_id)
            if not merchant:
                logger.error(f"Merchant not found for update: {merchant_id}")
                return False
            
            # Update the fields
            for key, value in updates.items():
                if key in merchant and key not in ["id", "created_at"]:
                    merchant[key] = value
            
            # Update timestamps
            merchant["updated_at"] = datetime.now().isoformat()
            
            # Generate new embedding if name was updated
            if "name" in updates:
                name_embedding = await self.embedding_service.get_embedding(merchant["name"])
            else:
                name_embedding = merchant.get("embedding", [])
            
            # Store updated merchant
            success = await self.qdrant_service.upsert_merchant(
                merchant_id=merchant_id,
                embedding=name_embedding,
                payload=merchant
            )
            
            if success:
                # Update cache if name is in it
                old_name = merchant.get("name", "").lower()
                if old_name in self._merchant_cache:
                    # Remove old entry if name changed
                    if "name" in updates and old_name != updates["name"].lower():
                        del self._merchant_cache[old_name]
                    
                    # Add updated entry
                    new_name = updates.get("name", merchant.get("name", "")).lower()
                    self._merchant_cache[new_name] = {
                        "id": str(merchant_id),
                        "name": updates.get("name", merchant.get("name", "")),
                        "display_name": updates.get("display_name", merchant.get("display_name", "")),
                        "category_id": updates.get("category_id", merchant.get("category_id")),
                        "score": 1.0
                    }
                
                return True
            else:
                logger.error(f"Failed to update merchant in Qdrant: {merchant_id}")
                return False
        except Exception as e:
            logger.error(f"Error updating merchant {merchant_id}: {str(e)}")
            return False