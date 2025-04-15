"""
Service for managing transaction categories.

This module provides functionality for retrieving, caching, and maintaining
the category hierarchy used for classifying financial transactions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta

from ..config.settings import settings
from ..config.logging_config import get_logger
from ..common.types import CATEGORY_CACHE_TTL
from ..config.constants import CATEGORY_LEVELS
from ..models.interfaces import CategoryServiceInterface

logger = get_logger(__name__)


class CategoryService(CategoryServiceInterface):
    """Service for managing transaction categories."""

    def __init__(self):
        """Initialize the category service."""
        # Self-import to avoid circular imports
        from .bridge_client import BridgeClient
        self.bridge_client = BridgeClient()
        
        self._categories_cache: Dict[int, Dict[str, Any]] = {}
        self._category_hierarchy_cache: Optional[List[Dict[str, Any]]] = None
        self._last_cache_update: Optional[datetime] = None
        self._category_keywords: Dict[int, List[str]] = {}
        self._category_name_to_id: Dict[str, int] = {}
        self._cache_lock = asyncio.Lock()
        
        logger.info("Category service initialized")

    async def preload_categories(self, force_refresh: bool = False):
        """
        Preload categories from Bridge API to local cache.
        
        Args:
            force_refresh: Whether to force a refresh even if cache is valid
        """
        async with self._cache_lock:
            # Check if cache needs refresh
            needs_refresh = (
                force_refresh or
                self._last_cache_update is None or
                self._categories_cache == {} or
                datetime.now() - self._last_cache_update > timedelta(seconds=CATEGORY_CACHE_TTL)
            )
            
            if needs_refresh:
                try:
                    # Fetch categories from Bridge API
                    raw_categories = await self.bridge_client.get_categories()
                    
                    # Process raw categories into structured format
                    self._categories_cache = {}
                    self._category_name_to_id = {}
                    self._category_keywords = {}
                    
                    # First pass: Store all categories with basic info
                    for category in raw_categories:
                        category_id = category.get("id")
                        if category_id is not None:
                            # Determine category level
                            level = "primary" if category.get("parent_id") is None else "secondary"
                            
                            # Create category object
                            self._categories_cache[category_id] = {
                                "id": category_id,
                                "name": category.get("name", ""),
                                "display_name": category.get("name", ""),
                                "parent_id": category.get("parent_id"),
                                "level": level,
                                "path": [],
                                "keywords": [],
                                "children": []
                            }
                            
                            # Map name to ID for lookups
                            name = category.get("name", "").lower()
                            if name:
                                self._category_name_to_id[name] = category_id
                    
                    # Second pass: Build paths and parent-child relationships
                    for category_id, category in self._categories_cache.items():
                        parent_id = category.get("parent_id")
                        if parent_id:
                            # Add current category to parent's children
                            if parent_id in self._categories_cache:
                                self._categories_cache[parent_id]["children"].append(category_id)
                            
                            # Build path (list of ancestors)
                            path = [parent_id]
                            self._categories_cache[category_id]["path"] = path
                    
                    # Generate keywords for categories
                    for category_id, category in self._categories_cache.items():
                        # Start with the category name as a keyword
                        keywords = [category["name"].lower()]
                        
                        # Add parent name if applicable
                        parent_id = category.get("parent_id")
                        if parent_id and parent_id in self._categories_cache:
                            parent_name = self._categories_cache[parent_id]["name"]
                            keywords.append(parent_name.lower())
                            
                        # Store keywords
                        self._category_keywords[category_id] = keywords
                        self._categories_cache[category_id]["keywords"] = keywords
                    
                    # Build the hierarchy cache
                    self._build_hierarchy_cache()
                    
                    # Update cache timestamp
                    self._last_cache_update = datetime.now()
                    logger.info(f"Category cache refreshed with {len(self._categories_cache)} categories")
                except Exception as e:
                    logger.error(f"Error refreshing category cache: {str(e)}")
                    # If this is the first load and it failed, initialize with empty values
                    if self._categories_cache == {}:
                        self._categories_cache = {}
                        self._category_name_to_id = {}
                        self._category_keywords = {}
                        self._category_hierarchy_cache = []

    def _build_hierarchy_cache(self):
        """Build the category hierarchy tree structure."""
        # Get all top-level categories (no parent)
        top_level_categories = [
            cat_id for cat_id, cat in self._categories_cache.items()
            if cat.get("parent_id") is None
        ]
        
        # Build the hierarchy recursively
        hierarchy = []
        for cat_id in top_level_categories:
            category_tree = self._build_category_subtree(cat_id)
            if category_tree:
                hierarchy.append(category_tree)
        
        self._category_hierarchy_cache = hierarchy

    def _build_category_subtree(self, category_id: int) -> Optional[Dict[str, Any]]:
        """
        Build a category subtree starting from the given category ID.
        
        Args:
            category_id: Root category ID for the subtree
            
        Returns:
            Dictionary representing the category subtree or None if category not found
        """
        if category_id not in self._categories_cache:
            return None
        
        category = self._categories_cache[category_id]
        result = {
            "id": category["id"],
            "name": category["name"],
            "display_name": category["display_name"],
            "level": category["level"],
            "children": []
        }
        
        # Add children recursively
        for child_id in category.get("children", []):
            child_tree = self._build_category_subtree(child_id)
            if child_tree:
                result["children"].append(child_tree)
        
        return result

    async def get_category(self, category_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a category by ID.
        
        Args:
            category_id: Category ID to retrieve
            
        Returns:
            Category data or None if not found
        """
        # Ensure cache is loaded
        if not self._categories_cache:
            await self.preload_categories()
            
        # Return from cache
        return self._categories_cache.get(category_id)

    async def get_category_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a category by name.
        
        Args:
            name: Category name to search for
            
        Returns:
            Category data or None if not found
        """
        # Ensure cache is loaded
        if not self._categories_cache:
            await self.preload_categories()
            
        # Look up ID from name mapping
        name_lower = name.lower()
        category_id = self._category_name_to_id.get(name_lower)
        
        if category_id:
            return self._categories_cache.get(category_id)
        
        return None

    async def predict_category(self, description: str, amount: float) -> Optional[int]:
        """
        Predict the most likely category for a transaction based on description and amount.
        
        Args:
            description: Transaction description
            amount: Transaction amount
            
        Returns:
            Predicted category ID or None
        """
        # This is a simple implementation. In a real system, this would use ML.
        # Ensure cache is loaded
        if not self._categories_cache:
            await self.preload_categories()
            
        description_lower = description.lower()
        
        # Check for exact matches with category names
        for name, category_id in self._category_name_to_id.items():
            if name in description_lower:
                return category_id
        
        # Check for keyword matches
        best_match = None
        longest_match = 0
        
        for category_id, keywords in self._category_keywords.items():
            for keyword in keywords:
                if keyword in description_lower and len(keyword) > longest_match:
                    longest_match = len(keyword)
                    best_match = category_id
        
        return best_match

    async def get_all_categories(self) -> List[Dict[str, Any]]:
        """
        Get all categories.
        
        Returns:
            List of all categories
        """
        # Ensure cache is loaded
        if not self._categories_cache:
            await self.preload_categories()
            
        return list(self._categories_cache.values())

    async def get_category_hierarchy(self) -> List[Dict[str, Any]]:
        """
        Get the full category hierarchy.
        
        Returns:
            Nested structure of categories
        """
        # Ensure cache is loaded
        if not self._categories_cache:
            await self.preload_categories()
            
        return self._category_hierarchy_cache or []

    async def get_category_keywords(self, category_id: int) -> List[str]:
        """
        Get keywords associated with a category.
        
        Args:
            category_id: Category ID to get keywords for
            
        Returns:
            List of keywords for the category
        """
        # Ensure cache is loaded
        if not self._categories_cache:
            await self.preload_categories()
            
        return self._category_keywords.get(category_id, [])

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the category cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "total_categories": len(self._categories_cache),
            "primary_categories": sum(1 for c in self._categories_cache.values() if c.get("level") == "primary"),
            "secondary_categories": sum(1 for c in self._categories_cache.values() if c.get("level") == "secondary"),
            "last_update": self._last_cache_update.isoformat() if self._last_cache_update else None,
            "cache_age_seconds": (datetime.now() - self._last_cache_update).total_seconds() if self._last_cache_update else None,
        }