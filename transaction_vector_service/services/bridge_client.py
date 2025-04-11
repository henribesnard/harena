# transaction_vector_service/services/bridge_client.py
"""
Client for Bridge API integration.

This module provides a client for interacting with the Bridge API,
handling authentication, request building, and response processing.
"""

import httpx
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from ..config.settings import settings
from ..config.logging_config import get_logger
from ..models.transaction import Transaction, TransactionCreate
from ..models.category import Category, CategoryCreate
from ..models.account import Account

logger = get_logger(__name__)


class BridgeClient:
    """Client for interacting with the Bridge API."""

    def __init__(self):
        """Initialize the Bridge API client."""
        self.base_url = settings.BRIDGE_API_URL
        self.api_version = settings.BRIDGE_API_VERSION
        self.client_id = settings.BRIDGE_CLIENT_ID
        self.client_secret = settings.BRIDGE_CLIENT_SECRET
        
        # Request timeout settings
        self.timeout = 30.0  # seconds
        
        # Cache for access tokens
        self._token_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Bridge API client initialized")

    async def _get_headers(self, user_token: Optional[str] = None) -> Dict[str, str]:
        """
        Generate headers for Bridge API requests.
        
        Args:
            user_token: Optional user-specific token for authenticated requests
            
        Returns:
            Dict of headers
        """
        headers = {
            "accept": "application/json",
            "Bridge-Version": self.api_version,
            "Client-Id": self.client_id,
            "Client-Secret": self.client_secret,
            "content-type": "application/json",
        }
        
        if user_token:
            headers["authorization"] = f"Bearer {user_token}"
            
        return headers

    async def get_user_token(self, user_uuid: str, external_user_id: Optional[str] = None) -> str:
        """
        Get an access token for a user.
        
        Args:
            user_uuid: Bridge user UUID
            external_user_id: Optional external user ID, used as fallback
            
        Returns:
            Access token for the user
        """
        # Check if we have a valid cached token
        cache_key = user_uuid or external_user_id
        if cache_key in self._token_cache:
            token_data = self._token_cache[cache_key]
            expires_at = datetime.fromisoformat(token_data["expires_at"].replace("Z", "+00:00"))
            
            # If token is still valid (with 5 minutes buffer)
            if expires_at > datetime.now() + timedelta(minutes=5):
                logger.debug(f"Using cached token for user {cache_key}")
                return token_data["access_token"]
        
        # Request new token
        url = f"{self.base_url}/aggregation/authorization/token"
        payload = {}
        
        if user_uuid:
            payload["user_uuid"] = user_uuid
        elif external_user_id:
            payload["external_user_id"] = external_user_id
        else:
            raise ValueError("Either user_uuid or external_user_id must be provided")
        
        headers = await self._get_headers()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                token_data = response.json()
                
                # Cache the token
                if cache_key:
                    self._token_cache[cache_key] = token_data
                    
                return token_data["access_token"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting user token: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting user token: {str(e)}")
            raise

    async def get_transactions(
        self, 
        user_token: str, 
        account_id: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get transactions from Bridge API.
        
        Args:
            user_token: User access token
            account_id: Optional account ID to filter transactions
            since: Optional timestamp to get transactions updated after
            until: Optional timestamp to get transactions updated before
            min_date: Optional date to filter transactions by date (lower bound)
            max_date: Optional date to filter transactions by date (upper bound)
            limit: Number of transactions to fetch (default: 50)
            
        Returns:
            List of transaction data
        """
        url = f"{self.base_url}/aggregation/transactions"
        params = {"limit": min(limit, 500)}  # Bridge API limit is 500
        
        if account_id:
            params["account_id"] = account_id
            
        if since:
            params["since"] = since.isoformat()
            
        if until:
            params["until"] = until.isoformat()
            
        if min_date:
            params["min_date"] = min_date.date().isoformat()
            
        if max_date:
            params["max_date"] = max_date.date().isoformat()
        
        headers = await self._get_headers(user_token)
        all_transactions = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # First request
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if "resources" in data:
                    all_transactions.extend(data["resources"])
                    
                    # Handle pagination if needed
                    while "pagination" in data and data["pagination"].get("next_uri"):
                        next_uri = data["pagination"]["next_uri"]
                        # Make sure we only take the path part if it's a full URL
                        if next_uri.startswith("http"):
                            from urllib.parse import urlparse
                            parts = urlparse(next_uri)
                            next_uri = parts.path
                            if parts.query:
                                next_uri += f"?{parts.query}"
                        
                        next_url = f"{self.base_url}{next_uri}"
                        response = await client.get(next_url, headers=headers)
                        response.raise_for_status()
                        data = response.json()
                        
                        if "resources" in data:
                            all_transactions.extend(data["resources"])
                        else:
                            break
                
                return all_transactions
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting transactions: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting transactions: {str(e)}")
            raise

    async def get_accounts(
        self, 
        user_token: str, 
        item_id: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get accounts from Bridge API.
        
        Args:
            user_token: User access token
            item_id: Optional item ID to filter accounts
            limit: Number of accounts to fetch (default: 50)
            
        Returns:
            List of account data
        """
        url = f"{self.base_url}/aggregation/accounts"
        params = {"limit": min(limit, 500)}  # Bridge API limit is 500
        
        if item_id:
            params["item_id"] = item_id
            
        headers = await self._get_headers(user_token)
        all_accounts = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # First request
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if "resources" in data:
                    all_accounts.extend(data["resources"])
                    
                    # Handle pagination if needed
                    while "pagination" in data and data["pagination"].get("next_uri"):
                        next_uri = data["pagination"]["next_uri"]
                        # Make sure we only take the path part if it's a full URL
                        if next_uri.startswith("http"):
                            from urllib.parse import urlparse
                            parts = urlparse(next_uri)
                            next_uri = parts.path
                            if parts.query:
                                next_uri += f"?{parts.query}"
                        
                        next_url = f"{self.base_url}{next_uri}"
                        response = await client.get(next_url, headers=headers)
                        response.raise_for_status()
                        data = response.json()
                        
                        if "resources" in data:
                            all_accounts.extend(data["resources"])
                        else:
                            break
                
                return all_accounts
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting accounts: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting accounts: {str(e)}")
            raise

    async def get_categories(self, language: str = "fr") -> List[Dict[str, Any]]:
        """
        Get categories from Bridge API.
        
        Args:
            language: Language code for category names (default: "fr")
            
        Returns:
            List of category data
        """
        url = f"{self.base_url}/aggregation/categories"
        headers = await self._get_headers()
        headers["Accept-Language"] = language
        
        all_categories = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # First request
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if "resources" in data:
                    # Bridge API returns categories in a nested structure
                    for parent in data["resources"]:
                        parent_id = parent.get("id")
                        parent_name = parent.get("name")
                        
                        # Add parent category
                        all_categories.append({
                            "id": parent_id,
                            "name": parent_name,
                            "parent_id": None
                        })
                        
                        # Add subcategories
                        for category in parent.get("categories", []):
                            category["parent_id"] = parent_id
                            all_categories.append(category)
                    
                    # Handle pagination if needed
                    while "pagination" in data and data["pagination"].get("next_uri") and data["pagination"]["next_uri"] != "null":
                        next_uri = data["pagination"]["next_uri"]
                        # Make sure we only take the path part if it's a full URL
                        if next_uri.startswith("http"):
                            from urllib.parse import urlparse
                            parts = urlparse(next_uri)
                            next_uri = parts.path
                            if parts.query:
                                next_uri += f"?{parts.query}"
                        
                        next_url = f"{self.base_url}{next_uri}"
                        response = await client.get(next_url, headers=headers)
                        response.raise_for_status()
                        data = response.json()
                        
                        if "resources" in data:
                            for parent in data["resources"]:
                                parent_id = parent.get("id")
                                parent_name = parent.get("name")
                                
                                # Add parent category
                                all_categories.append({
                                    "id": parent_id,
                                    "name": parent_name,
                                    "parent_id": None
                                })
                                
                                # Add subcategories
                                for category in parent.get("categories", []):
                                    category["parent_id"] = parent_id
                                    all_categories.append(category)
                        else:
                            break
                
                return all_categories
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting categories: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            raise

    async def get_items(self, user_token: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get items from Bridge API.
        
        Args:
            user_token: User access token
            limit: Number of items to fetch (default: 50)
            
        Returns:
            List of item data
        """
        url = f"{self.base_url}/aggregation/items"
        params = {"limit": min(limit, 500)}  # Bridge API limit is 500
        headers = await self._get_headers(user_token)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if "resources" in data:
                    return data["resources"]
                return []
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting items: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting items: {str(e)}")
            raise

    async def create_user(self, external_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new user in Bridge API.
        
        Args:
            external_user_id: Optional external user ID to associate with the Bridge user
            
        Returns:
            User data including UUID
        """
        url = f"{self.base_url}/aggregation/users"
        payload = {}
        
        if external_user_id:
            payload["external_user_id"] = external_user_id
            
        headers = await self._get_headers()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error creating user: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise

    async def delete_user(self, user_uuid: str) -> bool:
        """
        Delete a user from Bridge API.
        
        Args:
            user_uuid: Bridge user UUID
            
        Returns:
            True if deletion was successful
        """
        url = f"{self.base_url}/aggregation/users/{user_uuid}"
        headers = await self._get_headers()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(url, headers=headers)
                response.raise_for_status()
                return True
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error deleting user: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            raise

    async def create_connect_session(
        self,
        user_token: str,
        user_email: str,
        callback_url: Optional[str] = None,
        country_code: str = "FR",
        account_types: str = "payment",
        context: Optional[str] = None,
        provider_id: Optional[int] = None,
        item_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a connect session for bank authentication.
        
        Args:
            user_token: User access token
            user_email: User email address
            callback_url: Optional URL to redirect after authentication
            country_code: Country code for bank selection (default: "FR")
            account_types: Account types to connect (default: "payment")
            context: Optional context string for callback
            provider_id: Optional provider ID for preselected bank
            item_id: Optional item ID for managing existing connection
            
        Returns:
            Session data including URL
        """
        url = f"{self.base_url}/aggregation/connect-sessions"
        payload = {"user_email": user_email}
        
        if callback_url:
            payload["callback_url"] = callback_url
            
        if country_code:
            payload["country_code"] = country_code
            
        if account_types:
            payload["account_types"] = account_types
            
        if context:
            payload["context"] = context
            
        if provider_id:
            payload["provider_id"] = provider_id
            
        if item_id:
            payload["item_id"] = item_id
            
        headers = await self._get_headers(user_token)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error creating connect session: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error creating connect session: {str(e)}")
            raise