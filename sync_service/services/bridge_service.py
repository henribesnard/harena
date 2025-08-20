import logging
import httpx
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode

from config.settings import settings

logger = logging.getLogger(__name__)

class BridgeService:
    """Service pour interagir avec Bridge API."""
    
    def __init__(self):
        self.api_url = settings.BRIDGE_API_URL
        self.api_version = settings.BRIDGE_API_VERSION
        self.client_id = settings.BRIDGE_CLIENT_ID
        self.client_secret = settings.BRIDGE_CLIENT_SECRET
    
    async def get_headers(self, access_token: Optional[str] = None, language: str = None) -> Dict[str, str]:
        """Prépare les en-têtes standards pour les appels API Bridge."""
        headers = {
            "accept": "application/json",
            "Bridge-Version": self.api_version,
            "Client-Id": self.client_id,
            "Client-Secret": self.client_secret,
        }
        
        if access_token:
            headers["authorization"] = f"Bearer {access_token}"
            
        if language:
            headers["Accept-Language"] = language
            
        return headers
    
    async def fetch_paginated_resources(self, url: str, headers: Dict[str, str], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Récupère toutes les ressources d'un endpoint paginé de Bridge."""
        all_resources = []
        next_uri = url
        page_count = 0
        max_pages = 50  # Limiter pour éviter les boucles infinies
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            while next_uri and page_count < max_pages:
                page_count += 1
                logger.debug(f"Récupération de la page {page_count} depuis: {next_uri}")
                
                try:
                    response = await client.get(next_uri, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    resources = data.get("resources", [])
                    all_resources.extend(resources)
                    logger.debug(f"Récupération de {len(resources)} ressources de la page {page_count}.")
                    
                    # Gestion de la pagination
                    pagination = data.get("pagination", {})
                    next_uri = pagination.get("next_uri")
                    
                    if isinstance(next_uri, str) and next_uri.lower() == 'null':
                        next_uri = None
                    elif next_uri and not next_uri.startswith("http"):
                        next_uri = f"{self.api_url}{next_uri}"
                    
                    # Arrêter si on a atteint la limite demandée
                    if limit is not None and len(all_resources) >= limit:
                        logger.info(f"Limite de {limit} ressources atteinte, arrêt de la pagination.")
                        all_resources = all_resources[:limit]
                        break
                
                except httpx.HTTPStatusError as e:
                    logger.error(f"Erreur HTTP lors de la récupération des ressources: {e.response.status_code} - {e.response.text}")
                    raise
                except Exception as e:
                    logger.error(f"Erreur inattendue lors de la récupération des ressources: {e}", exc_info=True)
                    raise
        
        return all_resources
    
    async def get_items(self, access_token: str) -> List[Dict[str, Any]]:
        """Récupère les items d'un utilisateur depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/items"
        params = {"limit": "50"}
        request_url = f"{url}?{urlencode(params)}"
        
        return await self.fetch_paginated_resources(request_url, headers)
    
    async def get_item(self, access_token: str, item_id: int) -> Optional[Dict[str, Any]]:
        """Récupère un item spécifique depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/items/{item_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Item {item_id} non trouvé dans Bridge API")
                return None
            logger.error(f"Erreur HTTP lors de la récupération de l'item {item_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'item {item_id}: {e}", exc_info=True)
            raise
    
    async def refresh_item(self, access_token: str, item_id: int) -> Dict[str, Any]:
        """Rafraîchit un item Bridge."""
        headers = await self.get_headers(access_token)
        headers["content-type"] = "application/json"
        
        url = f"{self.api_url}/aggregation/items/{item_id}/refresh"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json={})
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP lors du rafraîchissement de l'item {item_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du rafraîchissement de l'item {item_id}: {e}", exc_info=True)
            raise
    
    async def delete_item(self, access_token: str, item_id: int) -> bool:
        """Supprime un item Bridge."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/items/{item_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(url, headers=headers)
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'item {item_id}: {e}", exc_info=True)
            return False
    
    async def get_accounts(self, access_token: str, item_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Récupère les comptes d'un utilisateur depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/accounts"
        params = {"limit": "200"}
        
        if item_id:
            params["item_id"] = str(item_id)
        
        request_url = f"{url}?{urlencode(params)}"
        
        return await self.fetch_paginated_resources(request_url, headers)
    
    async def get_account(self, access_token: str, account_id: int) -> Optional[Dict[str, Any]]:
        """Récupère un compte spécifique depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/accounts/{account_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Compte {account_id} non trouvé dans Bridge API")
                return None
            logger.error(f"Erreur HTTP lors de la récupération du compte {account_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du compte {account_id}: {e}", exc_info=True)
            raise
    
    async def get_transactions(
        self, 
        access_token: str, 
        account_id: Optional[int] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Récupère les transactions depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/transactions"
        params = {"limit": str(limit)}
        
        if account_id:
            params["account_id"] = str(account_id)
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if min_date:
            params["min_date"] = min_date
        if max_date:
            params["max_date"] = max_date
        
        request_url = f"{url}?{urlencode(params)}"
        
        return await self.fetch_paginated_resources(request_url, headers, limit)
    
    async def get_transaction(self, access_token: str, transaction_id: int) -> Optional[Dict[str, Any]]:
        """Récupère une transaction spécifique depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/transactions/{transaction_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Transaction {transaction_id} non trouvée dans Bridge API")
                return None
            logger.error(f"Erreur HTTP lors de la récupération de la transaction {transaction_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la transaction {transaction_id}: {e}", exc_info=True)
            raise
    
    async def get_categories(self, access_token: str, language: str = "fr") -> List[Dict[str, Any]]:
        """Récupère les catégories depuis Bridge API."""
        headers = await self.get_headers(access_token, language=language)
        
        url = f"{self.api_url}/aggregation/categories"
        params = {"limit": "200"}
        
        request_url = f"{url}?{urlencode(params)}"
        
        categories = await self.fetch_paginated_resources(request_url, headers)
        
        # Transformer en liste à plat pour faciliter l'utilisation
        flat_categories = []
        for parent_cat in categories:
            parent_id = parent_cat.get('id')
            parent_name = parent_cat.get('name')
            
            if 'categories' in parent_cat and isinstance(parent_cat['categories'], list):
                for sub_cat in parent_cat['categories']:
                    flat_categories.append({
                        "id": sub_cat.get('id'),
                        "name": sub_cat.get('name'),
                        "parent_id": parent_id,
                        "parent_name": parent_name
                    })
            else:
                # Ajouter la catégorie parente si elle n'a pas de sous-catégories
                flat_categories.append({
                    "id": parent_id,
                    "name": parent_name,
                    "parent_id": None,
                    "parent_name": None
                })
        
        return flat_categories
    
    async def get_category(self, access_token: str, category_id: int, language: str = "fr") -> Optional[Dict[str, Any]]:
        """Récupère une catégorie spécifique depuis Bridge API."""
        headers = await self.get_headers(access_token, language=language)
        
        url = f"{self.api_url}/aggregation/categories/{category_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Catégorie {category_id} non trouvée dans Bridge API")
                return None
            logger.error(f"Erreur HTTP lors de la récupération de la catégorie {category_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la catégorie {category_id}: {e}", exc_info=True)
            raise
    
    async def get_stocks(
        self, 
        access_token: str, 
        account_id: Optional[int] = None,
        since: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Récupère les stocks depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/stocks"
        params = {"limit": str(limit)}
        
        if account_id:
            params["account_id"] = str(account_id)
        if since:
            params["since"] = since
        
        request_url = f"{url}?{urlencode(params)}"
        
        return await self.fetch_paginated_resources(request_url, headers, limit)
    
    async def get_stock(self, access_token: str, stock_id: int) -> Optional[Dict[str, Any]]:
        """Récupère un stock spécifique depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/stocks/{stock_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Stock {stock_id} non trouvé dans Bridge API")
                return None
            logger.error(f"Erreur HTTP lors de la récupération du stock {stock_id}: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du stock {stock_id}: {e}", exc_info=True)
            raise
    
    async def get_insights(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Récupère les insights depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/insights/category"
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [403, 404]:
                logger.warning(f"Insights non disponibles dans Bridge API: {e.response.status_code}")
                return None
            logger.error(f"Erreur HTTP lors de la récupération des insights: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des insights: {e}", exc_info=True)
            raise
    
    async def get_accounts_information(self, access_token: str) -> Optional[List[Dict[str, Any]]]:
        """Récupère les informations de compte (IBAN, identité) depuis Bridge API."""
        headers = await self.get_headers(access_token)
        
        url = f"{self.api_url}/aggregation/accounts-information"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result.get("resources", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [403, 404]:
                logger.warning(f"Informations de compte non disponibles dans Bridge API: {e.response.status_code}")
                return None
            logger.error(f"Erreur HTTP lors de la récupération des informations de compte: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations de compte: {e}", exc_info=True)
            raise
    
    async def create_connect_session(
        self,
        access_token: str,
        user_email: Optional[str] = None,
        callback_url: Optional[str] = None,
        country_code: Optional[str] = "FR",
        account_types: Optional[str] = "payment",
        context: Optional[str] = None,
        provider_id: Optional[int] = None,
        item_id: Optional[int] = None
    ) -> str:
        """Crée une session de connexion Bridge."""
        headers = await self.get_headers(access_token)
        headers["content-type"] = "application/json"
        
        url = f"{self.api_url}/aggregation/connect-sessions"
        
        payload = {}
        if item_id:
            payload["item_id"] = item_id
        elif provider_id:
            payload["provider_id"] = provider_id
            if user_email:
                payload["user_email"] = user_email
        elif user_email:
            payload["user_email"] = user_email
        
        if callback_url:
            payload["callback_url"] = callback_url
        if country_code:
            payload["country_code"] = country_code
        if account_types:
            payload["account_types"] = account_types
        if context and len(context) <= 100:
            payload["context"] = context
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if "url" not in data:
                    logger.error(f"Session de connexion créée mais aucune URL retournée: {data}")
                    raise ValueError("No URL returned in connect session response")
                
                return data["url"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP lors de la création de la session de connexion: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la création de la session de connexion: {e}", exc_info=True)
            raise