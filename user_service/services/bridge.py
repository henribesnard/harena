"""
Service de communication avec l'API Bridge.

Ce module fournit les fonctionnalités pour interagir avec l'API Bridge,
gérer les tokens d'authentification et récupérer les données financières.
"""
import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Union
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from db_service.models.user import User, BridgeConnection
from config.settings import settings


# Configuration des logs
logger = logging.getLogger(__name__)

# --- Fonctions d'aide internes ---

async def _get_bridge_headers(
    db: Optional[Session] = None,
    user_id: Optional[int] = None,
    include_auth: bool = True,
    language: Optional[str] = None
) -> Dict[str, str]:
    """Prépare les en-têtes standards pour les appels API Bridge."""
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
    }
    
    if include_auth:
        if db is None or user_id is None:
            raise ValueError("Database session and user_id are required for authenticated headers.")
        try:
            token_data = await get_bridge_token(db, user_id)
            headers["authorization"] = f"Bearer {token_data['access_token']}"
        except HTTPException as e:
            logger.error(f"Erreur lors de la récupération du token pour user {user_id} dans _get_bridge_headers: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération du token pour user {user_id} dans _get_bridge_headers: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve authentication token")

    if language:
        headers["Accept-Language"] = language

    return headers


async def _fetch_paginated_resources(
    initial_url: str,
    headers: Dict[str, str],
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Récupère toutes les ressources d'un endpoint paginé de Bridge."""
    all_resources = []
    next_uri = initial_url
    page_count = 0
    max_pages = 50

    async with httpx.AsyncClient(timeout=60.0) as client:
        while next_uri and page_count < max_pages:
            page_count += 1
            logger.debug(f"Fetching page {page_count} from: {next_uri}")
            try:
                response = await client.get(next_uri, headers=headers)
                response.raise_for_status()

                data = response.json()
                resources = data.get("resources", [])
                all_resources.extend(resources)
                logger.debug(f"Fetched {len(resources)} resources from page {page_count}.")

                pagination = data.get("pagination", {})
                next_uri = pagination.get("next_uri")
                if isinstance(next_uri, str) and next_uri.lower() == 'null':
                    next_uri = None

                if limit is not None and len(all_resources) >= limit:
                    logger.info(f"Limite de {limit} ressources atteinte, arrêt de la pagination.")
                    all_resources = all_resources[:limit]
                    break

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching paginated resources from {e.request.url}: {e.response.status_code} - {e.response.text}")
                raise HTTPException(status_code=e.response.status_code, detail=f"Bridge API error: {e.response.text}")
            except httpx.RequestError as e:
                logger.error(f"Request error fetching paginated resources from {e.request.url}: {e}")
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Bridge API: {e}")
            except Exception as e:
                logger.error(f"Unexpected error fetching paginated resources: {e}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing Bridge API response: {e}")

    if page_count >= max_pages:
        logger.warning(f"Pagination stopped after reaching max pages ({max_pages}) for URL: {initial_url}")

    logger.info(f"Total resources fetched after pagination for {initial_url}: {len(all_resources)}")
    return all_resources


# --- Fonctions principales pour Bridge API ---

async def create_bridge_user(db: Session, user: User) -> BridgeConnection:
    """Crée un utilisateur dans Bridge API et enregistre sa connexion"""
    external_user_id = f"harena-user-{user.id}"
    logger.info(f"Attempting to create/retrieve Bridge user with external_user_id: {external_user_id}")

    # Vérifier si une connexion existe déjà
    existing_connection = db.query(BridgeConnection).filter(
        BridgeConnection.user_id == user.id
    ).first()

    if existing_connection:
        logger.info(f"Found existing Bridge connection for user {user.id}")
        return existing_connection

    # Utilisation de _get_bridge_headers sans authentification
    headers = await _get_bridge_headers(include_auth=False)
    headers["content-type"] = "application/json"  # Ajout spécifique pour POST

    payload = {"external_user_id": external_user_id}

    url = f"{settings.BRIDGE_API_URL}/aggregation/users"
    logger.debug(f"Calling Bridge API: POST {url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            logger.info(f"Bridge API user creation response status: {response.status_code}")
            logger.debug(f"Bridge API response content: {response.text[:500]}...")  # Limiter la taille du log

            if response.status_code not in [200, 201]:  # 200 OK ou 201 Created
                # Gestion du cas où l'utilisateur existe déjà
                if response.status_code == 409:
                    logger.warning(f"Bridge user with external_id {external_user_id} likely already exists. Attempting to fetch.")
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Bridge user conflict: {response.text}")
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Failed to create Bridge user: {response.text}"
                    )

            bridge_user = response.json()
            logger.info(f"Bridge user created/retrieved successfully: {bridge_user.get('uuid')}")

            if "uuid" not in bridge_user:
                logger.error(f"Bridge API response missing 'uuid' field: {bridge_user}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid Bridge user response: missing uuid")

            bridge_connection = BridgeConnection(
                user_id=user.id,
                bridge_user_uuid=bridge_user["uuid"],
                external_user_id=external_user_id
            )

            try:
                db.add(bridge_connection)
                db.commit()
                db.refresh(bridge_connection)
                logger.info(f"Bridge connection saved successfully for user {user.id}, connection ID: {bridge_connection.id}")
                return bridge_connection
            except Exception as db_error:
                logger.error(f"Database error saving Bridge connection for user {user.id}: {db_error}", exc_info=True)
                db.rollback()
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {db_error}")

    except httpx.RequestError as req_error:
        logger.error(f"HTTP request error creating Bridge user: {req_error}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Failed to connect to Bridge API: {req_error}")
    except HTTPException as http_exc:
        # Renvoyer les exceptions HTTP déjà formatées
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error creating Bridge user for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error creating Bridge user: {e}")


async def get_bridge_token(db: Session, user_id: int) -> Dict[str, Any]:
    """Récupère ou génère un token d'authentification Bridge"""
    bridge_connection = db.query(BridgeConnection).filter(BridgeConnection.user_id == user_id).first()

    if not bridge_connection:
        logger.error(f"No Bridge connection found for user {user_id} when trying to get token.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No Bridge connection found for this user")

    now = datetime.now(timezone.utc)
    # Vérifier l'expiration avec une marge de sécurité (60 secondes)
    if bridge_connection.token_expires_at and bridge_connection.token_expires_at > (now + timedelta(seconds=60)) and bridge_connection.last_token:
        logger.debug(f"Using existing valid token for user {user_id}, expires at {bridge_connection.token_expires_at}")
        return {
            "access_token": bridge_connection.last_token,
            "expires_at": bridge_connection.token_expires_at
        }

    logger.info(f"Requesting new Bridge token for user {user_id}")
    # Utilisation de _get_bridge_headers sans authentification user token
    headers = await _get_bridge_headers(include_auth=False)
    headers["content-type"] = "application/json"

    # Authentification par external_user_id
    payload = {"external_user_id": bridge_connection.external_user_id}
    logger.debug(f"Token request payload: {payload}")
    url = f"{settings.BRIDGE_API_URL}/aggregation/authorization/token"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            logger.info(f"Bridge token API response status: {response.status_code}")
            logger.debug(f"Bridge token API response content: {response.text[:200]}...")

            if response.status_code not in [200, 201]:
                logger.error(f"Failed to get Bridge token for user {user_id}: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"Failed to get Bridge token: {response.text}")

            token_data = response.json()

            if "access_token" not in token_data or "expires_at" not in token_data:
                logger.error(f"Bridge token API response missing required fields: {token_data}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid Bridge token response: missing required fields")

            try:
                bridge_connection.last_token = token_data["access_token"]
                # Conversion de la date d'expiration ISO en objet datetime avec timezone
                expires_at_dt = datetime.fromisoformat(token_data["expires_at"].replace('Z', '+00:00'))
                bridge_connection.token_expires_at = expires_at_dt

                db.add(bridge_connection)
                db.commit()
                db.refresh(bridge_connection)
                logger.info(f"New Bridge token saved to database for user {user_id}, expires at {expires_at_dt}")

                return {
                    "access_token": bridge_connection.last_token,
                    "expires_at": expires_at_dt
                }
            except Exception as db_error:
                logger.error(f"Database error saving new token for user {user_id}: {db_error}", exc_info=True)
                db.rollback()
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error saving token: {db_error}")

    except httpx.RequestError as req_error:
        logger.error(f"HTTP request error getting Bridge token: {req_error}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Failed to connect to Bridge API for token: {req_error}")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error getting Bridge token for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error getting Bridge token: {e}")


async def create_connect_session(
    db: Session,
    user_id: int,
    callback_url: Optional[str] = None,
    country_code: Optional[str] = "FR",
    account_types: Optional[str] = "payment",
    context: Optional[str] = None,
    provider_id: Optional[int] = None,
    item_id: Optional[int] = None
) -> str:
    """Crée une session de connexion Bridge pour connecter un compte bancaire."""
    logger.info(f"Creating connect session for user {user_id} (item_id: {item_id}, provider_id: {provider_id})")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        logger.error(f"User not found for connect session: {user_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Utilisation de _get_bridge_headers AVEC authentification
    headers = await _get_bridge_headers(db=db, user_id=user_id, include_auth=True)
    headers["content-type"] = "application/json"

    payload = {}
    if item_id:
        payload["item_id"] = item_id
        logger.debug("Connect session mode: Manage existing item.")
    elif provider_id:
        payload["provider_id"] = provider_id
        payload["user_email"] = user.email  # Nécessaire pour provider_id
        logger.debug("Connect session mode: Pre-selected provider.")
    else:
        payload["user_email"] = user.email
        logger.debug("Connect session mode: Standard connection.")

    if callback_url: payload["callback_url"] = callback_url
    if country_code: payload["country_code"] = country_code
    if account_types: payload["account_types"] = account_types
    if context and len(context) <= 100: payload["context"] = context

    logger.debug(f"Connect session payload: {payload}")
    url = f"{settings.BRIDGE_API_URL}/aggregation/connect-sessions"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            logger.info(f"Connect session API response status: {response.status_code}")
            logger.debug(f"Connect session API response content: {response.text[:200]}...")

            if response.status_code not in [200, 201]:
                logger.error(f"Failed to create connect session for user {user_id}: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"Failed to create connect session: {response.text}")

            session_data = response.json()

            if "url" not in session_data:
                logger.error(f"Connect session API response missing 'url' field: {session_data}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid connect session response: missing url")

            logger.info(f"Connect session created successfully for user {user_id}")
            return session_data["url"]

    except httpx.RequestError as req_error:
        logger.error(f"HTTP request error creating connect session: {req_error}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Failed to connect to Bridge API for session: {req_error}")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error creating connect session for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error creating connect session: {e}")


async def get_bridge_item(db: Session, user_id: int, access_token: str, bridge_item_id: int) -> Optional[Dict[str, Any]]:
    """Récupère les détails d'un item Bridge depuis l'API."""
    logger.info(f"Getting Bridge item {bridge_item_id} for user {user_id}")
    
    # Les en-têtes sont construits avec le token fourni
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}"
    }
    
    url = f"{settings.BRIDGE_API_URL}/aggregation/items/{bridge_item_id}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            item_data = response.json()
            logger.info(f"Successfully retrieved item {bridge_item_id} for user {user_id}")
            return item_data
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting item {bridge_item_id} for user {user_id}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Bridge API error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error getting item {bridge_item_id}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Bridge API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_bridge_item for item {bridge_item_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get item: {e}")


async def refresh_bridge_item(db: Session, user_id: int, access_token: str, bridge_item_id: int) -> Optional[Dict[str, Any]]:
    """Rafraîchit un item Bridge pour récupérer ses dernières données."""
    logger.info(f"Refreshing Bridge item {bridge_item_id} for user {user_id}")
    
    # Les en-têtes sont construits avec le token fourni
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json"
    }
    
    url = f"{settings.BRIDGE_API_URL}/aggregation/items/{bridge_item_id}/refresh"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json={})
            response.raise_for_status()
            refresh_result = response.json()
            logger.info(f"Successfully refreshed item {bridge_item_id} for user {user_id}")
            return refresh_result
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error refreshing item {bridge_item_id}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Bridge API error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error refreshing item {bridge_item_id}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Bridge API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in refresh_bridge_item for item {bridge_item_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to refresh item: {e}")


async def create_bridge_reconnect_session(db: Session, user_id: int, access_token: str, bridge_item_id: int) -> Dict[str, Any]:
    """Crée une session de reconnexion pour un item qui nécessite une action utilisateur."""
    logger.info(f"Creating reconnect session for item {bridge_item_id}, user {user_id}")
    
    # Les en-têtes sont construits avec le token fourni
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json"
    }
    
    url = f"{settings.BRIDGE_API_URL}/aggregation/connect-sessions"
    payload = {"item_id": bridge_item_id}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            session_data = response.json()
            
            if "url" not in session_data:
                logger.error(f"Reconnect session API response missing 'url' field: {session_data}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid reconnect session response")
            
            logger.info(f"Successfully created reconnect session for item {bridge_item_id}")
            return {
                "redirect_url": session_data["url"],
                "session_id": session_data.get("id")
            }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error creating reconnect session for item {bridge_item_id}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Bridge API error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error creating reconnect session: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Bridge API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in create_bridge_reconnect_session for item {bridge_item_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create reconnect session: {e}")


async def get_bridge_accounts(
    db: Session,
    user_id: int,
    access_token: Optional[str] = None,
    item_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Récupère les comptes bancaires d'un utilisateur depuis Bridge API via /accounts.
    """
    logger.info(f"Getting Bridge accounts for user {user_id}, item_id={item_id}")
    
    # Utiliser le token fourni ou récupérer un nouveau
    if access_token:
        headers = {
            "accept": "application/json",
            "Bridge-Version": settings.BRIDGE_API_VERSION,
            "Client-Id": settings.BRIDGE_CLIENT_ID,
            "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
            "authorization": f"Bearer {access_token}"
        }
    else:
        headers = await _get_bridge_headers(db=db, user_id=user_id)

    # Construire l'URL avec filtres optionnels
    url = f"{settings.BRIDGE_API_URL}/aggregation/accounts"
    params = {}
    if item_id:
        params["item_id"] = str(item_id)
    params["limit"] = "200"  # Demander 200 par page pour optimiser

    # Construction sécurisée de l'URL
    request_url = httpx.URL(url, params=params)

    try:
        # Utiliser le helper pour gérer la pagination
        all_accounts = await _fetch_paginated_resources(str(request_url), headers)
        logger.info(f"Successfully retrieved {len(all_accounts)} accounts total for user {user_id}.")
        return all_accounts
    except HTTPException as http_exc:
        # Renvoyer les exceptions HTTP gérées par les helpers
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in get_bridge_accounts for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get accounts: {e}")


async def get_bridge_transaction(
    db: Session,
    user_id: int,
    access_token: str,
    transaction_id: int
) -> Optional[Dict[str, Any]]:
    """Récupère les détails d'une transaction spécifique."""
    logger.info(f"Getting Bridge transaction {transaction_id} for user {user_id}")
    
    # Les en-têtes sont construits avec le token fourni
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}"
    }
    
    url = f"{settings.BRIDGE_API_URL}/aggregation/transactions/{transaction_id}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            transaction_data = response.json()
            logger.info(f"Successfully retrieved transaction {transaction_id}")
            return transaction_data
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting transaction {transaction_id}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Bridge API error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error getting transaction {transaction_id}: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Bridge API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_bridge_transaction for transaction {transaction_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get transaction: {e}")


async def get_bridge_transactions(
    db: Session,
    user_id: int,
    access_token: str,
    account_id: Optional[int] = None,
    since: Optional[Union[datetime, str]] = None,
    limit: int = 500  # Limite raisonnable
) -> List[Dict[str, Any]]:
    """Récupère les transactions d'un compte bancaire depuis Bridge API."""
    logger.info(f"Getting Bridge transactions for user {user_id}, account_id={account_id}, since={since}")
    
    # Les en-têtes sont construits avec le token fourni
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}"
    }

    url = f"{settings.BRIDGE_API_URL}/aggregation/transactions"
    params: Dict[str, Union[str, int]] = {"limit": limit}
    if account_id:
        params["account_id"] = account_id
    if since:
        # Vérifier le type de since et formatter en conséquence
        if isinstance(since, datetime):
            # Si c'est déjà un objet datetime, formater en ISO
            params["since"] = since.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        elif isinstance(since, str):
            # Si c'est une chaîne, l'utiliser directement
            params["since"] = since
        else:
            logger.warning(f"Type de paramètre since non reconnu: {type(since)}")

    request_url = httpx.URL(url, params=params)

    try:
        # Utiliser le helper pour gérer la pagination
        all_transactions = await _fetch_paginated_resources(str(request_url), headers, limit=limit)
        logger.info(f"Successfully retrieved {len(all_transactions)} transactions for account {account_id}.")
        return all_transactions
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in get_bridge_transactions for account {account_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get transactions: {e}")

async def get_bridge_categories(
    db: Session,
    user_id: int,
    language: str = "fr"
) -> List[Dict[str, Any]]:
    """Récupère la liste des catégories depuis Bridge API."""
    logger.info(f"Getting Bridge categories (language: {language}) using token for user {user_id}")
    headers = await _get_bridge_headers(db=db, user_id=user_id, language=language)

    url = f"{settings.BRIDGE_API_URL}/aggregation/categories"
    params = {"limit": "200"}
    request_url = httpx.URL(url, params=params)

    try:
        all_categories = await _fetch_paginated_resources(str(request_url), headers)
        logger.info(f"Successfully retrieved {len(all_categories)} categories.")
        
        # Transformer la structure (aplatir les sous-catégories)
        flat_categories = []
        for parent_cat in all_categories:
            parent_id = parent_cat.get('id')
            parent_name = parent_cat.get('name')
            if 'categories' in parent_cat and isinstance(parent_cat['categories'], list):
                for sub_cat in parent_cat['categories']:
                    flat_categories.append({
                        "bridge_category_id": sub_cat.get('id'),
                        "name": sub_cat.get('name'),
                        "parent_id": parent_id,
                        "parent_name": parent_name
                    })
            else:
                # Ajouter la catégorie parente elle-même
                flat_categories.append({
                    "bridge_category_id": parent_id,
                    "name": parent_name,
                    "parent_id": None,
                    "parent_name": None
                })

        logger.info(f"Transformed into {len(flat_categories)} flat categories.")
        return flat_categories
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in get_bridge_categories: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get categories: {e}")


async def get_bridge_insights(db: Session, user_id: int) -> Optional[Dict[str, Any]]:
    """Récupère les insights agrégés par catégorie pour un utilisateur."""
    logger.info(f"Getting Bridge insights for user {user_id}")
    headers = await _get_bridge_headers(db=db, user_id=user_id)
    url = f"{settings.BRIDGE_API_URL}/aggregation/insights/category"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            insights_data = response.json()
            logger.info(f"Successfully retrieved insights for user {user_id}.")
            return insights_data
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting insights for user {user_id}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code in [403, 404]:
            logger.warning(f"Insights endpoint might not be available or activated for user {user_id}.")
            return None
        raise HTTPException(status_code=e.response.status_code, detail=f"Bridge API error getting insights: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error getting insights for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Bridge API for insights: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_bridge_insights for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get insights: {e}")
    
async def get_bridge_items(db: Session, user_id: int, access_token: str = None) -> List[Dict[str, Any]]:
    """
    Récupère tous les items Bridge d'un utilisateur.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        access_token: Token d'accès Bridge (optionnel)
        
    Returns:
        List[Dict]: Liste des items Bridge
    """
    logger.info(f"Récupération des items Bridge pour l'utilisateur {user_id}")
    
    # Utiliser le token fourni ou récupérer un nouveau
    if not access_token:
        token_data = await get_bridge_token(db, user_id)
        access_token = token_data["access_token"]
    
    # Construire les en-têtes avec le token
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}"
    }
    
    url = f"{settings.BRIDGE_API_URL}/aggregation/items"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extraire les ressources (items) de la réponse
            items = data.get("resources", [])
            logger.info(f"Récupéré {len(items)} items Bridge pour l'utilisateur {user_id}")
            
            return items
    except httpx.HTTPStatusError as e:
        logger.error(f"Erreur HTTP lors de la récupération des items Bridge: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Bridge API error: {e.response.text}")
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des items Bridge: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get Bridge items: {e}")

async def get_bridge_stocks(
    db: Session,
    user_id: int,
    access_token: str,
    account_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Récupère les stocks/titres depuis Bridge API.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        access_token: Token d'accès Bridge
        account_id: ID du compte (optionnel)
        
    Returns:
        List[Dict]: Liste des stocks/titres
    """
    logger.info(f"Getting Bridge stocks for user {user_id}, account_id={account_id}")
    
    # Les en-têtes sont construits avec le token fourni
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}"
    }
    
    # Construire l'URL avec filtres optionnels
    url = f"{settings.BRIDGE_API_URL}/aggregation/stocks"
    params = {"limit": "500"}  # Nombre maximum de stocks par page
    
    if account_id:
        params["account_id"] = str(account_id)
    
    # Construction sécurisée de l'URL
    request_url = httpx.URL(url, params=params)
    
    try:
        # Utiliser le helper pour gérer la pagination
        all_stocks = await _fetch_paginated_resources(str(request_url), headers)
        logger.info(f"Successfully retrieved {len(all_stocks)} stocks for account {account_id}.")
        return all_stocks
    except HTTPException as http_exc:
        # Renvoyer les exceptions HTTP gérées par les helpers
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in get_bridge_stocks for account {account_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get stocks: {e}")
    
async def get_accounts_information(
    db: Session,
    user_id: int,
    access_token: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Récupère les informations détaillées des comptes (IBAN, identité) depuis Bridge API.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        access_token: Token d'accès Bridge (optionnel)
        
    Returns:
        List[Dict]: Liste des informations des comptes ou None si non disponible
    """
    logger.info(f"Getting accounts information for user {user_id}")
    
    # Utiliser le token fourni ou récupérer un nouveau
    if access_token:
        headers = {
            "accept": "application/json",
            "Bridge-Version": settings.BRIDGE_API_VERSION,
            "Client-Id": settings.BRIDGE_CLIENT_ID,
            "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
            "authorization": f"Bearer {access_token}"
        }
    else:
        headers = await _get_bridge_headers(db=db, user_id=user_id)
    
    url = f"{settings.BRIDGE_API_URL}/aggregation/accounts-information"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            accounts_info = data.get("resources", [])
            
            logger.info(f"Successfully retrieved account information for user {user_id}")
            return accounts_info
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting account information: {e.response.status_code} - {e.response.text}")
        if e.response.status_code in [403, 404]:
            logger.warning(f"Accounts-information endpoint might not be available in the current plan")
            return None
        raise HTTPException(status_code=e.response.status_code, detail=f"Bridge API error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error getting account information: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to Bridge API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_accounts_information for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get account information: {e}")