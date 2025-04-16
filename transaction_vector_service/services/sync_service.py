"""
Service for synchronizing data between Bridge API and vector storage.

This module provides functionality for coordinating the process of
fetching transaction data from Bridge API and storing it in the vector database.
"""

import logging
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from uuid import UUID

from ..config.logging_config import get_logger
from ..models.transaction import TransactionCreate, Transaction
from .bridge_client import BridgeClient
from .transaction_service import TransactionService
from .category_service import CategoryService
from ..api.dependencies import get_service
from user_service.db.session import get_db

logger = get_logger(__name__)


class SyncService:
    """Service for synchronizing data between Bridge API and vector storage."""
    
    def __init__(
        self,
        bridge_client: Optional[BridgeClient] = None,
        transaction_service: Optional[TransactionService] = None,
        category_service: Optional[CategoryService] = None
    ):
        """
        Initialize the sync service.
        
        Args:
            bridge_client: Optional Bridge API client
            transaction_service: Optional transaction service
            category_service: Optional category service
        """
        self.bridge_client = bridge_client or BridgeClient()
        
        # Initialiser les services - pour éviter les dépendances circulaires,
        # nous utilisons get_service
        if transaction_service:
            self.transaction_service = transaction_service
        else:
            try:
                self.transaction_service = get_service("transaction_service")
            except Exception as e:
                logger.warning(f"Could not get transaction_service: {str(e)}")
                self.transaction_service = TransactionService()
        
        if category_service:
            self.category_service = category_service
        else:
            try:
                self.category_service = get_service("category_service")
            except Exception as e:
                logger.warning(f"Could not get category_service: {str(e)}")
                self.category_service = CategoryService()
        
        # Keep track of ongoing syncs to prevent duplicates
        self._active_syncs = set()
        
        logger.info("Sync service initialized")
    
    async def sync_user_transactions(
        self,
        user_id: int,
        bridge_user_uuid: str,
        incremental: bool = True,
        account_id: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Synchronize transactions for a user.
        
        Args:
            user_id: Internal user ID
            bridge_user_uuid: Bridge user UUID
            incremental: Whether to do an incremental sync (default: True)
            account_id: Optional account ID to limit sync to
            since: Optional timestamp to limit sync to transactions updated after
            
        Returns:
            Dictionary with sync results
        """
        sync_key = f"{user_id}_{account_id or 'all'}"
        
        # Check if sync is already in progress
        if sync_key in self._active_syncs:
            logger.info(f"Sync already in progress for user {user_id}, account {account_id}")
            return {
                "status": "skipped",
                "reason": "Sync already in progress",
                "user_id": user_id,
                "account_id": account_id
            }
        
        self._active_syncs.add(sync_key)
        logger.info(f"Starting {'incremental' if incremental else 'full'} sync for user {user_id}, account {account_id}")
        
        try:
            # Get user token
            logger.debug(f"Getting token for user_uuid={bridge_user_uuid}")
            token = await self.bridge_client.get_user_token(user_uuid=bridge_user_uuid)
            if not token:
                logger.error(f"Failed to get user token for user_uuid={bridge_user_uuid}")
                return {
                    "status": "error",
                    "reason": "Could not get user token",
                    "user_id": user_id
                }
            
            # Get accounts if needed
            accounts_to_sync = []
            if account_id:
                accounts_to_sync = [account_id]
                logger.info(f"Will sync only account_id={account_id}")
            else:
                # Get all user accounts
                logger.debug(f"Getting all accounts for user_id={user_id}")
                bridge_accounts = await self.bridge_client.get_accounts(token)
                accounts_to_sync = [acc.get("id") for acc in bridge_accounts if acc.get("id")]
                logger.info(f"Found {len(accounts_to_sync)} accounts to sync for user_id={user_id}")
            
            if not accounts_to_sync:
                logger.warning(f"No accounts found for user {user_id}")
                return {
                    "status": "error",
                    "reason": "No accounts found",
                    "user_id": user_id
                }
            
            # Determine sync parameters
            sync_params = {}
            if incremental and since:
                sync_params["since"] = since
                logger.info(f"Incremental sync with provided since={since}")
            elif incremental and not since:
                # Default to last 30 days for incremental sync without timestamp
                sync_params["since"] = datetime.now() - timedelta(days=30)
                logger.info(f"Incremental sync with default since={sync_params['since']}")
            else:
                logger.info("Full sync requested (no 'since' parameter)")
            
            # Track sync results
            total_transactions = 0
            new_transactions = 0
            updated_transactions = 0
            errors = 0
            
            # Sync each account
            for acc_id in accounts_to_sync:
                logger.info(f"Starting sync for account_id={acc_id}")
                try:
                    # Get transactions from Bridge API
                    logger.debug(f"Getting transactions from Bridge API for account_id={acc_id}")
                    bridge_transactions = await self.bridge_client.get_transactions(
                        user_token=token,
                        account_id=acc_id,
                        **sync_params
                    )
                    
                    logger.info(f"Retrieved {len(bridge_transactions)} transactions from Bridge API for account_id={acc_id}")
                    
                    # Process transactions
                    for bridge_tx in bridge_transactions:
                        total_transactions += 1
                        tx_id = bridge_tx.get("id", "unknown")
                        
                        try:
                            # Convert to our model
                            logger.debug(f"Converting transaction id={tx_id} to internal model")
                            tx_create = self._convert_bridge_transaction(bridge_tx, user_id)
                            
                            # Process and store the transaction
                            logger.debug(f"Processing transaction id={tx_id}")
                            processed_tx = await self.transaction_service.process_transaction(tx_create)
                            
                            if processed_tx:
                                new_transactions += 1
                                logger.debug(f"Transaction id={tx_id} processed successfully")
                            else:
                                errors += 1
                                logger.warning(f"Failed to process transaction id={tx_id}")
                        except Exception as tx_error:
                            errors += 1
                            logger.error(f"Error processing transaction id={tx_id}: {str(tx_error)}")
                            logger.error(traceback.format_exc())
                except Exception as acc_error:
                    logger.error(f"Error syncing account {acc_id}: {str(acc_error)}")
                    logger.error(traceback.format_exc())
                    errors += 1
            
            sync_result = {
                "status": "success" if errors == 0 else "partial",
                "user_id": user_id,
                "total_transactions": total_transactions,
                "new_transactions": new_transactions,
                "updated_transactions": updated_transactions,
                "errors": errors,
                "accounts_synced": len(accounts_to_sync),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Sync completed for user {user_id}: {sync_result}")
            return sync_result
        except Exception as e:
            logger.error(f"Error syncing transactions for user {user_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "reason": str(e),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Remove from active syncs
            self._active_syncs.discard(sync_key)
            logger.info(f"Sync key {sync_key} removed from active syncs")
    
    def _convert_bridge_transaction(self, bridge_tx: Dict[str, Any], user_id: int) -> TransactionCreate:
        """
        Convert a Bridge API transaction to our transaction model.
        
        Args:
            bridge_tx: Bridge API transaction data
            user_id: Internal user ID
            
        Returns:
            TransactionCreate model
        """
        # Extract transaction date
        date_str = bridge_tx.get("date")
        transaction_date = None
        if date_str:
            try:
                if isinstance(date_str, str):
                    transaction_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                else:
                    transaction_date = date_str
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}, using current date")
                transaction_date = datetime.now().date()
        else:
            logger.warning("No date provided in transaction, using current date")
            transaction_date = datetime.now().date()
        
        # Extract value date
        value_date_str = bridge_tx.get("value_date")
        value_date = None
        if value_date_str:
            try:
                if isinstance(value_date_str, str):
                    value_date = datetime.strptime(value_date_str, "%Y-%m-%d").date()
                else:
                    value_date = value_date_str
            except ValueError:
                logger.warning(f"Invalid value_date format: {value_date_str}")
                value_date = None
        
        # Extract booking date
        booking_date_str = bridge_tx.get("booking_date")
        booking_date = None
        if booking_date_str:
            try:
                if isinstance(booking_date_str, str):
                    booking_date = datetime.strptime(booking_date_str, "%Y-%m-%d").date()
                else:
                    booking_date = booking_date_str
            except ValueError:
                logger.warning(f"Invalid booking_date format: {booking_date_str}")
                booking_date = None
        
        # Créer un identifiant unique pour cette transaction
        bridge_tx_id = bridge_tx.get("id")
        if not bridge_tx_id:
            logger.warning("Transaction without ID, generating placeholder")
            bridge_tx_id = f"unknown_{datetime.now().timestamp()}"
        
        # Extraire la description
        provider_description = bridge_tx.get("provider_description", "")
        clean_description = bridge_tx.get("clean_description", "")
        
        # Extraire le montant
        amount = bridge_tx.get("amount", 0.0)
        if not isinstance(amount, (int, float)):
            logger.warning(f"Invalid amount: {amount}, using 0.0")
            amount = 0.0
        
        # Create transaction
        logger.debug(f"Creating TransactionCreate for bridge_tx_id={bridge_tx_id}, amount={amount}, date={transaction_date}")
        return TransactionCreate(
            bridge_transaction_id=bridge_tx_id,
            account_id=bridge_tx.get("account_id"),
            user_id=user_id,
            amount=amount,
            currency_code=bridge_tx.get("currency_code", "EUR"),
            description=provider_description,
            clean_description=clean_description,
            transaction_date=transaction_date,
            value_date=value_date,
            booking_date=booking_date,
            category_id=bridge_tx.get("category_id"),
            operation_type=bridge_tx.get("operation_type"),
            is_future=bridge_tx.get("future", False),
            is_deleted=bridge_tx.get("deleted", False),
            raw_data=bridge_tx
        )
    
    async def sync_categories(self) -> Dict[str, Any]:
        """
        Synchronize category data from Bridge API.
        
        Returns:
            Dictionary with sync results
        """
        try:
            logger.info("Starting category sync")
            
            # Force refresh of the category cache
            await self.category_service.preload_categories(force_refresh=True)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error syncing categories: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def handle_webhook_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a webhook event from Bridge API.
        
        Args:
            event_type: Type of webhook event
            event_data: Event data
            
        Returns:
            Dictionary with handling results
        """
        try:
            logger.info(f"Handling webhook event: {event_type}")
            logger.debug(f"Event data: {event_data}")
            
            # Handle different event types
            if event_type == "item.refreshed":
                # Item has been refreshed, sync the accounts
                item_id = event_data.get("item_id")
                user_uuid = event_data.get("user_uuid")
                
                if not item_id or not user_uuid:
                    logger.error(f"Missing item_id or user_uuid in webhook event: {event_data}")
                    return {
                        "status": "error",
                        "reason": "Missing item_id or user_uuid",
                        "event_type": event_type
                    }
                
                # Chercher l'user_id correspondant au user_uuid
                user_id = await self._lookup_user_id_from_uuid(user_uuid)
                if not user_id:
                    logger.error(f"Could not find user_id for user_uuid={user_uuid}")
                    return {
                        "status": "error",
                        "reason": f"User not found for UUID {user_uuid}",
                        "event_type": event_type
                    }
                
                logger.info(f"Found user_id={user_id} for user_uuid={user_uuid}")
                
                # Get the accounts for this item
                token = await self.bridge_client.get_user_token(user_uuid=user_uuid)
                if not token:
                    logger.error(f"Failed to get token for user_uuid={user_uuid}")
                    return {
                        "status": "error",
                        "reason": "Could not get user token",
                        "event_type": event_type
                    }
                
                logger.debug(f"Getting accounts for item_id={item_id}")
                accounts = await self.bridge_client.get_accounts(token, item_id=item_id)
                logger.info(f"Found {len(accounts)} accounts for item_id={item_id}")
                
                # Sync each account
                results = []
                for account in accounts:
                    account_id = account.get("id")
                    if account_id:
                        logger.info(f"Syncing account_id={account_id} for user_id={user_id}")
                        sync_result = await self.sync_user_transactions(
                            user_id=user_id,
                            bridge_user_uuid=user_uuid,
                            incremental=True,
                            account_id=account_id,
                            since=datetime.now() - timedelta(days=7)  # Sync last week
                        )
                        results.append({
                            "account_id": account_id,
                            "result": sync_result
                        })
                
                return {
                    "status": "success",
                    "event_type": event_type,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif event_type == "item.created":
                # New item created, sync all accounts
                item_id = event_data.get("item_id")
                user_uuid = event_data.get("user_uuid")
                
                if not item_id or not user_uuid:
                    logger.error(f"Missing item_id or user_uuid in webhook event: {event_data}")
                    return {
                        "status": "error",
                        "reason": "Missing item_id or user_uuid",
                        "event_type": event_type
                    }
                
                # Chercher l'user_id correspondant au user_uuid
                user_id = await self._lookup_user_id_from_uuid(user_uuid)
                if not user_id:
                    logger.error(f"Could not find user_id for user_uuid={user_uuid}")
                    return {
                        "status": "error",
                        "reason": f"User not found for UUID {user_uuid}",
                        "event_type": event_type
                    }
                
                logger.info(f"Found user_id={user_id} for user_uuid={user_uuid}")
                
                # Sync all accounts for this user
                logger.info(f"Starting full sync for user_id={user_id} after item creation")
                sync_result = await self.sync_user_transactions(
                    user_id=user_id,
                    bridge_user_uuid=user_uuid,
                    incremental=False  # Full sync for new items
                )
                
                return {
                    "status": "success",
                    "event_type": event_type,
                    "result": sync_result,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif event_type == "item.account.updated" or event_type == "item.account.created":
                # Account updated or created, sync specific account
                account_id = event_data.get("account_id")
                user_uuid = event_data.get("user_uuid")
                
                if not account_id or not user_uuid:
                    logger.error(f"Missing account_id or user_uuid in webhook event: {event_data}")
                    return {
                        "status": "error",
                        "reason": "Missing account_id or user_uuid",
                        "event_type": event_type
                    }
                
                # Chercher l'user_id correspondant au user_uuid
                user_id = await self._lookup_user_id_from_uuid(user_uuid)
                if not user_id:
                    logger.error(f"Could not find user_id for user_uuid={user_uuid}")
                    return {
                        "status": "error",
                        "reason": f"User not found for UUID {user_uuid}",
                        "event_type": event_type
                    }
                
                logger.info(f"Found user_id={user_id} for user_uuid={user_uuid}")
                
                # Sync this specific account
                logger.info(f"Syncing account_id={account_id} for user_id={user_id} after {event_type}")
                sync_result = await self.sync_user_transactions(
                    user_id=user_id,
                    bridge_user_uuid=user_uuid,
                    incremental=True,
                    account_id=account_id,
                    since=datetime.now() - timedelta(hours=24)  # Last day's changes
                )
                
                return {
                    "status": "success",
                    "event_type": event_type,
                    "result": sync_result,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Default case - unhandled event type
            logger.warning(f"Unhandled webhook event type: {event_type}")
            return {
                "status": "warning",
                "reason": f"Unhandled event type: {event_type}",
                "event_type": event_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling webhook event {event_type}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "reason": str(e),
                "event_type": event_type,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _lookup_user_id_from_uuid(self, user_uuid: str) -> Optional[int]:
        """
        Lookup user_id from Bridge user UUID.
        
        Args:
            user_uuid: Bridge user UUID
            
        Returns:
            User ID if found, None otherwise
        """
        try:
            # Try to use session from bridge_client if available
            if hasattr(self.bridge_client, 'get_user_id_from_uuid'):
                user_id = await self.bridge_client.get_user_id_from_uuid(user_uuid)
                if user_id:
                    return user_id
            
            # Fallback to database query
            from sqlalchemy.ext.asyncio import AsyncSession
            from user_service.models.user import BridgeConnection
            
            # Get database session
            db = next(get_db())
            
            # Query for bridge connection
            bridge_connection = db.query(BridgeConnection).filter(
                BridgeConnection.bridge_user_uuid == user_uuid
            ).first()
            
            if bridge_connection:
                return bridge_connection.user_id
            
            logger.warning(f"No bridge connection found for user_uuid={user_uuid}")
            return None
            
        except Exception as e:
            logger.error(f"Error looking up user_id for user_uuid={user_uuid}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def get_sync_status(self, user_id: int) -> Dict[str, Any]:
        """
        Get the sync status for a user.
        
        Args:
            user_id: User ID to get status for
            
        Returns:
            Dictionary with sync status
        """
        try:
            # Check if sync is in progress
            is_syncing = any(key.startswith(f"{user_id}_") for key in self._active_syncs)
            
            # Récupérer les informations de synchronisation depuis la base de données
            from sqlalchemy.ext.asyncio import AsyncSession
            from sync_service.models.sync import SyncItem, SyncAccount
            
            # Get database session
            db = next(get_db())
            
            # Récupérer les items et comptes de l'utilisateur
            items = db.query(SyncItem).filter(SyncItem.user_id == user_id).all()
            
            if not items:
                logger.warning(f"No sync items found for user_id={user_id}")
                return {
                    "user_id": user_id,
                    "is_syncing": is_syncing,
                    "last_sync": None,
                    "sync_age_hours": None,
                    "accounts_status": "no_accounts",
                    "items_count": 0,
                    "accounts_count": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Compter les comptes
            accounts = []
            for item in items:
                item_accounts = db.query(SyncAccount).filter(SyncAccount.item_id == item.id).all()
                accounts.extend(item_accounts)
            
            # Déterminer la dernière synchronisation
            last_syncs = [account.last_sync_timestamp for account in accounts if account.last_sync_timestamp]
            last_sync = max(last_syncs) if last_syncs else None
            
            sync_age_hours = None
            if last_sync:
                sync_age_hours = (datetime.now(last_sync.tzinfo) - last_sync).total_seconds() / 3600
            
            # Déterminer le statut des comptes
            accounts_status = "unknown"
            if not accounts:
                accounts_status = "no_accounts"
            elif all(account.last_sync_timestamp for account in accounts):
                accounts_status = "synced"
            elif any(account.last_sync_timestamp for account in accounts):
                accounts_status = "partially_synced"
            else:
                accounts_status = "not_synced"
            
            return {
                "user_id": user_id,
                "is_syncing": is_syncing,
                "last_sync": last_sync.isoformat() if last_sync else None,
                "sync_age_hours": sync_age_hours,
                "accounts_status": accounts_status,
                "items_count": len(items),
                "accounts_count": len(accounts),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting sync status for user_id={user_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "reason": str(e),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }