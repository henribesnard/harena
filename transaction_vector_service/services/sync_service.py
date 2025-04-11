# transaction_vector_service/services/sync_service.py
"""
Service for synchronizing data between Bridge API and vector storage.

This module provides functionality for coordinating the process of
fetching transaction data from Bridge API and storing it in the vector database.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from uuid import UUID

from ..config.logging_config import get_logger
from ..models.transaction import TransactionCreate, Transaction
from .bridge_client import BridgeClient
from .transaction_service import TransactionService
from .category_service import CategoryService

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
        self.transaction_service = transaction_service or TransactionService()
        self.category_service = category_service or CategoryService()
        
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
        
        try:
            logger.info(f"Starting {'incremental' if incremental else 'full'} sync for user {user_id}")
            
            # Get user token
            token = await self.bridge_client.get_user_token(user_uuid=bridge_user_uuid)
            
            # Get accounts if needed
            accounts_to_sync = []
            if account_id:
                accounts_to_sync = [account_id]
            else:
                # Get all user accounts
                bridge_accounts = await self.bridge_client.get_accounts(token)
                accounts_to_sync = [acc.get("id") for acc in bridge_accounts if acc.get("id")]
            
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
            elif incremental and not since:
                # Default to last 30 days for incremental sync without timestamp
                sync_params["since"] = datetime.now() - timedelta(days=30)
            
            # Track sync results
            total_transactions = 0
            new_transactions = 0
            updated_transactions = 0
            errors = 0
            
            # Sync each account
            for acc_id in accounts_to_sync:
                try:
                    # Get transactions from Bridge API
                    bridge_transactions = await self.bridge_client.get_transactions(
                        user_token=token,
                        account_id=acc_id,
                        **sync_params
                    )
                    
                    # Process transactions
                    for bridge_tx in bridge_transactions:
                        total_transactions += 1
                        
                        try:
                            # Convert to our model
                            tx_create = self._convert_bridge_transaction(bridge_tx, user_id)
                            
                            # Process and store the transaction
                            processed_tx = await self.transaction_service.process_transaction(tx_create)
                            
                            if processed_tx:
                                new_transactions += 1
                            else:
                                errors += 1
                        except Exception as tx_error:
                            logger.error(f"Error processing transaction: {str(tx_error)}")
                            errors += 1
                except Exception as acc_error:
                    logger.error(f"Error syncing account {acc_id}: {str(acc_error)}")
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
            return {
                "status": "error",
                "reason": str(e),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Remove from active syncs
            self._active_syncs.discard(sync_key)
    
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
                transaction_date = datetime.now().date()
        else:
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
                booking_date = None
        
        # Create transaction
        return TransactionCreate(
            bridge_transaction_id=bridge_tx.get("id"),
            account_id=bridge_tx.get("account_id"),
            user_id=user_id,
            amount=bridge_tx.get("amount", 0.0),
            currency_code=bridge_tx.get("currency_code", "EUR"),
            description=bridge_tx.get("provider_description", ""),
            clean_description=bridge_tx.get("clean_description", ""),
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
            
            # Handle different event types
            if event_type == "item.refreshed":
                # Item has been refreshed, sync the accounts
                item_id = event_data.get("item_id")
                user_uuid = event_data.get("user_uuid")
                
                if not item_id or not user_uuid:
                    return {
                        "status": "error",
                        "reason": "Missing item_id or user_uuid",
                        "event_type": event_type
                    }
                
                # In a real implementation, you'd look up the internal user ID
                # For now, we'll use a placeholder
                user_id = 1  # Placeholder
                
                # Get the accounts for this item
                token = await self.bridge_client.get_user_token(user_uuid=user_uuid)
                accounts = await self.bridge_client.get_accounts(token, item_id=item_id)
                
                # Sync each account
                for account in accounts:
                    account_id = account.get("id")
                    if account_id:
                        await self.sync_user_transactions(
                            user_id=user_id,
                            bridge_user_uuid=user_uuid,
                            incremental=True,
                            account_id=account_id,
                            since=datetime.now() - timedelta(days=7)  # Sync last week
                        )
            
            elif event_type == "item.created":
                # New item created, sync all accounts
                item_id = event_data.get("item_id")
                user_uuid = event_data.get("user_uuid")
                
                if not item_id or not user_uuid:
                    return {
                        "status": "error",
                        "reason": "Missing item_id or user_uuid",
                        "event_type": event_type
                    }
                
                # In a real implementation, you'd look up the internal user ID
                user_id = 1  # Placeholder
                
                # Sync all accounts for this user
                await self.sync_user_transactions(
                    user_id=user_id,
                    bridge_user_uuid=user_uuid,
                    incremental=False  # Full sync for new items
                )
            
            elif event_type == "item.account.updated":
                # Account updated, sync specific account
                account_id = event_data.get("account_id")
                user_uuid = event_data.get("user_uuid")
                
                if not account_id or not user_uuid:
                    return {
                        "status": "error",
                        "reason": "Missing account_id or user_uuid",
                        "event_type": event_type
                    }
                
                # In a real implementation, you'd look up the internal user ID
                user_id = 1  # Placeholder
                
                # Sync this specific account
                await self.sync_user_transactions(
                    user_id=user_id,
                    bridge_user_uuid=user_uuid,
                    incremental=True,
                    account_id=account_id,
                    since=datetime.now() - timedelta(hours=24)  # Last day's changes
                )
            
            return {
                "status": "success",
                "event_type": event_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling webhook event {event_type}: {str(e)}")
            return {
                "status": "error",
                "reason": str(e),
                "event_type": event_type,
                "timestamp": datetime.now().isoformat()
            }
    
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
            
            # In a real implementation, you'd fetch the last sync timestamp from database
            last_sync = datetime.now() - timedelta(hours=6)  # Placeholder
            
            return {
                "user_id": user_id,
                "is_syncing": is_syncing,
                "last_sync": last_sync.isoformat(),
                "sync_age_hours": (datetime.now() - last_sync).total_seconds() / 3600,
                "accounts_status": "synced",  # Placeholder
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting sync status: {str(e)}")
            return {
                "status": "error",
                "reason": str(e),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }