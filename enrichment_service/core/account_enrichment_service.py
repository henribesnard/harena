import logging
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from enrichment_service.models import TransactionInput
from db_service.models.sync import SyncAccount, BridgeCategory

logger = logging.getLogger(__name__)


class AccountEnrichmentService:
    """Service providing additional information about accounts and categories."""

    def __init__(self, db: Session):
        self.db = db

    async def enrich_with_account_data(self, transaction: TransactionInput) -> Dict[str, Optional[Any]]:
        """Return additional metadata for a transaction.

        The returned dictionary can contain account metadata, category name and a
        merchant name extracted from the description.
        """

        account_info = {
            "account_name": transaction.account_name,
            "account_type": transaction.account_type,
            "account_balance": transaction.account_balance,
            "account_currency": transaction.account_currency,
            "account_last_sync": transaction.account_last_sync,
        }

        if not account_info["account_name"]:
            db_account = self.get_account_details(transaction.account_id)
            if db_account:
                account_info.update(db_account)

        category_name = self.resolve_category_name(transaction.category_id)
        description = transaction.clean_description or transaction.provider_description or ""
        merchant_name = self.extract_merchant_name(description)
        return {
            **account_info,
            "category_name": category_name,
            "merchant_name": merchant_name,
        }

    def get_account_details(self, account_id: int) -> Optional[Dict[str, Any]]:
        """Fetch the full account metadata for the given account id."""
        if not self.db:
            return None
        account = (
            self.db.query(SyncAccount)
            .filter(
                (SyncAccount.bridge_account_id == account_id)
                | (SyncAccount.id == account_id)
            )
            .first()
        )
        if not account:
            logger.debug(f"No account found for id {account_id}")
            return None
        return {
            "account_name": account.account_name,
            "account_type": account.account_type,
            "account_balance": account.balance,
            "account_currency": account.currency_code,
            "account_last_sync": account.last_sync_timestamp,
        }

    def resolve_category_name(self, category_id: Optional[int]) -> Optional[str]:
        """Return the category name for the given category id."""
        if not self.db or not category_id:
            return None
        category = (
            self.db.query(BridgeCategory)
            .filter(BridgeCategory.bridge_category_id == category_id)
            .first()
        )
        if not category:
            logger.debug(f"No category found for id {category_id}")
            return None
        return category.name

    def extract_merchant_name(self, description: str) -> Optional[str]:
        """Very naive merchant name extraction from the description."""
        if not description:
            return None
        # Use the first word as a placeholder merchant extraction
        return description.strip().split(" ")[0]
