from typing import Tuple, List, Optional
from enrichment_service.models import TransactionInput


class DataQualityValidator:
    """Basic validator to detect transaction anomalies."""

    def __init__(self, amount_threshold: float = 1_000_000):
        self.amount_threshold = amount_threshold

    def validate_transaction_consistency(self, tx: TransactionInput) -> bool:
        """Check mandatory fields of a transaction."""
        mandatory_present = all([
            tx.bridge_transaction_id,
            tx.user_id,
            tx.account_id,
            tx.amount is not None,
            tx.date is not None,
            tx.currency_code is not None,
        ])
        return mandatory_present

    def detect_amount_anomalies(self, tx: TransactionInput) -> bool:
        """Return True if amount is outside accepted range."""
        return abs(tx.amount) > self.amount_threshold or tx.amount == 0

    def validate_account_balance_consistency(
        self, tx: TransactionInput, account_balance: Optional[float] = None
    ) -> bool:
        """Validate account balance after applying transaction."""
        if account_balance is None:
            return True
        projected = account_balance + tx.amount
        # flag if transaction would drive balance negative beyond existing balance
        return not (projected < 0 and tx.amount < 0)

    def evaluate(
        self, tx: TransactionInput, account_balance: Optional[float] = None
    ) -> Tuple[bool, float, List[str]]:
        """Compute quality score and flags for a transaction."""
        score = 1.0
        flags: List[str] = []

        if not self.validate_transaction_consistency(tx):
            flags.append("inconsistent_transaction")
            score -= 0.4

        if self.detect_amount_anomalies(tx):
            flags.append("amount_anomaly")
            score -= 0.3

        if not self.validate_account_balance_consistency(tx, account_balance):
            flags.append("account_balance_mismatch")
            score -= 0.3

        score = max(0.0, score)
        is_valid = score >= 0.5 and not flags
        return is_valid, score, flags
