import logging
from typing import List, Dict


class DataQualityValidator:
    """Validates transaction and account consistency."""

    def __init__(self, threshold: float = 1.0) -> None:
        """Initialize the validator.

        Args:
            threshold: Maximum allowed difference between the account balance
                and the aggregated transactions total before triggering an
                alert.
        """
        self.threshold = threshold
        self.logger = logging.getLogger("data_quality")

    def validate_account_balance_consistency(
        self, account_balance: float, recent_transactions: List[float]
    ) -> Dict[str, float | bool]:
        """Check that the balance matches the sum of recent transactions.

        Args:
            account_balance: Reported balance for the account.
            recent_transactions: Monetary amounts for recent transactions.

        Returns:
            Dict containing ``balance_check_passed`` boolean and a
            ``quality_score`` between 0 and 1.
        """
        if account_balance is None or recent_transactions is None:
            return {"balance_check_passed": True, "quality_score": 1.0}

        total = sum(recent_transactions)
        difference = account_balance - total
        passed = abs(difference) <= self.threshold

        # Quality score is 1 when perfectly matched and degrades linearly with
        # the relative difference, capped between 0 and 1.
        if abs(account_balance) > 0:
            relative_diff = min(abs(difference) / abs(account_balance), 1.0)
            quality_score = 1.0 - relative_diff
        else:
            quality_score = 1.0 if passed else 0.0

        if not passed:
            self.logger.warning(
                "Balance inconsistency detected: balance=%s, total=%s, diff=%s",
                account_balance,
                total,
                difference,
            )

        return {
            "balance_check_passed": passed,
            "quality_score": round(quality_score, 3),
        }
