# transaction_vector_service/services/recurring_service.py
"""
Service for detecting and managing recurring transactions.

This module provides functionality for identifying recurring transaction patterns,
such as subscriptions, bills, and regular payments.
"""

import logging
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from uuid import UUID, uuid4
from collections import defaultdict

from ..models.transaction import TransactionSearch
from ..config.logging_config import get_logger
from ..config.constants import (
    MIN_RECURRING_OCCURRENCES,
    MAX_DATE_VARIANCE_DAYS,
    AMOUNT_VARIANCE_PERCENT
)
from ..models.recurring import (
    RecurringTransaction,
    RecurringPattern,
    RecurrenceFrequency,
    RecurrenceStatus
)
from .transaction_service import TransactionService
from .qdrant_client import QdrantService

logger = get_logger(__name__)


class RecurringService:
    """Service for detecting and managing recurring transactions."""

    def __init__(
        self,
        transaction_service: Optional[TransactionService] = None,
        qdrant_service: Optional[QdrantService] = None
    ):
        """
        Initialize the recurring service.
        
        Args:
            transaction_service: Optional transaction service instance
            qdrant_service: Optional Qdrant service instance
        """
        self.transaction_service = transaction_service or TransactionService()
        self.qdrant_service = qdrant_service or QdrantService()
        
        logger.info("Recurring service initialized")

    async def detect_recurring_transactions(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Detect recurring transaction patterns for a user.
        
        Args:
            user_id: User ID to detect patterns for
            
        Returns:
            List of detected recurring patterns
        """
        try:
            # Get all transactions for the user
            transactions, _ = await self.transaction_service.search_transactions(
                user_id=user_id,
                search_params=TransactionSearch(
                    limit=10000,  # Get as many as possible
                    include_future=False,
                    include_deleted=False
                )
            )
            
            if not transactions:
                logger.info(f"No transactions found for user {user_id}")
                return []
            
            # Group transactions by merchant
            merchant_groups = defaultdict(list)
            for tx in transactions:
                merchant_name = tx.get("normalized_merchant") or "unknown"
                merchant_groups[merchant_name].append(tx)
            
            # Detect patterns within each merchant group
            detected_patterns = []
            
            for merchant_name, merchant_txs in merchant_groups.items():
                # Skip merchants with too few transactions
                if len(merchant_txs) < MIN_RECURRING_OCCURRENCES:
                    continue
                
                # Group by approximate amount
                amount_groups = self._group_by_amount(merchant_txs)
                
                # For each amount group, check for recurring patterns
                for amount, amount_txs in amount_groups.items():
                    # Skip groups with too few transactions
                    if len(amount_txs) < MIN_RECURRING_OCCURRENCES:
                        continue
                    
                    # Sort by date
                    amount_txs.sort(key=lambda x: x.get("transaction_date", ""))
                    
                    # Detect frequency
                    frequency_info = self._detect_frequency(amount_txs)
                    if frequency_info:
                        frequency, confidence, interval_days = frequency_info
                        
                        # Create a pattern
                        pattern = {
                            "id": str(uuid4()),
                            "user_id": user_id,
                            "merchant_name": merchant_name,
                            "frequency": frequency.value,
                            "typical_amount": amount,
                            "amount_variance": self._calculate_amount_variance(amount_txs),
                            "transaction_count": len(amount_txs),
                            "first_occurrence": amount_txs[0].get("transaction_date"),
                            "last_occurrence": amount_txs[-1].get("transaction_date"),
                            "confidence_score": confidence,
                            "description_pattern": self._extract_description_pattern(amount_txs),
                            "transaction_ids": [tx.get("id") for tx in amount_txs],
                            "expected_next_date": self._calculate_next_date(
                                amount_txs[-1].get("transaction_date"),
                                frequency,
                                interval_days
                            )
                        }
                        
                        # Add merchant and category info if available
                        if "merchant_id" in amount_txs[0]:
                            pattern["merchant_id"] = amount_txs[0].get("merchant_id")
                        
                        if "category_id" in amount_txs[0]:
                            pattern["category_id"] = amount_txs[0].get("category_id")
                        
                        detected_patterns.append(pattern)
                        
                        # Mark transactions as recurring
                        for tx in amount_txs:
                            tx_id = tx.get("id")
                            if tx_id:
                                await self.transaction_service.update_transaction(
                                    transaction_id=tx_id,
                                    updates={
                                        "is_recurring": True,
                                        "recurring_pattern_id": pattern["id"]
                                    }
                                )
            
            return detected_patterns
        except Exception as e:
            logger.error(f"Error detecting recurring transactions: {str(e)}")
            return []

    def _group_by_amount(self, transactions: List[Dict[str, Any]]) -> Dict[float, List[Dict[str, Any]]]:
        """
        Group transactions by approximate amount.
        
        Args:
            transactions: List of transactions to group
            
        Returns:
            Dictionary of amount groups
        """
        amount_groups = defaultdict(list)
        
        for tx in transactions:
            amount = tx.get("amount", 0.0)
            
            # Find a matching group
            matched = False
            for group_amount in list(amount_groups.keys()):
                # Check if amount is within variance threshold
                variance_pct = abs(amount - group_amount) / max(abs(group_amount), 0.01)
                if variance_pct <= AMOUNT_VARIANCE_PERCENT:
                    amount_groups[group_amount].append(tx)
                    matched = True
                    break
            
            # If no match, create a new group
            if not matched:
                amount_groups[amount].append(tx)
        
        return amount_groups

    def _detect_frequency(
        self, 
        transactions: List[Dict[str, Any]]
    ) -> Optional[Tuple[RecurrenceFrequency, float, int]]:
        """
        Detect the frequency pattern in a series of transactions.
        
        Args:
            transactions: List of transactions sorted by date
            
        Returns:
            Tuple of (frequency, confidence score, interval in days) or None if no pattern
        """
        if len(transactions) < MIN_RECURRING_OCCURRENCES:
            return None
        
        # Extract dates
        dates = []
        for tx in transactions:
            date_str = tx.get("transaction_date")
            if date_str:
                try:
                    if isinstance(date_str, str):
                        tx_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    else:
                        tx_date = date_str
                    dates.append(tx_date)
                except ValueError:
                    continue
        
        if len(dates) < MIN_RECURRING_OCCURRENCES:
            return None
        
        # Calculate intervals between consecutive dates
        intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        
        if not intervals:
            return None
        
        # Calculate average interval
        avg_interval = sum(intervals) / len(intervals)
        
        # Calculate variance in intervals
        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
        std_dev = variance ** 0.5
        
        # Check if variance is within threshold
        if std_dev > MAX_DATE_VARIANCE_DAYS:
            # Too much variance, check if it might be a multiple of the interval
            # For example, quarterly payments might show up as 90, 92, 91 days
            for multiplier in [2, 3, 4, 6, 12]:
                adjusted_intervals = [i / multiplier for i in intervals]
                adjusted_avg = sum(adjusted_intervals) / len(adjusted_intervals)
                adjusted_variance = sum((i - adjusted_avg) ** 2 for i in adjusted_intervals) / len(adjusted_intervals)
                adjusted_std_dev = adjusted_variance ** 0.5
                
                if adjusted_std_dev <= MAX_DATE_VARIANCE_DAYS:
                    avg_interval = adjusted_avg
                    std_dev = adjusted_std_dev
                    break
            else:
                # Still too much variance
                return None
        
        # Determine frequency based on average interval
        if 25 <= avg_interval <= 32:
            frequency = RecurrenceFrequency.MONTHLY
        elif 13 <= avg_interval <= 16:
            frequency = RecurrenceFrequency.BIWEEKLY
        elif 6 <= avg_interval <= 8:
            frequency = RecurrenceFrequency.WEEKLY
        elif 85 <= avg_interval <= 95:
            frequency = RecurrenceFrequency.QUARTERLY
        elif 175 <= avg_interval <= 185:
            frequency = RecurrenceFrequency.SEMIANNUAL
        elif 350 <= avg_interval <= 380:
            frequency = RecurrenceFrequency.ANNUAL
        elif avg_interval <= 3:
            frequency = RecurrenceFrequency.DAILY
        else:
            frequency = RecurrenceFrequency.IRREGULAR
        
        # Calculate confidence score (higher is better, 1.0 is perfect)
        confidence = 1.0 - min(std_dev / MAX_DATE_VARIANCE_DAYS, 0.99)
        
        return (frequency, confidence, round(avg_interval))

    def _calculate_amount_variance(self, transactions: List[Dict[str, Any]]) -> float:
        """
        Calculate the variance in transaction amounts.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Amount variance as a decimal (0.0-1.0)
        """
        if not transactions:
            return 0.0
            
        amounts = [tx.get("amount", 0.0) for tx in transactions]
        avg_amount = sum(amounts) / len(amounts)
        
        if avg_amount == 0:
            return 0.0
            
        # Calculate max deviation as percentage
        max_deviation = max(abs(a - avg_amount) for a in amounts)
        return max_deviation / abs(avg_amount)

    def _extract_description_pattern(self, transactions: List[Dict[str, Any]]) -> str:
        """
        Extract a common pattern from transaction descriptions.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Common description pattern
        """
        if not transactions:
            return ""
            
        # Extract descriptions
        descriptions = [
            tx.get("clean_description") or tx.get("description", "")
            for tx in transactions
        ]
        
        # Simple implementation - use the shortest description as the pattern
        # In a real system, this would use more sophisticated NLP techniques
        if descriptions:
            return min(descriptions, key=len)
        
        return ""

    def _calculate_next_date(
        self, 
        last_date: Union[str, date], 
        frequency: RecurrenceFrequency,
        interval_days: int
    ) -> Optional[date]:
        """
        Calculate the expected next date for a recurring pattern.
        
        Args:
            last_date: Last occurrence date
            frequency: Detected frequency
            interval_days: Average interval in days
            
        Returns:
            Expected next date or None if can't be calculated
        """
        if not last_date:
            return None
            
        # Convert string to date if needed
        if isinstance(last_date, str):
            try:
                last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
            except ValueError:
                return None
        
        # Calculate next date based on frequency
        if frequency == RecurrenceFrequency.DAILY:
            return last_date + timedelta(days=1)
        elif frequency == RecurrenceFrequency.WEEKLY:
            return last_date + timedelta(weeks=1)
        elif frequency == RecurrenceFrequency.BIWEEKLY:
            return last_date + timedelta(weeks=2)
        elif frequency == RecurrenceFrequency.MONTHLY:
            # This is simplified - doesn't handle month boundaries correctly
            month = last_date.month + 1
            year = last_date.year
            if month > 12:
                month = 1
                year += 1
            day = min(last_date.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1])
            return date(year, month, day)
        elif frequency == RecurrenceFrequency.QUARTERLY:
            # Add 3 months approximately
            return last_date + timedelta(days=interval_days)
        elif frequency == RecurrenceFrequency.SEMIANNUAL:
            # Add 6 months approximately
            return last_date + timedelta(days=interval_days)
        elif frequency == RecurrenceFrequency.ANNUAL:
            return date(last_date.year + 1, last_date.month, last_date.day)
        else:
            # For irregular patterns, just add the average interval
            return last_date + timedelta(days=interval_days)

    async def get_recurring_patterns(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get recurring patterns for a user.
        
        Args:
            user_id: User ID to get patterns for
            
        Returns:
            List of recurring patterns
        """
        try:
            # This function would typically retrieve patterns from a database
            # For now, we'll just detect them on demand
            return await self.detect_recurring_transactions(user_id)
        except Exception as e:
            logger.error(f"Error getting recurring patterns: {str(e)}")
            return []

    async def get_recurring_pattern(self, pattern_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """
        Get a specific recurring pattern.
        
        Args:
            pattern_id: Pattern ID to retrieve
            
        Returns:
            Recurring pattern or None if not found
        """
        # This is a stub - in a real implementation, this would retrieve from a database
        return None

    async def update_recurring_pattern(
        self, 
        pattern_id: Union[str, UUID], 
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a recurring pattern.
        
        Args:
            pattern_id: Pattern ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful
        """
        # This is a stub - in a real implementation, this would update in a database
        return False

    async def delete_recurring_pattern(self, pattern_id: Union[str, UUID]) -> bool:
        """
        Delete a recurring pattern.
        
        Args:
            pattern_id: Pattern ID to delete
            
        Returns:
            True if deletion was successful
        """
        # This is a stub - in a real implementation, this would delete from a database
        return False

    async def get_upcoming_payments(
        self, 
        user_id: int, 
        days_ahead: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming recurring payments for a user.
        
        Args:
            user_id: User ID to get upcoming payments for
            days_ahead: Number of days ahead to look
            
        Returns:
            List of upcoming payments
        """
        try:
            # Get all recurring patterns for the user
            patterns = await self.get_recurring_patterns(user_id)
            
            # Filter to active patterns
            active_patterns = [p for p in patterns if p.get("confidence_score", 0) > 0.5]
            
            # Calculate upcoming payments
            upcoming = []
            end_date = datetime.now().date() + timedelta(days=days_ahead)
            
            for pattern in active_patterns:
                next_date = pattern.get("expected_next_date")
                if next_date:
                    # Convert string to date if needed
                    if isinstance(next_date, str):
                        try:
                            next_date = datetime.strptime(next_date, "%Y-%m-%d").date()
                        except ValueError:
                            continue
                    
                    # Check if it's within the requested time range
                    if next_date <= end_date:
                        upcoming.append({
                            "pattern_id": pattern.get("id"),
                            "merchant_name": pattern.get("merchant_name"),
                            "amount": pattern.get("typical_amount"),
                            "expected_date": next_date.isoformat(),
                            "frequency": pattern.get("frequency"),
                            "confidence": pattern.get("confidence_score")
                        })
            
            # Sort by date
            upcoming.sort(key=lambda x: x.get("expected_date", ""))
            
            return upcoming
        except Exception as e:
            logger.error(f"Error getting upcoming payments: {str(e)}")
            return []