# transaction_vector_service/services/insight_service.py
"""
Service for generating financial insights.

This module provides functionality for analyzing transaction data and
generating personalized financial insights and recommendations.
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
    INSIGHT_TIMEFRAMES,
    MIN_TRANSACTIONS_FOR_INSIGHTS
)
from ..models.insight import (
    Insight,
    InsightCreate,
    InsightType,
    InsightSeverity,
    InsightTimeframe
)
from .transaction_service import TransactionService
from .category_service import CategoryService
from .merchant_service import MerchantService
from .recurring_service import RecurringService

logger = get_logger(__name__)


class InsightService:
    """Service for generating and managing financial insights."""
    
    def __init__(
        self,
        transaction_service: Optional[TransactionService] = None,
        category_service: Optional[CategoryService] = None,
        merchant_service: Optional[MerchantService] = None,
        recurring_service: Optional[RecurringService] = None
    ):
        """
        Initialize the insight service.
        
        Args:
            transaction_service: Optional transaction service instance
            category_service: Optional category service instance
            merchant_service: Optional merchant service instance
            recurring_service: Optional recurring service instance
        """
        self.transaction_service = transaction_service or TransactionService()
        self.category_service = category_service or CategoryService()
        self.merchant_service = merchant_service or MerchantService()
        self.recurring_service = recurring_service or RecurringService(self.transaction_service)
        
        # Define insight generators
        self.insight_generators = {
            InsightType.SPENDING_PATTERN: self._generate_spending_pattern_insights,
            InsightType.BUDGET_ALERT: self._generate_budget_alerts,
            InsightType.RECURRING_DETECTION: self._generate_recurring_detection_insights,
            InsightType.ANOMALY_DETECTION: self._generate_anomaly_insights,
            InsightType.MERCHANT_PATTERN: self._generate_merchant_pattern_insights,
            InsightType.SAVING_OPPORTUNITY: self._generate_saving_opportunity_insights,
            InsightType.FINANCIAL_HEALTH: self._generate_financial_health_insights
        }
        
        logger.info("Insight service initialized")
    
    async def generate_insights(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Generate all types of insights for a user.
        
        Args:
            user_id: User ID to generate insights for
            
        Returns:
            List of generated insights
        """
        try:
            all_insights = []
            
            # Run all insight generators
            for insight_type, generator in self.insight_generators.items():
                try:
                    insights = await generator(user_id)
                    all_insights.extend(insights)
                except Exception as e:
                    logger.error(f"Error generating {insight_type} insights: {str(e)}")
            
            # Sort insights by severity (most critical first) and then by creation time
            all_insights.sort(
                key=lambda x: (
                    -self._get_severity_score(x.get("severity", "info")),
                    x.get("created_at", datetime.now().isoformat())
                )
            )
            
            return all_insights
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    def _get_severity_score(self, severity: str) -> int:
        """
        Convert severity string to numeric score for sorting.
        
        Args:
            severity: Severity string
            
        Returns:
            Numeric score (higher is more severe)
        """
        severity_scores = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "info": 0
        }
        return severity_scores.get(severity.lower(), 0)
    
    async def _generate_spending_pattern_insights(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Generate insights about spending patterns.
        
        Args:
            user_id: User ID to generate insights for
            
        Returns:
            List of spending pattern insights
        """
        insights = []
        
        try:
            # Get transaction stats
            month_stats = await self.transaction_service.get_transaction_stats(
                user_id=user_id,
                start_date=datetime.now().replace(day=1).date(),  # First day of current month
                end_date=datetime.now().date()
            )
            
            # Get previous month stats
            last_month = datetime.now().replace(day=1) - timedelta(days=1)
            prev_month_start = last_month.replace(day=1).date()
            prev_month_end = last_month.date()
            
            prev_month_stats = await self.transaction_service.get_transaction_stats(
                user_id=user_id,
                start_date=prev_month_start,
                end_date=prev_month_end
            )
            
            # Compare spending by category
            current_by_category = month_stats.get("by_category", {})
            prev_by_category = prev_month_stats.get("by_category", {})
            
            # Check for significant increases
            for category_id, current in current_by_category.items():
                if category_id in prev_by_category:
                    prev = prev_by_category[category_id]
                    current_total = current.get("total", 0)
                    prev_total = prev.get("total", 0)
                    
                    # Skip if not enough spending to analyze
                    if abs(current_total) < 100 or abs(prev_total) < 100:
                        continue
                    
                    # Calculate percent change
                    if prev_total != 0:
                        percent_change = (current_total - prev_total) / abs(prev_total) * 100
                    else:
                        percent_change = 100  # Arbitrary large increase
                    
                    # Get category name
                    category = await self.category_service.get_category(int(category_id))
                    category_name = category.get("name", "Unknown") if category else "Unknown"
                    
                    # Generate insight for significant increase (>20%)
                    if percent_change > 20:
                        insights.append({
                            "id": str(uuid4()),
                            "type": InsightType.SPENDING_PATTERN.value,
                            "title": f"Spending increase in {category_name}",
                            "description": f"Your spending in {category_name} has increased by {abs(percent_change):.1f}% compared to last month.",
                            "severity": InsightSeverity.MEDIUM.value,
                            "timeframe": InsightTimeframe.MONTHLY.value,
                            "start_date": datetime.now().replace(day=1).date().isoformat(),
                            "end_date": datetime.now().date().isoformat(),
                            "category_id": int(category_id),
                            "category_name": category_name,
                            "amount": current_total,
                            "previous_amount": prev_total,
                            "percent_change": percent_change,
                            "created_at": datetime.now().isoformat(),
                            "is_read": False,
                            "is_dismissed": False,
                            "user_id": user_id
                        })
                    # Generate insight for significant decrease (>20% for non-negative categories)
                    elif percent_change < -20 and prev_total > 0:
                        insights.append({
                            "id": str(uuid4()),
                            "type": InsightType.SPENDING_PATTERN.value,
                            "title": f"Spending decrease in {category_name}",
                            "description": f"Your spending in {category_name} has decreased by {abs(percent_change):.1f}% compared to last month.",
                            "severity": InsightSeverity.INFO.value,
                            "timeframe": InsightTimeframe.MONTHLY.value,
                            "start_date": datetime.now().replace(day=1).date().isoformat(),
                            "end_date": datetime.now().date().isoformat(),
                            "category_id": int(category_id),
                            "category_name": category_name,
                            "amount": current_total,
                            "previous_amount": prev_total,
                            "percent_change": percent_change,
                            "created_at": datetime.now().isoformat(),
                            "is_read": False,
                            "is_dismissed": False,
                            "user_id": user_id
                        })
        except Exception as e:
            logger.error(f"Error generating spending pattern insights: {str(e)}")
        
        return insights
    
    async def _generate_budget_alerts(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Generate budget alert insights.
        
        Args:
            user_id: User ID to generate insights for
            
        Returns:
            List of budget alert insights
        """
        # In a real implementation, this would check against user-defined budgets
        # For now, return an empty list as a placeholder
        return []
    
    async def _generate_recurring_detection_insights(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Generate insights about newly detected recurring transactions.
        
        Args:
            user_id: User ID to generate insights for
            
        Returns:
            List of recurring detection insights
        """
        insights = []
        
        try:
            # Get recurring patterns
            recurring_patterns = await self.recurring_service.get_recurring_patterns(user_id)
            
            # Focus on high-confidence patterns
            high_confidence_patterns = [
                p for p in recurring_patterns 
                if p.get("confidence_score", 0) > 0.8
            ]
            
            # Generate insight for each high-confidence pattern
            for pattern in high_confidence_patterns:
                merchant_name = pattern.get("merchant_name", "Unknown")
                amount = pattern.get("typical_amount", 0)
                frequency = pattern.get("frequency", "unknown")
                
                # Make frequency more readable
                frequency_display = {
                    "daily": "daily",
                    "weekly": "weekly",
                    "biweekly": "bi-weekly",
                    "monthly": "monthly",
                    "quarterly": "quarterly",
                    "semiannual": "every six months",
                    "annual": "yearly",
                    "irregular": "recurring"
                }.get(frequency, frequency)
                
                insights.append({
                    "id": str(uuid4()),
                    "type": InsightType.RECURRING_DETECTION.value,
                    "title": f"Recurring payment detected: {merchant_name}",
                    "description": f"We've detected a {frequency_display} payment of {abs(amount):.2f} € to {merchant_name}.",
                    "severity": InsightSeverity.INFO.value,
                    "timeframe": InsightTimeframe.CUSTOM.value,
                    "start_date": pattern.get("first_occurrence", datetime.now().date().isoformat()),
                    "end_date": pattern.get("last_occurrence", datetime.now().date().isoformat()),
                    "merchant_name": merchant_name,
                    "amount": amount,
                    "data_points": {
                        "frequency": frequency,
                        "next_date": pattern.get("expected_next_date"),
                        "transaction_count": pattern.get("transaction_count", 0)
                    },
                    "created_at": datetime.now().isoformat(),
                    "is_read": False,
                    "is_dismissed": False,
                    "user_id": user_id
                })
        except Exception as e:
            logger.error(f"Error generating recurring detection insights: {str(e)}")
        
        return insights
    
    async def _generate_anomaly_insights(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Generate insights about anomalous transactions.
        
        Args:
            user_id: User ID to generate insights for
            
        Returns:
            List of anomaly insights
        """
        insights = []
        
        try:
            # Get recent transactions
            transactions, _ = await self.transaction_service.search_transactions(
                user_id=user_id,
                search_params=TransactionSearch(
                    start_date=(datetime.now() - timedelta(days=30)).date(),
                    end_date=datetime.now().date(),
                    limit=1000,
                    include_future=False,
                    include_deleted=False
                )
            )
            
            # Group by merchant
            merchant_groups = defaultdict(list)
            for tx in transactions:
                merchant = tx.get("normalized_merchant") or "unknown"
                merchant_groups[merchant].append(tx)
            
            # Check for unusually large transactions
            for merchant, txs in merchant_groups.items():
                if len(txs) < 3:
                    continue
                
                # Calculate average and standard deviation
                amounts = [abs(tx.get("amount", 0)) for tx in txs]
                avg_amount = sum(amounts) / len(amounts)
                std_dev = (sum((a - avg_amount) ** 2 for a in amounts) / len(amounts)) ** 0.5
                
                # Check for outliers (more than 2.5 standard deviations above average)
                for tx in txs:
                    amount = abs(tx.get("amount", 0))
                    if amount > avg_amount + 2.5 * std_dev and amount > 50:  # Minimum amount threshold
                        insights.append({
                            "id": str(uuid4()),
                            "type": InsightType.ANOMALY_DETECTION.value,
                            "title": f"Unusually large payment to {merchant}",
                            "description": f"Your payment of {amount:.2f} € to {merchant} is {(amount/avg_amount):.1f}x your usual amount.",
                            "severity": InsightSeverity.MEDIUM.value,
                            "timeframe": InsightTimeframe.CUSTOM.value,
                            "start_date": tx.get("transaction_date", datetime.now().date().isoformat()),
                            "end_date": tx.get("transaction_date", datetime.now().date().isoformat()),
                            "merchant_name": merchant,
                            "amount": tx.get("amount", 0),
                            "previous_amount": avg_amount,
                            "percent_change": ((amount - avg_amount) / avg_amount) * 100,
                            "transaction_ids": [tx.get("id")],
                            "created_at": datetime.now().isoformat(),
                            "is_read": False,
                            "is_dismissed": False,
                            "user_id": user_id
                        })
        except Exception as e:
            logger.error(f"Error generating anomaly insights: {str(e)}")
        
        return insights
    
    async def _generate_merchant_pattern_insights(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Generate insights about merchant spending patterns.
        
        Args:
            user_id: User ID to generate insights for
            
        Returns:
            List of merchant pattern insights
        """
        insights = []
        
        try:
            # Get transactions from last 3 months
            transactions, _ = await self.transaction_service.search_transactions(
                user_id=user_id,
                search_params=TransactionSearch(
                    start_date=(datetime.now() - timedelta(days=90)).date(),
                    end_date=datetime.now().date(),
                    limit=2000,
                    include_future=False,
                    include_deleted=False
                )
            )
            
            # Group by merchant
            merchant_groups = defaultdict(list)
            for tx in transactions:
                merchant = tx.get("normalized_merchant") or "unknown"
                merchant_groups[merchant].append(tx)
            
            # Identify frequently visited merchants
            for merchant, txs in merchant_groups.items():
                if len(txs) >= 5:  # At least 5 visits in 3 months
                    # Calculate total spent
                    total_spent = sum(abs(tx.get("amount", 0)) for tx in txs)
                    avg_transaction = total_spent / len(txs)
                    
                    # Calculate average interval between visits
                    dates = []
                    for tx in txs:
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
                    
                    dates.sort()
                    if len(dates) >= 5:
                        intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                        avg_interval = sum(intervals) / len(intervals)
                        
                        # Generate insight for frequent merchants
                        insights.append({
                            "id": str(uuid4()),
                            "type": InsightType.MERCHANT_PATTERN.value,
                            "title": f"Frequent spending at {merchant}",
                            "description": f"You've made {len(txs)} purchases at {merchant} in the last 3 months, spending a total of {total_spent:.2f} €.",
                            "severity": InsightSeverity.INFO.value,
                            "timeframe": InsightTimeframe.CUSTOM.value,
                            "start_date": dates[0].isoformat(),
                            "end_date": dates[-1].isoformat(),
                            "merchant_name": merchant,
                            "amount": total_spent,
                            "data_points": {
                                "visit_count": len(txs),
                                "average_transaction": avg_transaction,
                                "average_interval_days": avg_interval,
                                "first_visit": dates[0].isoformat(),
                                "last_visit": dates[-1].isoformat()
                            },
                            "created_at": datetime.now().isoformat(),
                            "is_read": False,
                            "is_dismissed": False,
                            "user_id": user_id
                        })
        except Exception as e:
            logger.error(f"Error generating merchant pattern insights: {str(e)}")
        
        return insights
    
    async def _generate_saving_opportunity_insights(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Generate insights about potential saving opportunities.
        
        Args:
            user_id: User ID to generate insights for
            
        Returns:
            List of saving opportunity insights
        """
        insights = []
        
        try:
            # Get recurring patterns
            recurring_patterns = await self.recurring_service.get_recurring_patterns(user_id)
            
            # Look for potentially unnecessary subscriptions
            for pattern in recurring_patterns:
                amount = pattern.get("typical_amount", 0)
                
                # Only consider negative amounts (payments)
                if amount >= 0:
                    continue
                
                merchant_name = pattern.get("merchant_name", "Unknown")
                frequency = pattern.get("frequency", "unknown")
                
                # Calculate annual cost
                annual_cost = abs(amount)
                if frequency == "monthly":
                    annual_cost *= 12
                elif frequency == "weekly":
                    annual_cost *= 52
                elif frequency == "biweekly":
                    annual_cost *= 26
                elif frequency == "quarterly":
                    annual_cost *= 4
                elif frequency == "semiannual":
                    annual_cost *= 2
                
                # Generate insight for potentially high-cost subscriptions
                if annual_cost > 200:
                    frequency_display = {
                        "monthly": "monthly",
                        "weekly": "weekly",
                        "biweekly": "bi-weekly",
                        "quarterly": "quarterly",
                        "semiannual": "semi-annual",
                        "annual": "annual"
                    }.get(frequency, frequency)
                    
                    insights.append({
                        "id": str(uuid4()),
                        "type": InsightType.SAVING_OPPORTUNITY.value,
                        "title": f"Review your {merchant_name} subscription",
                        "description": f"Your {frequency_display} payment of {abs(amount):.2f} € to {merchant_name} costs you {annual_cost:.2f} € per year.",
                        "severity": InsightSeverity.LOW.value,
                        "timeframe": InsightTimeframe.YEARLY.value,
                        "start_date": datetime.now().date().isoformat(),
                        "merchant_name": merchant_name,
                        "amount": annual_cost,
                        "data_points": {
                            "recurring_id": pattern.get("id"),
                            "payment_amount": abs(amount),
                            "frequency": frequency,
                            "payment_count_per_year": annual_cost / abs(amount)
                        },
                        "is_actionable": True,
                        "created_at": datetime.now().isoformat(),
                        "is_read": False,
                        "is_dismissed": False,
                        "user_id": user_id
                    })
        except Exception as e:
            logger.error(f"Error generating saving opportunity insights: {str(e)}")
        
        return insights
    
    async def _generate_financial_health_insights(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Generate insights about overall financial health.
        
        Args:
            user_id: User ID to generate insights for
            
        Returns:
            List of financial health insights
        """
        # In a real implementation, this would analyze income vs. expenses, savings rate, etc.
        # For now, return an empty list as a placeholder
        return []
    
    async def get_insights(
        self, 
        user_id: int,
        types: Optional[List[str]] = None,
        limit: int = 10,
        include_read: bool = False,
        include_dismissed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get insights for a user with filtering options.
        
        Args:
            user_id: User ID to get insights for
            types: Optional list of insight types to filter by
            limit: Maximum number of insights to return
            include_read: Whether to include already read insights
            include_dismissed: Whether to include dismissed insights
            
        Returns:
            List of filtered insights
        """
        try:
            # Generate insights (in a real system, these would be retrieved from a database)
            all_insights = await self.generate_insights(user_id)
            
            # Apply filters
            filtered_insights = []
            for insight in all_insights:
                # Filter by type
                if types and insight.get("type") not in types:
                    continue
                
                # Filter by read status
                if not include_read and insight.get("is_read", False):
                    continue
                
                # Filter by dismissed status
                if not include_dismissed and insight.get("is_dismissed", False):
                    continue
                
                filtered_insights.append(insight)
            
            # Sort by severity and creation time
            filtered_insights.sort(
                key=lambda x: (
                    -self._get_severity_score(x.get("severity", "info")),
                    x.get("created_at", datetime.now().isoformat())
                )
            )
            
            # Apply limit
            return filtered_insights[:limit]
        except Exception as e:
            logger.error(f"Error getting insights: {str(e)}")
            return []
    
    async def mark_insight_as_read(self, insight_id: Union[str, UUID], user_id: int) -> bool:
        """
        Mark an insight as read.
        
        Args:
            insight_id: Insight ID to mark as read
            user_id: User ID for security validation
            
        Returns:
            True if successful
        """
        # In a real implementation, this would update a database record
        # For now, return True as a placeholder
        return True
    
    async def dismiss_insight(self, insight_id: Union[str, UUID], user_id: int) -> bool:
        """
        Dismiss an insight.
        
        Args:
            insight_id: Insight ID to dismiss
            user_id: User ID for security validation
            
        Returns:
            True if successful
        """
        # In a real implementation, this would update a database record
        # For now, return True as a placeholder
        return True
    
    async def take_action_on_insight(
        self, 
        insight_id: Union[str, UUID], 
        user_id: int,
        action: str
    ) -> bool:
        """
        Record an action taken on an actionable insight.
        
        Args:
            insight_id: Insight ID
            user_id: User ID for security validation
            action: Description of action taken
            
        Returns:
            True if successful
        """
        # In a real implementation, this would update a database record
        # For now, return True as a placeholder
        return True