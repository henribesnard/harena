"""
Enhanced Response Generator - Phase 1 Integration
Adds Analytics Agent and User Profile support to Response Generator

This module extends the base Response Generator with:
- Analytics Agent integration for advanced metrics (MoM, YoY, trends, anomalies)
- User Profile Service for personalization
- Automatic detection of analytics needs from user query
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from conversation_service.agents.response_generator import (
    ResponseGeneratorAgent,
    ResponseGenerationConfig,
    GeneratedResponse
)
from conversation_service.agents.logic.analytics_agent import (
    AnalyticsAgent,
    PeriodType,
    AnomalyMethod
)
from conversation_service.agents.logic.context_manager import OptimizedContext
from conversation_service.services.user_profile import UserProfileService
from conversation_service.models.user_profile.entities import QueryPattern

logger = logging.getLogger(__name__)


class EnhancedResponseGeneratorAgent(ResponseGeneratorAgent):
    """
    Enhanced Response Generator with Analytics Agent and User Profile

    Extends base ResponseGeneratorAgent with:
    - Advanced analytics (comparisons, trends, anomalies)
    - User profile integration
    - Query pattern detection
    """

    def __init__(
        self,
        deepseek_client,
        profile_service: Optional[UserProfileService] = None,
        config: Optional[ResponseGenerationConfig] = None
    ):
        super().__init__(deepseek_client, config)

        # Initialize Analytics Agent
        self.analytics_agent = AnalyticsAgent()

        # User Profile Service
        self.profile_service = profile_service

        logger.info("EnhancedResponseGeneratorAgent v3.0 initialized with Analytics Agent")

    async def generate_response(
        self,
        optimized_context: OptimizedContext,
        user_id: int,
        streaming: Optional[bool] = None
    ) -> GeneratedResponse:
        """
        Generate enhanced response with analytics and profile integration

        Args:
            optimized_context: Optimized context from ContextManager
            user_id: User ID
            streaming: Override streaming config

        Returns:
            GeneratedResponse with enhanced insights
        """
        # Detect if analytics are needed
        analytics_type = await self._detect_analytics_need(optimized_context)

        # Update user profile with query pattern
        if self.profile_service and analytics_type:
            await self._update_user_profile(user_id, analytics_type, optimized_context)

        # Run analytics if needed
        analytics_results = None
        if analytics_type and optimized_context.search_results:
            analytics_results = await self._run_analytics(
                analytics_type,
                optimized_context
            )

        # Inject analytics into context for LLM
        if analytics_results:
            optimized_context = self._enrich_context_with_analytics(
                optimized_context,
                analytics_results
            )

        # Call parent generate_response with enriched context
        response = await super().generate_response(optimized_context, user_id, streaming)

        # Add analytics metadata
        if analytics_results:
            response.metadata['analytics_applied'] = analytics_type
            response.metadata['analytics_results'] = self._serialize_analytics(analytics_results)

        return response

    async def _detect_analytics_need(self, context: OptimizedContext) -> Optional[str]:
        """
        Detect if advanced analytics are needed based on user query

        Returns:
            Analytics type needed: 'mom', 'yoy', 'trend', 'anomaly', 'pivot', None
        """
        user_message = context.current_query.get('user_message', '').lower()
        intent = context.current_query.get('intent_classification', {}).get('intent', '')

        # Detect MoM (Month-over-Month)
        if any(keyword in user_message for keyword in [
            'mois dernier', 'mois précédent', 'par rapport au mois',
            'compare mois', 'versus mois', 'vs mois'
        ]):
            return 'mom'

        # Detect YoY (Year-over-Year)
        if any(keyword in user_message for keyword in [
            'année dernière', 'année précédente', 'par rapport à l\'année',
            'même mois l\'année', 'compare année', 'yoy', 'year over year'
        ]):
            return 'yoy'

        # Detect trend analysis
        if any(keyword in user_message for keyword in [
            'évolution', 'tendance', 'progression', 'trend',
            'derniers mois', 'sur 3 mois', 'sur 6 mois', 'sur 12 mois'
        ]):
            return 'trend'

        # Detect anomaly detection
        if any(keyword in user_message for keyword in [
            'anormal', 'inhabituel', 'suspect', 'anomalie',
            'transaction étrange', 'dépense inhabituelle'
        ]):
            return 'anomaly'

        # Detect pivot/breakdown analysis
        if any(keyword in user_message for keyword in [
            'par catégorie', 'répartition', 'breakdown', 'distribution',
            'top catégories', 'par marchand'
        ]):
            return 'pivot'

        return None

    async def _update_user_profile(
        self,
        user_id: int,
        analytics_type: str,
        context: OptimizedContext
    ):
        """Update user profile with detected query pattern"""
        try:
            # Map analytics type to QueryPattern
            pattern_mapping = {
                'mom': QueryPattern.MOM_COMPARISONS,
                'yoy': QueryPattern.YOY_COMPARISONS,
                'trend': QueryPattern.MONTHLY_REPORTS,
                'anomaly': QueryPattern.ANOMALY_DETECTION,
                'pivot': QueryPattern.CATEGORY_ANALYSIS
            }

            pattern = pattern_mapping.get(analytics_type)
            if pattern and self.profile_service:
                await self.profile_service.add_query_pattern(user_id, pattern)
                await self.profile_service.increment_query_count(user_id)

                logger.info(f"Updated user profile for user_id={user_id}, pattern={pattern.value}")

        except Exception as e:
            logger.warning(f"Failed to update user profile: {str(e)}")

    async def _run_analytics(
        self,
        analytics_type: str,
        context: OptimizedContext
    ) -> Optional[Any]:
        """
        Run appropriate analytics based on type

        Returns:
            Analytics result object
        """
        try:
            transactions = self._extract_transactions_list(context.search_results)

            if not transactions:
                return None

            if analytics_type == 'mom':
                return await self._run_mom_comparison(transactions, context)

            elif analytics_type == 'yoy':
                return await self._run_yoy_comparison(transactions, context)

            elif analytics_type == 'trend':
                return await self._run_trend_analysis(transactions)

            elif analytics_type == 'anomaly':
                return await self._run_anomaly_detection(transactions)

            elif analytics_type == 'pivot':
                return await self._run_pivot_analysis(transactions)

            return None

        except Exception as e:
            logger.error(f"Error running analytics: {str(e)}")
            return None

    def _extract_transactions_list(self, search_results) -> List[Dict[str, Any]]:
        """Extract transactions list from search results"""
        if not search_results or not search_results.data:
            return []

        return search_results.data

    async def _run_mom_comparison(self, transactions: List[Dict], context) -> Any:
        """Run Month-over-Month comparison"""
        # TODO: Split transactions by current and previous month
        # For now, simple comparison using all transactions as current period
        # In real implementation, would need date filtering

        # Placeholder: Use first half vs second half
        mid_point = len(transactions) // 2
        if mid_point == 0:
            return None

        period_1 = transactions[:mid_point]
        period_2 = transactions[mid_point:]

        result = await self.analytics_agent.compare_periods(
            period_1,
            period_2,
            PeriodType.MOM,
            top_contributors_limit=5
        )

        return result

    async def _run_yoy_comparison(self, transactions: List[Dict], context) -> Any:
        """Run Year-over-Year comparison"""
        # Placeholder: Similar to MoM
        mid_point = len(transactions) // 2
        if mid_point == 0:
            return None

        period_1 = transactions[:mid_point]
        period_2 = transactions[mid_point:]

        result = await self.analytics_agent.compare_periods(
            period_1,
            period_2,
            PeriodType.YOY,
            top_contributors_limit=5
        )

        return result

    async def _run_trend_analysis(self, transactions: List[Dict]) -> Any:
        """Run trend analysis"""
        result = await self.analytics_agent.calculate_trend(
            transactions,
            aggregation="monthly",
            forecast_periods=2
        )

        return result

    async def _run_anomaly_detection(self, transactions: List[Dict]) -> Any:
        """Run anomaly detection"""
        result = await self.analytics_agent.detect_anomalies(
            transactions,
            method=AnomalyMethod.ZSCORE,
            threshold=2.0
        )

        return result

    async def _run_pivot_analysis(self, transactions: List[Dict]) -> Any:
        """Run pivot table analysis"""
        result = await self.analytics_agent.pivot_analysis(
            transactions,
            rows="category",
            columns="month",
            values="amount",
            aggfunc="sum"
        )

        return result

    def _enrich_context_with_analytics(
        self,
        context: OptimizedContext,
        analytics_results: Any
    ) -> OptimizedContext:
        """
        Enrich context with analytics results for LLM

        This injects analytics metrics into search_results context
        so the LLM can use them in response generation
        """
        # Add analytics summary to metadata
        if not hasattr(context, 'analytics_summary'):
            context.analytics_summary = {}

        # Serialize analytics results
        context.analytics_summary = self._serialize_analytics(analytics_results)

        return context

    def _serialize_analytics(self, analytics_results: Any) -> Dict[str, Any]:
        """Serialize analytics results for JSON/metadata"""
        if analytics_results is None:
            return {}

        # Handle ComparisonResult
        if hasattr(analytics_results, 'period_1_total'):
            return {
                'type': 'comparison',
                'period_1_total': analytics_results.period_1_total,
                'period_2_total': analytics_results.period_2_total,
                'absolute_change': analytics_results.absolute_change,
                'percentage_change': analytics_results.percentage_change,
                'top_contributors': analytics_results.top_contributors[:3]
            }

        # Handle TrendAnalysis
        if hasattr(analytics_results, 'direction'):
            return {
                'type': 'trend',
                'direction': analytics_results.direction.value,
                'slope': analytics_results.slope,
                'r_squared': analytics_results.r_squared,
                'forecast_values': analytics_results.forecast_values
            }

        # Handle Anomaly list
        if isinstance(analytics_results, list) and analytics_results and hasattr(analytics_results[0], 'score'):
            return {
                'type': 'anomaly',
                'anomalies_count': len(analytics_results),
                'top_anomalies': [
                    {
                        'transaction_id': a.transaction_id,
                        'amount': a.amount,
                        'score': a.score,
                        'severity': a.severity
                    }
                    for a in analytics_results[:5]
                ]
            }

        # Handle PivotTable
        if hasattr(analytics_results, 'grand_total'):
            return {
                'type': 'pivot',
                'grand_total': analytics_results.grand_total,
                'rows_count': len(analytics_results.row_totals),
                'columns_count': len(analytics_results.column_totals)
            }

        return {'type': 'unknown'}

    async def _extract_financial_insights(
        self,
        search_results,
        current_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Override parent method to add analytics-powered insights
        """
        # Get base insights from parent
        insights = await super()._extract_financial_insights(search_results, current_query)

        # Add analytics-powered insights if available
        # (These would have been computed in generate_response)

        return insights


__all__ = ["EnhancedResponseGeneratorAgent"]
