"""
Analytics Agent - Logic-based agent for advanced statistical analysis

This agent provides:
- Temporal comparisons (MoM, YoY, QoQ)
- Trend analysis and linear regression
- Anomaly detection (Z-score, IQR)
- Pivot table analysis (multi-dimensional aggregations)

No LLM calls - pure Python logic using pandas, numpy, scipy.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class PeriodType(str, Enum):
    """Types of period comparisons"""
    MOM = "month_over_month"  # Month-over-Month
    YOY = "year_over_year"    # Year-over-Year
    QOQ = "quarter_over_quarter"  # Quarter-over-Quarter
    WOW = "week_over_week"    # Week-over-Week


class AnomalyMethod(str, Enum):
    """Anomaly detection methods"""
    ZSCORE = "zscore"
    IQR = "iqr"


class TrendDirection(str, Enum):
    """Trend directions"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass
class ComparisonResult:
    """Result of period comparison"""
    period_1_total: float
    period_2_total: float
    absolute_change: float
    percentage_change: float
    period_1_count: int
    period_2_count: int
    period_1_average: float
    period_2_average: float
    top_contributors: List[Dict[str, Any]]
    period_type: str


@dataclass
class TrendAnalysis:
    """Result of trend analysis"""
    direction: TrendDirection
    slope: float  # Regression slope
    intercept: float
    r_squared: float  # Quality of fit
    forecast_values: List[float]
    forecast_periods: List[str]
    confidence_interval_95: Tuple[List[float], List[float]]


@dataclass
class Anomaly:
    """Detected anomaly"""
    transaction_id: str
    amount: float
    date: str
    merchant_name: str
    score: float  # Z-score or IQR-based score
    reason: str
    severity: str  # "low", "medium", "high"


@dataclass
class PivotTable:
    """Multi-dimensional pivot table"""
    data: pd.DataFrame
    row_totals: pd.Series
    column_totals: pd.Series
    grand_total: float


class AnalyticsAgent:
    """
    Analytics Agent for advanced statistical analysis on transactions.

    This agent operates on transaction data and provides insights without
    requiring LLM calls.
    """

    def __init__(self):
        """Initialize Analytics Agent"""
        self.logger = logging.getLogger(__name__)

    async def compare_periods(
        self,
        transactions_period_1: List[Dict[str, Any]],
        transactions_period_2: List[Dict[str, Any]],
        period_type: PeriodType = PeriodType.MOM,
        top_contributors_limit: int = 5
    ) -> ComparisonResult:
        """
        Compare two periods of transactions (MoM, YoY, QoQ, WoW).

        Args:
            transactions_period_1: Transactions for first period (e.g., current month)
            transactions_period_2: Transactions for second period (e.g., previous month)
            period_type: Type of comparison (MOM, YOY, QOQ, WOW)
            top_contributors_limit: Number of top contributors to identify

        Returns:
            ComparisonResult with metrics and top contributors
        """
        try:
            # Convert to DataFrames
            df1 = pd.DataFrame(transactions_period_1)
            df2 = pd.DataFrame(transactions_period_2)

            # Calculate totals
            total_1 = df1['amount'].sum() if len(df1) > 0 else 0.0
            total_2 = df2['amount'].sum() if len(df2) > 0 else 0.0

            # Calculate changes
            absolute_change = total_1 - total_2
            percentage_change = ((total_1 - total_2) / total_2 * 100) if total_2 != 0 else 0.0

            # Calculate averages
            avg_1 = df1['amount'].mean() if len(df1) > 0 else 0.0
            avg_2 = df2['amount'].mean() if len(df2) > 0 else 0.0

            # Identify top contributors to change
            top_contributors = self._identify_top_contributors(
                df1, df2, top_contributors_limit
            )

            self.logger.info(
                f"Period comparison {period_type.value}: "
                f"P1={total_1:.2f}, P2={total_2:.2f}, Change={percentage_change:.1f}%"
            )

            return ComparisonResult(
                period_1_total=float(total_1),
                period_2_total=float(total_2),
                absolute_change=float(absolute_change),
                percentage_change=float(percentage_change),
                period_1_count=len(df1),
                period_2_count=len(df2),
                period_1_average=float(avg_1),
                period_2_average=float(avg_2),
                top_contributors=top_contributors,
                period_type=period_type.value
            )

        except Exception as e:
            self.logger.error(f"Error comparing periods: {str(e)}")
            raise

    def _identify_top_contributors(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Identify top contributors to spending change between periods"""
        try:
            if len(df1) == 0 or len(df2) == 0:
                return []

            # Group by category (or merchant if category not available)
            group_field = 'category' if 'category' in df1.columns else 'merchant_name'

            # Aggregate by group field
            agg1 = df1.groupby(group_field)['amount'].sum() if group_field in df1.columns else pd.Series()
            agg2 = df2.groupby(group_field)['amount'].sum() if group_field in df2.columns else pd.Series()

            # Calculate changes
            changes = pd.DataFrame({
                'period_1': agg1,
                'period_2': agg2
            }).fillna(0)

            changes['absolute_change'] = changes['period_1'] - changes['period_2']
            changes['percentage_change'] = (
                (changes['period_1'] - changes['period_2']) / changes['period_2'] * 100
            ).replace([np.inf, -np.inf], 0).fillna(0)

            # Sort by absolute change (descending)
            changes = changes.sort_values('absolute_change', ascending=False)

            # Convert to list of dicts
            contributors = []
            for idx, row in changes.head(limit).iterrows():
                contributors.append({
                    'name': str(idx),
                    'period_1_amount': float(row['period_1']),
                    'period_2_amount': float(row['period_2']),
                    'absolute_change': float(row['absolute_change']),
                    'percentage_change': float(row['percentage_change'])
                })

            return contributors

        except Exception as e:
            self.logger.error(f"Error identifying top contributors: {str(e)}")
            return []

    async def calculate_trend(
        self,
        transactions: List[Dict[str, Any]],
        aggregation: str = "monthly",  # "daily", "weekly", "monthly"
        forecast_periods: int = 3
    ) -> TrendAnalysis:
        """
        Calculate trend with linear regression and forecast.

        Args:
            transactions: List of transactions
            aggregation: Time aggregation level ("daily", "weekly", "monthly")
            forecast_periods: Number of periods to forecast

        Returns:
            TrendAnalysis with regression metrics and forecast
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for trend analysis")

            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])

            # Aggregate by time period
            if aggregation == "daily":
                df['period'] = df['date'].dt.date
                freq = 'D'
            elif aggregation == "weekly":
                df['period'] = df['date'].dt.to_period('W').dt.start_time
                freq = 'W'
            else:  # monthly
                df['period'] = df['date'].dt.to_period('M').dt.start_time
                freq = 'M'

            # Group by period
            aggregated = df.groupby('period')['amount'].sum().sort_index()

            # Prepare data for regression
            X = np.arange(len(aggregated)).reshape(-1, 1)
            y = aggregated.values

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
            r_squared = r_value ** 2

            # Determine trend direction
            if abs(slope) < 0.01 * np.mean(y):  # Less than 1% of mean
                direction = TrendDirection.STABLE
            elif slope > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING

            # Forecast future periods
            last_period_idx = len(aggregated)
            forecast_X = np.arange(last_period_idx, last_period_idx + forecast_periods)
            forecast_y = slope * forecast_X + intercept

            # Generate forecast period labels
            last_date = aggregated.index[-1]
            forecast_dates = pd.date_range(
                start=last_date,
                periods=forecast_periods + 1,
                freq=freq
            )[1:]  # Skip first (duplicate of last historical)

            # Calculate 95% confidence interval
            # Using standard error of prediction
            residuals = y - (slope * X.flatten() + intercept)
            residual_std = np.std(residuals)
            margin = 1.96 * residual_std  # 95% CI

            ci_lower = (forecast_y - margin).tolist()
            ci_upper = (forecast_y + margin).tolist()

            self.logger.info(
                f"Trend analysis: direction={direction.value}, slope={slope:.2f}, "
                f"R²={r_squared:.3f}"
            )

            return TrendAnalysis(
                direction=direction,
                slope=float(slope),
                intercept=float(intercept),
                r_squared=float(r_squared),
                forecast_values=forecast_y.tolist(),
                forecast_periods=[d.strftime('%Y-%m-%d') for d in forecast_dates],
                confidence_interval_95=(ci_lower, ci_upper)
            )

        except Exception as e:
            self.logger.error(f"Error calculating trend: {str(e)}")
            raise

    async def detect_anomalies(
        self,
        transactions: List[Dict[str, Any]],
        method: AnomalyMethod = AnomalyMethod.ZSCORE,
        threshold: float = 2.0
    ) -> List[Anomaly]:
        """
        Detect anomalous transactions using statistical methods.

        Args:
            transactions: List of transactions
            method: Detection method (ZSCORE or IQR)
            threshold: Threshold for detection (Z-score >= threshold or IQR multiplier)

        Returns:
            List of detected anomalies
        """
        try:
            if not transactions:
                return []

            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            amounts = df['amount'].values

            anomalies = []

            if method == AnomalyMethod.ZSCORE:
                # Z-score method
                mean = np.mean(amounts)
                std = np.std(amounts)

                if std == 0:
                    return []  # No variation, no anomalies

                z_scores = np.abs((amounts - mean) / std)

                for idx, z_score in enumerate(z_scores):
                    if z_score >= threshold:
                        transaction = transactions[idx]
                        severity = self._calculate_severity(z_score, threshold)

                        anomalies.append(Anomaly(
                            transaction_id=str(transaction.get('id', idx)),
                            amount=float(transaction['amount']),
                            date=transaction['date'],
                            merchant_name=transaction.get('merchant_name', 'Unknown'),
                            score=float(z_score),
                            reason=f"Amount is {z_score:.1f}σ from mean (mean={mean:.2f}, std={std:.2f})",
                            severity=severity
                        ))

            elif method == AnomalyMethod.IQR:
                # IQR method
                q1 = np.percentile(amounts, 25)
                q3 = np.percentile(amounts, 75)
                iqr = q3 - q1

                if iqr == 0:
                    return []  # No variation

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                for idx, amount in enumerate(amounts):
                    if amount < lower_bound or amount > upper_bound:
                        transaction = transactions[idx]
                        distance = max(lower_bound - amount, amount - upper_bound, 0)
                        score = distance / iqr if iqr > 0 else 0

                        severity = self._calculate_severity(score, threshold)

                        reason_type = "below" if amount < lower_bound else "above"
                        anomalies.append(Anomaly(
                            transaction_id=str(transaction.get('id', idx)),
                            amount=float(transaction['amount']),
                            date=transaction['date'],
                            merchant_name=transaction.get('merchant_name', 'Unknown'),
                            score=float(score),
                            reason=f"Amount is {reason_type} IQR bounds (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})",
                            severity=severity
                        ))

            # Sort by score (descending)
            anomalies.sort(key=lambda a: a.score, reverse=True)

            self.logger.info(
                f"Detected {len(anomalies)} anomalies using {method.value} method "
                f"(threshold={threshold})"
            )

            return anomalies

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            raise

    def _calculate_severity(self, score: float, threshold: float) -> str:
        """Calculate anomaly severity based on score"""
        if score >= threshold * 2:
            return "high"
        elif score >= threshold * 1.5:
            return "medium"
        else:
            return "low"

    async def pivot_analysis(
        self,
        transactions: List[Dict[str, Any]],
        rows: str = "category",
        columns: str = "month",
        values: str = "amount",
        aggfunc: str = "sum"
    ) -> PivotTable:
        """
        Generate pivot table (multi-dimensional aggregation).

        Args:
            transactions: List of transactions
            rows: Field for rows (e.g., "category", "merchant_name")
            columns: Field for columns (e.g., "month", "week")
            values: Field to aggregate (e.g., "amount")
            aggfunc: Aggregation function ("sum", "mean", "count")

        Returns:
            PivotTable with aggregated data
        """
        try:
            if not transactions:
                raise ValueError("No transactions provided for pivot analysis")

            # Convert to DataFrame
            df = pd.DataFrame(transactions)

            # Prepare column field (temporal aggregation)
            if columns == "month":
                df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
                pivot_columns = 'month'
            elif columns == "week":
                df['week'] = pd.to_datetime(df['date']).dt.to_period('W').astype(str)
                pivot_columns = 'week'
            elif columns == "day":
                df['day'] = pd.to_datetime(df['date']).dt.date.astype(str)
                pivot_columns = 'day'
            else:
                pivot_columns = columns

            # Create pivot table
            pivot = pd.pivot_table(
                df,
                values=values,
                index=rows,
                columns=pivot_columns,
                aggfunc=aggfunc,
                fill_value=0
            )

            # Calculate totals
            row_totals = pivot.sum(axis=1)
            column_totals = pivot.sum(axis=0)
            grand_total = pivot.sum().sum()

            self.logger.info(
                f"Pivot analysis: {len(pivot)} rows × {len(pivot.columns)} columns, "
                f"grand total={grand_total:.2f}"
            )

            return PivotTable(
                data=pivot,
                row_totals=row_totals,
                column_totals=column_totals,
                grand_total=float(grand_total)
            )

        except Exception as e:
            self.logger.error(f"Error creating pivot table: {str(e)}")
            raise

    async def calculate_metrics_summary(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics summary for transactions.

        Args:
            transactions: List of transactions

        Returns:
            Dictionary with various metrics
        """
        try:
            if not transactions:
                return {
                    "count": 0,
                    "total": 0.0,
                    "average": 0.0,
                    "median": 0.0,
                    "std_dev": 0.0,
                    "min": 0.0,
                    "max": 0.0
                }

            df = pd.DataFrame(transactions)
            amounts = df['amount']

            return {
                "count": len(df),
                "total": float(amounts.sum()),
                "average": float(amounts.mean()),
                "median": float(amounts.median()),
                "std_dev": float(amounts.std()),
                "min": float(amounts.min()),
                "max": float(amounts.max()),
                "q1": float(amounts.quantile(0.25)),
                "q3": float(amounts.quantile(0.75))
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics summary: {str(e)}")
            raise
