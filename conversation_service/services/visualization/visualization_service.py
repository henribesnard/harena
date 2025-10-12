"""
Visualization Service - Generate Chart.js Specifications

Automatically generates visualizations based on:
- Intent type (transaction_search, analytics, etc.)
- Search results (transaction data)
- Aggregations (pre-computed metrics)
- User preferences

Sprint 1.3 - T3.1 & T3.2
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from collections import defaultdict

from conversation_service.models.visualization.schemas import (
    VisualizationType,
    KPICard,
    ChartVisualization,
    ChartData,
    ChartDataset,
    ChartOptions,
    VisualizationResponse
)

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for generating visualization specifications"""

    def __init__(self):
        # Mapping intent → types de visualisations recommandées
        self.intent_visualization_map = {
            "transaction_search.simple": [VisualizationType.KPI_CARD],
            "transaction_search.by_category": [
                VisualizationType.KPI_CARD,
                VisualizationType.PIE_CHART
            ],
            "transaction_search.by_merchant": [
                VisualizationType.KPI_CARD,
                VisualizationType.BAR_CHART
            ],
            "analytics.comparison": [
                VisualizationType.KPI_CARD,
                VisualizationType.BAR_CHART
            ],
            "analytics.trend": [
                VisualizationType.KPI_CARD,
                VisualizationType.LINE_CHART
            ],
            "analytics.mom": [
                VisualizationType.KPI_CARD,
                VisualizationType.BAR_CHART
            ],
            "analytics.yoy": [
                VisualizationType.KPI_CARD,
                VisualizationType.LINE_CHART
            ],
        }

        # Color palettes
        self.colors = {
            "primary": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40"],
            "spending": "#FF6384",
            "income": "#4BC0C0",
            "neutral": "#36A2EB"
        }

    def generate_visualizations(
        self,
        intent_group: str,
        intent_subtype: str,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> VisualizationResponse:
        """
        Generate visualizations based on intent and data

        Args:
            intent_group: Intent group (e.g., "transaction_search")
            intent_subtype: Intent subtype (e.g., "by_category")
            search_results: Transaction data
            aggregations: Aggregated data from search
            user_preferences: User visualization preferences

        Returns:
            VisualizationResponse with specs for frontend
        """
        intent_key = f"{intent_group}.{intent_subtype}"
        viz_types = self.intent_visualization_map.get(
            intent_key,
            [VisualizationType.KPI_CARD]  # Default fallback
        )

        visualizations = []

        try:
            for viz_type in viz_types:
                if viz_type == VisualizationType.KPI_CARD:
                    kpis = self._generate_kpi_cards(
                        search_results,
                        aggregations,
                        intent_group,
                        intent_subtype
                    )
                    visualizations.extend(kpis)

                elif viz_type == VisualizationType.PIE_CHART:
                    pie = self._generate_pie_chart(
                        search_results,
                        aggregations,
                        intent_group,
                        intent_subtype
                    )
                    if pie:
                        visualizations.append(pie)

                elif viz_type == VisualizationType.BAR_CHART:
                    bar = self._generate_bar_chart(
                        search_results,
                        aggregations,
                        intent_group,
                        intent_subtype
                    )
                    if bar:
                        visualizations.append(bar)

                elif viz_type == VisualizationType.LINE_CHART:
                    line = self._generate_line_chart(
                        search_results,
                        aggregations,
                        intent_group,
                        intent_subtype
                    )
                    if line:
                        visualizations.append(line)

            logger.info(f"Generated {len(visualizations)} visualizations for {intent_key}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
            # Graceful degradation - return empty list

        return VisualizationResponse(
            visualizations=visualizations,
            intent_group=intent_group,
            intent_subtype=intent_subtype,
            metadata={"source": "VisualizationService", "version": "1.0"}
        )

    def _generate_kpi_cards(
        self,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]],
        intent_group: str,
        intent_subtype: str
    ) -> List[KPICard]:
        """
        Generate KPI cards from transaction data

        Examples:
        - Total spending
        - Transaction count
        - Average transaction amount
        - Comparison vs previous period
        """
        kpis = []

        if not search_results:
            return kpis

        # Calculate totals
        total_spending = 0
        total_income = 0

        for tx in search_results:
            amount = tx.get('amount', 0)
            tx_type = tx.get('transaction_type', '')

            if tx_type == 'debit' or amount < 0:
                total_spending += abs(amount)
            elif tx_type == 'credit' or amount > 0:
                total_income += abs(amount)

        # KPI 1: Total Spending
        kpis.append(KPICard(
            title="Total Dépenses",
            value=round(total_spending, 2),
            unit="€",
            icon="credit-card",
            color="red"
        ))

        # KPI 2: Transaction Count
        kpis.append(KPICard(
            title="Nombre de Transactions",
            value=len(search_results),
            unit="transactions",
            icon="list",
            color="blue"
        ))

        # KPI 3: Average Transaction
        if search_results:
            avg_amount = total_spending / len(search_results)
            kpis.append(KPICard(
                title="Montant Moyen",
                value=round(avg_amount, 2),
                unit="€",
                icon="trending-up",
                color="green"
            ))

        # Add comparison data if available
        if aggregations and 'comparison' in aggregations:
            comparison = aggregations['comparison']
            change_percent = comparison.get('change_percent', 0)

            if change_percent != 0:
                direction = "up" if change_percent > 0 else "down" if change_percent < 0 else "stable"

                kpis[0].change_percent = abs(round(change_percent, 1))
                kpis[0].change_direction = direction
                kpis[0].comparison_text = comparison.get('period_text', 'vs période précédente')

        return kpis

    def _generate_pie_chart(
        self,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]],
        intent_group: str,
        intent_subtype: str
    ) -> Optional[ChartVisualization]:
        """
        Generate pie chart for category breakdown

        Shows spending distribution by category
        """
        # Group by category
        category_totals = defaultdict(float)

        for tx in search_results:
            category = tx.get('category', 'Autre')
            amount = abs(tx.get('amount', 0))

            # Only count debits for spending breakdown
            if tx.get('transaction_type') == 'debit' or tx.get('amount', 0) < 0:
                category_totals[category] += amount

        if not category_totals:
            return None

        # Sort by amount descending
        sorted_categories = sorted(
            category_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top 5 categories + "Autres"
        top_categories = sorted_categories[:5]
        other_amount = sum(amount for cat, amount in sorted_categories[5:])

        labels = [cat for cat, _ in top_categories]
        values = [round(amount, 2) for _, amount in top_categories]

        if other_amount > 0:
            labels.append("Autres")
            values.append(round(other_amount, 2))

        return ChartVisualization(
            type=VisualizationType.PIE_CHART,
            title="Répartition par Catégorie",
            description=f"Top {len(top_categories)} catégories de dépenses",
            data=ChartData(
                labels=labels,
                datasets=[ChartDataset(
                    label="Montant",
                    data=values,
                    backgroundColor=self.colors["primary"][:len(values)]
                )]
            ),
            options=ChartOptions(
                plugins={
                    "legend": {"position": "right"},
                    "tooltip": {
                        "callbacks": {
                            "label": "function(context) { return context.label + ': ' + context.parsed.toFixed(2) + '€'; }"
                        }
                    }
                }
            ),
            metadata={"category_count": len(category_totals)}
        )

    def _generate_bar_chart(
        self,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]],
        intent_group: str,
        intent_subtype: str
    ) -> Optional[ChartVisualization]:
        """
        Generate bar chart for comparisons

        Examples:
        - Month-over-month comparison
        - Category comparison
        - Merchant comparison
        """
        if not aggregations or 'comparison_data' not in aggregations:
            # Generate from category data if available
            return self._generate_bar_from_categories(search_results)

        comparison_data = aggregations['comparison_data']

        labels = comparison_data.get('labels', [])
        current_values = comparison_data.get('current', [])
        previous_values = comparison_data.get('previous', [])

        if not labels or not current_values:
            return None

        datasets = [
            ChartDataset(
                label="Période Actuelle",
                data=current_values,
                backgroundColor=self.colors["neutral"],
                borderColor=self.colors["neutral"],
                borderWidth=2
            )
        ]

        if previous_values:
            datasets.append(ChartDataset(
                label="Période Précédente",
                data=previous_values,
                backgroundColor=self.colors["spending"],
                borderColor=self.colors["spending"],
                borderWidth=2
            ))

        return ChartVisualization(
            type=VisualizationType.BAR_CHART,
            title="Comparaison des Dépenses",
            description="Évolution par rapport à la période précédente",
            data=ChartData(
                labels=labels,
                datasets=datasets
            ),
            options=ChartOptions(
                scales={
                    "y": {
                        "beginAtZero": True,
                        "ticks": {
                            "callback": "function(value) { return value.toFixed(0) + '€'; }"
                        }
                    }
                },
                plugins={
                    "legend": {"position": "top"}
                }
            )
        )

    def _generate_bar_from_categories(
        self,
        search_results: List[Dict[str, Any]]
    ) -> Optional[ChartVisualization]:
        """Generate bar chart from category totals"""
        category_totals = defaultdict(float)

        for tx in search_results:
            category = tx.get('category', 'Autre')
            amount = abs(tx.get('amount', 0))

            if tx.get('transaction_type') == 'debit' or tx.get('amount', 0) < 0:
                category_totals[category] += amount

        if not category_totals:
            return None

        # Top 5 categories
        sorted_categories = sorted(
            category_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        labels = [cat for cat, _ in sorted_categories]
        values = [round(amount, 2) for _, amount in sorted_categories]

        return ChartVisualization(
            type=VisualizationType.BAR_CHART,
            title="Top Catégories",
            description="Les 5 catégories avec le plus de dépenses",
            data=ChartData(
                labels=labels,
                datasets=[ChartDataset(
                    label="Montant",
                    data=values,
                    backgroundColor=self.colors["spending"],
                    borderColor=self.colors["spending"],
                    borderWidth=2
                )]
            ),
            options=ChartOptions(
                scales={
                    "y": {
                        "beginAtZero": True,
                        "ticks": {
                            "callback": "function(value) { return value.toFixed(0) + '€'; }"
                        }
                    }
                },
                plugins={
                    "legend": {"display": False}
                }
            )
        )

    def _generate_line_chart(
        self,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]],
        intent_group: str,
        intent_subtype: str
    ) -> Optional[ChartVisualization]:
        """
        Generate line chart for trends

        Shows spending evolution over time
        """
        if not aggregations or 'trend_data' not in aggregations:
            # Try to generate from search_results
            return self._generate_line_from_transactions(search_results)

        trend_data = aggregations['trend_data']

        labels = trend_data.get('periods', [])
        values = trend_data.get('values', [])

        if not labels or not values:
            return None

        return ChartVisualization(
            type=VisualizationType.LINE_CHART,
            title="Évolution des Dépenses",
            description="Tendance sur les derniers mois",
            data=ChartData(
                labels=labels,
                datasets=[ChartDataset(
                    label="Dépenses",
                    data=values,
                    borderColor=self.colors["neutral"],
                    backgroundColor="rgba(54, 162, 235, 0.1)",
                    fill=True,
                    tension=0.4,
                    borderWidth=3
                )]
            ),
            options=ChartOptions(
                scales={
                    "y": {
                        "beginAtZero": True,
                        "ticks": {
                            "callback": "function(value) { return value.toFixed(0) + '€'; }"
                        }
                    }
                },
                plugins={
                    "legend": {"display": False}
                }
            )
        )

    def _generate_line_from_transactions(
        self,
        search_results: List[Dict[str, Any]]
    ) -> Optional[ChartVisualization]:
        """Generate line chart from transaction dates"""
        # Group by month
        monthly_totals = defaultdict(float)

        for tx in search_results:
            date_str = tx.get('date', '')
            amount = abs(tx.get('amount', 0))

            if date_str and (tx.get('transaction_type') == 'debit' or tx.get('amount', 0) < 0):
                try:
                    # Extract YYYY-MM
                    month_key = date_str[:7]  # "2025-01"
                    monthly_totals[month_key] += amount
                except Exception:
                    continue

        if not monthly_totals:
            return None

        # Sort by month
        sorted_months = sorted(monthly_totals.items())

        labels = [month for month, _ in sorted_months]
        values = [round(amount, 2) for _, amount in sorted_months]

        return ChartVisualization(
            type=VisualizationType.LINE_CHART,
            title="Évolution Mensuelle",
            description="Dépenses par mois",
            data=ChartData(
                labels=labels,
                datasets=[ChartDataset(
                    label="Dépenses",
                    data=values,
                    borderColor=self.colors["spending"],
                    backgroundColor="rgba(255, 99, 132, 0.1)",
                    fill=True,
                    tension=0.4,
                    borderWidth=3
                )]
            ),
            options=ChartOptions(
                scales={
                    "y": {
                        "beginAtZero": True,
                        "ticks": {
                            "callback": "function(value) { return value.toFixed(0) + '€'; }"
                        }
                    }
                },
                plugins={
                    "legend": {"display": False}
                }
            )
        )


__all__ = ['VisualizationService']
