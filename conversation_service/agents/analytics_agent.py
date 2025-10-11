"""
Analytics Agent - Advanced Financial Metrics
Architecture v3.0 - Phase 1 Quick Wins

Responsabilité: Calculs statistiques et comparaisons temporelles avancées
- Comparaisons YoY (Year-over-Year), MoM (Month-over-Month), YTD
- Moyennes mobiles (rolling averages)
- Détection anomalies statistiques (z-score, IQR)
- Calcul trends (régression linéaire simple)
- Agrégations multi-dimensionnelles
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class ComparisonPeriod(Enum):
    """Périodes de comparaison supportées"""
    YEAR_OVER_YEAR = "yoy"
    MONTH_OVER_MONTH = "mom"
    QUARTER_OVER_QUARTER = "qoq"
    WEEK_OVER_WEEK = "wow"
    YEAR_TO_DATE = "ytd"


class AnomalyDetectionMethod(Enum):
    """Méthodes de détection d'anomalies"""
    Z_SCORE = "zscore"  # Distance en écarts-types
    IQR = "iqr"  # Interquartile Range
    ISOLATION_FOREST = "isolation_forest"  # ML-based (future)


@dataclass
class PeriodMetrics:
    """Métriques pour une période donnée"""
    period_label: str  # Ex: "2025-01", "Q1 2025"
    total_amount: float
    transaction_count: int
    average_amount: float
    median_amount: float
    min_amount: float
    max_amount: float
    std_deviation: float


@dataclass
class ComparisonResult:
    """Résultat de comparaison entre deux périodes"""
    current_period: PeriodMetrics
    previous_period: PeriodMetrics
    comparison_type: ComparisonPeriod

    # Deltas absolus
    delta_amount: float
    delta_count: int
    delta_average: float

    # Variations en %
    percentage_change_amount: float
    percentage_change_count: float
    percentage_change_average: float

    # Insights
    trend: str  # "increasing", "decreasing", "stable"
    significance: str  # "major", "moderate", "minor"


@dataclass
class Anomaly:
    """Transaction anormale détectée"""
    transaction_id: int
    amount: float
    date: str
    merchant: str
    anomaly_score: float  # Distance par rapport à la normale
    method: AnomalyDetectionMethod
    reason: str  # Explication humaine


@dataclass
class TrendAnalysis:
    """Analyse de tendance temporelle"""
    period: str  # "daily", "weekly", "monthly"
    data_points: List[Tuple[str, float]]  # (date, amount)
    trend_direction: str  # "up", "down", "flat"
    slope: float  # Pente régression linéaire
    r_squared: float  # Qualité du fit (0-1)
    forecast_next_periods: List[Tuple[str, float]]  # Prédictions simples


class AnalyticsAgent:
    """
    Agent analytique pour calculs financiers avancés

    Capacités:
    - Comparaisons temporelles (YoY, MoM, etc.)
    - Détection anomalies
    - Analyse trends
    - Statistiques descriptives
    """

    def __init__(self):
        self.stats = {
            "comparisons_performed": 0,
            "anomalies_detected": 0,
            "trends_calculated": 0
        }

        logger.info("AnalyticsAgent initialized")

    async def compare_periods(
        self,
        current_transactions: List[Dict[str, Any]],
        previous_transactions: List[Dict[str, Any]],
        comparison_type: ComparisonPeriod = ComparisonPeriod.MONTH_OVER_MONTH,
        current_label: str = "Current Period",
        previous_label: str = "Previous Period"
    ) -> ComparisonResult:
        """
        Compare deux périodes de transactions avec métriques détaillées

        Args:
            current_transactions: Transactions période actuelle
            previous_transactions: Transactions période précédente
            comparison_type: Type de comparaison (YoY, MoM, etc.)
            current_label: Label période actuelle
            previous_label: Label période précédente

        Returns:
            ComparisonResult avec deltas et variations %
        """
        try:
            # Calcul métriques période actuelle
            current_metrics = self._calculate_period_metrics(
                current_transactions, current_label
            )

            # Calcul métriques période précédente
            previous_metrics = self._calculate_period_metrics(
                previous_transactions, previous_label
            )

            # Calcul deltas absolus
            delta_amount = current_metrics.total_amount - previous_metrics.total_amount
            delta_count = current_metrics.transaction_count - previous_metrics.transaction_count
            delta_average = current_metrics.average_amount - previous_metrics.average_amount

            # Calcul variations %
            percentage_change_amount = self._safe_percentage_change(
                previous_metrics.total_amount, current_metrics.total_amount
            )
            percentage_change_count = self._safe_percentage_change(
                previous_metrics.transaction_count, current_metrics.transaction_count
            )
            percentage_change_average = self._safe_percentage_change(
                previous_metrics.average_amount, current_metrics.average_amount
            )

            # Détermination trend et significativité
            trend = self._determine_trend(percentage_change_amount)
            significance = self._determine_significance(abs(percentage_change_amount))

            self.stats["comparisons_performed"] += 1

            return ComparisonResult(
                current_period=current_metrics,
                previous_period=previous_metrics,
                comparison_type=comparison_type,
                delta_amount=delta_amount,
                delta_count=delta_count,
                delta_average=delta_average,
                percentage_change_amount=percentage_change_amount,
                percentage_change_count=percentage_change_count,
                percentage_change_average=percentage_change_average,
                trend=trend,
                significance=significance
            )

        except Exception as e:
            logger.error(f"Error comparing periods: {str(e)}")
            raise

    async def detect_anomalies(
        self,
        transactions: List[Dict[str, Any]],
        method: AnomalyDetectionMethod = AnomalyDetectionMethod.Z_SCORE,
        threshold: float = 2.0
    ) -> List[Anomaly]:
        """
        Détecte les transactions anormales

        Args:
            transactions: Liste de transactions
            method: Méthode de détection
            threshold: Seuil de détection (ex: 2.0 = 2 écarts-types)

        Returns:
            Liste d'anomalies détectées
        """
        if not transactions or len(transactions) < 3:
            return []

        try:
            anomalies = []

            if method == AnomalyDetectionMethod.Z_SCORE:
                anomalies = self._detect_anomalies_zscore(transactions, threshold)
            elif method == AnomalyDetectionMethod.IQR:
                anomalies = self._detect_anomalies_iqr(transactions)

            self.stats["anomalies_detected"] += len(anomalies)

            logger.info(f"Detected {len(anomalies)} anomalies using {method.value}")

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []

    async def calculate_trend(
        self,
        transactions: List[Dict[str, Any]],
        aggregation_period: str = "monthly",
        forecast_periods: int = 3
    ) -> TrendAnalysis:
        """
        Calcule la tendance temporelle avec régression linéaire simple

        Args:
            transactions: Liste de transactions
            aggregation_period: Période d'agrégation ("daily", "weekly", "monthly")
            forecast_periods: Nombre de périodes à prévoir

        Returns:
            TrendAnalysis avec pente et prédictions
        """
        try:
            # Agrégation par période
            aggregated_data = self._aggregate_by_period(transactions, aggregation_period)

            if len(aggregated_data) < 2:
                logger.warning("Not enough data points for trend analysis")
                return TrendAnalysis(
                    period=aggregation_period,
                    data_points=aggregated_data,
                    trend_direction="flat",
                    slope=0.0,
                    r_squared=0.0,
                    forecast_next_periods=[]
                )

            # Régression linéaire simple
            slope, r_squared = self._calculate_linear_regression(aggregated_data)

            # Direction tendance
            if slope > 0.05:  # Augmentation >5%
                trend_direction = "up"
            elif slope < -0.05:  # Diminution >5%
                trend_direction = "down"
            else:
                trend_direction = "flat"

            # Forecast simple (extrapolation linéaire)
            forecast = self._forecast_simple(aggregated_data, slope, forecast_periods)

            self.stats["trends_calculated"] += 1

            return TrendAnalysis(
                period=aggregation_period,
                data_points=aggregated_data,
                trend_direction=trend_direction,
                slope=slope,
                r_squared=r_squared,
                forecast_next_periods=forecast
            )

        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            raise

    async def calculate_rolling_average(
        self,
        transactions: List[Dict[str, Any]],
        window_days: int = 30
    ) -> List[Tuple[str, float]]:
        """
        Calcule la moyenne mobile sur N jours

        Args:
            transactions: Liste de transactions triées par date
            window_days: Fenêtre de calcul en jours

        Returns:
            Liste de (date, moyenne_mobile)
        """
        try:
            # Agrégation par jour
            daily_data = self._aggregate_by_period(transactions, "daily")

            if len(daily_data) < window_days:
                logger.warning(f"Not enough data for {window_days}-day rolling average")
                return daily_data

            rolling_averages = []

            for i in range(len(daily_data)):
                if i < window_days - 1:
                    # Pas assez de données pour cette fenêtre
                    rolling_averages.append((daily_data[i][0], daily_data[i][1]))
                else:
                    # Calculer moyenne sur la fenêtre
                    window_values = [daily_data[j][1] for j in range(i - window_days + 1, i + 1)]
                    rolling_avg = sum(window_values) / len(window_values)
                    rolling_averages.append((daily_data[i][0], rolling_avg))

            return rolling_averages

        except Exception as e:
            logger.error(f"Error calculating rolling average: {str(e)}")
            return []

    # === MÉTHODES PRIVÉES ===

    def _calculate_period_metrics(
        self,
        transactions: List[Dict[str, Any]],
        period_label: str
    ) -> PeriodMetrics:
        """Calcule les métriques statistiques pour une période"""

        if not transactions:
            return PeriodMetrics(
                period_label=period_label,
                total_amount=0.0,
                transaction_count=0,
                average_amount=0.0,
                median_amount=0.0,
                min_amount=0.0,
                max_amount=0.0,
                std_deviation=0.0
            )

        amounts = [abs(float(tx.get("amount", 0))) for tx in transactions]

        return PeriodMetrics(
            period_label=period_label,
            total_amount=sum(amounts),
            transaction_count=len(amounts),
            average_amount=statistics.mean(amounts),
            median_amount=statistics.median(amounts),
            min_amount=min(amounts),
            max_amount=max(amounts),
            std_deviation=statistics.stdev(amounts) if len(amounts) > 1 else 0.0
        )

    def _safe_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calcule variation % avec gestion division par zéro"""

        if old_value == 0:
            return 100.0 if new_value > 0 else 0.0

        return ((new_value - old_value) / abs(old_value)) * 100

    def _determine_trend(self, percentage_change: float) -> str:
        """Détermine direction du trend"""

        if percentage_change > 5:
            return "increasing"
        elif percentage_change < -5:
            return "decreasing"
        else:
            return "stable"

    def _determine_significance(self, abs_percentage: float) -> str:
        """Détermine significativité du changement"""

        if abs_percentage > 30:
            return "major"
        elif abs_percentage > 10:
            return "moderate"
        else:
            return "minor"

    def _detect_anomalies_zscore(
        self,
        transactions: List[Dict[str, Any]],
        threshold: float
    ) -> List[Anomaly]:
        """Détection par z-score (distance en écarts-types)"""

        amounts = [abs(float(tx.get("amount", 0))) for tx in transactions]

        if len(amounts) < 3:
            return []

        mean = statistics.mean(amounts)
        std = statistics.stdev(amounts)

        if std == 0:
            return []

        anomalies = []

        for i, tx in enumerate(transactions):
            amount = abs(float(tx.get("amount", 0)))
            z_score = abs((amount - mean) / std)

            if z_score > threshold:
                anomalies.append(Anomaly(
                    transaction_id=tx.get("id", i),
                    amount=amount,
                    date=tx.get("date", "N/A"),
                    merchant=tx.get("merchant_name", tx.get("merchant", "N/A")),
                    anomaly_score=z_score,
                    method=AnomalyDetectionMethod.Z_SCORE,
                    reason=f"Montant {z_score:.1f}x écart-type au-dessus de la moyenne ({mean:.2f}€)"
                ))

        return anomalies

    def _detect_anomalies_iqr(
        self,
        transactions: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Détection par IQR (Interquartile Range)"""

        amounts = [abs(float(tx.get("amount", 0))) for tx in transactions]

        if len(amounts) < 4:
            return []

        q1 = np.percentile(amounts, 25)
        q3 = np.percentile(amounts, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        anomalies = []

        for i, tx in enumerate(transactions):
            amount = abs(float(tx.get("amount", 0)))

            if amount < lower_bound or amount > upper_bound:
                distance = max(abs(amount - upper_bound), abs(amount - lower_bound))

                anomalies.append(Anomaly(
                    transaction_id=tx.get("id", i),
                    amount=amount,
                    date=tx.get("date", "N/A"),
                    merchant=tx.get("merchant_name", tx.get("merchant", "N/A")),
                    anomaly_score=distance,
                    method=AnomalyDetectionMethod.IQR,
                    reason=f"Montant hors plage normale (Q1-Q3: {q1:.2f}€ - {q3:.2f}€)"
                ))

        return anomalies

    def _aggregate_by_period(
        self,
        transactions: List[Dict[str, Any]],
        period: str
    ) -> List[Tuple[str, float]]:
        """Agrège transactions par période"""

        period_totals = defaultdict(float)

        for tx in transactions:
            date_str = tx.get("date", "")
            amount = abs(float(tx.get("amount", 0)))

            if not date_str:
                continue

            try:
                date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

                if period == "daily":
                    key = date_obj.strftime("%Y-%m-%d")
                elif period == "weekly":
                    key = date_obj.strftime("%Y-W%W")
                elif period == "monthly":
                    key = date_obj.strftime("%Y-%m")
                else:
                    key = date_obj.strftime("%Y-%m")

                period_totals[key] += amount

            except Exception as e:
                logger.warning(f"Error parsing date {date_str}: {e}")
                continue

        # Trier par date
        sorted_data = sorted(period_totals.items(), key=lambda x: x[0])

        return sorted_data

    def _calculate_linear_regression(
        self,
        data_points: List[Tuple[str, float]]
    ) -> Tuple[float, float]:
        """Calcule régression linéaire simple (pente et R²)"""

        if len(data_points) < 2:
            return 0.0, 0.0

        # Conversion en indices numériques
        x = np.arange(len(data_points))
        y = np.array([point[1] for point in data_points])

        # Régression linéaire
        z = np.polyfit(x, y, 1)
        slope = z[0]

        # Calcul R² (qualité du fit)
        y_pred = np.polyval(z, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return float(slope), float(r_squared)

    def _forecast_simple(
        self,
        historical_data: List[Tuple[str, float]],
        slope: float,
        periods: int
    ) -> List[Tuple[str, float]]:
        """Forecast simple par extrapolation linéaire"""

        if not historical_data:
            return []

        last_date_str, last_value = historical_data[-1]

        forecast = []

        for i in range(1, periods + 1):
            # Extrapolation
            predicted_value = last_value + (slope * i)
            predicted_value = max(0, predicted_value)  # Pas de valeurs négatives

            # Date prédite (approximation simple)
            try:
                last_date = datetime.fromisoformat(last_date_str)
                predicted_date = last_date + timedelta(days=30 * i)  # Approximation mensuelle
                predicted_date_str = predicted_date.strftime("%Y-%m")
            except:
                predicted_date_str = f"Period +{i}"

            forecast.append((predicted_date_str, predicted_value))

        return forecast

    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de l'agent"""
        return self.stats
