"""
Analytics Agent - Module pour calculs statistiques avancés

Ce module fournit des capacités analytiques avancées pour l'analyse
des transactions financières :
- Comparaisons temporelles (MoM, YoY, QoQ)
- Détection d'anomalies (Z-score, IQR, Isolation Forest)
- Analyse de tendances (régression linéaire + forecast)

Auteur: Harena Team
Version: 3.3.0-analytics-agent
Date: 2025-01-12
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ============================================
# MODÈLES DE DONNÉES
# ============================================

class TimeSeriesMetrics(BaseModel):
    """
    Métriques pour comparaisons temporelles (MoM, YoY, QoQ)

    Exemple d'utilisation:
        result = await analytics_agent.compare_periods(
            transactions_current=january_txs,
            transactions_previous=december_txs,
            metric='sum'
        )
        print(f"Variation MoM: {result.delta_percentage:+.1f}%")
    """
    period_current: str = Field(description="Label période actuelle (ex: '2025-01')")
    period_previous: str = Field(description="Label période précédente (ex: '2024-12')")
    value_current: float = Field(description="Valeur métrique période actuelle")
    value_previous: float = Field(description="Valeur métrique période précédente")
    delta: float = Field(description="Différence absolue (current - previous)")
    delta_percentage: float = Field(description="Variation en pourcentage")
    trend: str = Field(description="Direction tendance: 'up', 'down', 'stable'")

    @field_validator('trend')
    @classmethod
    def validate_trend(cls, v):
        if v not in ['up', 'down', 'stable']:
            raise ValueError("Trend must be 'up', 'down', or 'stable'")
        return v


class AnomalyDetectionResult(BaseModel):
    """
    Résultat détection anomalie pour une transaction

    Exemple d'utilisation:
        anomalies = await analytics_agent.detect_anomalies(
            transactions=january_txs,
            method='zscore',
            threshold=2.0
        )
        for anomaly in anomalies:
            print(f"Transaction anormale: {anomaly.merchant_name} - {anomaly.amount}€")
            print(f"Raison: {anomaly.explanation}")
    """
    transaction_id: int = Field(description="ID transaction anormale")
    amount: float = Field(description="Montant transaction")
    date: datetime = Field(description="Date transaction")
    merchant_name: str = Field(description="Nom du marchand")
    anomaly_score: float = Field(description="Score anomalie (>2 = anormal)")
    method: str = Field(description="Méthode détection: 'zscore', 'iqr', 'isolation_forest'")
    threshold_exceeded: bool = Field(description="Seuil dépassé ?")
    explanation: str = Field(description="Explication humaine de l'anomalie")


class TrendAnalysis(BaseModel):
    """
    Analyse de tendance avec régression linéaire et forecast

    Exemple d'utilisation:
        trend = await analytics_agent.calculate_trend(
            transactions=last_6_months_txs,
            aggregation='monthly',
            forecast_periods=3
        )
        print(f"Tendance: {trend.trend_direction}")
        print(f"Forecast 3 mois: {trend.forecast_next_periods}")
    """
    period: str = Field(description="Granularité: 'daily', 'weekly', 'monthly'")
    trend_direction: str = Field(description="Direction: 'increasing', 'decreasing', 'stable'")
    slope: float = Field(description="Pente régression linéaire (€/période)")
    r_squared: float = Field(description="Coefficient détermination R² (0-1, qualité fit)")
    forecast_next_periods: List[float] = Field(description="Prédictions N périodes suivantes")
    confidence_interval_95: List[Tuple[float, float]] = Field(description="Intervalles confiance 95%")


# ============================================
# ANALYTICS AGENT
# ============================================

class AnalyticsAgent:
    """
    Agent spécialisé pour calculs analytiques avancés sur transactions financières.

    Capabilities:
    - Comparaisons temporelles (MoM, YoY, QoQ)
    - Détection anomalies (Z-score, IQR, Isolation Forest)
    - Calcul tendances (régression linéaire + forecast)
    - Agrégations multi-dimensionnelles

    Usage:
        agent = AnalyticsAgent()

        # Comparaison MoM
        mom_result = await agent.compare_periods(
            transactions_current=january_txs,
            transactions_previous=december_txs,
            metric='sum'
        )

        # Détection anomalies
        anomalies = await agent.detect_anomalies(
            transactions=january_txs,
            method='zscore'
        )

        # Calcul tendance
        trend = await agent.calculate_trend(
            transactions=last_6_months_txs,
            aggregation='monthly',
            forecast_periods=3
        )
    """

    def __init__(
        self,
        zscore_threshold: float = 2.0,
        iqr_multiplier: float = 1.5,
        stable_threshold_pct: float = 5.0
    ):
        """
        Initialize Analytics Agent

        Args:
            zscore_threshold: Seuil Z-score pour anomalies (défaut: 2.0)
            iqr_multiplier: Multiplicateur IQR (défaut: 1.5)
            stable_threshold_pct: % variation considéré stable (défaut: 5%)
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.stable_threshold_pct = stable_threshold_pct

        logger.info(
            f"AnalyticsAgent initialized (zscore={zscore_threshold}, "
            f"iqr={iqr_multiplier}, stable={stable_threshold_pct}%)"
        )

    # ============================================
    # COMPARAISONS TEMPORELLES
    # ============================================

    async def compare_periods(
        self,
        transactions_current: List[Dict[str, Any]],
        transactions_previous: List[Dict[str, Any]],
        metric: str = "sum"
    ) -> TimeSeriesMetrics:
        """
        Compare deux périodes avec calculs delta et variation %

        Args:
            transactions_current: Liste transactions période actuelle
            transactions_previous: Liste transactions période précédente
            metric: Métrique à calculer ('sum', 'avg', 'count', 'median')

        Returns:
            TimeSeriesMetrics avec comparaison détaillée

        Raises:
            ValueError: Si metric invalide ou données insuffisantes

        Exemple:
            >>> january_txs = [{'amount': 120, 'date': '2025-01-05'}, ...]
            >>> december_txs = [{'amount': 110, 'date': '2024-12-05'}, ...]
            >>> result = await agent.compare_periods(january_txs, december_txs, 'sum')
            >>> print(f"MoM: {result.delta_percentage:+.1f}%")
            MoM: +12.5%
        """
        try:
            # Validation données
            if not transactions_current or not transactions_previous:
                raise ValueError("Transactions lists cannot be empty")

            # Conversion en DataFrames
            df_current = pd.DataFrame(transactions_current)
            df_previous = pd.DataFrame(transactions_previous)

            # Validation colonnes requises
            if 'amount' not in df_current.columns or 'amount' not in df_previous.columns:
                raise ValueError("'amount' column required in transactions")

            # Calcul métrique selon type
            metric_funcs = {
                'sum': lambda df: df['amount'].sum(),
                'avg': lambda df: df['amount'].mean(),
                'count': lambda df: len(df),
                'median': lambda df: df['amount'].median()
            }

            if metric not in metric_funcs:
                raise ValueError(f"Invalid metric: {metric}. Must be one of {list(metric_funcs.keys())}")

            value_current = float(metric_funcs[metric](df_current))
            value_previous = float(metric_funcs[metric](df_previous))

            # Calcul delta et %
            delta = value_current - value_previous
            delta_pct = (delta / value_previous * 100) if value_previous != 0 else 0

            # Détermination tendance
            if abs(delta_pct) < self.stable_threshold_pct:
                trend = 'stable'
            elif delta > 0:
                trend = 'up'
            else:
                trend = 'down'

            # Extraction labels périodes
            period_current = self._extract_period_label(df_current)
            period_previous = self._extract_period_label(df_previous)

            logger.info(
                f"Period comparison completed: {period_previous} → {period_current} "
                f"({delta_pct:+.1f}%, {trend})"
            )

            return TimeSeriesMetrics(
                period_current=period_current,
                period_previous=period_previous,
                value_current=value_current,
                value_previous=value_previous,
                delta=delta,
                delta_percentage=delta_pct,
                trend=trend
            )

        except Exception as e:
            logger.error(f"Error in compare_periods: {str(e)}")
            raise

    # ============================================
    # DÉTECTION ANOMALIES
    # ============================================

    async def detect_anomalies(
        self,
        transactions: List[Dict[str, Any]],
        method: str = "zscore",
        threshold: Optional[float] = None
    ) -> List[AnomalyDetectionResult]:
        """
        Détecte transactions anormales selon méthode statistique

        Args:
            transactions: Liste transactions à analyser
            method: Méthode détection ('zscore', 'iqr', 'isolation_forest')
            threshold: Seuil personnalisé (None = utilise config par défaut)

        Returns:
            Liste anomalies détectées, triée par score décroissant

        Exemple:
            >>> txs = [
            ...     {'id': 1, 'amount': 120, 'date': '2025-01-05', 'merchant_name': 'Amazon'},
            ...     {'id': 2, 'amount': 1200, 'date': '2025-01-15', 'merchant_name': 'Tesla'},
            ... ]
            >>> anomalies = await agent.detect_anomalies(txs, method='zscore')
            >>> print(f"Found {len(anomalies)} anomalies")
            >>> for a in anomalies:
            ...     print(f"- {a.merchant_name}: {a.amount}€ (score: {a.anomaly_score:.2f})")
            Found 1 anomalies
            - Tesla: 1200.0€ (score: 3.45)
        """
        try:
            if not transactions:
                logger.warning("detect_anomalies called with empty transactions list")
                return []

            df = pd.DataFrame(transactions)

            # Validation colonnes requises
            required_cols = ['id', 'amount', 'date']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' missing in transactions")

            # Détection selon méthode
            if method == "zscore":
                anomalies = self._detect_zscore_anomalies(df, threshold)
            elif method == "iqr":
                anomalies = self._detect_iqr_anomalies(df)
            elif method == "isolation_forest":
                anomalies = self._detect_ml_anomalies(df)
            else:
                raise ValueError(f"Invalid method: {method}. Must be 'zscore', 'iqr', or 'isolation_forest'")

            # Tri par score décroissant
            anomalies.sort(key=lambda a: a.anomaly_score, reverse=True)

            logger.info(f"Anomaly detection completed: {len(anomalies)} anomalies found (method={method})")

            return anomalies

        except Exception as e:
            logger.error(f"Error in detect_anomalies: {str(e)}")
            raise

    def _detect_zscore_anomalies(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[AnomalyDetectionResult]:
        """
        Détection anomalies via Z-Score (écarts-types de la moyenne)

        Méthode:
        - Calcule moyenne μ et écart-type σ des montants
        - Z-score = (montant - μ) / σ
        - Si |Z-score| > threshold → anomalie

        Args:
            df: DataFrame transactions
            threshold: Seuil Z-score (None = utilise self.zscore_threshold)

        Returns:
            Liste anomalies détectées
        """
        threshold = threshold or self.zscore_threshold

        mean = df['amount'].mean()
        std = df['amount'].std()

        if std == 0:
            logger.warning("Standard deviation is 0, no anomalies can be detected")
            return []

        # Calcul Z-scores
        df['zscore'] = (df['amount'] - mean) / std

        # Sélection anomalies
        anomalies_df = df[abs(df['zscore']) > threshold]

        results = []
        for _, row in anomalies_df.iterrows():
            results.append(AnomalyDetectionResult(
                transaction_id=int(row['id']),
                amount=float(row['amount']),
                date=pd.to_datetime(row['date']),
                merchant_name=str(row.get('merchant_name', 'Unknown')),
                anomaly_score=float(abs(row['zscore'])),
                method='zscore',
                threshold_exceeded=True,
                explanation=f"Montant {abs(row['zscore']):.1f}σ de la moyenne ({mean:.2f}€, σ={std:.2f}€)"
            ))

        return results

    def _detect_iqr_anomalies(self, df: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Détection anomalies via IQR (Interquartile Range)

        Méthode:
        - Calcule Q1 (25e percentile) et Q3 (75e percentile)
        - IQR = Q3 - Q1
        - Bornes: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        - Valeurs hors bornes → anomalies

        Args:
            df: DataFrame transactions

        Returns:
            Liste anomalies détectées
        """
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            logger.warning("IQR is 0, no anomalies can be detected")
            return []

        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR

        # Sélection anomalies (hors bornes)
        anomalies_df = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]

        results = []
        for _, row in anomalies_df.iterrows():
            # Score normalisé (distance à la borne / IQR)
            if row['amount'] < lower_bound:
                score = (lower_bound - row['amount']) / IQR
            else:
                score = (row['amount'] - upper_bound) / IQR

            results.append(AnomalyDetectionResult(
                transaction_id=int(row['id']),
                amount=float(row['amount']),
                date=pd.to_datetime(row['date']),
                merchant_name=str(row.get('merchant_name', 'Unknown')),
                anomaly_score=float(score),
                method='iqr',
                threshold_exceeded=True,
                explanation=f"Montant hors intervalle IQR [{lower_bound:.2f}€, {upper_bound:.2f}€] (IQR={IQR:.2f}€)"
            ))

        return results

    def _detect_ml_anomalies(self, df: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Détection anomalies via Isolation Forest (ML)

        Note: Implémentation simplifiée pour Phase 1.
        Phase 3 aura modèle ML complet avec features multi-dimensionnelles.

        Args:
            df: DataFrame transactions

        Returns:
            Liste anomalies détectées
        """
        from sklearn.ensemble import IsolationForest

        if len(df) < 10:
            logger.warning("Not enough data for Isolation Forest (<10 samples)")
            return []

        # Feature engineering simple (Phase 1)
        X = df[['amount']].values

        # Entraînement Isolation Forest
        clf = IsolationForest(contamination=0.1, random_state=42)
        predictions = clf.fit_predict(X)
        scores = clf.score_samples(X)

        # Sélection anomalies (prediction = -1)
        df['prediction'] = predictions
        df['anomaly_score'] = -scores  # Inverser pour que score élevé = anormal

        anomalies_df = df[df['prediction'] == -1]

        results = []
        for _, row in anomalies_df.iterrows():
            results.append(AnomalyDetectionResult(
                transaction_id=int(row['id']),
                amount=float(row['amount']),
                date=pd.to_datetime(row['date']),
                merchant_name=str(row.get('merchant_name', 'Unknown')),
                anomaly_score=float(row['anomaly_score']),
                method='isolation_forest',
                threshold_exceeded=True,
                explanation=f"Transaction isolée détectée par ML (score: {row['anomaly_score']:.2f})"
            ))

        return results

    # ============================================
    # CALCUL TENDANCES
    # ============================================

    async def calculate_trend(
        self,
        transactions: List[Dict[str, Any]],
        aggregation: str = "monthly",
        forecast_periods: int = 3
    ) -> TrendAnalysis:
        """
        Calcule tendance avec régression linéaire et forecast

        Args:
            transactions: Liste transactions à analyser
            aggregation: Granularité temporelle ('daily', 'weekly', 'monthly')
            forecast_periods: Nombre périodes à prédire

        Returns:
            TrendAnalysis avec régression, forecast et intervalle confiance

        Exemple:
            >>> last_6_months = [
            ...     {'amount': 400, 'date': '2024-08-01'},
            ...     {'amount': 420, 'date': '2024-09-01'},
            ...     {'amount': 450, 'date': '2024-10-01'},
            ...     {'amount': 480, 'date': '2024-11-01'},
            ...     {'amount': 500, 'date': '2024-12-01'},
            ...     {'amount': 530, 'date': '2025-01-01'},
            ... ]
            >>> trend = await agent.calculate_trend(last_6_months, 'monthly', 3)
            >>> print(f"Tendance: {trend.trend_direction} (R²={trend.r_squared:.2f})")
            >>> print(f"Forecast 3 mois: {trend.forecast_next_periods}")
            Tendance: increasing (R²=0.98)
            Forecast 3 mois: [560.5, 590.2, 620.0]
        """
        try:
            if not transactions:
                raise ValueError("Transactions list cannot be empty")

            df = pd.DataFrame(transactions)

            # Validation colonnes
            if 'amount' not in df.columns or 'date' not in df.columns:
                raise ValueError("'amount' and 'date' columns required")

            # Conversion dates
            df['date'] = pd.to_datetime(df['date'])

            # Agrégation temporelle
            if aggregation == 'daily':
                df_agg = df.groupby(df['date'].dt.date)['amount'].sum()
            elif aggregation == 'weekly':
                df_agg = df.groupby(df['date'].dt.to_period('W'))['amount'].sum()
            elif aggregation == 'monthly':
                df_agg = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
            else:
                raise ValueError(f"Invalid aggregation: {aggregation}. Must be 'daily', 'weekly', or 'monthly'")

            if len(df_agg) < 3:
                raise ValueError("Need at least 3 data points for trend analysis")

            # Régression linéaire
            x = np.arange(len(df_agg))
            y = df_agg.values

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Forecast
            forecast_x = np.arange(len(df_agg), len(df_agg) + forecast_periods)
            forecast_y = slope * forecast_x + intercept

            # Intervalle confiance 95% (approximation via std_err)
            confidence_interval = [
                (float(pred - 1.96 * std_err * np.sqrt(1 + 1/len(x))),
                 float(pred + 1.96 * std_err * np.sqrt(1 + 1/len(x))))
                for pred in forecast_y
            ]

            # Direction tendance
            if abs(slope) < 0.1:  # Seuil arbitraire pour stabilité
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'

            logger.info(
                f"Trend analysis completed: {trend_direction} "
                f"(slope={slope:.2f}, R²={r_value**2:.2f})"
            )

            return TrendAnalysis(
                period=aggregation,
                trend_direction=trend_direction,
                slope=float(slope),
                r_squared=float(r_value**2),
                forecast_next_periods=forecast_y.tolist(),
                confidence_interval_95=confidence_interval
            )

        except Exception as e:
            logger.error(f"Error in calculate_trend: {str(e)}")
            raise

    # ============================================
    # UTILITAIRES
    # ============================================

    def _extract_period_label(self, df: pd.DataFrame) -> str:
        """
        Extrait label période lisible depuis DataFrame

        Args:
            df: DataFrame avec colonne 'date'

        Returns:
            Label période (ex: "2025-01-01 to 2025-01-31")
        """
        if df.empty or 'date' not in df.columns:
            return "N/A"

        dates = pd.to_datetime(df['date'])
        period_start = dates.min().strftime('%Y-%m-%d')
        period_end = dates.max().strftime('%Y-%m-%d')

        return f"{period_start} to {period_end}"
