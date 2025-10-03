"""
Prophet Forecaster pour les prévisions financières
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Import conditionnel de Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Prophet non installé - prévisions désactivées")
    PROPHET_AVAILABLE = False

class FinancialForecaster:
    """Prévisions financières avec Prophet"""

    def __init__(self):
        self.model: Optional[Any] = None

    def forecast_balance(
        self,
        transactions: List[Dict[str, Any]],
        current_balance: float,
        periods: int = 90
    ) -> Dict[str, Any]:
        """
        Prévision de solde avec Prophet

        Args:
            transactions: Liste des transactions historiques
            current_balance: Solde actuel
            periods: Nombre de jours à prévoir

        Returns:
            Dict avec prévisions (dates, montants, limites confiance)
        """
        if not PROPHET_AVAILABLE:
            return self._fallback_balance_forecast(transactions, current_balance, periods)

        try:
            # Préparer les données pour Prophet
            df = self._prepare_balance_data(transactions, current_balance)

            if len(df) < 10:
                logger.warning("Pas assez de données pour Prophet - fallback linéaire")
                return self._fallback_balance_forecast(transactions, current_balance, periods)

            # Créer et entraîner le modèle
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.80
            )

            model.fit(df)

            # Générer les prévisions
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            # Extraire les résultats
            last_idx = len(df) - 1
            forecast_data = forecast.iloc[last_idx:]

            return {
                "forecast_type": "prophet",
                "periods": periods,
                "current_balance": current_balance,
                "predictions": [
                    {
                        "date": row['ds'].strftime('%Y-%m-%d'),
                        "balance": float(row['yhat']),
                        "balance_lower": float(row['yhat_lower']),
                        "balance_upper": float(row['yhat_upper'])
                    }
                    for _, row in forecast_data.iterrows()
                ],
                "trend": self._calculate_trend(forecast_data),
                "confidence": "high" if len(df) > 30 else "medium"
            }

        except Exception as e:
            logger.error(f"❌ Erreur Prophet forecast: {e}")
            return self._fallback_balance_forecast(transactions, current_balance, periods)

    def forecast_expenses(
        self,
        transactions: List[Dict[str, Any]],
        periods: int = 30
    ) -> Dict[str, Any]:
        """
        Prévision des dépenses avec Prophet

        Args:
            transactions: Transactions historiques (dépenses uniquement)
            periods: Nombre de jours à prévoir

        Returns:
            Dict avec prévisions de dépenses
        """
        if not PROPHET_AVAILABLE:
            return self._fallback_expense_forecast(transactions, periods)

        try:
            # Préparer les données (somme des dépenses par jour)
            df = self._prepare_expense_data(transactions)

            if len(df) < 14:
                logger.warning("Pas assez de données pour Prophet - fallback moyenne")
                return self._fallback_expense_forecast(transactions, periods)

            # Modèle Prophet pour dépenses
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.1
            )

            model.fit(df)

            # Prévisions
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            # Extraire prévisions futures uniquement
            last_idx = len(df) - 1
            forecast_data = forecast.iloc[last_idx:]

            return {
                "forecast_type": "prophet",
                "periods": periods,
                "predictions": [
                    {
                        "date": row['ds'].strftime('%Y-%m-%d'),
                        "amount": abs(float(row['yhat'])),
                        "amount_lower": abs(float(row['yhat_lower'])),
                        "amount_upper": abs(float(row['yhat_upper']))
                    }
                    for _, row in forecast_data.iterrows()
                ],
                "total_forecast": float(forecast_data['yhat'].sum()),
                "confidence": "high" if len(df) > 30 else "medium"
            }

        except Exception as e:
            logger.error(f"❌ Erreur Prophet expense forecast: {e}")
            return self._fallback_expense_forecast(transactions, periods)

    def _prepare_balance_data(self, transactions: List[Dict], current_balance: float) -> pd.DataFrame:
        """Préparer les données de solde pour Prophet"""
        # Trier par date
        sorted_txs = sorted(transactions, key=lambda x: x['transaction_date'])

        # Calculer le solde jour par jour
        daily_balances = []
        running_balance = current_balance

        # Travailler à rebours depuis aujourd'hui
        for tx in reversed(sorted_txs):
            tx_date = tx['transaction_date']
            if isinstance(tx_date, str):
                tx_date = datetime.fromisoformat(tx_date.replace('Z', '+00:00'))

            amount = float(tx.get('amount', 0))
            running_balance -= amount

            daily_balances.append({
                'ds': tx_date,
                'y': running_balance
            })

        # Ajouter le solde actuel
        daily_balances.append({
            'ds': datetime.now(),
            'y': current_balance
        })

        df = pd.DataFrame(daily_balances)
        df = df.sort_values('ds').reset_index(drop=True)

        return df

    def _prepare_expense_data(self, transactions: List[Dict]) -> pd.DataFrame:
        """Préparer les données de dépenses pour Prophet"""
        # Filtrer les dépenses (montants négatifs)
        expenses = [tx for tx in transactions if float(tx.get('amount', 0)) < 0]

        # Grouper par jour
        daily_expenses = {}
        for tx in expenses:
            tx_date = tx['transaction_date']
            if isinstance(tx_date, str):
                tx_date = datetime.fromisoformat(tx_date.replace('Z', '+00:00'))

            date_key = tx_date.date()
            amount = abs(float(tx.get('amount', 0)))

            if date_key not in daily_expenses:
                daily_expenses[date_key] = 0
            daily_expenses[date_key] += amount

        # Convertir en DataFrame
        df = pd.DataFrame([
            {'ds': datetime.combine(date, datetime.min.time()), 'y': amount}
            for date, amount in daily_expenses.items()
        ])

        df = df.sort_values('ds').reset_index(drop=True)

        return df

    def _calculate_trend(self, forecast_data: pd.DataFrame) -> str:
        """Calculer la tendance (hausse/baisse/stable)"""
        first_val = forecast_data.iloc[0]['yhat']
        last_val = forecast_data.iloc[-1]['yhat']

        change_pct = ((last_val - first_val) / abs(first_val)) * 100

        if change_pct > 5:
            return "increasing"
        elif change_pct < -5:
            return "decreasing"
        else:
            return "stable"

    def _fallback_balance_forecast(
        self,
        transactions: List[Dict],
        current_balance: float,
        periods: int
    ) -> Dict[str, Any]:
        """Prévision simple sans Prophet (moyenne mobile)"""
        # Calculer la moyenne des variations quotidiennes
        sorted_txs = sorted(transactions[-90:], key=lambda x: x['transaction_date'])

        if not sorted_txs:
            daily_avg = 0
        else:
            total_change = sum(float(tx.get('amount', 0)) for tx in sorted_txs)
            days_span = (datetime.now() - datetime.fromisoformat(
                sorted_txs[0]['transaction_date'].replace('Z', '+00:00')
            )).days or 1
            daily_avg = total_change / days_span

        # Générer prévisions linéaires
        predictions = []
        for i in range(1, periods + 1):
            forecast_date = datetime.now() + timedelta(days=i)
            forecast_balance = current_balance + (daily_avg * i)

            predictions.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "balance": forecast_balance,
                "balance_lower": forecast_balance * 0.9,
                "balance_upper": forecast_balance * 1.1
            })

        return {
            "forecast_type": "linear",
            "periods": periods,
            "current_balance": current_balance,
            "predictions": predictions,
            "trend": "increasing" if daily_avg > 0 else "decreasing" if daily_avg < 0 else "stable",
            "confidence": "low"
        }

    def _fallback_expense_forecast(
        self,
        transactions: List[Dict],
        periods: int
    ) -> Dict[str, Any]:
        """Prévision simple des dépenses sans Prophet"""
        expenses = [abs(float(tx.get('amount', 0))) for tx in transactions if float(tx.get('amount', 0)) < 0]

        if not expenses:
            daily_avg = 0
        else:
            daily_avg = sum(expenses) / len(expenses)

        predictions = []
        for i in range(1, periods + 1):
            forecast_date = datetime.now() + timedelta(days=i)
            predictions.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "amount": daily_avg,
                "amount_lower": daily_avg * 0.8,
                "amount_upper": daily_avg * 1.2
            })

        return {
            "forecast_type": "average",
            "periods": periods,
            "predictions": predictions,
            "total_forecast": daily_avg * periods,
            "confidence": "low"
        }

# Instance globale
forecaster = FinancialForecaster()
