"""
Core Calculator - Calcule toutes les métriques financières
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from metric_service.core.forecaster import forecaster
from metric_service.models.metrics import (
    MoMMetric, YoYMetric, SavingsRateMetric, ExpenseRatioMetric,
    BurnRateMetric, RecurringExpensesMetric, RecurringExpense,
    TrendDirection
)

logger = logging.getLogger(__name__)

class MetricCalculator:
    """Calculateur de métriques financières"""

    def __init__(self):
        pass

    async def _fetch_transactions(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les transactions depuis la DB

        TODO: Remplacer par vraie requête SQL
        """
        # Import conditionnel pour éviter dépendances circulaires
        try:
            from sqlalchemy import text
            from db_service.session import get_db

            db = next(get_db())
            try:
                query = """
                    SELECT
                        rt.id,
                        rt.transaction_date,
                        rt.amount,
                        rt.description,
                        rt.merchant_name,
                        c.category_name
                    FROM raw_transactions rt
                    LEFT JOIN categories c ON rt.category_id = c.category_id
                    WHERE rt.user_id = :user_id
                """

                params = {"user_id": user_id}

                if start_date:
                    query += " AND rt.transaction_date >= :start_date"
                    params["start_date"] = start_date

                if end_date:
                    query += " AND rt.transaction_date <= :end_date"
                    params["end_date"] = end_date

                if category:
                    query += " AND c.category_name = :category"
                    params["category"] = category

                query += " ORDER BY rt.transaction_date DESC"

                result = db.execute(text(query), params)

                transactions = []
                for row in result:
                    transactions.append({
                        "id": row.id,
                        "transaction_date": row.transaction_date,
                        "amount": float(row.amount),
                        "description": row.description,
                        "merchant_name": row.merchant_name,
                        "category_name": row.category_name
                    })

                return transactions

            finally:
                db.close()

        except Exception as e:
            logger.error(f"❌ Error fetching transactions: {e}")
            return []

    async def _fetch_account_balance(self, user_id: int) -> float:
        """Récupère le solde actuel du compte"""
        try:
            from sqlalchemy import text
            from db_service.session import get_db

            db = next(get_db())
            try:
                result = db.execute(text("""
                    SELECT balance
                    FROM accounts
                    WHERE user_id = :user_id
                    LIMIT 1
                """), {"user_id": user_id})

                row = result.fetchone()
                return float(row.balance) if row else 0.0

            finally:
                db.close()

        except Exception as e:
            logger.error(f"❌ Error fetching balance: {e}")
            return 0.0

    # === TRENDS ===

    async def calculate_mom(
        self,
        user_id: int,
        month: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Month-over-Month comparison

        Args:
            user_id: ID utilisateur
            month: YYYY-MM format (default: current month)
            category: Filter by category
        """
        # Déterminer les périodes
        if month:
            current_date = datetime.strptime(month, "%Y-%m")
        else:
            current_date = datetime.now()

        current_start = current_date.replace(day=1)
        current_end = (current_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)

        previous_start = (current_start - timedelta(days=1)).replace(day=1)
        previous_end = current_start - timedelta(days=1)

        # Récupérer les transactions
        current_txs = await self._fetch_transactions(user_id, current_start, current_end, category)
        previous_txs = await self._fetch_transactions(user_id, previous_start, previous_end, category)

        # Calculer totaux
        current_amount = sum(tx['amount'] for tx in current_txs)
        previous_amount = sum(tx['amount'] for tx in previous_txs)

        change_amount = current_amount - previous_amount
        change_percent = (change_amount / abs(previous_amount) * 100) if previous_amount != 0 else 0

        # Déterminer tendance
        if change_percent > 5:
            trend = TrendDirection.INCREASING
        elif change_percent < -5:
            trend = TrendDirection.DECREASING
        else:
            trend = TrendDirection.STABLE

        return {
            "current_month": current_start.strftime("%Y-%m"),
            "previous_month": previous_start.strftime("%Y-%m"),
            "current_amount": current_amount,
            "previous_amount": previous_amount,
            "change_amount": change_amount,
            "change_percent": round(change_percent, 2),
            "trend": trend.value
        }

    async def calculate_yoy(
        self,
        user_id: int,
        year: Optional[int] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Year-over-Year comparison"""
        current_year = year or datetime.now().year
        previous_year = current_year - 1

        current_start = datetime(current_year, 1, 1)
        current_end = datetime(current_year, 12, 31)
        previous_start = datetime(previous_year, 1, 1)
        previous_end = datetime(previous_year, 12, 31)

        current_txs = await self._fetch_transactions(user_id, current_start, current_end, category)
        previous_txs = await self._fetch_transactions(user_id, previous_start, previous_end, category)

        current_amount = sum(tx['amount'] for tx in current_txs)
        previous_amount = sum(tx['amount'] for tx in previous_txs)

        change_amount = current_amount - previous_amount
        change_percent = (change_amount / abs(previous_amount) * 100) if previous_amount != 0 else 0

        if change_percent > 5:
            trend = TrendDirection.INCREASING
        elif change_percent < -5:
            trend = TrendDirection.DECREASING
        else:
            trend = TrendDirection.STABLE

        return {
            "current_year": current_year,
            "previous_year": previous_year,
            "current_amount": current_amount,
            "previous_amount": previous_amount,
            "change_amount": change_amount,
            "change_percent": round(change_percent, 2),
            "trend": trend.value
        }

    # === HEALTH ===

    async def calculate_savings_rate(
        self,
        user_id: int,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """Calcule le taux d'épargne"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        transactions = await self._fetch_transactions(user_id, start_date, end_date)

        total_income = sum(tx['amount'] for tx in transactions if tx['amount'] > 0)
        total_expenses = sum(abs(tx['amount']) for tx in transactions if tx['amount'] < 0)
        net_savings = total_income - total_expenses

        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0

        # Déterminer le statut
        if savings_rate >= 20:
            health_status = "excellent"
            recommendation = None
        elif savings_rate >= 10:
            health_status = "good"
            recommendation = "Vous êtes sur la bonne voie, essayez d'augmenter votre épargne à 20%"
        elif savings_rate >= 0:
            health_status = "fair"
            recommendation = "Votre taux d'épargne est faible, essayez de réduire vos dépenses"
        else:
            health_status = "poor"
            recommendation = "Attention: vous dépensez plus que vous ne gagnez"

        return {
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net_savings": net_savings,
            "savings_rate": round(savings_rate, 2),
            "health_status": health_status,
            "recommendation": recommendation
        }

    async def calculate_expense_ratio(
        self,
        user_id: int,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """Calcule les ratios de dépenses (50/30/20 rule)"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        transactions = await self._fetch_transactions(user_id, start_date, end_date)

        # Catégoriser les dépenses
        essentials_categories = [
            "Loyer/Charges", "Alimentation/Supermarché", "Transport/Carburant",
            "Eau/Gaz/Électricité", "Assurance/Mutuelle", "Santé/Pharmacie"
        ]

        lifestyle_categories = [
            "Restaurants/Sorties", "Loisirs/Divertissement", "Culture/Loisirs",
            "Sport/Bien-être", "Vêtements/Mode", "Achats en ligne"
        ]

        essentials = sum(
            abs(tx['amount']) for tx in transactions
            if tx['amount'] < 0 and tx.get('category_name') in essentials_categories
        )

        lifestyle = sum(
            abs(tx['amount']) for tx in transactions
            if tx['amount'] < 0 and tx.get('category_name') in lifestyle_categories
        )

        total_expenses = sum(abs(tx['amount']) for tx in transactions if tx['amount'] < 0)
        total_income = sum(tx['amount'] for tx in transactions if tx['amount'] > 0)

        savings = total_income - total_expenses

        # Calculer les pourcentages
        essentials_percent = (essentials / total_income * 100) if total_income > 0 else 0
        lifestyle_percent = (lifestyle / total_income * 100) if total_income > 0 else 0
        savings_percent = (savings / total_income * 100) if total_income > 0 else 0

        # Vérifier l'équilibre 50/30/20
        is_balanced = (
            40 <= essentials_percent <= 60 and
            20 <= lifestyle_percent <= 40 and
            savings_percent >= 15
        )

        recommendations = []
        if essentials_percent > 60:
            recommendations.append("Vos dépenses essentielles sont trop élevées (>60%)")
        if lifestyle_percent > 40:
            recommendations.append("Réduisez vos dépenses lifestyle (<30%)")
        if savings_percent < 15:
            recommendations.append("Augmentez votre épargne à au moins 20%")

        return {
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "total_expenses": total_expenses,
            "essentials": essentials,
            "essentials_percent": round(essentials_percent, 2),
            "lifestyle": lifestyle,
            "lifestyle_percent": round(lifestyle_percent, 2),
            "savings": savings,
            "savings_percent": round(savings_percent, 2),
            "is_balanced": is_balanced,
            "recommendations": recommendations
        }

    async def calculate_burn_rate(
        self,
        user_id: int,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """Calcule le burn rate et runway"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        transactions = await self._fetch_transactions(user_id, start_date, end_date)
        current_balance = await self._fetch_account_balance(user_id)

        # Calculer le burn rate mensuel
        total_expenses = sum(abs(tx['amount']) for tx in transactions if tx['amount'] < 0)
        monthly_burn_rate = (total_expenses / period_days) * 30

        # Calculer le runway
        if monthly_burn_rate > 0:
            runway_months = current_balance / monthly_burn_rate
            runway_days = int(runway_months * 30)
        else:
            runway_months = None
            runway_days = None

        # Déterminer le niveau de risque
        if runway_days is None:
            risk_level = "low"
            alert = None
        elif runway_days < 30:
            risk_level = "critical"
            alert = f"⚠️ URGENT: Il ne vous reste que {runway_days} jours de runway"
        elif runway_days < 60:
            risk_level = "high"
            alert = f"Attention: runway faible ({runway_days} jours)"
        elif runway_days < 90:
            risk_level = "medium"
            alert = None
        else:
            risk_level = "low"
            alert = None

        return {
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "current_balance": current_balance,
            "monthly_burn_rate": round(monthly_burn_rate, 2),
            "runway_days": runway_days,
            "runway_months": round(runway_months, 2) if runway_months else None,
            "risk_level": risk_level,
            "alert": alert
        }

    async def calculate_balance_forecast(
        self,
        user_id: int,
        periods: int = 90
    ) -> Dict[str, Any]:
        """Prévision de solde avec Prophet"""
        # Récupérer historique de transactions
        lookback_days = max(periods * 2, 180)  # Au moins 2x la période de prévision
        start_date = datetime.now() - timedelta(days=lookback_days)

        transactions = await self._fetch_transactions(user_id, start_date)
        current_balance = await self._fetch_account_balance(user_id)

        # Convertir en format pour forecaster
        tx_list = [
            {
                "transaction_date": tx['transaction_date'].isoformat() if isinstance(tx['transaction_date'], datetime) else tx['transaction_date'],
                "amount": tx['amount']
            }
            for tx in transactions
        ]

        # Utiliser le forecaster Prophet
        forecast = forecaster.forecast_balance(tx_list, current_balance, periods)

        return forecast

    # === PATTERNS ===

    async def calculate_recurring_expenses(
        self,
        user_id: int,
        min_occurrences: int = 3,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """Détecte les dépenses récurrentes"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        transactions = await self._fetch_transactions(user_id, start_date, end_date)

        # Filtrer les dépenses uniquement
        expenses = [tx for tx in transactions if tx['amount'] < 0]

        # Grouper par marchand
        merchant_groups = defaultdict(list)
        for tx in expenses:
            merchant = tx.get('merchant_name') or tx.get('description', 'Unknown')
            merchant_groups[merchant].append(tx)

        # Détecter les récurrences
        recurring_expenses = []

        for merchant, txs in merchant_groups.items():
            if len(txs) < min_occurrences:
                continue

            # Trier par date
            sorted_txs = sorted(txs, key=lambda x: x['transaction_date'])

            # Calculer les intervalles
            intervals = []
            for i in range(1, len(sorted_txs)):
                prev_date = sorted_txs[i-1]['transaction_date']
                curr_date = sorted_txs[i]['transaction_date']

                if isinstance(prev_date, str):
                    prev_date = datetime.fromisoformat(prev_date.replace('Z', '+00:00'))
                if isinstance(curr_date, str):
                    curr_date = datetime.fromisoformat(curr_date.replace('Z', '+00:00'))

                interval = (curr_date - prev_date).days
                intervals.append(interval)

            if not intervals:
                continue

            # Déterminer la fréquence
            avg_interval = statistics.mean(intervals)
            std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0

            # Si variance est faible, c'est récurrent
            if std_interval < avg_interval * 0.3:  # Variance < 30%
                if 6 <= avg_interval <= 8:
                    frequency = "weekly"
                elif 25 <= avg_interval <= 35:
                    frequency = "monthly"
                elif 350 <= avg_interval <= 380:
                    frequency = "yearly"
                else:
                    continue  # Pas une fréquence standard

                # Calculer montant moyen
                average_amount = abs(statistics.mean(tx['amount'] for tx in sorted_txs))

                # Dernière occurrence et prochaine prévue
                last_tx = sorted_txs[-1]
                last_date = last_tx['transaction_date']
                if isinstance(last_date, str):
                    last_date = datetime.fromisoformat(last_date.replace('Z', '+00:00'))

                next_expected = last_date + timedelta(days=int(avg_interval))

                recurring_expenses.append({
                    "merchant": merchant,
                    "category": last_tx.get('category_name'),
                    "frequency": frequency,
                    "average_amount": round(average_amount, 2),
                    "last_occurrence": last_date.strftime("%Y-%m-%d"),
                    "next_expected": next_expected.strftime("%Y-%m-%d"),
                    "confidence": 1 - (std_interval / avg_interval) if avg_interval > 0 else 0,
                    "occurrences": len(sorted_txs)
                })

        # Calculer le total mensuel récurrent
        total_monthly = sum(
            exp['average_amount']
            for exp in recurring_expenses
            if exp['frequency'] == "monthly"
        )

        # Ajouter les hebdomadaires * 4
        total_monthly += sum(
            exp['average_amount'] * 4
            for exp in recurring_expenses
            if exp['frequency'] == "weekly"
        )

        # Total des dépenses de la période
        total_expenses = sum(abs(tx['amount']) for tx in expenses)

        recurring_percent = (total_monthly / total_expenses * 100) if total_expenses > 0 else 0

        return {
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "recurring_expenses": sorted(recurring_expenses, key=lambda x: x['average_amount'], reverse=True),
            "total_monthly_recurring": round(total_monthly, 2),
            "recurring_percent_of_expenses": round(recurring_percent, 2)
        }

# Instance globale
metric_calculator = MetricCalculator()
