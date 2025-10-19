"""
Service de récupération et préparation des transactions pour analyse
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
import logging

from db_service.models.sync import RawTransaction, Category

logger = logging.getLogger(__name__)


class TransactionService:
    """
    Service de récupération et formatage des transactions
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self._category_cache = {}

    def get_user_transactions(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        months_back: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les transactions d'un utilisateur sur une période

        Args:
            user_id: ID utilisateur
            start_date: Date début (optionnel)
            end_date: Date fin (optionnel)
            months_back: Nombre de mois en arrière (None = toutes les transactions)

        Returns:
            Liste de transactions formatées pour analyse
        """
        try:
            if not end_date:
                end_date = datetime.now()

            if not start_date:
                if months_back is None:
                    # Récupérer TOUTES les transactions - mettre une date très ancienne
                    start_date = datetime(2000, 1, 1)
                else:
                    start_date = datetime.now() - timedelta(days=30 * months_back)

            logger.info(f"Récupération transactions user {user_id} de {start_date} à {end_date}")

            query = select(RawTransaction).where(
                and_(
                    RawTransaction.user_id == user_id,
                    RawTransaction.date >= start_date,
                    RawTransaction.date <= end_date,
                    RawTransaction.deleted == False
                )
            ).order_by(RawTransaction.date.desc())

            result = self.db.execute(query)
            transactions = result.scalars().all()

            logger.info(f"Trouvé {len(transactions)} transactions")

            # Formater pour analyse
            formatted = []
            for tx in transactions:
                formatted.append({
                    'id': tx.bridge_transaction_id,
                    'user_id': tx.user_id,
                    'merchant_name': tx.merchant_name or tx.clean_description,
                    'amount': float(tx.amount) if tx.amount else 0.0,
                    'date': tx.date,
                    'category': self._get_category_name(tx.category_id),
                    'operation_type': tx.operation_type,
                    'is_debit': tx.amount < 0 if tx.amount else False,
                    'is_credit': tx.amount > 0 if tx.amount else False
                })

            return formatted

        except Exception as e:
            logger.error(f"Erreur récupération transactions: {e}", exc_info=True)
            return []

    def _get_category_name(self, category_id: Optional[int]) -> str:
        """
        Récupère nom catégorie depuis category_id avec cache
        """
        if not category_id:
            return 'uncategorized'

        # Vérifier cache
        if category_id in self._category_cache:
            return self._category_cache[category_id]

        # Récupérer depuis DB
        try:
            result = self.db.execute(
                select(Category).where(Category.category_id == category_id)
            )
            category = result.scalar_one_or_none()

            if category:
                self._category_cache[category_id] = category.category_name
                return category.category_name
            else:
                return 'uncategorized'

        except Exception as e:
            logger.error(f"Erreur récupération catégorie {category_id}: {e}")
            return 'uncategorized'

    def get_monthly_aggregates(
        self,
        user_id: int,
        months: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Agrégations mensuelles (revenus, dépenses)

        Args:
            user_id: ID utilisateur
            months: Nombre de mois (None = toutes les transactions)

        Returns:
            [
                {
                    'month': '2025-01',
                    'total_income': 3000.0,
                    'total_expenses': 2400.0,
                    'net_cashflow': 600.0,
                    'transaction_count': 145
                },
                ...
            ]
        """
        try:
            transactions = self.get_user_transactions(
                user_id,
                months_back=months
            )

            # Grouper par mois
            monthly_data = {}
            for tx in transactions:
                month_key = tx['date'].strftime('%Y-%m')

                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        'month': month_key,
                        'total_income': 0.0,
                        'total_expenses': 0.0,
                        'transaction_count': 0
                    }

                if tx['is_credit']:
                    monthly_data[month_key]['total_income'] += tx['amount']
                elif tx['is_debit']:
                    monthly_data[month_key]['total_expenses'] += abs(tx['amount'])

                monthly_data[month_key]['transaction_count'] += 1

            # Calculer net cashflow et trier
            result = []
            for month_key in sorted(monthly_data.keys()):
                data = monthly_data[month_key]
                data['net_cashflow'] = data['total_income'] - data['total_expenses']
                result.append(data)

            return result

        except Exception as e:
            logger.error(f"Erreur calcul agrégats mensuels: {e}", exc_info=True)
            return []

    def get_category_breakdown(
        self,
        user_id: int,
        months: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Répartition des dépenses par catégorie (MOYENNES MENSUELLES)

        Returns:
            {
                'alimentation': 450.0,  # Moyenne mensuelle
                'transport': 120.0,
                'loisirs': 230.0,
                ...
            }
        """
        try:
            transactions = self.get_user_transactions(
                user_id,
                months_back=months
            )

            if not transactions:
                return {}

            # Grouper transactions par mois pour compter les mois TOTAUX (avec n'importe quelle transaction)
            all_months = set()
            for tx in transactions:
                month_key = f"{tx['date'].year}-{tx['date'].month:02d}"
                all_months.add(month_key)

            nb_months = len(all_months)

            if nb_months == 0:
                return {}

            # Grouper par catégorie (seulement dépenses)
            category_totals = {}
            for tx in transactions:
                if tx['is_debit']:
                    category = tx['category']
                    if category not in category_totals:
                        category_totals[category] = 0.0
                    category_totals[category] += abs(tx['amount'])

            # Diviser par le nombre de mois RÉELS (tous mois avec transactions)
            category_averages = {
                cat: total / nb_months
                for cat, total in category_totals.items()
            }

            return category_averages

        except Exception as e:
            logger.error(f"Erreur calcul breakdown catégories: {e}", exc_info=True)
            return {}

    def get_merchant_transactions(
        self,
        user_id: int,
        merchant_name: str,
        months_back: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Récupère toutes les transactions pour un marchand spécifique
        Utile pour la détection de charges fixes
        """
        try:
            transactions = self.get_user_transactions(
                user_id,
                months_back=months_back
            )

            # Filtrer par merchant_name
            merchant_txs = [
                tx for tx in transactions
                if tx['merchant_name'] and merchant_name.lower() in tx['merchant_name'].lower()
            ]

            return merchant_txs

        except Exception as e:
            logger.error(f"Erreur récupération transactions marchand: {e}", exc_info=True)
            return []
