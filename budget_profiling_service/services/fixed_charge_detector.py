"""
Service de détection automatique des charges fixes
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import select, and_
import logging
import statistics

from db_service.models.budget_profiling import FixedCharge
from budget_profiling_service.services.transaction_service import TransactionService

logger = logging.getLogger(__name__)


class FixedChargeDetector:
    """
    Détecte automatiquement les charges fixes récurrentes

    Critères de détection:
    - Récurrence mensuelle stable (±5 jours autour de la même date)
    - Montant identique ou avec faible variance (±10%)
    - Minimum 3 occurrences pour confirmer
    """

    # Catégories typiques de charges fixes
    FIXED_CHARGE_CATEGORIES = [
        'loyer',
        'eau_electricite',
        'gaz',
        'telephone_internet',
        'assurances',
        'garde_enfants',
        'pension_alimentaire',
        'credits_prets',
        'abonnements'
    ]

    def __init__(self, db_session: Session):
        self.db = db_session
        self.transaction_service = TransactionService(db_session)

    def detect_fixed_charges(
        self,
        user_id: int,
        months_back: Optional[int] = None,
        min_occurrences: int = 3,
        max_amount_variance_pct: float = 10.0,
        max_day_variance: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Détecte les charges fixes pour un utilisateur

        Args:
            user_id: ID utilisateur
            months_back: Nombre de mois à analyser (None = toutes les transactions)
            min_occurrences: Nombre minimum d'occurrences pour confirmer
            max_amount_variance_pct: Variance max du montant (%)
            max_day_variance: Variance max du jour du mois

        Returns:
            Liste de charges fixes détectées
        """
        try:
            logger.info(f"Détection charges fixes pour user {user_id}")

            # Récupérer transactions
            transactions = self.transaction_service.get_user_transactions(
                user_id,
                months_back=months_back
            )

            # Filtrer uniquement les débits
            debits = [tx for tx in transactions if tx['is_debit']]

            # Grouper par marchand
            merchant_groups = defaultdict(list)
            for tx in debits:
                merchant = tx['merchant_name']
                if merchant:
                    merchant_groups[merchant].append(tx)

            # Analyser chaque groupe
            detected_charges = []
            for merchant, txs in merchant_groups.items():
                # Besoin d'au moins min_occurrences
                if len(txs) < min_occurrences:
                    continue

                # Analyser régularité
                charge_info = self._analyze_recurrence(
                    merchant,
                    txs,
                    max_amount_variance_pct,
                    max_day_variance
                )

                if charge_info and charge_info['recurrence_confidence'] >= 0.7:
                    detected_charges.append(charge_info)

            logger.info(f"Détecté {len(detected_charges)} charges fixes")
            return detected_charges

        except Exception as e:
            logger.error(f"Erreur détection charges fixes: {e}", exc_info=True)
            return []

    def _analyze_recurrence(
        self,
        merchant: str,
        transactions: List[Dict[str, Any]],
        max_amount_variance_pct: float,
        max_day_variance: int
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse la récurrence d'un groupe de transactions
        """
        try:
            # Trier par date
            sorted_txs = sorted(transactions, key=lambda x: x['date'])

            # Calculer variance montant
            amounts = [abs(tx['amount']) for tx in sorted_txs]
            avg_amount = statistics.mean(amounts)

            if len(amounts) > 1:
                amount_variance = (statistics.stdev(amounts) / avg_amount) * 100
            else:
                amount_variance = 0.0

            # Vérifier variance montant
            if amount_variance > max_amount_variance_pct:
                return None

            # Calculer jour moyen du mois
            days_of_month = [tx['date'].day for tx in sorted_txs]
            avg_day = int(statistics.mean(days_of_month))

            # Vérifier variance jour
            if len(days_of_month) > 1:
                day_variance = statistics.stdev(days_of_month)
                if day_variance > max_day_variance:
                    return None
            else:
                day_variance = 0.0

            # Vérifier intervalle entre transactions (devrait être ~30 jours)
            intervals = []
            for i in range(1, len(sorted_txs)):
                delta = (sorted_txs[i]['date'] - sorted_txs[i-1]['date']).days
                intervals.append(delta)

            if intervals:
                avg_interval = statistics.mean(intervals)
                # Devrait être proche de 30 jours (±10 jours)
                if not (20 <= avg_interval <= 40):
                    return None
            else:
                avg_interval = 30

            # Calculer score de confiance
            confidence = self._calculate_confidence(
                len(sorted_txs),
                amount_variance,
                day_variance,
                avg_interval
            )

            # Déterminer catégorie
            category = self._infer_category(merchant, sorted_txs[0])

            return {
                'merchant_name': merchant,
                'category': category,
                'avg_amount': round(avg_amount, 2),
                'amount_variance': round(amount_variance, 2),
                'recurrence_day': avg_day,
                'recurrence_confidence': round(confidence, 2),
                'transaction_count': len(sorted_txs),
                'first_detected_date': sorted_txs[0]['date'].date(),
                'last_transaction_date': sorted_txs[-1]['date'].date()
            }

        except Exception as e:
            logger.error(f"Erreur analyse récurrence pour {merchant}: {e}")
            return None

    def _calculate_confidence(
        self,
        occurrence_count: int,
        amount_variance: float,
        day_variance: float,
        avg_interval: float
    ) -> float:
        """
        Calcule un score de confiance pour la détection

        Returns:
            Score entre 0.0 et 1.0
        """
        # Facteur occurrences (max 0.4)
        occurrence_score = min(occurrence_count / 6.0, 1.0) * 0.4

        # Facteur variance montant (max 0.3)
        amount_score = max(0, 1.0 - (amount_variance / 10.0)) * 0.3

        # Facteur variance jour (max 0.2)
        day_score = max(0, 1.0 - (day_variance / 5.0)) * 0.2

        # Facteur régularité intervalle (max 0.1)
        interval_score = max(0, 1.0 - abs(avg_interval - 30) / 10.0) * 0.1

        total_score = occurrence_score + amount_score + day_score + interval_score
        return min(max(total_score, 0.0), 1.0)

    def _infer_category(
        self,
        merchant: str,
        sample_tx: Dict[str, Any]
    ) -> str:
        """
        Infère la catégorie de charge fixe
        """
        merchant_lower = merchant.lower()

        # Mapping keywords -> catégories
        keyword_mapping = {
            'edf': 'eau_electricite',
            'enedis': 'eau_electricite',
            'engie': 'gaz',
            'veolia': 'eau_electricite',
            'orange': 'telephone_internet',
            'sfr': 'telephone_internet',
            'free': 'telephone_internet',
            'bouygues': 'telephone_internet',
            'netflix': 'abonnements',
            'spotify': 'abonnements',
            'amazon prime': 'abonnements',
            'assurance': 'assurances',
            'maif': 'assurances',
            'macif': 'assurances',
            'axa': 'assurances',
            'loyer': 'loyer',
            'bail': 'loyer',
            'crédit': 'credits_prets',
            'prêt': 'credits_prets',
        }

        for keyword, category in keyword_mapping.items():
            if keyword in merchant_lower:
                return category

        # Utiliser catégorie de la transaction si disponible
        if sample_tx.get('category'):
            return sample_tx['category']

        return 'autres_charges_fixes'

    def save_detected_charges(
        self,
        user_id: int,
        detected_charges: List[Dict[str, Any]]
    ) -> int:
        """
        Sauvegarde les charges fixes détectées en base

        Returns:
            Nombre de charges sauvegardées
        """
        try:
            saved_count = 0

            for charge_data in detected_charges:
                # Vérifier si existe déjà
                existing = self.db.execute(
                    select(FixedCharge).where(
                        and_(
                            FixedCharge.user_id == user_id,
                            FixedCharge.merchant_name == charge_data['merchant_name']
                        )
                    )
                )
                existing_charge = existing.scalar_one_or_none()

                if existing_charge:
                    # Mettre à jour
                    existing_charge.avg_amount = charge_data['avg_amount']
                    existing_charge.amount_variance = charge_data['amount_variance']
                    existing_charge.recurrence_day = charge_data['recurrence_day']
                    existing_charge.recurrence_confidence = charge_data['recurrence_confidence']
                    existing_charge.transaction_count = charge_data['transaction_count']
                    existing_charge.last_transaction_date = charge_data['last_transaction_date']
                else:
                    # Créer nouvelle
                    new_charge = FixedCharge(
                        user_id=user_id,
                        merchant_name=charge_data['merchant_name'],
                        category=charge_data['category'],
                        avg_amount=charge_data['avg_amount'],
                        amount_variance=charge_data['amount_variance'],
                        recurrence_day=charge_data['recurrence_day'],
                        recurrence_confidence=charge_data['recurrence_confidence'],
                        validated_by_user=False,
                        is_active=True,
                        first_detected_date=charge_data['first_detected_date'],
                        last_transaction_date=charge_data['last_transaction_date'],
                        transaction_count=charge_data['transaction_count']
                    )
                    self.db.add(new_charge)

                saved_count += 1

            self.db.commit()
            logger.info(f"Sauvegardé {saved_count} charges fixes pour user {user_id}")
            return saved_count

        except Exception as e:
            self.db.rollback()
            logger.error(f"Erreur sauvegarde charges fixes: {e}", exc_info=True)
            return 0

    def get_user_fixed_charges(
        self,
        user_id: int,
        active_only: bool = True
    ) -> List[FixedCharge]:
        """
        Récupère les charges fixes d'un utilisateur depuis la DB
        """
        try:
            query = select(FixedCharge).where(FixedCharge.user_id == user_id)

            if active_only:
                query = query.where(FixedCharge.is_active == True)

            result = self.db.execute(query)
            charges = result.scalars().all()

            return charges

        except Exception as e:
            logger.error(f"Erreur récupération charges fixes: {e}", exc_info=True)
            return []
