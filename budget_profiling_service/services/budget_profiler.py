"""
Service de calcul du profil budgétaire utilisateur
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import select
from decimal import Decimal
import logging

from db_service.models.budget_profiling import UserBudgetProfile
from budget_profiling_service.services.transaction_service import TransactionService
from budget_profiling_service.services.fixed_charge_detector import FixedChargeDetector

logger = logging.getLogger(__name__)


class BudgetProfiler:
    """
    Calcule et maintient le profil budgétaire d'un utilisateur
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self.transaction_service = TransactionService(db_session)
        self.fixed_charge_detector = FixedChargeDetector(db_session)

    def calculate_user_profile(
        self,
        user_id: int,
        months_analysis: int = 3
    ) -> Dict[str, Any]:
        """
        Calcule le profil budgétaire complet d'un utilisateur

        Args:
            user_id: ID utilisateur
            months_analysis: Nombre de mois à analyser (défaut 3)

        Returns:
            Profil budgétaire complet
        """
        try:
            logger.info(f"Calcul profil budgétaire pour user {user_id}")

            # 1. Récupérer agrégats mensuels
            monthly_aggregates = self.transaction_service.get_monthly_aggregates(
                user_id,
                months=months_analysis
            )

            if not monthly_aggregates:
                logger.warning(f"Pas de données pour user {user_id}")
                return self._empty_profile()

            # 2. Calculer moyennes
            avg_income = sum(m['total_income'] for m in monthly_aggregates) / len(monthly_aggregates)
            avg_expenses = sum(m['total_expenses'] for m in monthly_aggregates) / len(monthly_aggregates)
            avg_savings = avg_income - avg_expenses

            # 3. Taux d'épargne (avec gestion robuste)
            savings_rate = self._calculate_savings_rate(avg_savings, avg_income)

            # 4. Récupérer charges fixes
            fixed_charges = self.fixed_charge_detector.get_user_fixed_charges(user_id)
            fixed_charges_total = sum(float(charge.avg_amount) for charge in fixed_charges)

            # 5. Récupérer breakdown par catégorie
            category_breakdown = self.transaction_service.get_category_breakdown(
                user_id,
                months=months_analysis
            )

            # 6. Calculer charges variables et semi-fixes
            semi_fixed_total, variable_total, structural_fixed_total = self._categorize_expenses(
                category_breakdown,
                fixed_charges_total
            )

            # 6b. Ajouter charges fixes structurelles aux charges fixes détectées
            total_fixed_charges = fixed_charges_total + structural_fixed_total

            # 7. Reste à vivre
            remaining_to_live = avg_income - total_fixed_charges

            # 8. Déterminer segment utilisateur
            user_segment = self._determine_segment(avg_income, avg_expenses)

            # 9. Déterminer pattern comportemental
            behavioral_pattern = self._determine_behavioral_pattern(user_id)

            # 10. Calculer complétude profil
            profile_completeness = self._calculate_completeness(
                monthly_aggregates,
                fixed_charges,
                months_analysis
            )

            profile = {
                'user_id': user_id,
                'user_segment': user_segment,
                'behavioral_pattern': behavioral_pattern,
                'avg_monthly_income': round(avg_income, 2),
                'avg_monthly_expenses': round(avg_expenses, 2),
                'avg_monthly_savings': round(avg_savings, 2),
                'savings_rate': round(savings_rate, 2),
                'fixed_charges_total': round(total_fixed_charges, 2),
                'semi_fixed_charges_total': round(semi_fixed_total, 2),
                'variable_charges_total': round(variable_total, 2),
                'remaining_to_live': round(remaining_to_live, 2),
                'profile_completeness': round(profile_completeness, 2),
                'last_analyzed_at': datetime.now(timezone.utc)
            }

            logger.info(f"Profil calculé: segment={user_segment}, savings_rate={savings_rate:.1f}%")
            return profile

        except Exception as e:
            logger.error(f"Erreur calcul profil: {e}", exc_info=True)
            return self._empty_profile()

    def _calculate_savings_rate(self, avg_savings: float, avg_income: float) -> float:
        """
        Calcule le taux d'épargne avec gestion robuste des cas limites

        Args:
            avg_savings: Épargne moyenne mensuelle (peut être négative)
            avg_income: Revenus moyens mensuels

        Returns:
            Taux d'épargne en pourcentage (-100 à +100)
        """
        if avg_income <= 0:
            logger.warning(f"Revenus nuls ou négatifs ({avg_income}), taux d'épargne indéterminé")
            return 0.0

        rate = (avg_savings / avg_income) * 100

        # Limiter à des valeurs réalistes
        if rate > 100:
            logger.warning(f"Taux d'épargne anormalement élevé: {rate:.2f}% (limité à 100%)")
            return 100.0
        elif rate < -100:
            logger.warning(f"Taux d'épargne anormalement bas: {rate:.2f}% (limité à -100%)")
            return -100.0

        return round(rate, 2)

    def _categorize_expenses(
        self,
        category_breakdown: Dict[str, float],
        fixed_charges_total: float
    ) -> tuple[float, float, float]:
        """
        Catégorise les dépenses en fixes structurelles, semi-fixes et variables

        Returns:
            (semi_fixed_total, variable_total, structural_fixed_total)
        """
        # Catégories semi-fixes (dépenses récurrentes mais ajustables)
        semi_fixed_categories = [
            'alimentation',
            'courses',
            'carburant',
            'transport',
            'santé',
            'pharmacie',
            'entretien',
            'électricité',
            'eau',
            'énergie',
            'essence',
            'garage'
        ]

        # Catégories variables (dépenses discrétionnaires)
        variable_categories = [
            'loisirs',
            'restaurant',
            'shopping',
            'vêtement',
            'cadeau',
            'voyage',
            'divertissement',
            'streaming',
            'paris',
            'jeux',
            'loterie',
            'ligne'  # achats en ligne
        ]

        # Catégories fixes (à exclure car déjà comptées dans fixed_charges ou charges structurelles)
        fixed_categories = [
            'prêt',
            'crédit',
            'assurance',
            'loyer',
            'bail',
            'pension',
            'garde',
            'scolarité',
            'téléphone',
            'internet',
            'abonnement',
            'impôt',
            'taxe'
        ]

        semi_fixed_total = 0.0
        variable_total = 0.0
        structural_fixed_total = 0.0  # Charges fixes structurelles (prêts, impôts, etc.)
        other_total = 0.0  # Autres dépenses non classifiées (incluant moyens de paiement non typés)

        for category, amount in category_breakdown.items():
            category_lower = category.lower()

            # Priorité : fixed > semi-fixed > variable > other
            if any(fixed in category_lower for fixed in fixed_categories):
                structural_fixed_total += amount
            elif any(semi in category_lower for semi in semi_fixed_categories):
                semi_fixed_total += amount
            elif any(var in category_lower for var in variable_categories):
                variable_total += amount
            else:
                # Tout le reste va dans "autres" (y compris moyens de paiement non typés)
                other_total += amount

        # Ajouter "autres" aux charges variables (considérées comme discrétionnaires par défaut)
        # Note: Cela inclut virements, chèques, espèces, etc. dont on ne connaît pas la destination
        variable_total += other_total

        # Retourner les 3 types
        # Note: structural_fixed sera ajouté à fixed_charges_total
        return semi_fixed_total, variable_total, structural_fixed_total

    def _determine_segment(self, avg_income: float, avg_expenses: float) -> str:
        """
        Détermine le segment budgétaire de l'utilisateur

        Returns:
            'budget_serré', 'équilibré', 'confortable'
        """
        if avg_income <= 0:
            return 'indéterminé'

        ratio = avg_expenses / avg_income

        if ratio > 0.90:
            return 'budget_serré'
        elif ratio >= 0.70:
            return 'équilibré'
        else:
            return 'confortable'

    def _determine_behavioral_pattern(self, user_id: int) -> str:
        """
        Détermine le pattern comportemental de l'utilisateur

        Returns:
            'dépensier_hebdomadaire', 'acheteur_impulsif', 'planificateur'
        """
        try:
            # Récupérer transactions du dernier mois
            transactions = self.transaction_service.get_user_transactions(
                user_id,
                months_back=1
            )

            if not transactions:
                return 'indéterminé'

            # Filtrer débits uniquement
            debits = [tx for tx in transactions if tx['is_debit']]

            if not debits:
                return 'indéterminé'

            # Nombre de transactions par semaine
            tx_per_week = len(debits) / 4.0

            # Montant moyen par transaction
            avg_tx_amount = sum(abs(tx['amount']) for tx in debits) / len(debits)

            # Critères de classification
            if tx_per_week > 10 and avg_tx_amount < 20:
                return 'acheteur_impulsif'
            elif tx_per_week < 5 and avg_tx_amount > 50:
                return 'planificateur'
            else:
                return 'dépensier_hebdomadaire'

        except Exception as e:
            logger.error(f"Erreur détermination pattern: {e}")
            return 'indéterminé'

    def _calculate_completeness(
        self,
        monthly_aggregates: list,
        fixed_charges: list,
        months_required: Optional[int]
    ) -> float:
        """
        Calcule le score de complétude du profil (0.0 - 1.0)
        """
        score = 0.0

        # Facteur 1: Nombre de mois de données (max 0.4)
        if months_required is None:
            # Toutes les transactions: Score basé sur nombre absolu de mois
            # 12+ mois = score complet, progressif avant
            months_score = min(len(monthly_aggregates) / 12.0, 1.0) * 0.4
        else:
            months_score = min(len(monthly_aggregates) / months_required, 1.0) * 0.4

        # Facteur 2: Présence de charges fixes détectées (max 0.3)
        fixed_charges_score = min(len(fixed_charges) / 5.0, 1.0) * 0.3

        # Facteur 3: Présence de revenus (max 0.3)
        has_income = any(m['total_income'] > 0 for m in monthly_aggregates)
        income_score = 0.3 if has_income else 0.0

        score = months_score + fixed_charges_score + income_score
        return min(max(score, 0.0), 1.0)

    def _empty_profile(self) -> Dict[str, Any]:
        """
        Retourne un profil vide
        """
        return {
            'user_segment': 'indéterminé',
            'behavioral_pattern': 'indéterminé',
            'avg_monthly_income': 0.0,
            'avg_monthly_expenses': 0.0,
            'avg_monthly_savings': 0.0,
            'savings_rate': 0.0,
            'fixed_charges_total': 0.0,
            'semi_fixed_charges_total': 0.0,
            'variable_charges_total': 0.0,
            'remaining_to_live': 0.0,
            'profile_completeness': 0.0,
            'last_analyzed_at': datetime.now(timezone.utc)
        }

    def save_profile(
        self,
        user_id: int,
        profile_data: Dict[str, Any]
    ) -> Optional[UserBudgetProfile]:
        """
        Sauvegarde le profil budgétaire en base

        Returns:
            Le profil sauvegardé ou None en cas d'erreur
        """
        try:
            # Vérifier si profil existe
            result = self.db.execute(
                select(UserBudgetProfile).where(UserBudgetProfile.user_id == user_id)
            )
            existing_profile = result.scalar_one_or_none()

            if existing_profile:
                # Mettre à jour
                for key, value in profile_data.items():
                    if key != 'user_id' and hasattr(existing_profile, key):
                        setattr(existing_profile, key, value)

                profile = existing_profile
            else:
                # Créer nouveau profil - filtrer user_id de profile_data
                filtered_data = {k: v for k, v in profile_data.items() if k != 'user_id'}
                profile = UserBudgetProfile(
                    user_id=user_id,
                    **filtered_data
                )
                self.db.add(profile)

            self.db.commit()
            self.db.refresh(profile)

            logger.info(f"Profil sauvegardé pour user {user_id}")
            return profile

        except Exception as e:
            self.db.rollback()
            logger.error(f"Erreur sauvegarde profil: {e}", exc_info=True)
            return None

    def get_user_profile(self, user_id: int) -> Optional[UserBudgetProfile]:
        """
        Récupère le profil budgétaire d'un utilisateur
        """
        try:
            result = self.db.execute(
                select(UserBudgetProfile).where(UserBudgetProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()
            return profile

        except Exception as e:
            logger.error(f"Erreur récupération profil: {e}", exc_info=True)
            return None
