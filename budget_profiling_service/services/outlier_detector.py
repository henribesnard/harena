"""
Détection de pics et anomalies dans les dépenses
Permet d'identifier les mois avec dépenses exceptionnelles pour calculer des moyennes plus représentatives
"""
from typing import List, Dict, Any, Tuple
import statistics
import logging

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Détecte les mois avec dépenses anormalement élevées ou basses
    """

    @staticmethod
    def detect_spending_outliers(
        monthly_aggregates: List[Dict[str, Any]],
        method: str = 'iqr'  # 'iqr' ou 'zscore'
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Détecte les outliers dans les dépenses mensuelles

        Args:
            monthly_aggregates: Liste des agrégats mensuels
            method: Méthode de détection ('iqr' = Interquartile Range, 'zscore' = Z-score)

        Returns:
            (clean_data, outliers)
            - clean_data: Agrégats sans les outliers
            - outliers: Liste des mois identifiés comme outliers avec détails
        """
        if len(monthly_aggregates) < 6:
            logger.info("Moins de 6 mois de données, pas de détection d'outliers")
            return monthly_aggregates, []

        expenses = [m['total_expenses'] for m in monthly_aggregates]

        if method == 'iqr':
            lower_bound, upper_bound = OutlierDetector._calculate_iqr_bounds(expenses)
        else:  # zscore
            lower_bound, upper_bound = OutlierDetector._calculate_zscore_bounds(expenses)

        logger.info(f"Seuils outliers ({method}): [{lower_bound:.2f}, {upper_bound:.2f}]")

        # Identifier les outliers
        outliers = []
        clean_data = []

        mean_expense = statistics.mean(expenses)

        for month_data in monthly_aggregates:
            expense = month_data['total_expenses']

            if lower_bound <= expense <= upper_bound:
                clean_data.append(month_data)
            else:
                deviation_pct = ((expense - mean_expense) / mean_expense) * 100 if mean_expense > 0 else 0

                outliers.append({
                    'month': month_data['month'],
                    'expense': expense,
                    'type': 'spike' if expense > upper_bound else 'drop',
                    'deviation_pct': round(deviation_pct, 2),
                    'amount_above_normal': round(expense - mean_expense, 2)
                })

        logger.info(f"Détecté {len(outliers)} outliers sur {len(monthly_aggregates)} mois")

        return clean_data, outliers

    @staticmethod
    def _calculate_iqr_bounds(expenses: List[float]) -> Tuple[float, float]:
        """
        Calcule les bornes avec la méthode IQR (Interquartile Range)

        Returns:
            (lower_bound, upper_bound)
        """
        if len(expenses) < 4:
            return 0, float('inf')

        # Trier les dépenses
        sorted_expenses = sorted(expenses)

        # Calculer quartiles
        n = len(sorted_expenses)
        q1_idx = n // 4
        q3_idx = 3 * n // 4

        q1 = sorted_expenses[q1_idx]
        q3 = sorted_expenses[q3_idx]

        iqr = q3 - q1

        # Bornes (facteur 1.5 = standard pour outliers)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Assurer bornes positives
        lower_bound = max(0, lower_bound)

        return lower_bound, upper_bound

    @staticmethod
    def _calculate_zscore_bounds(expenses: List[float]) -> Tuple[float, float]:
        """
        Calcule les bornes avec la méthode Z-score

        Returns:
            (lower_bound, upper_bound)
        """
        if len(expenses) < 2:
            return 0, float('inf')

        mean = statistics.mean(expenses)
        stdev = statistics.stdev(expenses)

        # Seuil à 2 écarts-types (95% des données)
        lower_bound = mean - 2 * stdev
        upper_bound = mean + 2 * stdev

        # Assurer bornes positives
        lower_bound = max(0, lower_bound)

        return lower_bound, upper_bound

    @staticmethod
    def categorize_outlier_reason(
        outlier_month: str,
        category_breakdown: Dict[str, float]
    ) -> str:
        """
        Tente de catégoriser la raison du pic en analysant les catégories dominantes

        Args:
            outlier_month: Mois de l'outlier (format 'YYYY-MM')
            category_breakdown: Répartition des dépenses par catégorie pour ce mois

        Returns:
            Raison probable ('vacation', 'shopping', 'medical', 'home', 'unknown')
        """
        if not category_breakdown:
            return 'unknown'

        # Trouver la catégorie dominante
        sorted_categories = sorted(
            category_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if not sorted_categories:
            return 'unknown'

        top_category, top_amount = sorted_categories[0]
        category_lower = top_category.lower()

        # Mapping catégories -> raisons
        reason_mapping = {
            'vacation': ['voyage', 'hotel', 'avion', 'train', 'vacances'],
            'shopping': ['shopping', 'vêtement', 'cadeau', 'ligne'],
            'medical': ['santé', 'médical', 'pharmacie', 'hopital'],
            'home': ['travaux', 'maison', 'meuble', 'décoration', 'bricolage'],
            'vehicle': ['voiture', 'garage', 'réparation', 'automobile'],
            'celebration': ['restaurant', 'traiteur', 'réception', 'mariage']
        }

        for reason, keywords in reason_mapping.items():
            if any(keyword in category_lower for keyword in keywords):
                return reason

        return 'unknown'

    @staticmethod
    def calculate_baseline_metrics(
        monthly_aggregates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calcule les métriques de base en excluant les outliers

        Args:
            monthly_aggregates: Agrégats mensuels complets

        Returns:
            Métriques sans outliers (avg_expenses, avg_income, avg_savings)
        """
        clean_data, outliers = OutlierDetector.detect_spending_outliers(monthly_aggregates)

        if not clean_data:
            logger.warning("Aucune donnée après suppression outliers, utiliser données complètes")
            clean_data = monthly_aggregates

        avg_expenses = sum(m['total_expenses'] for m in clean_data) / len(clean_data)
        avg_income = sum(m['total_income'] for m in clean_data) / len(clean_data)
        avg_savings = avg_income - avg_expenses

        return {
            'avg_monthly_expenses': round(avg_expenses, 2),
            'avg_monthly_income': round(avg_income, 2),
            'avg_monthly_savings': round(avg_savings, 2),
            'months_analyzed': len(clean_data),
            'outliers_excluded': len(outliers)
        }
