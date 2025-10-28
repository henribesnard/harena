"""
Méthodes analytiques avancées pour le profiling budgétaire
Segmentation multi-critères, détection de patterns, calcul de health score, alertes
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from budget_profiling_service.services import alert_thresholds

logger = logging.getLogger(__name__)


class AdvancedBudgetAnalytics:
    """
    Classe contenant les méthodes analytiques avancées pour le profiling
    """

    @staticmethod
    def determine_segment_v2(
        avg_income: float,
        avg_expenses: float,
        remaining_to_live: float,
        fixed_charges_total: float
    ) -> Dict[str, Any]:
        """
        Segmentation améliorée avec scoring multi-critères

        Returns:
            {
                'segment': str,
                'score': float (0-100),
                'factors': dict,
                'risk_level': str
            }
        """
        if avg_income <= 0:
            return {
                'segment': 'indéterminé',
                'score': 0,
                'factors': {},
                'risk_level': 'unknown'
            }

        # Critère 1: Ratio dépenses/revenus (poids 30%)
        expense_ratio = avg_expenses / avg_income
        expense_score = max(0, 100 - (expense_ratio * 100)) * 0.3

        # Critère 2: Reste à vivre absolu (poids 30%)
        # Seuils adaptatifs selon les revenus
        min_remaining_needed = min(500, avg_income * 0.2)  # 20% min ou 500€
        remaining_score = min(100, (remaining_to_live / min_remaining_needed) * 100) * 0.3

        # Critère 3: Taux de charges fixes (poids 20%)
        # Idéalement < 50% des revenus
        fixed_ratio = fixed_charges_total / avg_income if avg_income > 0 else 1.0
        fixed_score = max(0, 100 - (fixed_ratio * 200)) * 0.2  # 50% = score 0

        # Critère 4: Capacité d'épargne absolue (poids 20%)
        monthly_savings = avg_income - avg_expenses
        # Seuil minimum : 10% des revenus
        savings_target = avg_income * 0.1
        savings_score = min(100, (monthly_savings / savings_target) * 100) * 0.2 if savings_target > 0 else 0

        # Score global (0-100)
        total_score = expense_score + remaining_score + fixed_score + savings_score

        # Déterminer le segment selon le score
        if total_score >= 70:
            segment = 'confortable'
            risk_level = 'low'
        elif total_score >= 40:
            segment = 'équilibré'
            risk_level = 'medium'
        elif total_score >= 20:
            segment = 'budget_serré'
            risk_level = 'high'
        else:
            segment = 'précaire'
            risk_level = 'critical'

        return {
            'segment': segment,
            'score': round(total_score, 1),
            'factors': {
                'expense_ratio': round(expense_ratio * 100, 1),
                'remaining_to_live': round(remaining_to_live, 2),
                'fixed_charges_ratio': round(fixed_ratio * 100, 1),
                'monthly_savings': round(monthly_savings, 2)
            },
            'risk_level': risk_level,
            'recommendations': AdvancedBudgetAnalytics._get_segment_recommendations(segment, total_score)
        }

    @staticmethod
    def _get_segment_recommendations(segment: str, score: float) -> List[str]:
        """Génère des recommandations selon le segment"""
        recommendations = []

        if segment == 'précaire' or score < 20:
            recommendations = [
                "Réduire immédiatement les dépenses variables",
                "Identifier les charges fixes réductibles",
                "Consulter un conseiller financier",
                "Établir un budget d'urgence strict"
            ]
        elif segment == 'budget_serré':
            recommendations = [
                "Augmenter votre taux d'épargne de 5%",
                "Optimiser les dépenses semi-fixes (courses, carburant)",
                "Renégocier vos abonnements et assurances",
                "Planifier les grosses dépenses"
            ]
        elif segment == 'équilibré':
            recommendations = [
                "Maintenir votre équilibre actuel",
                "Constituer une épargne de précaution (3-6 mois)",
                "Optimiser les petites dépenses récurrentes",
                "Envisager des placements pour l'épargne excédentaire"
            ]
        else:  # confortable
            recommendations = [
                "Maximiser votre épargne et vos investissements",
                "Optimiser la fiscalité de votre épargne",
                "Diversifier vos placements",
                "Planifier des objectifs financiers à long terme"
            ]

        return recommendations

    @staticmethod
    def calculate_expense_volatility(monthly_aggregates: List[Dict[str, Any]]) -> float:
        """
        Calcule la volatilité des dépenses (coefficient de variation)

        Args:
            monthly_aggregates: Liste des agrégats mensuels

        Returns:
            Coefficient de variation (écart-type / moyenne)
        """
        if not monthly_aggregates or len(monthly_aggregates) < 2:
            return 0.0

        expenses = [m['total_expenses'] for m in monthly_aggregates]

        # Calculer moyenne
        mean_expenses = sum(expenses) / len(expenses)

        if mean_expenses == 0:
            return 0.0

        # Calculer écart-type
        variance = sum((x - mean_expenses) ** 2 for x in expenses) / len(expenses)
        std_dev = variance ** 0.5

        # Coefficient de variation
        cv = std_dev / mean_expenses if mean_expenses > 0 else 0

        return cv

    @staticmethod
    def analyze_spending_trend(monthly_aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse la tendance des dépenses avec moyenne mobile 3 mois (plus robuste)

        Returns:
            {
                'trend': str ('increasing', 'decreasing', 'stable'),
                'change_pct': float,
                'prev_income': float,
                'prev_expenses': float,
                'income_change_pct': float,
                'expense_change_pct': float
            }
        """
        if not monthly_aggregates or len(monthly_aggregates) < 2:
            return {
                'trend': 'stable',
                'change_pct': 0.0,
                'prev_income': 0.0,
                'prev_expenses': 0.0,
                'income_change_pct': 0.0,
                'expense_change_pct': 0.0
            }

        # Trier par date (du plus récent au plus ancien)
        sorted_months = sorted(monthly_aggregates, key=lambda x: x.get('month', ''), reverse=True)

        # Si moins de 6 mois, fallback sur comparaison simple 2 mois
        if len(sorted_months) < 6:
            current_month = sorted_months[0]
            prev_month = sorted_months[1] if len(sorted_months) > 1 else sorted_months[0]

            current_expenses = current_month['total_expenses']
            prev_expenses = prev_month['total_expenses']
            current_income = current_month['total_income']
            prev_income = prev_month['total_income']

            expense_change_pct = ((current_expenses - prev_expenses) / prev_expenses * 100) if prev_expenses > 0 else 0
            income_change_pct = ((current_income - prev_income) / prev_income * 100) if prev_income > 0 else 0

            # Seuil plus élevé pour éviter faux signaux
            if abs(expense_change_pct) < 10:
                trend = 'stable'
            elif expense_change_pct > 10:
                trend = 'increasing'
            else:
                trend = 'decreasing'

            return {
                'trend': trend,
                'change_pct': round(expense_change_pct, 2),
                'prev_income': round(prev_income, 2),
                'prev_expenses': round(prev_expenses, 2),
                'income_change_pct': round(income_change_pct, 2),
                'expense_change_pct': round(expense_change_pct, 2)
            }

        # Moyenne mobile actuelle (3 derniers mois)
        current_avg_expenses = sum(m['total_expenses'] for m in sorted_months[0:3]) / 3
        current_avg_income = sum(m['total_income'] for m in sorted_months[0:3]) / 3

        # Moyenne mobile précédente (mois 3 à 6)
        prev_avg_expenses = sum(m['total_expenses'] for m in sorted_months[3:6]) / 3
        prev_avg_income = sum(m['total_income'] for m in sorted_months[3:6]) / 3

        # Calculer variation
        expense_change_pct = ((current_avg_expenses - prev_avg_expenses) / prev_avg_expenses * 100) if prev_avg_expenses > 0 else 0
        income_change_pct = ((current_avg_income - prev_avg_income) / prev_avg_income * 100) if prev_avg_income > 0 else 0

        # Déterminer tendance avec seuil plus élevé (moins de faux signaux)
        if abs(expense_change_pct) < 10:  # Moins de 10% = stable
            trend = 'stable'
        elif expense_change_pct > 10:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        return {
            'trend': trend,
            'change_pct': round(expense_change_pct, 2),
            'prev_income': round(prev_avg_income, 2),
            'prev_expenses': round(prev_avg_expenses, 2),
            'income_change_pct': round(income_change_pct, 2),
            'expense_change_pct': round(expense_change_pct, 2),
            'current_avg_expenses': round(current_avg_expenses, 2),
            'current_avg_income': round(current_avg_income, 2)
        }

    @staticmethod
    def calculate_debt_to_income_ratio(fixed_charges_total: float, avg_income: float) -> float:
        """
        Calcule le ratio d'endettement (charges fixes / revenus)

        Args:
            fixed_charges_total: Total des charges fixes mensuelles
            avg_income: Revenus moyens mensuels

        Returns:
            Ratio en pourcentage (0-100+)
        """
        if avg_income <= 0:
            return 0.0

        ratio = (fixed_charges_total / avg_income) * 100
        return min(ratio, 100.0)  # Cap à 100% pour éviter valeurs aberrantes

    @staticmethod
    def calculate_financial_health_score(profile_data: Dict[str, Any]) -> float:
        """
        Score de santé financière global (0-100)

        Facteurs:
        - 30%: Taux d'épargne
        - 25%: Ratio charges fixes/revenus
        - 20%: Reste à vivre
        - 15%: Régularité (faible volatilité)
        - 10%: Tendance (amélioration vs détérioration)
        """
        score = 0.0

        # 1. Taux d'épargne (30 points max)
        savings_rate = profile_data.get('savings_rate', 0)
        if savings_rate >= 20:
            score += 30
        elif savings_rate >= 10:
            score += 20
        elif savings_rate >= 5:
            score += 10
        elif savings_rate > 0:
            score += 5
        # Négatif = 0 points

        # 2. Ratio charges fixes (25 points max)
        income = profile_data.get('avg_monthly_income', 1)
        fixed_charges = profile_data.get('fixed_charges_total', 0)
        fixed_ratio = fixed_charges / income if income > 0 else 1
        if fixed_ratio < 0.3:
            score += 25
        elif fixed_ratio < 0.5:
            score += 15
        elif fixed_ratio < 0.7:
            score += 8
        # > 70% = 0 points (dangereux)

        # 3. Reste à vivre (20 points max)
        remaining = profile_data.get('remaining_to_live', 0)
        if remaining > income * 0.4:  # > 40% des revenus
            score += 20
        elif remaining > income * 0.2:
            score += 12
        elif remaining > 0:
            score += 5

        # 4. Volatilité (15 points max)
        volatility = profile_data.get('expense_volatility', 1.0)
        if volatility < 0.3:  # Très régulier
            score += 15
        elif volatility < 0.6:
            score += 10
        elif volatility < 1.0:
            score += 5

        # 5. Tendance (10 points max)
        trend = profile_data.get('spending_trend', 'stable')
        trend_pct = profile_data.get('spending_trend_pct', 0)
        if trend == 'decreasing' and trend_pct < -5:
            score += 10  # En amélioration
        elif trend == 'stable':
            score += 7
        elif trend == 'increasing' and trend_pct < 10:
            score += 3
        # Forte hausse = 0 points

        return min(100, max(0, score))

    @staticmethod
    def generate_alerts(profile_data: Dict[str, Any], user_id: int) -> List[Dict[str, Any]]:
        """
        Génère des alertes selon le profil avec seuils personnalisés

        Types d'alertes:
        - CRITICAL: Situation financière précaire
        - WARNING: Tendance négative
        - INFO: Opportunité d'optimisation
        """
        alerts = []

        # Récupérer seuils personnalisés selon le profil utilisateur
        thresholds = alert_thresholds.get_thresholds_for_user(profile_data)
        profile_type = thresholds.get('profile_type', 'default')

        # Alerte 1: Taux d'épargne négatif
        if profile_data.get('savings_rate', 0) < 0:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'negative_savings',
                'title': 'Dépenses supérieures aux revenus',
                'message': f"Vous dépensez {abs(profile_data['savings_rate']):.1f}% de plus que vos revenus. Action urgente requise.",
                'priority': 1,
                'actionable': True,
                'suggested_actions': [
                    'Analyser les dépenses variables',
                    'Identifier les charges fixes réductibles',
                    'Contacter un conseiller financier'
                ]
            })

        # Alerte 2: Taux d'épargne faible (seuil personnalisé)
        elif profile_data.get('savings_rate', 0) < thresholds['min_savings_rate']:
            min_savings = thresholds['min_savings_rate']
            message = alert_thresholds.get_contextual_alert_message(
                'savings',
                profile_type,
                profile_data['savings_rate'],
                min_savings
            )
            alerts.append({
                'level': 'WARNING',
                'type': 'low_savings',
                'title': 'Épargne insuffisante',
                'message': message,
                'priority': 2,
                'actionable': True,
                'suggested_actions': [
                    'Réduire les dépenses variables de 10%',
                    f"Économie potentielle: {profile_data.get('variable_charges_total', 0) * 0.1:.2f}€/mois"
                ]
            })

        # Alerte 3: Charges fixes élevées (seuil personnalisé)
        income = profile_data.get('avg_monthly_income', 1)
        fixed_charges = profile_data.get('fixed_charges_total', 0)
        fixed_ratio = fixed_charges / income if income > 0 else 0
        max_fixed_ratio = thresholds['max_fixed_ratio']

        if fixed_ratio > max_fixed_ratio:
            message = alert_thresholds.get_contextual_alert_message(
                'fixed_charges',
                profile_type,
                fixed_ratio * 100,
                max_fixed_ratio
            )
            alerts.append({
                'level': 'CRITICAL',
                'type': 'high_fixed_charges',
                'title': 'Charges fixes trop élevées',
                'message': message,
                'priority': 1,
                'actionable': True,
                'suggested_actions': [
                    'Renégocier vos abonnements',
                    'Revoir vos assurances',
                    'Analyser les prélèvements récurrents'
                ]
            })

        # Alerte 4: Tendance négative (seuil personnalisé)
        if profile_data.get('spending_trend') == 'increasing':
            trend_pct = profile_data.get('spending_trend_pct', 0)
            max_increase = thresholds['max_spending_increase']

            if trend_pct > max_increase:
                message = alert_thresholds.get_contextual_alert_message(
                    'spending_increase',
                    profile_type,
                    trend_pct,
                    max_increase
                )
                alerts.append({
                    'level': 'WARNING',
                    'type': 'increasing_expenses',
                    'title': 'Dépenses en forte hausse',
                    'message': message,
                    'priority': 2,
                    'actionable': True,
                    'suggested_actions': [
                        'Identifier les nouvelles dépenses',
                        'Vérifier les abonnements récents',
                        'Comparer avec les 3 derniers mois'
                    ]
                })

        # Alerte 5: Pas de fonds d'urgence (seuil personnalisé)
        months_saved = profile_data.get('months_of_expenses_saved', 0)
        emergency_months = thresholds['emergency_fund_months']

        if months_saved < emergency_months:
            avg_expenses = profile_data.get('avg_monthly_expenses', 0)
            avg_savings = profile_data.get('avg_monthly_savings', 0)
            savings_rate = profile_data.get('savings_rate', 1)

            months_to_goal = 0
            if avg_savings > 0:
                months_to_goal = (emergency_months - months_saved) / (savings_rate / 100) if savings_rate > 0 else 99

            alerts.append({
                'level': 'INFO',
                'type': 'emergency_fund',
                'title': 'Fonds d\'urgence insuffisant',
                'message': f"Vous pouvez tenir {months_saved:.1f} mois. Recommandé pour votre profil: {emergency_months} mois.",
                'priority': 3,
                'actionable': True,
                'suggested_actions': [
                    f"Objectif: épargner {avg_expenses * emergency_months:.0f}€",
                    f"Avec {avg_savings:.0f}€/mois, objectif atteint en {months_to_goal:.0f} mois"
                ]
            })

        # Trier par priorité
        alerts.sort(key=lambda x: x['priority'])

        return alerts

    @staticmethod
    def determine_behavioral_patterns_v2(
        transaction_service,
        user_id: int,
        account_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Détection multi-patterns avec scoring avancé

        Args:
            account_ids: Liste de bridge_account_ids à filtrer (optionnel)

        Returns:
            {
                'primary_pattern': str,
                'confidence': float,
                'all_patterns': List[Dict],
                'spending_style': Dict,
                'temporal_insights': Dict
            }
        """
        try:
            # Récupérer 6 mois de transactions pour meilleure analyse et patterns saisonniers
            transactions = transaction_service.get_user_transactions(
                user_id,
                months_back=6,
                account_ids=account_ids
            )

            if not transactions:
                return {
                    'primary_pattern': 'indéterminé',
                    'confidence': 0,
                    'all_patterns': [],
                    'spending_style': {},
                    'temporal_insights': {}
                }

            debits = [tx for tx in transactions if tx['is_debit']]

            if len(debits) < 10:
                return {
                    'primary_pattern': 'données_insuffisantes',
                    'confidence': 0,
                    'all_patterns': [],
                    'spending_style': {},
                    'temporal_insights': {}
                }

            # Analyse multi-dimensionnelle
            patterns = []

            # Pattern 1: Fréquence d'achat
            tx_per_week = len(debits) / 24.0  # Sur 6 mois = 24 semaines
            if tx_per_week > 15:
                patterns.append({
                    'type': 'high_frequency_spender',
                    'score': min(100, (tx_per_week / 20) * 100),
                    'description': f'{tx_per_week:.1f} transactions/semaine - Acheteur très fréquent'
                })
            elif tx_per_week < 3:
                patterns.append({
                    'type': 'low_frequency_spender',
                    'score': min(100, (5 - tx_per_week) / 5 * 100),
                    'description': f'{tx_per_week:.1f} transactions/semaine - Acheteur occasionnel'
                })

            # Pattern 2: Montant moyen (impulsivité)
            avg_amount = sum(abs(tx['amount']) for tx in debits) / len(debits)
            std_amount = AdvancedBudgetAnalytics._calculate_std_dev([abs(tx['amount']) for tx in debits])
            cv_amount = (std_amount / avg_amount) if avg_amount > 0 else 0

            if avg_amount < 15 and tx_per_week > 10:
                patterns.append({
                    'type': 'micro_spender',
                    'score': 85,
                    'description': f'Nombreux petits achats (moy: {avg_amount:.2f}€) - Tendance impulsive'
                })
            elif avg_amount > 100 and cv_amount < 0.3:
                patterns.append({
                    'type': 'planificateur',
                    'score': 90,
                    'description': f'Achats espacés et réfléchis (moy: {avg_amount:.2f}€) - Planificateur'
                })

            # Pattern 3: Variance (régularité)
            if cv_amount < 0.4:
                patterns.append({
                    'type': 'regular_spender',
                    'score': 80,
                    'description': 'Dépenses très régulières - Comportement prévisible'
                })
            elif cv_amount > 1.0:
                patterns.append({
                    'type': 'erratic_spender',
                    'score': 75,
                    'description': 'Forte variabilité des dépenses - Comportement imprévisible'
                })

            # Pattern 4: Temporalité (weekend vs semaine)
            weekend_spending = AdvancedBudgetAnalytics._analyze_weekend_pattern(debits)
            if weekend_spending['weekend_ratio'] > 0.4:
                patterns.append({
                    'type': 'weekend_spender',
                    'score': weekend_spending['confidence'],
                    'description': f'{weekend_spending["weekend_ratio"]*100:.0f}% des dépenses le weekend'
                })

            # Trier par score et prendre le pattern dominant
            patterns.sort(key=lambda x: x['score'], reverse=True)
            primary_pattern = patterns[0]['type'] if patterns else 'standard'

            return {
                'primary_pattern': primary_pattern,
                'confidence': patterns[0]['score'] if patterns else 0,
                'all_patterns': patterns[:3],  # Top 3 patterns
                'spending_style': {
                    'frequency': f'{tx_per_week:.1f} tx/semaine',
                    'avg_amount': round(avg_amount, 2),
                    'regularity': 'élevée' if cv_amount < 0.5 else 'moyenne' if cv_amount < 1.0 else 'faible',
                    'coefficient_variation': round(cv_amount, 2)
                },
                'temporal_insights': weekend_spending
            }

        except Exception as e:
            logger.error(f"Erreur détection patterns: {e}", exc_info=True)
            return {
                'primary_pattern': 'erreur',
                'confidence': 0,
                'all_patterns': [],
                'spending_style': {},
                'temporal_insights': {}
            }

    @staticmethod
    def _analyze_weekend_pattern(transactions: List[Dict]) -> Dict[str, Any]:
        """Analyse les dépenses weekend vs semaine"""
        weekend_amount = 0
        weekday_amount = 0

        for tx in transactions:
            date_str = tx.get('date')
            if not date_str:
                continue

            try:
                date = datetime.fromisoformat(date_str) if isinstance(date_str, str) else date_str
                amount = abs(tx['amount'])

                if date.weekday() >= 5:  # Samedi=5, Dimanche=6
                    weekend_amount += amount
                else:
                    weekday_amount += amount
            except Exception:
                continue

        total = weekend_amount + weekday_amount
        weekend_ratio = weekend_amount / total if total > 0 else 0

        return {
            'weekend_ratio': round(weekend_ratio, 2),
            'weekend_amount': round(weekend_amount, 2),
            'weekday_amount': round(weekday_amount, 2),
            'confidence': min(100, len(transactions) / 30 * 100)  # Plus de données = plus de confiance
        }

    @staticmethod
    def _calculate_std_dev(values: List[float]) -> float:
        """Calcule l'écart-type"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
