"""
Seuils d'alertes personnalisés selon le profil utilisateur
Adapte les seuils en fonction du niveau de revenus, segment, et situation
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


# Seuils par profil utilisateur
ALERT_THRESHOLDS = {
    'student': {
        'min_savings_rate': 0,       # Étudiant: 0% OK
        'max_fixed_ratio': 0.80,     # 80% max en charges fixes
        'max_spending_increase': 20,  # 20% variation acceptable
        'min_remaining': 100,         # Minimum 100€ de reste à vivre
        'emergency_fund_months': 1    # 1 mois de réserve minimum
    },
    'young_professional': {
        'min_savings_rate': 5,
        'max_fixed_ratio': 0.70,
        'max_spending_increase': 15,
        'min_remaining': 300,
        'emergency_fund_months': 3
    },
    'family': {
        'min_savings_rate': 10,
        'max_fixed_ratio': 0.60,
        'max_spending_increase': 10,
        'min_remaining': 500,
        'emergency_fund_months': 4
    },
    'senior': {
        'min_savings_rate': 15,
        'max_fixed_ratio': 0.50,
        'max_spending_increase': 5,
        'min_remaining': 400,
        'emergency_fund_months': 6
    },
    'default': {
        'min_savings_rate': 5,
        'max_fixed_ratio': 0.70,
        'max_spending_increase': 15,
        'min_remaining': 300,
        'emergency_fund_months': 3
    }
}


def get_thresholds_for_user(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Détermine les seuils d'alertes adaptés au profil utilisateur

    Args:
        profile_data: Données du profil budgétaire

    Returns:
        Seuils personnalisés pour ce profil
    """
    avg_income = profile_data.get('avg_monthly_income', 0)
    user_segment = profile_data.get('user_segment', 'default')

    # Classification basée sur revenus et segment
    user_profile_type = _classify_user_profile(avg_income, user_segment)

    thresholds = ALERT_THRESHOLDS.get(user_profile_type, ALERT_THRESHOLDS['default'])

    logger.info(
        f"Seuils sélectionnés pour profil '{user_profile_type}': "
        f"épargne min={thresholds['min_savings_rate']}%, "
        f"charges fixes max={thresholds['max_fixed_ratio']*100}%"
    )

    return {
        **thresholds,
        'profile_type': user_profile_type
    }


def _classify_user_profile(avg_income: float, user_segment: str) -> str:
    """
    Classifie le profil utilisateur selon revenus et segment

    Args:
        avg_income: Revenus mensuels moyens
        user_segment: Segment budgétaire

    Returns:
        Type de profil ('student', 'young_professional', 'family', 'senior', 'default')
    """
    # Revenus très bas + budget serré = probablement étudiant
    if avg_income < 1200 and user_segment in ['budget_serré', 'précaire']:
        return 'student'

    # Revenus bas/moyens
    elif avg_income < 2500:
        return 'young_professional'

    # Revenus moyens/élevés avec dépenses importantes = famille
    elif avg_income < 5000:
        # Si budget serré malgré revenus corrects = charges famille
        if user_segment in ['budget_serré', 'équilibré']:
            return 'family'
        else:
            return 'young_professional'

    # Revenus élevés
    else:
        # Si confortable = profil senior/stable
        if user_segment == 'confortable':
            return 'senior'
        else:
            return 'family'


def adjust_threshold_for_trend(
    threshold: float,
    spending_trend: str,
    trend_pct: float
) -> float:
    """
    Ajuste un seuil en fonction de la tendance des dépenses

    Args:
        threshold: Seuil de base
        spending_trend: Tendance ('increasing', 'decreasing', 'stable')
        trend_pct: Pourcentage de variation

    Returns:
        Seuil ajusté
    """
    # Si dépenses en hausse, être plus strict sur épargne
    if spending_trend == 'increasing' and abs(trend_pct) > 10:
        return threshold * 1.2  # +20% plus strict

    # Si dépenses en baisse, être plus souple
    elif spending_trend == 'decreasing' and abs(trend_pct) > 10:
        return threshold * 0.8  # -20% plus souple

    return threshold


def get_contextual_alert_message(
    alert_type: str,
    profile_type: str,
    value: float,
    threshold: float
) -> str:
    """
    Génère un message d'alerte contextualisé selon le profil

    Args:
        alert_type: Type d'alerte ('savings', 'fixed_charges', 'spending_increase')
        profile_type: Type de profil utilisateur
        value: Valeur actuelle
        threshold: Seuil dépassé

    Returns:
        Message personnalisé
    """
    messages = {
        'student': {
            'savings': f"Votre taux d'épargne ({value:.1f}%) est bas. Essayez d'économiser au moins {threshold}% pour constituer une réserve de sécurité.",
            'fixed_charges': f"Vos charges fixes représentent {value:.1f}% de vos revenus (max recommandé: {threshold*100}%). Renégociez vos abonnements si possible.",
            'spending_increase': f"Vos dépenses ont augmenté de {value:.1f}%. Surveillez vos achats impulsifs."
        },
        'young_professional': {
            'savings': f"Taux d'épargne: {value:.1f}% (objectif: {threshold}%). Augmentez votre épargne de précaution pour 3 mois de dépenses.",
            'fixed_charges': f"Charges fixes élevées: {value:.1f}% (max: {threshold*100}%). Optimisez vos contrats (assurance, télécom).",
            'spending_increase': f"Hausse de {value:.1f}% des dépenses. Analysez vos postes de dépenses variables."
        },
        'family': {
            'savings': f"Épargne familiale insuffisante: {value:.1f}% (objectif: {threshold}%). Visez 4 mois de réserve pour protéger votre famille.",
            'fixed_charges': f"Charges fixes: {value:.1f}% (max recommandé: {threshold*100}%). Réduisez les dépenses contraintes pour plus de flexibilité.",
            'spending_increase': f"Dépenses en hausse de {value:.1f}%. Impliquez toute la famille dans la gestion budgétaire."
        },
        'senior': {
            'savings': f"Taux d'épargne: {value:.1f}% (objectif: {threshold}%). Maximisez votre épargne et diversifiez vos placements.",
            'fixed_charges': f"Charges fixes: {value:.1f}% (optimum: {threshold*100}%). Réduisez vos engagements pour plus de sérénité.",
            'spending_increase': f"Variation de {value:.1f}%. Maintenez la stabilité de vos dépenses pour préserver votre épargne."
        }
    }

    default_messages = {
        'savings': f"Taux d'épargne faible: {value:.1f}% (recommandé: {threshold}%).",
        'fixed_charges': f"Charges fixes élevées: {value:.1f}% (max: {threshold*100}%).",
        'spending_increase': f"Dépenses en hausse de {value:.1f}%."
    }

    return messages.get(profile_type, {}).get(alert_type, default_messages.get(alert_type, ""))
