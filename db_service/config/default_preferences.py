"""
Configuration des valeurs par défaut pour les préférences utilisateur du Budget Profiling Service
"""

# Valeurs par défaut pour tous les paramètres de profiling budgétaire
DEFAULT_BUDGET_SETTINGS = {
    # === Profondeur d'analyse ===
    "months_analysis": 12,  # Nombre de mois à analyser par défaut

    # === Sélection des comptes ===
    # NOTE: Seuls les comptes "checking" et "card" sont éligibles (WHITELIST)
    # Les autres types (savings, loan, investment, other) sont automatiquement exclus
    "account_selection": {
        "mode": "all",                     # Mode de sélection: "all", "exclude_types", "include_specific"
        "excluded_types": [],              # Types additionnels à exclure parmi les éligibles (ex: ["card"] pour ne garder que checking)
        "included_accounts": []            # Liste d'IDs de comptes spécifiquement inclus (si mode="include_specific", parmi checking/card uniquement)
    },

    # === Détection des charges fixes ===
    "fixed_charge_detection": {
        "min_occurrences": 5,              # Nombre minimum de transactions récurrentes (prend en compte paiements en 4x)
        "max_amount_variance_pct": 20.0,   # Variance maximale acceptée du montant (%)
        "max_day_variance": 7,             # Variance maximale du jour du mois (jours)
        "min_amount_threshold": 10.0,      # Montant minimum pour considérer comme charge fixe (€)
        "recurrence_confidence_min": 0.7   # Confiance minimale pour valider une charge fixe (0.0 - 1.0)
    },

    # === Préférences d'analyse ===
    "analysis_preferences": {
        "include_outliers": True,          # Inclure les outliers dans le calcul des moyennes
        "savings_calculation_method": "net_cashflow",  # Méthode de calcul de l'épargne : "net_cashflow" ou "difference"
        "outlier_detection_method": "iqr"  # Méthode de détection des outliers : "iqr" ou "zscore"
    },

    # === Seuils d'alertes (par défaut, seront ajustés selon le profil) ===
    "alert_thresholds": {
        "min_savings_rate": 5.0,           # Taux d'épargne minimum recommandé (%)
        "max_fixed_charges_ratio": 0.70,   # Ratio maximum charges fixes / revenus (70%)
        "max_spending_increase": 15.0,     # Variation maximum des dépenses acceptable (%)
        "min_remaining_to_live": 300.0,    # Reste à vivre minimum recommandé (€)
        "emergency_fund_months": 3         # Nombre de mois de fonds d'urgence recommandé
    },

    # === Calcul du score de santé ===
    "health_score_weights": {
        "savings_rate": 0.30,      # Poids du taux d'épargne (30%)
        "fixed_charges": 0.25,     # Poids du ratio charges fixes (25%)
        "remaining_to_live": 0.20, # Poids du reste à vivre (20%)
        "volatility": 0.15,        # Poids de la volatilité (15%)
        "trend": 0.10              # Poids de la tendance (10%)
    },

    # === Complétude du profil ===
    "completeness_weights": {
        "months_data": 0.30,       # Poids du nombre de mois de données (30%)
        "fixed_charges": 0.25,     # Poids des charges fixes détectées (25%)
        "income_presence": 0.25,   # Poids de la présence de revenus (25%)
        "categorization": 0.20     # Poids de la qualité de catégorisation (20%)
    }
}


def get_default_budget_settings() -> dict:
    """
    Retourne une copie des paramètres par défaut

    Returns:
        Dictionnaire des paramètres par défaut
    """
    import copy
    return copy.deepcopy(DEFAULT_BUDGET_SETTINGS)


def merge_with_defaults(user_settings: dict) -> dict:
    """
    Fusionne les paramètres utilisateur avec les valeurs par défaut
    Les valeurs utilisateur ont la priorité, les valeurs manquantes sont complétées avec les défauts

    Args:
        user_settings: Paramètres personnalisés de l'utilisateur (peut être incomplet)

    Returns:
        Paramètres complets avec valeurs par défaut pour les clés manquantes
    """
    import copy

    def deep_merge(default: dict, user: dict) -> dict:
        """Fusion récursive de dictionnaires"""
        result = copy.deepcopy(default)
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    if not user_settings:
        return get_default_budget_settings()

    return deep_merge(DEFAULT_BUDGET_SETTINGS, user_settings)


def validate_budget_settings(settings: dict) -> tuple[bool, list[str]]:
    """
    Valide les paramètres budgétaires

    Args:
        settings: Paramètres à valider

    Returns:
        (is_valid, errors) - Tuple avec booléen de validité et liste d'erreurs
    """
    errors = []

    # Validation months_analysis
    if "months_analysis" in settings:
        months = settings["months_analysis"]
        if not isinstance(months, int) or months < 1 or months > 60:
            errors.append("months_analysis doit être entre 1 et 60")

    # Validation fixed_charge_detection
    if "fixed_charge_detection" in settings:
        fcd = settings["fixed_charge_detection"]

        if "min_occurrences" in fcd:
            if not isinstance(fcd["min_occurrences"], int) or fcd["min_occurrences"] < 2:
                errors.append("fixed_charge_detection.min_occurrences doit être >= 2")

        if "max_amount_variance_pct" in fcd:
            if not isinstance(fcd["max_amount_variance_pct"], (int, float)) or fcd["max_amount_variance_pct"] < 0 or fcd["max_amount_variance_pct"] > 100:
                errors.append("fixed_charge_detection.max_amount_variance_pct doit être entre 0 et 100")

        if "max_day_variance" in fcd:
            if not isinstance(fcd["max_day_variance"], int) or fcd["max_day_variance"] < 1 or fcd["max_day_variance"] > 15:
                errors.append("fixed_charge_detection.max_day_variance doit être entre 1 et 15")

        if "min_amount_threshold" in fcd:
            if not isinstance(fcd["min_amount_threshold"], (int, float)) or fcd["min_amount_threshold"] < 0:
                errors.append("fixed_charge_detection.min_amount_threshold doit être >= 0")

    # Validation analysis_preferences
    if "analysis_preferences" in settings:
        ap = settings["analysis_preferences"]

        if "savings_calculation_method" in ap:
            if ap["savings_calculation_method"] not in ["net_cashflow", "difference"]:
                errors.append("analysis_preferences.savings_calculation_method doit être 'net_cashflow' ou 'difference'")

        if "outlier_detection_method" in ap:
            if ap["outlier_detection_method"] not in ["iqr", "zscore"]:
                errors.append("analysis_preferences.outlier_detection_method doit être 'iqr' ou 'zscore'")

    # Validation account_selection
    if "account_selection" in settings:
        ac = settings["account_selection"]

        if "mode" in ac:
            if ac["mode"] not in ["all", "exclude_types", "include_specific"]:
                errors.append("account_selection.mode doit être 'all', 'exclude_types' ou 'include_specific'")

        if "excluded_types" in ac:
            if not isinstance(ac["excluded_types"], list):
                errors.append("account_selection.excluded_types doit être une liste")
            else:
                # Note: Seuls checking et card sont éligibles, donc excluded_types ne peut contenir que ceux-ci
                eligible_types = ["checking", "card"]
                for account_type in ac["excluded_types"]:
                    if account_type not in eligible_types:
                        errors.append(
                            f"account_selection.excluded_types contient un type invalide: {account_type}. "
                            f"Seuls 'checking' et 'card' sont valides (les autres types sont déjà exclus)"
                        )

        if "included_accounts" in ac:
            if not isinstance(ac["included_accounts"], list):
                errors.append("account_selection.included_accounts doit être une liste")
            else:
                for account_id in ac["included_accounts"]:
                    if not isinstance(account_id, int):
                        errors.append(f"account_selection.included_accounts doit contenir uniquement des IDs entiers")
                        break

    return len(errors) == 0, errors
