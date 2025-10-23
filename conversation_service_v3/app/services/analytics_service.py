"""
AnalyticsService - Service de calculs analytiques post-agrégation

Responsabilités:
- Comparaisons temporelles (période vs période)
- Calculs de ratios financiers (taux d'épargne, etc.)
- Classification de transactions (charges fixes vs variables)
- Détection de tendances
- Prévisions simples (moyennes mobiles)
- Recommandations d'optimisation

Author: Claude Code
Date: 2025-10-23
"""

import logging
from typing import Dict, Any, List, Optional
from statistics import mean
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service de calculs analytiques avancés

    Effectue des calculs post-agrégation sur les résultats Elasticsearch
    pour générer des insights actionnables.
    """

    def __init__(self):
        logger.info("AnalyticsService initialized")

    def compare_periods(
        self,
        period1_results,
        period2_results,
        metric: str = "total_amount"
    ) -> Dict[str, Any]:
        """
        Compare deux périodes et retourne variation absolue et relative

        Args:
            period1_results: SearchResults de la première période
            period2_results: SearchResults de la deuxième période
            metric: Métrique à comparer (total_amount, transaction_count, etc.)

        Returns:
            Dict avec comparaison détaillée
        """
        try:
            logger.info(f"Comparing periods for metric: {metric}")

            # Extraire les totaux de chaque période
            total1 = self._extract_metric(period1_results, metric)
            total2 = self._extract_metric(period2_results, metric)

            # Calculer variations
            variation_abs = total2 - total1
            variation_pct = (variation_abs / total1 * 100) if total1 != 0 else 0

            # Déterminer direction
            direction = "hausse" if variation_abs > 0 else "baisse" if variation_abs < 0 else "stable"

            result = {
                "period1": {
                    "value": total1,
                    "transaction_count": period1_results.total,
                    "aggregations": period1_results.aggregations
                },
                "period2": {
                    "value": total2,
                    "transaction_count": period2_results.total,
                    "aggregations": period2_results.aggregations
                },
                "comparison": {
                    "variation_absolute": variation_abs,
                    "variation_percent": variation_pct,
                    "direction": direction,
                    "significance": self._assess_significance(variation_pct)
                },
                "metric": metric
            }

            logger.info(f"Comparison complete: {direction} de {variation_pct:.1f}%")
            return result

        except Exception as e:
            logger.error(f"Error comparing periods: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "period1": {"value": 0},
                "period2": {"value": 0},
                "comparison": {"variation_absolute": 0, "variation_percent": 0}
            }

    def calculate_savings_rate(
        self,
        income_results,
        expense_results
    ) -> Dict[str, Any]:
        """
        Calcule le taux d'épargne et ses composants

        Args:
            income_results: SearchResults des revenus (credits)
            expense_results: SearchResults des dépenses (debits)

        Returns:
            Dict avec taux d'épargne et détails
        """
        try:
            logger.info("Calculating savings rate")

            # Extraire totaux
            total_income = self._extract_metric(income_results, "total_amount")
            total_expenses = self._extract_metric(expense_results, "total_amount")

            # Calculer épargne
            savings = total_income - total_expenses
            savings_rate = (savings / total_income * 100) if total_income != 0 else 0

            # Évaluation
            evaluation = self._evaluate_savings_rate(savings_rate)

            result = {
                "income": {
                    "total": total_income,
                    "transaction_count": income_results.total
                },
                "expenses": {
                    "total": total_expenses,
                    "transaction_count": expense_results.total
                },
                "savings": {
                    "amount": savings,
                    "rate_percent": savings_rate,
                    "evaluation": evaluation
                },
                "recommendations": self._generate_savings_recommendations(
                    savings_rate,
                    expense_results
                )
            }

            logger.info(f"Savings rate: {savings_rate:.1f}% ({evaluation})")
            return result

        except Exception as e:
            logger.error(f"Error calculating savings rate: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "savings": {"amount": 0, "rate_percent": 0}
            }

    def detect_trend(
        self,
        time_series_results,
        aggregation_name: str = "monthly_trend"
    ) -> Dict[str, Any]:
        """
        Détecte la tendance dans une série temporelle

        Utilise régression linéaire simple pour identifier:
        - Direction de la tendance (increasing, decreasing, stable)
        - Magnitude du changement moyen
        - Volatilité

        Args:
            time_series_results: SearchResults avec agrégation temporelle
            aggregation_name: Nom de l'agrégation date_histogram

        Returns:
            Dict avec analyse de tendance
        """
        try:
            logger.info(f"Detecting trend in {aggregation_name}")

            # Extraire buckets temporels
            buckets = time_series_results.aggregations.get(aggregation_name, {}).get("buckets", [])

            if len(buckets) < 3:
                return {
                    "trend": "insufficient_data",
                    "reason": "Au moins 3 périodes nécessaires pour détecter une tendance",
                    "periods_available": len(buckets)
                }

            # Extraire valeurs
            periods = [b.get("key_as_string") or b.get("key") for b in buckets]
            values = []
            for b in buckets:
                # Chercher la métrique de valeur (total_spent, total, etc.)
                value = None
                for key in ["total_spent", "total_amount", "total", "sum"]:
                    if key in b:
                        metric_data = b[key]
                        if isinstance(metric_data, dict):
                            value = metric_data.get("value", 0)
                        else:
                            value = metric_data
                        break
                values.append(value or 0)

            # Régression linéaire simple
            n = len(values)
            x = list(range(n))
            y = values

            x_mean = mean(x)
            y_mean = mean(y)

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean)**2 for i in range(n))

            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            # Déterminer tendance
            # Seuil: 5% de variation relative moyenne
            avg_value = mean(values) if values else 1
            threshold = abs(avg_value * 0.05)

            if abs(slope) < threshold:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            # Calculer volatilité (coefficient de variation)
            std_dev = self._std_dev(values)
            volatility = (std_dev / avg_value * 100) if avg_value != 0 else 0

            result = {
                "trend": trend,
                "slope": slope,
                "average_change_per_period": slope,
                "volatility_percent": volatility,
                "data": {
                    "periods": periods,
                    "values": values,
                    "average_value": avg_value
                },
                "interpretation": self._interpret_trend(trend, slope, volatility)
            }

            logger.info(f"Trend detected: {trend}, slope={slope:.2f}, volatility={volatility:.1f}%")
            return result

        except Exception as e:
            logger.error(f"Error detecting trend: {str(e)}", exc_info=True)
            return {
                "trend": "error",
                "error": str(e)
            }

    def forecast_next_period(
        self,
        time_series_results,
        aggregation_name: str = "monthly_trend",
        method: str = "moving_average"
    ) -> Dict[str, Any]:
        """
        Prévoit la prochaine période basée sur l'historique

        Méthodes disponibles:
        - moving_average: Moyenne mobile sur 3 dernières périodes
        - linear_regression: Projection linéaire

        Args:
            time_series_results: SearchResults avec agrégation temporelle
            aggregation_name: Nom de l'agrégation date_histogram
            method: Méthode de prévision

        Returns:
            Dict avec prévision et métadonnées
        """
        try:
            logger.info(f"Forecasting next period using {method}")

            # Extraire buckets
            buckets = time_series_results.aggregations.get(aggregation_name, {}).get("buckets", [])

            if len(buckets) < 3:
                return {
                    "forecast": None,
                    "reason": "Données insuffisantes (minimum 3 périodes)",
                    "periods_available": len(buckets)
                }

            # Extraire valeurs
            values = []
            for b in buckets:
                value = None
                for key in ["total_spent", "total_amount", "total", "sum"]:
                    if key in b:
                        metric_data = b[key]
                        if isinstance(metric_data, dict):
                            value = metric_data.get("value", 0)
                        else:
                            value = metric_data
                        break
                values.append(value or 0)

            # Calcul selon méthode
            if method == "moving_average":
                # Moyenne mobile sur 3 dernières périodes
                last_3_values = values[-3:]
                forecast = sum(last_3_values) / 3
                confidence = "medium"

            elif method == "linear_regression":
                # Projection linéaire
                trend_analysis = self.detect_trend(time_series_results, aggregation_name)
                slope = trend_analysis.get("slope", 0)
                last_value = values[-1]
                forecast = last_value + slope
                confidence = "low" if trend_analysis.get("volatility_percent", 0) > 30 else "medium"

            else:
                return {
                    "forecast": None,
                    "reason": f"Méthode inconnue: {method}"
                }

            # Intervalles de confiance approximatifs (±15% pour medium, ±25% pour low)
            confidence_range = 0.15 if confidence == "medium" else 0.25

            result = {
                "forecast": {
                    "value": forecast,
                    "confidence_level": confidence,
                    "range": {
                        "min": forecast * (1 - confidence_range),
                        "max": forecast * (1 + confidence_range)
                    }
                },
                "method": method,
                "based_on": {
                    "periods_count": len(values),
                    "recent_values": values[-3:],
                    "average_recent": mean(values[-3:])
                },
                "interpretation": self._interpret_forecast(forecast, values)
            }

            logger.info(f"Forecast: {forecast:.2f} (confidence: {confidence})")
            return result

        except Exception as e:
            logger.error(f"Error forecasting: {str(e)}", exc_info=True)
            return {
                "forecast": None,
                "error": str(e)
            }

    def recommend_savings_opportunities(
        self,
        by_category_results,
        target_reduction_pct: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Identifie les catégories de dépenses à réduire pour augmenter l'épargne

        Se concentre sur les catégories discrétionnaires (non-fixes)

        Args:
            by_category_results: SearchResults avec agrégation by_category
            target_reduction_pct: Pourcentage de réduction cible (default 10%)

        Returns:
            Liste de recommandations triées par potentiel d'économie
        """
        try:
            logger.info(f"Generating savings recommendations (target: {target_reduction_pct}%)")

            # Extraire buckets de catégories
            buckets = by_category_results.aggregations.get("by_category", {}).get("buckets", [])

            if not buckets:
                return []

            # Catégories discrétionnaires (où réductions possibles)
            discretionary_categories = [
                "Restaurant", "Loisirs", "achats en ligne", "Shopping",
                "Alimentation", "Transport", "Vêtements"
            ]

            # Catégories fixes (à exclure des recommandations)
            fixed_categories = [
                "Loyer", "Assurance", "Impôts", "Électricité/eau",
                "Abonnements", "Téléphones/internet"
            ]

            recommendations = []

            for bucket in buckets:
                category = bucket.get("key", "")

                # Ignorer catégories fixes
                if category in fixed_categories:
                    continue

                # Extraire métriques
                total = self._extract_bucket_metric(bucket, "total_amount")
                count = bucket.get("doc_count", 0)
                avg = self._extract_bucket_metric(bucket, "avg_transaction")

                if total == 0:
                    continue

                # Calculer potentiel d'économie
                potential_savings = total * (target_reduction_pct / 100)

                # Déterminer priorité
                is_discretionary = category in discretionary_categories
                priority = "high" if is_discretionary and total > 200 else "medium" if is_discretionary else "low"

                recommendations.append({
                    "category": category,
                    "current_spending": total,
                    "transaction_count": count,
                    "avg_per_transaction": avg,
                    "potential_savings": potential_savings,
                    "priority": priority,
                    "recommendation": self._generate_category_recommendation(
                        category, total, count, avg, potential_savings
                    )
                })

            # Trier par potentiel d'économie décroissant
            recommendations.sort(key=lambda x: x["potential_savings"], reverse=True)

            # Limiter aux top 5
            top_recommendations = recommendations[:5]

            logger.info(f"Generated {len(top_recommendations)} recommendations")
            return top_recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
            return []

    def classify_fixed_vs_variable(
        self,
        by_category_results
    ) -> Dict[str, Any]:
        """
        Classifie les dépenses en charges fixes vs variables

        Args:
            by_category_results: SearchResults avec agrégation by_category

        Returns:
            Dict avec classification et totaux
        """
        try:
            logger.info("Classifying fixed vs variable expenses")

            # Extraire buckets
            buckets = by_category_results.aggregations.get("by_category", {}).get("buckets", [])

            # Définition des catégories fixes
            fixed_categories = [
                "Loyer", "Assurance", "Impôts", "Électricité/eau",
                "Abonnements", "Téléphones/internet", "Garde d'enfants",
                "Frais scolarité"
            ]

            fixed_expenses = []
            variable_expenses = []
            fixed_total = 0
            variable_total = 0

            for bucket in buckets:
                category = bucket.get("key", "")
                total = self._extract_bucket_metric(bucket, "total_amount")
                count = bucket.get("doc_count", 0)

                expense_item = {
                    "category": category,
                    "amount": total,
                    "transaction_count": count
                }

                if category in fixed_categories:
                    fixed_expenses.append(expense_item)
                    fixed_total += total
                else:
                    variable_expenses.append(expense_item)
                    variable_total += total

            total_expenses = fixed_total + variable_total
            fixed_percent = (fixed_total / total_expenses * 100) if total_expenses != 0 else 0
            variable_percent = (variable_total / total_expenses * 100) if total_expenses != 0 else 0

            result = {
                "fixed_expenses": {
                    "categories": fixed_expenses,
                    "total": fixed_total,
                    "percent_of_total": fixed_percent,
                    "count": len(fixed_expenses)
                },
                "variable_expenses": {
                    "categories": variable_expenses,
                    "total": variable_total,
                    "percent_of_total": variable_percent,
                    "count": len(variable_expenses)
                },
                "summary": {
                    "total_expenses": total_expenses,
                    "fixed_to_variable_ratio": fixed_total / variable_total if variable_total != 0 else 0,
                    "evaluation": self._evaluate_fixed_variable_ratio(fixed_percent)
                }
            }

            logger.info(f"Classification complete: {fixed_percent:.1f}% fixed, {variable_percent:.1f}% variable")
            return result

        except Exception as e:
            logger.error(f"Error classifying expenses: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "fixed_expenses": {"total": 0},
                "variable_expenses": {"total": 0}
            }

    # === Méthodes utilitaires privées ===

    def _extract_metric(self, search_results, metric: str) -> float:
        """Extrait une métrique des résultats de recherche"""
        if not search_results or not search_results.aggregations:
            return 0.0

        aggs = search_results.aggregations

        # Chercher la métrique dans les agrégations
        if metric in aggs:
            value = aggs[metric]
            if isinstance(value, dict) and "value" in value:
                return float(value["value"] or 0)
            return float(value or 0)

        # Fallback: chercher dans les agrégations composites
        for agg_name, agg_data in aggs.items():
            if isinstance(agg_data, dict):
                if metric in agg_data:
                    value = agg_data[metric]
                    if isinstance(value, dict) and "value" in value:
                        return float(value["value"] or 0)

        return 0.0

    def _extract_bucket_metric(self, bucket: Dict, metric_name: str) -> float:
        """Extrait une métrique d'un bucket d'agrégation"""
        if metric_name in bucket:
            value = bucket[metric_name]
            if isinstance(value, dict) and "value" in value:
                return float(value["value"] or 0)
            return float(value or 0)
        return 0.0

    def _assess_significance(self, variation_pct: float) -> str:
        """Évalue la significativité d'une variation"""
        abs_var = abs(variation_pct)
        if abs_var < 5:
            return "négligeable"
        elif abs_var < 15:
            return "modérée"
        elif abs_var < 30:
            return "significative"
        else:
            return "très significative"

    def _evaluate_savings_rate(self, rate: float) -> str:
        """Évalue un taux d'épargne"""
        if rate < 0:
            return "déficitaire"
        elif rate < 5:
            return "très faible"
        elif rate < 10:
            return "faible"
        elif rate < 20:
            return "correct"
        elif rate < 30:
            return "bon"
        else:
            return "excellent"

    def _generate_savings_recommendations(self, rate: float, expense_results) -> List[str]:
        """Génère des recommandations basées sur le taux d'épargne"""
        recommendations = []

        if rate < 10:
            recommendations.append("Votre taux d'épargne est faible. Identifiez vos dépenses discrétionnaires à réduire.")

        if rate < 0:
            recommendations.append("⚠️ Vos dépenses dépassent vos revenus. Action urgente requise.")

        if rate >= 20:
            recommendations.append("✅ Excellent taux d'épargne ! Continuez ainsi.")

        return recommendations

    def _interpret_trend(self, trend: str, slope: float, volatility: float) -> str:
        """Interprète une tendance pour message utilisateur"""
        if trend == "stable":
            return "Vos dépenses sont stables sur la période analysée."
        elif trend == "increasing":
            return f"Vos dépenses sont en hausse (+{abs(slope):.2f}€ par période en moyenne)."
        else:
            return f"Vos dépenses sont en baisse (-{abs(slope):.2f}€ par période en moyenne)."

    def _interpret_forecast(self, forecast: float, historical_values: List[float]) -> str:
        """Interprète une prévision"""
        avg_historical = mean(historical_values[-3:])
        diff_pct = ((forecast - avg_historical) / avg_historical * 100) if avg_historical != 0 else 0

        if abs(diff_pct) < 5:
            return "Prévision similaire à la moyenne récente"
        elif diff_pct > 0:
            return f"Prévision en hausse de {diff_pct:.1f}% vs moyenne récente"
        else:
            return f"Prévision en baisse de {abs(diff_pct):.1f}% vs moyenne récente"

    def _generate_category_recommendation(
        self,
        category: str,
        total: float,
        count: int,
        avg: float,
        potential_savings: float
    ) -> str:
        """Génère une recommandation pour une catégorie"""
        if category == "Restaurant":
            return f"Réduire de 10% vos sorties au restaurant économiserait {potential_savings:.2f}€"
        elif category == "Loisirs":
            return f"Optimiser vos dépenses de loisirs pourrait économiser {potential_savings:.2f}€"
        elif category == "Alimentation":
            return f"Planifier vos courses et réduire le gaspillage: {potential_savings:.2f}€ d'économies"
        else:
            return f"Réduire de 10% économiserait {potential_savings:.2f}€"

    def _evaluate_fixed_variable_ratio(self, fixed_percent: float) -> str:
        """Évalue le ratio charges fixes/variables"""
        if fixed_percent < 30:
            return "Ratio sain - bonnes marges de manœuvre"
        elif fixed_percent < 50:
            return "Ratio équilibré"
        elif fixed_percent < 70:
            return "Charges fixes élevées - flexibilité limitée"
        else:
            return "⚠️ Charges fixes très élevées - peu de flexibilité budgétaire"

    def _std_dev(self, values: List[float]) -> float:
        """Calcule l'écart-type"""
        if len(values) < 2:
            return 0.0
        avg = mean(values)
        variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def analyze_multi_period_budget(
        self,
        period_results: List[Dict[str, Any]],
        period_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Analyse budgétaire sur plusieurs périodes

        Args:
            period_results: Liste de SearchResults pour chaque période
            period_labels: Labels des périodes (ex: ["2025-01", "2025-02", "2025-03"])

        Returns:
            Dict avec analyse complète multi-périodes
        """
        try:
            logger.info(f"Analyzing budget across {len(period_results)} periods")

            periods_data = []
            total_expenses_series = []
            total_income_series = []

            for idx, period_result in enumerate(period_results):
                result_obj = period_result.get("results")
                label = period_labels[idx] if idx < len(period_labels) else f"Period {idx+1}"

                # Extraire dépenses et revenus
                expenses = 0
                income = 0

                if result_obj and result_obj.aggregations:
                    # Chercher dans les agrégations
                    aggs = result_obj.aggregations

                    # Essayer d'extraire expenses
                    if "expenses" in aggs:
                        exp_data = aggs["expenses"]
                        if "total_expenses" in exp_data:
                            expenses = exp_data["total_expenses"].get("value", 0) or 0

                    # Essayer d'extraire income
                    if "income" in aggs:
                        inc_data = aggs["income"]
                        if "total_income" in inc_data:
                            income = inc_data["total_income"].get("value", 0) or 0

                    # Fallback: chercher directement
                    if expenses == 0:
                        expenses = self._extract_metric(result_obj, "total_amount")

                savings = income - expenses if income > 0 else 0
                savings_rate = (savings / income * 100) if income > 0 else 0

                periods_data.append({
                    "period": label,
                    "expenses": expenses,
                    "income": income,
                    "savings": savings,
                    "savings_rate": savings_rate
                })

                total_expenses_series.append(expenses)
                if income > 0:
                    total_income_series.append(income)

            # Calculs agrégés
            avg_expenses = mean(total_expenses_series) if total_expenses_series else 0
            avg_income = mean(total_income_series) if total_income_series else 0
            avg_savings = avg_income - avg_expenses if avg_income > 0 else 0

            # Tendances
            expense_trend = self._calculate_simple_trend(total_expenses_series)
            income_trend = self._calculate_simple_trend(total_income_series) if total_income_series else {"trend": "no_data"}

            # Identification meilleur/pire mois
            best_period = min(periods_data, key=lambda x: x["expenses"]) if periods_data else None
            worst_period = max(periods_data, key=lambda x: x["expenses"]) if periods_data else None

            result = {
                "periods": periods_data,
                "summary": {
                    "num_periods": len(period_results),
                    "average_expenses": avg_expenses,
                    "average_income": avg_income,
                    "average_savings": avg_savings,
                    "total_expenses": sum(total_expenses_series),
                    "total_income": sum(total_income_series)
                },
                "trends": {
                    "expenses": expense_trend,
                    "income": income_trend
                },
                "insights": {
                    "best_period": best_period,
                    "worst_period": worst_period,
                    "volatility": self._std_dev(total_expenses_series) if len(total_expenses_series) > 1 else 0
                }
            }

            logger.info(f"Multi-period analysis complete: {len(periods_data)} periods analyzed")
            return result

        except Exception as e:
            logger.error(f"Error in multi-period analysis: {str(e)}", exc_info=True)
            return {"error": str(e), "periods": []}

    def _calculate_simple_trend(self, values: List[float]) -> Dict[str, Any]:
        """
        Calcule une tendance simple sur une série de valeurs

        Args:
            values: Série de valeurs

        Returns:
            Dict avec tendance et pente
        """
        if len(values) < 2:
            return {"trend": "insufficient_data", "slope": 0}

        # Régression linéaire simple
        n = len(values)
        x = list(range(n))
        y = values

        x_mean = mean(x)
        y_mean = mean(y)

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean)**2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Déterminer tendance
        avg_value = mean(values)
        threshold = abs(avg_value * 0.05)

        if abs(slope) < threshold:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            "trend": trend,
            "slope": slope,
            "change_per_period": slope
        }

    def identify_spending_patterns(
        self,
        by_category_results,
        time_series_results
    ) -> Dict[str, Any]:
        """
        Identifie les patterns de dépenses

        Args:
            by_category_results: Résultats agrégés par catégorie
            time_series_results: Résultats de série temporelle

        Returns:
            Dict avec patterns identifiés
        """
        try:
            logger.info("Identifying spending patterns")

            patterns = {
                "top_categories": [],
                "recurring_expenses": [],
                "irregular_expenses": [],
                "seasonal_patterns": []
            }

            # Extraire top catégories
            if by_category_results and by_category_results.aggregations:
                buckets = by_category_results.aggregations.get("by_category", {}).get("buckets", [])

                for bucket in buckets[:10]:
                    category = bucket.get("key", "")
                    total = self._extract_bucket_metric(bucket, "total_amount")
                    count = bucket.get("doc_count", 0)
                    avg = self._extract_bucket_metric(bucket, "avg_transaction")

                    # Classifier comme récurrent si transactions régulières
                    is_recurring = count > 0 and (avg / total > 0.8) if total > 0 else False

                    pattern_item = {
                        "category": category,
                        "total": total,
                        "count": count,
                        "avg_per_transaction": avg,
                        "pattern_type": "recurring" if is_recurring else "irregular"
                    }

                    patterns["top_categories"].append(pattern_item)

                    if is_recurring:
                        patterns["recurring_expenses"].append(pattern_item)
                    else:
                        patterns["irregular_expenses"].append(pattern_item)

            logger.info(f"Identified {len(patterns['top_categories'])} spending patterns")
            return patterns

        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def calculate_budget_health_score(
        self,
        income: float,
        expenses: float,
        fixed_expenses: float,
        discretionary_expenses: float
    ) -> Dict[str, Any]:
        """
        Calcule un score de santé budgétaire (0-100)

        Critères:
        - Taux d'épargne (30 points)
        - Ratio charges fixes / revenus (25 points)
        - Ratio dépenses discrétionnaires / revenus (25 points)
        - Marge de sécurité (20 points)

        Args:
            income: Revenus totaux
            expenses: Dépenses totales
            fixed_expenses: Charges fixes
            discretionary_expenses: Dépenses discrétionnaires

        Returns:
            Dict avec score et détails
        """
        try:
            if income <= 0:
                return {
                    "score": 0,
                    "grade": "N/A",
                    "reason": "Revenus insuffisants pour calculer le score"
                }

            # Critère 1: Taux d'épargne (30 points)
            savings = income - expenses
            savings_rate = (savings / income * 100)

            if savings_rate >= 30:
                savings_score = 30
            elif savings_rate >= 20:
                savings_score = 25
            elif savings_rate >= 10:
                savings_score = 20
            elif savings_rate >= 5:
                savings_score = 15
            elif savings_rate >= 0:
                savings_score = 10
            else:
                savings_score = 0

            # Critère 2: Ratio charges fixes / revenus (25 points)
            fixed_ratio = (fixed_expenses / income * 100)

            if fixed_ratio <= 30:
                fixed_score = 25
            elif fixed_ratio <= 50:
                fixed_score = 20
            elif fixed_ratio <= 60:
                fixed_score = 15
            elif fixed_ratio <= 70:
                fixed_score = 10
            else:
                fixed_score = 5

            # Critère 3: Ratio dépenses discrétionnaires / revenus (25 points)
            discretionary_ratio = (discretionary_expenses / income * 100)

            if discretionary_ratio <= 20:
                discretionary_score = 25
            elif discretionary_ratio <= 30:
                discretionary_score = 20
            elif discretionary_ratio <= 40:
                discretionary_score = 15
            elif discretionary_ratio <= 50:
                discretionary_score = 10
            else:
                discretionary_score = 5

            # Critère 4: Marge de sécurité (20 points)
            # Équivalent 3 mois de dépenses fixes
            security_margin = savings / (fixed_expenses * 3) if fixed_expenses > 0 else 0

            if security_margin >= 1.0:
                security_score = 20
            elif security_margin >= 0.5:
                security_score = 15
            elif security_margin >= 0.25:
                security_score = 10
            else:
                security_score = 5

            # Score total
            total_score = savings_score + fixed_score + discretionary_score + security_score

            # Grade
            if total_score >= 85:
                grade = "A"
                evaluation = "Excellent"
            elif total_score >= 70:
                grade = "B"
                evaluation = "Bon"
            elif total_score >= 55:
                grade = "C"
                evaluation = "Correct"
            elif total_score >= 40:
                grade = "D"
                evaluation = "À améliorer"
            else:
                grade = "F"
                evaluation = "Critique"

            result = {
                "score": total_score,
                "grade": grade,
                "evaluation": evaluation,
                "breakdown": {
                    "savings_rate": {
                        "score": savings_score,
                        "max": 30,
                        "value": savings_rate
                    },
                    "fixed_expenses": {
                        "score": fixed_score,
                        "max": 25,
                        "value": fixed_ratio
                    },
                    "discretionary_expenses": {
                        "score": discretionary_score,
                        "max": 25,
                        "value": discretionary_ratio
                    },
                    "security_margin": {
                        "score": security_score,
                        "max": 20,
                        "value": security_margin
                    }
                },
                "recommendations": self._generate_health_score_recommendations(
                    savings_score, fixed_score, discretionary_score, security_score
                )
            }

            logger.info(f"Budget health score: {total_score}/100 ({grade})")
            return result

        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}", exc_info=True)
            return {"error": str(e), "score": 0}

    def _generate_health_score_recommendations(
        self,
        savings_score: float,
        fixed_score: float,
        discretionary_score: float,
        security_score: float
    ) -> List[str]:
        """Génère des recommandations basées sur les scores"""
        recommendations = []

        if savings_score < 15:
            recommendations.append("⚠️ Taux d'épargne insuffisant. Visez au moins 10% de vos revenus.")

        if fixed_score < 15:
            recommendations.append("⚠️ Charges fixes trop élevées. Cherchez à renégocier certains contrats.")

        if discretionary_score < 15:
            recommendations.append("💡 Dépenses discrétionnaires élevées. Identifiez les catégories à réduire.")

        if security_score < 10:
            recommendations.append("🛡️ Constituez un fonds d'urgence équivalent à 3-6 mois de dépenses fixes.")

        if not recommendations:
            recommendations.append("✅ Votre santé budgétaire est bonne. Maintenez vos bonnes habitudes !")

        return recommendations
