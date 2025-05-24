"""
Générateur de narratifs pour les résumés financiers.

Ce module contient la logique pour générer des descriptions
textuelles et des narratifs pour les résumés financiers.
"""

import logging
from typing import List
from enrichment_service.enrichers.summary.data_models import FinancialSummary

logger = logging.getLogger(__name__)

class SummaryNarrator:
    """
    Générateur de narratifs pour les résumés financiers.
    
    Cette classe génère des descriptions textuelles et des highlights
    pour rendre les résumés plus compréhensibles.
    """
    
    def generate_monthly_narrative(self, summary: FinancialSummary):
        """
        Génère le narratif pour un résumé mensuel.
        
        Args:
            summary: Résumé à enrichir
        """
        highlights = []
        
        # Performance générale
        if summary.net_flow > 0:
            highlights.append(f"Mois positif avec {summary.net_flow:.2f}€ d'épargne nette")
        else:
            highlights.append(f"Mois déficitaire de {abs(summary.net_flow):.2f}€")
        
        # Top catégorie de dépenses
        if summary.top_categories:
            top_expense_cat = next((cat for cat in summary.top_categories if cat.category_name in summary.expense_breakdown), None)
            if top_expense_cat:
                highlights.append(f"Principale dépense: {top_expense_cat.category_name} ({top_expense_cat.total_amount:.2f}€)")
        
        # Dépenses récurrentes
        if summary.recurring_percentage > 0:
            highlights.append(f"Dépenses récurrentes: {summary.recurring_percentage:.1f}% du total ({summary.recurring_spending:.2f}€)")
        
        # Comparaison avec mois précédent
        if summary.vs_previous_period:
            expense_change = summary.vs_previous_period.get("expense_change", 0)
            if abs(expense_change) > 10:
                direction = "hausse" if expense_change > 0 else "baisse"
                highlights.append(f"Dépenses en {direction} de {abs(expense_change):.1f}% vs mois précédent")
        
        # Anomalies
        if summary.anomalies:
            highlights.append(f"{len(summary.anomalies)} transaction(s) inhabituelle(s) détectée(s)")
        
        summary.narrative_highlights = highlights
        summary.summary_text = self._create_narrative_text(summary, "mensuel")
        summary.tags = self._generate_summary_tags(summary)
    
    def generate_quarterly_narrative(self, summary: FinancialSummary):
        """
        Génère le narratif pour un résumé trimestriel.
        
        Args:
            summary: Résumé à enrichir
        """
        highlights = []
        
        # Performance trimestrielle
        monthly_avg = summary.net_flow / 3
        if monthly_avg > 0:
            highlights.append(f"Trimestre positif avec une épargne moyenne de {monthly_avg:.2f}€/mois")
        else:
            highlights.append(f"Trimestre déficitaire avec un déficit moyen de {abs(monthly_avg):.2f}€/mois")
        
        # Évolution sur le trimestre
        if summary.transaction_count > 0:
            avg_transaction = (summary.total_income + summary.total_expenses) / summary.transaction_count
            highlights.append(f"Montant moyen par transaction: {avg_transaction:.2f}€")
        
        # Santé financière
        health_status = summary.financial_health_indicators.get("savings_status", "unknown")
        if health_status != "unknown":
            highlights.append(f"Santé financière: {health_status}")
        
        summary.narrative_highlights = highlights
        summary.summary_text = self._create_narrative_text(summary, "trimestriel")
        summary.tags = self._generate_summary_tags(summary)
    
    def generate_yearly_narrative(self, summary: FinancialSummary):
        """
        Génère le narratif pour un résumé annuel.
        
        Args:
            summary: Résumé à enrichir
        """
        highlights = []
        
        # Bilan annuel
        highlights.append(f"Bilan annuel: {summary.total_income:.2f}€ de revenus, {summary.total_expenses:.2f}€ de dépenses")
        
        if summary.savings_rate > 0:
            highlights.append(f"Taux d'épargne annuel: {summary.savings_rate:.1f}%")
        
        # Évolution mensuelle moyenne
        monthly_income = summary.total_income / 12
        monthly_expenses = summary.total_expenses / 12
        highlights.append(f"Moyennes mensuelles: {monthly_income:.2f}€ revenus, {monthly_expenses:.2f}€ dépenses")
        
        # Répartition des dépenses
        if summary.expense_breakdown:
            top_category = max(summary.expense_breakdown.items(), key=lambda x: x[1])
            percentage = (top_category[1] / summary.total_expenses) * 100 if summary.total_expenses > 0 else 0
            highlights.append(f"Principale catégorie: {top_category[0]} ({percentage:.1f}% des dépenses)")
        
        summary.narrative_highlights = highlights
        summary.summary_text = self._create_narrative_text(summary, "annuel")
        summary.tags = self._generate_summary_tags(summary)
    
    def _create_narrative_text(self, summary: FinancialSummary, period_type: str) -> str:
        """
        Crée un texte narratif complet pour le résumé.
        
        Args:
            summary: Résumé financier
            period_type: Type de période (mensuel, trimestriel, annuel)
            
        Returns:
            str: Texte narratif
        """
        narrative_parts = [
            f"Résumé {period_type} pour {summary.period_name}:",
            f"Total des revenus: {summary.total_income:.2f}€",
            f"Total des dépenses: {summary.total_expenses:.2f}€",
            f"Solde net: {summary.net_flow:.2f}€"
        ]
        
        if summary.savings_rate != 0:
            narrative_parts.append(f"Taux d'épargne: {summary.savings_rate:.1f}%")
        
        # Ajouter les points saillants
        if summary.narrative_highlights:
            narrative_parts.append("Points saillants:")
            for highlight in summary.narrative_highlights[:3]:  # Limiter à 3 points
                narrative_parts.append(f"• {highlight}")
        
        # Ajouter les indicateurs de santé
        savings_status = summary.financial_health_indicators.get("savings_status")
        if savings_status:
            status_texts = {
                "excellent": "Excellente capacité d'épargne",
                "good": "Bonne gestion financière",
                "fair": "Gestion financière correcte",
                "concerning": "Situation financière à surveiller"
            }
            narrative_parts.append(f"Évaluation: {status_texts.get(savings_status, '')}")
        
        return " | ".join(narrative_parts)
    
    def _generate_summary_tags(self, summary: FinancialSummary) -> List[str]:
        """
        Génère des tags pour le résumé.
        
        Args:
            summary: Résumé financier
            
        Returns:
            List[str]: Liste de tags
        """
        tags = [summary.period_type]
        
        # Tags basés sur la performance
        if summary.net_flow > 0:
            tags.append("épargne_positive")
        else:
            tags.append("déficit")
        
        # Tags basés sur le taux d'épargne
        if summary.savings_rate >= 20:
            tags.append("épargnant_excellent")
        elif summary.savings_rate >= 10:
            tags.append("bon_épargnant")
        elif summary.savings_rate < 0:
            tags.append("dépenses_excessives")
        
        # Tags basés sur l'activité
        activity_level = summary.financial_health_indicators.get("activity_level")
        if activity_level:
            tags.append(f"activité_{activity_level}")
        
        # Tags basés sur les dépenses récurrentes
        if summary.recurring_percentage > 50:
            tags.append("dépenses_rigides")
        elif summary.recurring_percentage < 30:
            tags.append("dépenses_flexibles")
        
        # Tags basés sur les anomalies
        if summary.anomalies:
            tags.append("anomalies_détectées")
        
        # Tags basés sur les changements
        if summary.vs_previous_period:
            expense_change = summary.vs_previous_period.get("expense_change", 0)
            if expense_change > 20:
                tags.append("hausse_dépenses")
            elif expense_change < -20:
                tags.append("baisse_dépenses")
        
        return tags