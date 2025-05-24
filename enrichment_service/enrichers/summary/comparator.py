"""
Comparateur de résumés financiers.

Ce module contient la logique pour comparer les résumés financiers
entre différentes périodes et identifier les changements significatifs.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from enrichment_service.enrichers.summary.data_models import FinancialSummary

logger = logging.getLogger(__name__)

class SummaryComparator:
    """
    Comparateur de résumés financiers.
    
    Cette classe compare les résumés entre différentes périodes
    et identifie les changements significatifs.
    """
    
    def __init__(self, summary_generator):
        """
        Initialise le comparateur.
        
        Args:
            summary_generator: Instance du générateur de résumés
        """
        self.summary_generator = summary_generator
    
    async def add_monthly_comparisons(self, summary: FinancialSummary, user_id: int, year: int, month: int):
        """
        Ajoute les comparaisons pour un résumé mensuel.
        
        Args:
            summary: Résumé à enrichir
            user_id: ID de l'utilisateur
            year: Année
            month: Mois
        """
        # Comparaison avec le mois précédent
        if month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, month - 1
        
        try:
            prev_summary = await self.summary_generator.generate_monthly_summary(user_id, prev_year, prev_month)
            
            summary.vs_previous_period = {
                "income_change": self._calculate_change(summary.total_income, prev_summary.total_income),
                "expense_change": self._calculate_change(summary.total_expenses, prev_summary.total_expenses),
                "net_flow_change": self._calculate_change(summary.net_flow, prev_summary.net_flow),
                "savings_rate_change": summary.savings_rate - prev_summary.savings_rate
            }
            
            # Identifier les changements significatifs
            if abs(summary.vs_previous_period["expense_change"]) > 20:
                change_type = "increase" if summary.vs_previous_period["expense_change"] > 0 else "decrease"
                summary.significant_changes.append({
                    "type": f"expense_{change_type}",
                    "description": f"Dépenses {change_type} de {abs(summary.vs_previous_period['expense_change']):.1f}% vs mois précédent",
                    "impact": "significant" if abs(summary.vs_previous_period["expense_change"]) > 50 else "moderate"
                })
                
        except Exception as e:
            logger.warning(f"Impossible de comparer avec le mois précédent: {e}")
    
    async def add_quarterly_comparisons(self, summary: FinancialSummary, user_id: int, year: int, quarter: int):
        """
        Ajoute les comparaisons pour un résumé trimestriel.
        
        Args:
            summary: Résumé à enrichir
            user_id: ID de l'utilisateur
            year: Année
            quarter: Trimestre
        """
        # Comparaison avec le trimestre précédent
        if quarter == 1:
            prev_year, prev_quarter = year - 1, 4
        else:
            prev_year, prev_quarter = year, quarter - 1
        
        try:
                        
            summary.vs_previous_period = {
                "income_change": self._calculate_change(summary.total_income, prev_summary.total_income),
                "expense_change": self._calculate_change(summary.total_expenses, prev_summary.total_expenses),
                "net_flow_change": self._calculate_change(summary.net_flow, prev_summary.net_flow),
                "savings_rate_change": summary.savings_rate - prev_summary.savings_rate
            }
            
        except Exception as e:
            logger.warning(f"Impossible de comparer avec le trimestre précédent: {e}")
    
    async def add_yearly_comparisons(self, summary: FinancialSummary, user_id: int, year: int):
        """
        Ajoute les comparaisons pour un résumé annuel.
        
        Args:
            summary: Résumé à enrichir
            user_id: ID de l'utilisateur
            year: Année
        """
        # Comparaison avec l'année précédente
        try:
            prev_summary = await self.summary_generator.generate_yearly_summary(user_id, year - 1)
            
            summary.vs_previous_period = {
                "income_change": self._calculate_change(summary.total_income, prev_summary.total_income),
                "expense_change": self._calculate_change(summary.total_expenses, prev_summary.total_expenses),
                "net_flow_change": self._calculate_change(summary.net_flow, prev_summary.net_flow),
                "savings_rate_change": summary.savings_rate - prev_summary.savings_rate
            }
            
        except Exception as e:
            logger.warning(f"Impossible de comparer avec l'année précédente: {e}")
    
    def _calculate_change(self, current: float, previous: float) -> float:
        """
        Calcule le pourcentage de changement entre deux valeurs.
        
        Args:
            current: Valeur actuelle
            previous: Valeur précédente
            
        Returns:
            float: Pourcentage de changement
        """
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        
        return ((current - previous) / abs(previous)) * 100