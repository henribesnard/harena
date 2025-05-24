"""
Modèles de données pour les résumés financiers.

Ce module contient les structures de données utilisées pour représenter
les résumés financiers et leurs composants.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

@dataclass
class CategorySummary:
    """Résumé d'une catégorie de dépenses/revenus."""
    category_id: int
    category_name: str
    total_amount: float
    transaction_count: int
    percentage: float
    avg_amount: float
    trend_vs_previous: Optional[float] = None

@dataclass
class MerchantSummary:
    """Résumé d'un marchand."""
    merchant_name: str
    total_amount: float
    transaction_count: int
    percentage: float
    avg_amount: float
    category: Optional[str] = None

@dataclass
class FinancialSummary:
    """Résumé financier complet pour une période."""
    summary_id: str
    user_id: int
    period_type: str
    period_name: str
    period_start: datetime
    period_end: datetime
    is_complete: bool
    
    # Métriques globales
    total_income: float = 0.0
    total_expenses: float = 0.0
    net_flow: float = 0.0
    savings_rate: float = 0.0
    transaction_count: int = 0
    
    # Répartitions
    income_breakdown: Dict[str, float] = field(default_factory=dict)
    expense_breakdown: Dict[str, float] = field(default_factory=dict)
    top_categories: List[CategorySummary] = field(default_factory=list)
    top_merchants: List[MerchantSummary] = field(default_factory=list)
    
    # Dépenses récurrentes
    recurring_spending: float = 0.0
    recurring_percentage: float = 0.0
    
    # Comparaisons
    vs_previous_period: Dict[str, Any] = field(default_factory=dict)
    vs_average: Dict[str, Any] = field(default_factory=dict)
    significant_changes: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Narratif
    narrative_highlights: List[str] = field(default_factory=list)
    financial_health_indicators: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""
    tags: List[str] = field(default_factory=list)