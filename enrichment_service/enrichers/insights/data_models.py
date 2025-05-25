"""
Modèles de données pour les insights financiers.

Ce module contient les structures de données utilisées pour représenter
les insights financiers et leurs composants.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

class InsightType(Enum):
    """Types d'insights financiers."""
    SPENDING_INCREASE = "spending_increase"
    SPENDING_DECREASE = "spending_decrease"
    CATEGORY_CONCENTRATION = "category_concentration"
    SAVINGS_IMPROVEMENT = "savings_improvement"
    LOW_SAVINGS_RATE = "low_savings_rate"
    EXPENSE_TREND_INCREASE = "expense_trend_increase"
    INCOME_TREND_INCREASE = "income_trend_increase"
    LARGE_TRANSACTION = "large_transaction"
    SUBSCRIPTION_ANALYSIS = "subscription_analysis"
    SAVINGS_OPPORTUNITY = "savings_opportunity"
    INCOME_VARIABILITY = "income_variability"
    SPENDING_PATTERN = "spending_pattern"
    BUDGET_ALERT = "budget_alert"
    ANOMALY_DETECTED = "anomaly_detected"

class TimeScope(Enum):
    """Portée temporelle des insights."""
    SHORT_TERM = "short_term"  # 1-7 jours
    MEDIUM_TERM = "medium_term"  # 1-3 mois
    LONG_TERM = "long_term"  # 3+ mois

class FinancialScope(Enum):
    """Portée financière des insights."""
    GENERAL = "general"
    SPENDING = "spending"
    SAVING = "saving"
    INCOME = "income"
    BUDGETING = "budgeting"
    INVESTMENT = "investment"

class Priority(Enum):
    """Niveaux de priorité des insights."""
    LOW = 1
    MEDIUM = 2
    NORMAL = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class InsightMetrics:
    """Métriques associées à un insight."""
    numerical_value: Optional[float] = None
    comparative_value: Optional[float] = None
    change_percentage: Optional[float] = None
    confidence_score: float = 0.0
    potential_impact: Optional[float] = None

@dataclass
class InsightContext:
    """Contexte d'un insight."""
    categories_concerned: List[str] = field(default_factory=list)
    merchants_concerned: List[str] = field(default_factory=list)
    accounts_concerned: List[str] = field(default_factory=list)
    related_transactions: List[int] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)

@dataclass
class InsightAction:
    """Action recommandée pour un insight."""
    action_text: str
    priority: str = "medium"  # low, medium, high
    estimated_impact: Optional[str] = None
    difficulty: str = "easy"  # easy, medium, hard

@dataclass
class FinancialInsight:
    """Insight financier complet."""
    insight_id: str
    user_id: int
    insight_type: InsightType
    title: str
    description: str
    highlight: str
    
    # Portée
    time_scope: TimeScope
    financial_scope: FinancialScope
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    # Importance
    priority: Priority = Priority.NORMAL
    
    # Métriques
    metrics: InsightMetrics = field(default_factory=InsightMetrics)
    
    # Contexte
    context: InsightContext = field(default_factory=InsightContext)
    
    # Actions recommandées
    suggested_actions: List[InsightAction] = field(default_factory=list)
    
    # Métadonnées
    generation_method: str = "rule_based"  # rule_based, ml, hybrid
    narrative: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Dates et statut
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    status: str = "active"  # active, read, dismissed

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'insight en dictionnaire pour le stockage."""
        return {
            "id": self.insight_id,
            "user_id": self.user_id,
            "insight_type": self.insight_type.value,
            "title": self.title,
            "description": self.description,
            "highlight": self.highlight,
            
            # Portée
            "time_scope": self.time_scope.value,
            "financial_scope": self.financial_scope.value,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            
            # Importance
            "priority": self.priority.value,
            
            # Métriques
            "numerical_value": self.metrics.numerical_value,
            "comparative_value": self.metrics.comparative_value,
            "change_percentage": self.metrics.change_percentage,
            "confidence_score": self.metrics.confidence_score,
            "potential_impact": self.metrics.potential_impact,
            
            # Contexte
            "categories_concerned": self.context.categories_concerned,
            "merchants_concerned": self.context.merchants_concerned,
            "accounts_concerned": self.context.accounts_concerned,
            "related_transactions": self.context.related_transactions,
            "related_patterns": self.context.related_patterns,
            
            # Actions
            "suggested_actions": [
                {
                    "action_text": action.action_text,
                    "priority": action.priority,
                    "estimated_impact": action.estimated_impact,
                    "difficulty": action.difficulty
                }
                for action in self.suggested_actions
            ],
            
            # Métadonnées
            "generation_method": self.generation_method,
            "narrative": self.narrative,
            "tags": self.tags,
            
            # Dates et statut
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialInsight':
        """Crée un insight depuis un dictionnaire."""
        metrics = InsightMetrics(
            numerical_value=data.get("numerical_value"),
            comparative_value=data.get("comparative_value"),
            change_percentage=data.get("change_percentage"),
            confidence_score=data.get("confidence_score", 0.0),
            potential_impact=data.get("potential_impact")
        )
        
        context = InsightContext(
            categories_concerned=data.get("categories_concerned", []),
            merchants_concerned=data.get("merchants_concerned", []),
            accounts_concerned=data.get("accounts_concerned", []),
            related_transactions=data.get("related_transactions", []),
            related_patterns=data.get("related_patterns", [])
        )
        
        suggested_actions = []
        for action_data in data.get("suggested_actions", []):
            if isinstance(action_data, str):
                # Support pour l'ancien format (liste de strings)
                suggested_actions.append(InsightAction(action_text=action_data))
            else:
                suggested_actions.append(InsightAction(
                    action_text=action_data.get("action_text", ""),
                    priority=action_data.get("priority", "medium"),
                    estimated_impact=action_data.get("estimated_impact"),
                    difficulty=action_data.get("difficulty", "easy")
                ))
        
        insight = cls(
            insight_id=data["id"],
            user_id=data["user_id"],
            insight_type=InsightType(data["insight_type"]),
            title=data["title"],
            description=data["description"],
            highlight=data["highlight"],
            time_scope=TimeScope(data["time_scope"]),
            financial_scope=FinancialScope(data["financial_scope"]),
            priority=Priority(data.get("priority", 3)),
            metrics=metrics,
            context=context,
            suggested_actions=suggested_actions,
            generation_method=data.get("generation_method", "rule_based"),
            narrative=data.get("narrative", ""),
            tags=data.get("tags", []),
            status=data.get("status", "active")
        )
        
        # Dates
        if data.get("created_at"):
            insight.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            insight.expires_at = datetime.fromisoformat(data["expires_at"])
        if data.get("read_at"):
            insight.read_at = datetime.fromisoformat(data["read_at"])
        if data.get("period_start"):
            insight.period_start = datetime.fromisoformat(data["period_start"])
        if data.get("period_end"):
            insight.period_end = datetime.fromisoformat(data["period_end"])
        
        return insight

@dataclass
class InsightTemplate:
    """Template pour générer des insights standardisés."""
    insight_type: InsightType
    title_template: str
    description_template: str
    highlight_template: str
    time_scope: TimeScope
    financial_scope: FinancialScope
    default_priority: Priority
    action_templates: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

# Templates prédéfinis pour les insights courants
INSIGHT_TEMPLATES = {
    InsightType.SPENDING_INCREASE: InsightTemplate(
        insight_type=InsightType.SPENDING_INCREASE,
        title_template="Augmentation significative des dépenses",
        description_template="Vos dépenses ont augmenté de {change_percentage:.1f}% ce mois-ci",
        highlight_template="+{change_percentage:.1f}% vs moyenne",
        time_scope=TimeScope.SHORT_TERM,
        financial_scope=FinancialScope.SPENDING,
        default_priority=Priority.HIGH,
        action_templates=[
            "Examiner les dépenses récentes pour identifier les causes",
            "Réviser votre budget pour le mois suivant",
            "Considérer reporter les achats non essentiels"
        ],
        tags=["dépenses", "augmentation", "budget"]
    ),
    
    InsightType.LOW_SAVINGS_RATE: InsightTemplate(
        insight_type=InsightType.LOW_SAVINGS_RATE,
        title_template="Taux d'épargne à améliorer",
        description_template="Votre taux d'épargne de {savings_rate:.1f}% peut être optimisé",
        highlight_template="Seulement {net_amount:.2f}€ épargnés",
        time_scope=TimeScope.SHORT_TERM,
        financial_scope=FinancialScope.SAVING,
        default_priority=Priority.HIGH,
        action_templates=[
            "Examiner vos dépenses pour identifier des économies",
            "Fixer un objectif d'épargne mensuel (recommandé: 10-20%)",
            "Automatiser votre épargne via un virement programmé"
        ],
        tags=["épargne", "amélioration", "objectif"]
    ),
    
    InsightType.LARGE_TRANSACTION: InsightTemplate(
        insight_type=InsightType.LARGE_TRANSACTION,
        title_template="Transaction importante détectée",
        description_template="Transaction de {amount:.2f}€ inhabituellement élevée",
        highlight_template="{amount:.2f}€",
        time_scope=TimeScope.SHORT_TERM,
        financial_scope=FinancialScope.SPENDING,
        default_priority=Priority.NORMAL,
        action_templates=[
            "Vérifier que cette transaction est correcte",
            "S'assurer qu'elle s'inscrit dans votre budget",
            "Ajuster votre budget si nécessaire"
        ],
        tags=["anomalie", "transaction_importante"]
    ),
    
    InsightType.SUBSCRIPTION_ANALYSIS: InsightTemplate(
        insight_type=InsightType.SUBSCRIPTION_ANALYSIS,
        title_template="Analyse de vos abonnements",
        description_template="~{monthly_cost:.0f}€/mois en abonnements détectés ({count} services)",
        highlight_template="{monthly_cost:.0f}€/mois d'abonnements",
        time_scope=TimeScope.MEDIUM_TERM,
        financial_scope=FinancialScope.SPENDING,
        default_priority=Priority.NORMAL,
        action_templates=[
            "Faire le point sur vos abonnements actifs",
            "Résilier les services non utilisés",
            "Considérer des alternatives moins coûteuses"
        ],
        tags=["abonnements", "optimisation", "économies"]
    )
}

@dataclass
class InsightAnalytics:
    """Analytics des insights d'un utilisateur."""
    total_insights: int = 0
    insights_by_type: Dict[str, int] = field(default_factory=dict)
    insights_by_priority: Dict[int, int] = field(default_factory=dict)
    insights_by_scope: Dict[str, int] = field(default_factory=dict)
    engagement_rate: float = 0.0
    average_confidence: float = 0.0
    total_potential_impact: float = 0.0
    most_common_type: Optional[str] = None
    period_days: int = 30

@dataclass
class InsightRecommendation:
    """Recommandation personnalisée basée sur les insights."""
    recommendation_type: str
    priority: str
    title: str
    description: str
    actions: List[str]
    estimated_impact: str
    affected_insights: List[str] = field(default_factory=list)