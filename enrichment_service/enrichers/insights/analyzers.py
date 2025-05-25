"""
Analyseurs spécialisés pour la génération d'insights financiers.

Ce module contient les analyseurs spécialisés qui examinent les données
financières pour identifier des patterns et générer des insights spécifiques.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from sqlalchemy.orm import Session

from enrichment_service.db.models import RawTransaction, BridgeCategory
from enrichment_service.enrichers.insights.data_models import (
    FinancialInsight, InsightType, TimeScope, FinancialScope, Priority,
    InsightMetrics, InsightContext, InsightAction, INSIGHT_TEMPLATES
)

logger = logging.getLogger(__name__)

class BaseInsightAnalyzer:
    """Analyseur de base pour les insights."""
    
    def __init__(self, db: Session):
        """
        Initialise l'analyseur.
        
        Args:
            db: Session de base de données
        """
        self.db = db
        self.significant_change_threshold = 0.20  # 20%
        self.large_transaction_threshold = 200.0  # euros
        
        # Cache des catégories
        self._categories_cache = {}
        self._load_categories_cache()
    
    def _load_categories_cache(self):
        """Charge les catégories en cache."""
        try:
            categories = self.db.query(BridgeCategory).all()
            self._categories_cache = {cat.bridge_category_id: cat for cat in categories}
        except Exception as e:
            logger.warning(f"Impossible de charger le cache des catégories: {e}")
    
    def _calculate_change_percentage(self, current: float, previous: float) -> float:
        """Calcule le pourcentage de changement entre deux valeurs."""
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return ((current - previous) / abs(previous)) * 100
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calcule la tendance d'une série de valeurs."""
        if len(values) < 2:
            return {"slope": 0.0, "confidence": 0.0}
        
        n = len(values)
        x_values = list(range(n))
        
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return {"slope": 0.0, "confidence": 0.0}
        
        slope = numerator / denominator
        
        # Calculer le coefficient de corrélation pour la confiance
        if len(values) > 2:
            try:
                correlation = abs(statistics.correlation(x_values, values))
                confidence = min(1.0, correlation)
            except:
                confidence = 0.5
        else:
            confidence = 0.5
        
        return {"slope": slope, "confidence": confidence}

class SpendingAnalyzer(BaseInsightAnalyzer):
    """Analyseur spécialisé dans les insights de dépenses."""
    
    async def analyze_spending_patterns(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse les patterns de dépenses et génère des insights.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights générés
        """
        insights = []
        
        # Analyser l'augmentation des dépenses
        insights.extend(await self._analyze_spending_increase(
            user_id, recent_transactions, historical_transactions
        ))
        
        # Analyser la concentration par catégorie
        insights.extend(await self._analyze_category_concentration(
            user_id, recent_transactions
        ))
        
        # Analyser les patterns journaliers
        insights.extend(await self._analyze_daily_patterns(
            user_id, recent_transactions
        ))
        
        return insights
    
    async def _analyze_spending_increase(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse l'augmentation des dépenses."""
        insights = []
        
        recent_expenses = [t for t in recent_transactions if t.amount < 0]
        historical_expenses = [t for t in historical_transactions if t.amount < 0]
        
        if not recent_expenses or len(historical_expenses) < 30:
            return insights
        
        recent_total = sum(abs(t.amount) for t in recent_expenses)
        historical_monthly_avg = sum(abs(t.amount) for t in historical_expenses) / 3  # 3 mois
        
        if recent_total > historical_monthly_avg * (1 + self.significant_change_threshold):
            change_pct = self._calculate_change_percentage(recent_total, historical_monthly_avg)
            
            template = INSIGHT_TEMPLATES[InsightType.SPENDING_INCREASE]
            
            metrics = InsightMetrics(
                numerical_value=recent_total,
                comparative_value=historical_monthly_avg,
                change_percentage=change_pct,
                confidence_score=0.9
            )
            
            actions = [
                InsightAction(action_text=template.action_templates[0], priority="high"),
                InsightAction(action_text=template.action_templates[1], priority="medium"),
                InsightAction(action_text=template.action_templates[2], priority="medium")
            ]
            
            insight = FinancialInsight(
                insight_id=f"spending_increase_{user_id}_{datetime.now().strftime('%Y%m')}",
                user_id=user_id,
                insight_type=InsightType.SPENDING_INCREASE,
                title=template.title_template,
                description=template.description_template.format(change_percentage=change_pct),
                highlight=template.highlight_template.format(change_percentage=change_pct),
                time_scope=template.time_scope,
                financial_scope=template.financial_scope,
                priority=template.default_priority,
                metrics=metrics,
                suggested_actions=actions,
                narrative=f"Vos dépenses de {recent_total:.2f}€ ce mois dépassent votre moyenne de {historical_monthly_avg:.2f}€",
                tags=template.tags.copy()
            )
            
            insights.append(insight)
        
        return insights
    
    async def _analyze_category_concentration(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse la concentration des dépenses par catégorie."""
        insights = []
        
        recent_expenses = [t for t in recent_transactions if t.amount < 0]
        if not recent_expenses:
            return insights
        
        recent_total = sum(abs(t.amount) for t in recent_expenses)
        category_expenses = defaultdict(float)
        
        for transaction in recent_expenses:
            category_id = transaction.category_id or 0
            category_expenses[category_id] += abs(transaction.amount)
        
        # Identifier la catégorie avec le plus de dépenses
        if category_expenses:
            top_category_id = max(category_expenses.items(), key=lambda x: x[1])[0]
            top_category_amount = category_expenses[top_category_id]
            
            category_name = "Non catégorisé"
            if top_category_id in self._categories_cache:
                category_name = self._categories_cache[top_category_id].name
            
            category_percentage = (top_category_amount / recent_total) * 100 if recent_total > 0 else 0
            
            if category_percentage > 30:  # Plus de 30% des dépenses dans une catégorie
                metrics = InsightMetrics(
                    numerical_value=top_category_amount,
                    change_percentage=category_percentage,
                    confidence_score=0.8
                )
                
                context = InsightContext(
                    categories_concerned=[category_name]
                )
                
                actions = [
                    InsightAction(
                        action_text=f"Examiner si les dépenses en {category_name} sont justifiées",
                        priority="high"
                    ),
                    InsightAction(
                        action_text="Chercher des moyens d'optimiser cette catégorie",
                        priority="medium"
                    ),
                    InsightAction(
                        action_text="Diversifier vos dépenses si possible",
                        priority="low"
                    )
                ]
                
                insight = FinancialInsight(
                    insight_id=f"category_concentration_{user_id}_{category_name.lower().replace(' ', '_')}",
                    user_id=user_id,
                    insight_type=InsightType.CATEGORY_CONCENTRATION,
                    title=f"Concentration élevée des dépenses: {category_name}",
                    description=f"{category_percentage:.1f}% de vos dépenses concernent {category_name}",
                    highlight=f"{top_category_amount:.2f}€ en {category_name}",
                    time_scope=TimeScope.SHORT_TERM,
                    financial_scope=FinancialScope.SPENDING,
                    priority=Priority.NORMAL,
                    metrics=metrics,
                    context=context,
                    suggested_actions=actions,
                    narrative=f"La catégorie {category_name} représente une part importante de votre budget mensuel",
                    tags=["dépenses", "concentration", category_name.lower()]
                )
                
                insights.append(insight)
        
        return insights
    
    async def _analyze_daily_patterns(
        self, 
        user_id: int, 
        transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse les patterns de dépenses par jour de la semaine."""
        insights = []
        
        expenses = [t for t in transactions if t.amount < 0]
        if len(expenses) < 10:
            return insights
        
        # Analyser les patterns journaliers
        daily_spending = defaultdict(float)
        for transaction in expenses:
            day_name = transaction.date.strftime("%A")
            daily_spending[day_name] += abs(transaction.amount)
        
        if len(daily_spending) >= 5:  # Au moins 5 jours différents
            max_day = max(daily_spending.items(), key=lambda x: x[1])
            min_day = min(daily_spending.items(), key=lambda x: x[1])
            
            # Si l'écart est significatif
            if max_day[1] > min_day[1] * 2:
                metrics = InsightMetrics(
                    numerical_value=max_day[1],
                    comparative_value=min_day[1],
                    confidence_score=0.7
                )
                
                actions = [
                    InsightAction(
                        action_text=f"Planifier vos dépenses du {max_day[0]}",
                        priority="medium"
                    ),
                    InsightAction(
                        action_text="Faire une liste avant les gros achats",
                        priority="medium"
                    ),
                    InsightAction(
                        action_text="Répartir vos achats sur la semaine",
                        priority="low"
                    )
                ]
                
                insight = FinancialInsight(
                    insight_id=f"spending_pattern_{user_id}_{max_day[0].lower()}",
                    user_id=user_id,
                    insight_type=InsightType.SPENDING_PATTERN,
                    title=f"Pattern de dépenses: forte activité le {max_day[0]}",
                    description=f"Vous dépensez plus le {max_day[0]} ({max_day[1]:.0f}€) vs {min_day[0]} ({min_day[1]:.0f}€)",
                    highlight=f"Pic le {max_day[0]}: {max_day[1]:.0f}€",
                    time_scope=TimeScope.MEDIUM_TERM,
                    financial_scope=FinancialScope.SPENDING,
                    priority=Priority.LOW,
                    metrics=metrics,
                    suggested_actions=actions,
                    narrative=f"Vous avez tendance à dépenser plus le {max_day[0]}, ce qui peut déséquilibrer votre budget",
                    tags=["pattern", "comportement", "planification"]
                )
                
                insights.append(insight)
        
        return insights

class SavingsAnalyzer(BaseInsightAnalyzer):
    """Analyseur spécialisé dans les insights d'épargne."""
    
    async def analyze_savings_patterns(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse les patterns d'épargne et génère des insights.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights générés
        """
        insights = []
        
        # Analyser le taux d'épargne
        insights.extend(await self._analyze_savings_rate(
            user_id, recent_transactions, historical_transactions
        ))
        
        # Analyser l'amélioration de l'épargne
        insights.extend(await self._analyze_savings_improvement(
            user_id, recent_transactions, historical_transactions
        ))
        
        # Analyser la variabilité des revenus
        insights.extend(await self._analyze_income_variability(
            user_id, historical_transactions
        ))
        
        return insights
    
    async def _analyze_savings_rate(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse le taux d'épargne actuel."""
        insights = []
        
        # Calculer le flux net récent
        recent_income = sum(t.amount for t in recent_transactions if t.amount > 0)
        recent_expenses = abs(sum(t.amount for t in recent_transactions if t.amount < 0))
        recent_net = recent_income - recent_expenses
        
        if recent_income == 0:
            return insights
        
        recent_savings_rate = (recent_net / recent_income) * 100
        
        # Générer un insight si le taux d'épargne est faible
        if recent_savings_rate < 10:  # Taux d'épargne faible
            priority = Priority.CRITICAL if recent_savings_rate < 0 else Priority.HIGH
            
            template = INSIGHT_TEMPLATES[InsightType.LOW_SAVINGS_RATE]
            
            metrics = InsightMetrics(
                numerical_value=recent_savings_rate,
                confidence_score=0.9
            )
            
            actions = [
                InsightAction(action_text=template.action_templates[0], priority="high"),
                InsightAction(action_text=template.action_templates[1], priority="high"),
                InsightAction(action_text=template.action_templates[2], priority="medium")
            ]
            
            insight = FinancialInsight(
                insight_id=f"low_savings_rate_{user_id}_{datetime.now().strftime('%Y%m')}",
                user_id=user_id,
                insight_type=InsightType.LOW_SAVINGS_RATE,
                title=template.title_template,
                description=template.description_template.format(savings_rate=recent_savings_rate),
                highlight=template.highlight_template.format(net_amount=recent_net),
                time_scope=template.time_scope,
                financial_scope=template.financial_scope,
                priority=priority,
                metrics=metrics,
                suggested_actions=actions,
                narrative="Un taux d'épargne de 10-20% est recommandé pour une santé financière optimale",
                tags=template.tags.copy()
            )
            
            insights.append(insight)
        
        return insights
    
    async def _analyze_savings_improvement(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse l'amélioration du taux d'épargne."""
        insights = []
        
        if len(historical_transactions) < 30:
            return insights
        
        # Calculer les taux d'épargne récent et historique
        recent_income = sum(t.amount for t in recent_transactions if t.amount > 0)
        recent_expenses = abs(sum(t.amount for t in recent_transactions if t.amount < 0))
        recent_net = recent_income - recent_expenses
        
        historical_income = sum(t.amount for t in historical_transactions if t.amount > 0)
        historical_expenses = abs(sum(t.amount for t in historical_transactions if t.amount < 0))
        historical_net = historical_income - historical_expenses
        
        if recent_income == 0 or historical_income == 0:
            return insights
        
        recent_savings_rate = (recent_net / recent_income) * 100
        historical_savings_rate = (historical_net / historical_income * 3) * 100  # Ajusté sur 3 mois
        
        # Insight sur l'amélioration du taux d'épargne
        if recent_savings_rate > historical_savings_rate + 5:  # Amélioration de 5%+
            metrics = InsightMetrics(
                numerical_value=recent_savings_rate,
                comparative_value=historical_savings_rate,
                change_percentage=recent_savings_rate - historical_savings_rate,
                confidence_score=0.8
            )
            
            actions = [
                InsightAction(
                    action_text="Continuer sur cette lancée positive",
                    priority="low",
                    estimated_impact="Maintien de la progression"
                ),
                InsightAction(
                    action_text="Considérer augmenter vos objectifs d'épargne",
                    priority="medium",
                    estimated_impact="Accélération de l'épargne"
                ),
                InsightAction(
                    action_text="Explorer des placements pour optimiser votre épargne",
                    priority="medium",
                    estimated_impact="Rendement supplémentaire"
                )
            ]
            
            insight = FinancialInsight(
                insight_id=f"savings_improvement_{user_id}_{datetime.now().strftime('%Y%m')}",
                user_id=user_id,
                insight_type=InsightType.SAVINGS_IMPROVEMENT,
                title="Amélioration de votre taux d'épargne",
                description=f"Votre taux d'épargne s'améliore: {recent_savings_rate:.1f}% ce mois",
                highlight=f"Taux d'épargne: {recent_savings_rate:.1f}%",
                time_scope=TimeScope.SHORT_TERM,
                financial_scope=FinancialScope.SAVING,
                priority=Priority.LOW,
                metrics=metrics,
                suggested_actions=actions,
                narrative=f"Félicitations ! Votre discipline financière porte ses fruits avec {recent_net:.2f}€ épargnés",
                tags=["épargne", "amélioration", "félicitations"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def _analyze_income_variability(
        self, 
        user_id: int, 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse la variabilité des revenus."""
        insights = []
        
        # Analyser la régularité des revenus
        income_transactions = [t for t in historical_transactions if t.amount > 0]
        
        if len(income_transactions) < 6:  # Au moins 6 revenus
            return insights
        
        # Grouper par mois
        monthly_income = defaultdict(float)
        for transaction in income_transactions:
            month_key = transaction.date.strftime("%Y-%m")
            monthly_income[month_key] += transaction.amount
        
        if len(monthly_income) < 2:
            return insights
        
        income_values = list(monthly_income.values())
        avg_income = sum(income_values) / len(income_values)
        
        # Calculer la variation
        income_variation = max(income_values) - min(income_values)
        variation_percentage = (income_variation / avg_income) * 100 if avg_income > 0 else 0
        
        if variation_percentage > 25:  # Revenus variables
            metrics = InsightMetrics(
                numerical_value=variation_percentage,
                confidence_score=0.8
            )
            
            actions = [
                InsightAction(
                    action_text="Budgeter sur la base du revenu minimum",
                    priority="high"
                ),
                InsightAction(
                    action_text="Constituer un fonds d'urgence plus important",
                    priority="high"
                ),
                InsightAction(
                    action_text="Lisser vos dépenses sur les mois de revenus élevés",
                    priority="medium"
                )
            ]
            
            insight = FinancialInsight(
                insight_id=f"income_variability_{user_id}",
                user_id=user_id,
                insight_type=InsightType.INCOME_VARIABILITY,
                title="Revenus variables détectés",
                description=f"Vos revenus varient de {variation_percentage:.0f}% (moyenne: {avg_income:.0f}€)",
                highlight=f"Variation: {variation_percentage:.0f}%",
                time_scope=TimeScope.MEDIUM_TERM,
                financial_scope=FinancialScope.BUDGETING,
                priority=Priority.NORMAL,
                metrics=metrics,
                suggested_actions=actions,
                narrative=f"Avec des revenus variables, une gestion budgétaire prudente est recommandée",
                tags=["budget", "revenus_variables", "planification"]
            )
            
            insights.append(insight)
        
        return insights

class TrendAnalyzer(BaseInsightAnalyzer):
    """Analyseur spécialisé dans les tendances financières."""
    
    async def analyze_financial_trends(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse les tendances financières et génère des insights.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights générés
        """
        insights = []
        
        if len(historical_transactions) < 60:  # Besoin d'au moins 2 mois de données
            return insights
        
        # Analyser les tendances mensuelles
        monthly_data = defaultdict(lambda: {"income": 0.0, "expenses": 0.0})
        
        for transaction in historical_transactions:
            month_key = transaction.date.strftime("%Y-%m")
            if transaction.amount > 0:
                monthly_data[month_key]["income"] += transaction.amount
            else:
                monthly_data[month_key]["expenses"] += abs(transaction.amount)
        
        months = sorted(monthly_data.keys())
        if len(months) < 2:
            return insights
        
        # Analyser la tendance des dépenses
        expense_values = [monthly_data[month]["expenses"] for month in months]
        expense_trend = self._calculate_trend(expense_values)
        
        if expense_trend["slope"] > 50:  # Augmentation > 50€/mois
            metrics = InsightMetrics(
                numerical_value=expense_trend["slope"],
                confidence_score=expense_trend["confidence"]
            )
            
            actions = [
                InsightAction(
                    action_text="Identifier les causes de cette augmentation",
                    priority="high"
                ),
                InsightAction(
                    action_text="Mettre en place un budget plus strict",
                    priority="high"
                ),
                InsightAction(
                    action_text="Revoir vos abonnements et dépenses récurrentes",
                    priority="medium"
                )
            ]
            
            insight = FinancialInsight(
                insight_id=f"expense_trend_increase_{user_id}",
                user_id=user_id,
                insight_type=InsightType.EXPENSE_TREND_INCREASE,
                title="Tendance à la hausse des dépenses",
                description=f"Vos dépenses augmentent de ~{expense_trend['slope']:.0f}€ par mois",
                highlight=f"Tendance: +{expense_trend['slope']:.0f}€/mois",
                time_scope=TimeScope.MEDIUM_TERM,
                financial_scope=FinancialScope.SPENDING,
                priority=Priority.HIGH,
                metrics=metrics,
                suggested_actions=actions,
                narrative=f"Vos dépenses mensuelles ont une tendance haussière de {expense_trend['slope']:.0f}€/mois",
                tags=["tendance", "dépenses", "augmentation"]
            )
            
            insights.append(insight)
        
        # Analyser la tendance des revenus
        income_values = [monthly_data[month]["income"] for month in months]
        income_trend = self._calculate_trend(income_values)
        
        if income_trend["slope"] > 100:  # Augmentation > 100€/mois
            metrics = InsightMetrics(
                numerical_value=income_trend["slope"],
                confidence_score=income_trend["confidence"]
            )
            
            actions = [
                InsightAction(
                    action_text="Profiter de cette croissance pour augmenter votre épargne",
                    priority="high",
                    estimated_impact="Amélioration de l'épargne"
                ),
                InsightAction(
                    action_text="Considérer investir le surplus",
                    priority="medium",
                    estimated_impact="Rendement à long terme"
                ),
                InsightAction(
                    action_text="Revoir vos objectifs financiers à la hausse",
                    priority="medium"
                )
            ]
            
            insight = FinancialInsight(
                insight_id=f"income_trend_increase_{user_id}",
                user_id=user_id,
                insight_type=InsightType.INCOME_TREND_INCREASE,
                title="Progression positive de vos revenus",
                description=f"Vos revenus progressent de ~{income_trend['slope']:.0f}€ par mois",
                highlight=f"Croissance: +{income_trend['slope']:.0f}€/mois",
                time_scope=TimeScope.MEDIUM_TERM,
                financial_scope=FinancialScope.INCOME,
                priority=Priority.LOW,
                metrics=metrics,
                suggested_actions=actions,
                narrative=f"Excellente nouvelle ! Vos revenus croissent de {income_trend['slope']:.0f}€ par mois",
                tags=["revenus", "croissance", "opportunité"]
            )
            
            insights.append(insight)
        
        return insights

class AnomalyAnalyzer(BaseInsightAnalyzer):
    """Analyseur spécialisé dans la détection d'anomalies."""
    
    async def analyze_anomalies(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse les anomalies dans les transactions récentes.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            
        Returns:
            List[FinancialInsight]: Insights générés
        """
        insights = []
        
        if len(recent_transactions) < 10:
            return insights
        
        # Détecter les transactions inhabituellement importantes
        amounts = [abs(t.amount) for t in recent_transactions]
        
        if len(amounts) > 5:
            # Utiliser les percentiles pour détecter les anomalies
            amounts_sorted = sorted(amounts)
            p95_index = int(len(amounts_sorted) * 0.95)
            threshold = amounts_sorted[p95_index] if p95_index < len(amounts_sorted) else amounts_sorted[-1]
            
            # Si le seuil est significatif (> 200€)
            if threshold > self.large_transaction_threshold:
                large_transactions = [t for t in recent_transactions if abs(t.amount) >= threshold]
                
                for transaction in large_transactions[:3]:  # Limiter à 3 transactions
                    template = INSIGHT_TEMPLATES[InsightType.LARGE_TRANSACTION]
                    
                    metrics = InsightMetrics(
                        numerical_value=abs(transaction.amount),
                        confidence_score=0.9
                    )
                    
                    context = InsightContext(
                        related_transactions=[transaction.id]
                    )
                    
                    actions = [
                        InsightAction(action_text=template.action_templates[0], priority="high"),
                        InsightAction(
                            action_text="S'assurer qu'elle s'inscrit dans votre budget" if transaction.amount < 0 else "Considérer placer ce montant si c'est un revenu exceptionnel",
                            priority="medium"
                        ),
                        InsightAction(action_text=template.action_templates[2], priority="low")
                    ]
                    
                    insight = FinancialInsight(
                        insight_id=f"large_transaction_{user_id}_{transaction.id}",
                        user_id=user_id,
                        insight_type=InsightType.LARGE_TRANSACTION,
                        title=template.title_template,
                        description=template.description_template.format(amount=abs(transaction.amount)),
                        highlight=template.highlight_template.format(amount=abs(transaction.amount)),
                        time_scope=template.time_scope,
                        financial_scope=FinancialScope.SPENDING if transaction.amount < 0 else FinancialScope.INCOME,
                        priority=template.default_priority,
                        metrics=metrics,
                        context=context,
                        suggested_actions=actions,
                        narrative=f"Transaction de {abs(transaction.amount):.2f}€ le {transaction.date.strftime('%d/%m/%Y')}",
                        tags=template.tags.copy()
                    )
                    
                    insights.append(insight)
        
        return insights

class OpportunityAnalyzer(BaseInsightAnalyzer):
    """Analyseur spécialisé dans les opportunités d'optimisation."""
    
    async def analyze_optimization_opportunities(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse les opportunités d'optimisation financière.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights générés
        """
        insights = []
        
        # Analyser les abonnements potentiels
        insights.extend(await self._analyze_subscription_opportunities(
            user_id, historical_transactions
        ))
        
        # Analyser les opportunités d'économies
        insights.extend(await self._analyze_savings_opportunities(
            user_id, recent_transactions, historical_transactions
        ))
        
        return insights
    
    async def _analyze_subscription_opportunities(
        self, 
        user_id: int, 
        transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse les abonnements pour identifier des opportunités d'optimisation."""
        insights = []
        
        # Identifier les abonnements potentiels (montants récurrents similaires)
        potential_subscriptions = defaultdict(list)
        
        for transaction in transactions:
            if transaction.amount < 0:  # Seulement les dépenses
                # Grouper par montant arrondi et description similaire
                amount_key = round(abs(transaction.amount))
                description = (transaction.clean_description or transaction.provider_description or "").lower()
                
                # Simplifier la description pour le groupement
                simplified_desc = ' '.join(description.split()[:2])  # 2 premiers mots
                
                key = (amount_key, simplified_desc)
                potential_subscriptions[key].append(transaction)
        
        # Analyser les groupes pour détecter les abonnements
        total_subscription_cost = 0
        subscription_count = 0
        detected_subscriptions = []
        
        for (amount, desc), transactions_group in potential_subscriptions.items():
            if len(transactions_group) >= 3:  # Au moins 3 occurrences
                # Vérifier la régularité
                dates = [t.date for t in transactions_group]
                dates.sort()
                
                intervals = []
                for i in range(1, len(dates)):
                    interval = (dates[i] - dates[i-1]).days
                    intervals.append(interval)
                
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    
                    # Si l'intervalle moyen est proche de 30 jours (mensuel)
                    if 25 <= avg_interval <= 35:
                        monthly_cost = amount
                        total_subscription_cost += monthly_cost
                        subscription_count += 1
                        detected_subscriptions.append({
                            "description": desc,
                            "amount": monthly_cost,
                            "frequency": avg_interval
                        })
        
        # Générer un insight si des abonnements significatifs sont détectés
        if total_subscription_cost > 50:  # Plus de 50€ d'abonnements
            template = INSIGHT_TEMPLATES[InsightType.SUBSCRIPTION_ANALYSIS]
            
            metrics = InsightMetrics(
                numerical_value=total_subscription_cost,
                confidence_score=0.7,
                potential_impact=total_subscription_cost * 0.2  # 20% d'économie potentielle
            )
            
            # Identifier les marchands concernés
            merchants = [sub["description"] for sub in detected_subscriptions[:5]]
            context = InsightContext(
                merchants_concerned=merchants
            )
            
            actions = [
                InsightAction(action_text=template.action_templates[0], priority="high"),
                InsightAction(action_text=template.action_templates[1], priority="high"),
                InsightAction(
                    action_text=f"Économie potentielle: ~{total_subscription_cost * 0.2:.0f}€/mois",
                    priority="medium",
                    estimated_impact=f"{total_subscription_cost * 0.2:.0f}€/mois"
                )
            ]
            
            insight = FinancialInsight(
                insight_id=f"subscription_analysis_{user_id}",
                user_id=user_id,
                insight_type=InsightType.SUBSCRIPTION_ANALYSIS,
                title=template.title_template,
                description=template.description_template.format(
                    monthly_cost=total_subscription_cost,
                    count=subscription_count
                ),
                highlight=template.highlight_template.format(monthly_cost=total_subscription_cost),
                time_scope=template.time_scope,
                financial_scope=template.financial_scope,
                priority=template.default_priority,
                metrics=metrics,
                context=context,
                suggested_actions=actions,
                narrative=f"Vos abonnements représentent {total_subscription_cost:.0f}€ par mois, soit {total_subscription_cost * 12:.0f}€ par an",
                tags=template.tags.copy()
            )
            
            insights.append(insight)
        
        return insights
    
    async def _analyze_savings_opportunities(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse les opportunités d'économies."""
        insights = []
        
        # Analyser les catégories avec potentiel d'optimisation
        category_expenses = defaultdict(list)
        
        for transaction in recent_transactions:
            if transaction.amount < 0:
                category_id = transaction.category_id or 0
                category_expenses[category_id].append(abs(transaction.amount))
        
        # Identifier les catégories avec forte variabilité (potentiel d'optimisation)
        for category_id, amounts in category_expenses.items():
            if len(amounts) >= 5:  # Au moins 5 transactions
                avg_amount = sum(amounts) / len(amounts)
                max_amount = max(amounts)
                min_amount = min(amounts)
                
                # Calculer l'écart-type pour mesurer la variabilité
                variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
                std_dev = variance ** 0.5
                coefficient_variation = std_dev / avg_amount if avg_amount > 0 else 0
                
                # Si l'écart max est significatif ET forte variabilité
                if max_amount > avg_amount * 2 and avg_amount > 20 and coefficient_variation > 0.3:
                    category_name = "Non catégorisé"
                    if category_id in self._categories_cache:
                        category_name = self._categories_cache[category_id].name
                    
                    # Estimation conservative de l'économie potentielle
                    potential_savings = (max_amount - avg_amount) * len(amounts) / 6
                    
                    metrics = InsightMetrics(
                        numerical_value=potential_savings,
                        confidence_score=0.6,
                        potential_impact=potential_savings
                    )
                    
                    context = InsightContext(
                        categories_concerned=[category_name]
                    )
                    
                    actions = [
                        InsightAction(
                            action_text=f"Examiner vos dépenses en {category_name}",
                            priority="high"
                        ),
                        InsightAction(
                            action_text="Comparer les prix et négocier si possible",
                            priority="medium",
                            estimated_impact="Réduction des coûts"
                        ),
                        InsightAction(
                            action_text="Fixer un budget maximum pour cette catégorie",
                            priority="medium",
                            estimated_impact="Contrôle des dépenses"
                        )
                    ]
                    
                    insight = FinancialInsight(
                        insight_id=f"savings_opportunity_{user_id}_{category_name.lower().replace(' ', '_')}",
                        user_id=user_id,
                        insight_type=InsightType.SAVINGS_OPPORTUNITY,
                        title=f"Opportunité d'économies: {category_name}",
                        description=f"Forte variabilité dans {category_name} (moyenne: {avg_amount:.0f}€, max: {max_amount:.0f}€)",
                        highlight=f"Économie potentielle: {potential_savings:.0f}€",
                        time_scope=TimeScope.SHORT_TERM,
                        financial_scope=FinancialScope.SPENDING,
                        priority=Priority.NORMAL,
                        metrics=metrics,
                        context=context,
                        suggested_actions=actions,
                        narrative=f"Optimiser vos dépenses en {category_name} pourrait vous faire économiser ~{potential_savings:.0f}€",
                        tags=["économies", "optimisation", category_name.lower()]
                    )
                    
                    insights.append(insight)
        
        return insights

class BudgetAnalyzer(BaseInsightAnalyzer):
    """Analyseur spécialisé dans la gestion budgétaire."""
    
    async def analyze_budget_patterns(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse les patterns de gestion budgétaire.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights générés
        """
        insights = []
        
        # Analyser la régularité des flux financiers
        insights.extend(await self._analyze_cash_flow_regularity(
            user_id, historical_transactions
        ))
        
        # Analyser les pics et creux de dépenses
        insights.extend(await self._analyze_spending_cycles(
            user_id, recent_transactions
        ))
        
        return insights
    
    async def _analyze_cash_flow_regularity(
        self, 
        user_id: int, 
        transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse la régularité des flux de trésorerie."""
        insights = []
        
        if len(transactions) < 30:
            return insights
        
        # Grouper les transactions par semaine
        weekly_flows = defaultdict(float)
        for transaction in transactions:
            week_key = f"{transaction.date.year}-W{transaction.date.isocalendar()[1]}"
            weekly_flows[week_key] += transaction.amount
        
        if len(weekly_flows) < 4:  # Au moins 4 semaines
            return insights
        
        flow_values = list(weekly_flows.values())
        avg_flow = sum(flow_values) / len(flow_values)
        
        # Calculer la stabilité
        variance = sum((flow - avg_flow) ** 2 for flow in flow_values) / len(flow_values)
        std_dev = variance ** 0.5
        
        # Score de stabilité (inverse du coefficient de variation)
        stability_score = 1 / (1 + std_dev / abs(avg_flow)) if avg_flow != 0 else 0
        
        if stability_score < 0.4:  # Flux très irréguliers
            metrics = InsightMetrics(
                numerical_value=stability_score,
                confidence_score=0.8
            )
            
            actions = [
                InsightAction(
                    action_text="Identifier les causes de variation de vos flux",
                    priority="high"
                ),
                InsightAction(
                    action_text="Lisser vos dépenses sur le mois",
                    priority="medium"
                ),
                InsightAction(
                    action_text="Constituer une réserve pour les fluctuations",
                    priority="medium"
                )
            ]
            
            insight = FinancialInsight(
                insight_id=f"cash_flow_irregularity_{user_id}",
                user_id=user_id,
                insight_type=InsightType.BUDGET_ALERT,
                title="Flux de trésorerie irréguliers",
                description=f"Vos flux financiers varient beaucoup (stabilité: {stability_score:.1%})",
                highlight=f"Stabilité: {stability_score:.1%}",
                time_scope=TimeScope.MEDIUM_TERM,
                financial_scope=FinancialScope.BUDGETING,
                priority=Priority.NORMAL,
                metrics=metrics,
                suggested_actions=actions,
                narrative="Des flux réguliers facilitent la planification financière",
                tags=["budget", "stabilité", "planification"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def _analyze_spending_cycles(
        self, 
        user_id: int, 
        transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse les cycles de dépenses dans le mois."""
        insights = []
        
        expenses = [t for t in transactions if t.amount < 0]
        if len(expenses) < 20:
            return insights
        
        # Grouper par jour du mois
        daily_spending = defaultdict(float)
        for transaction in expenses:
            day_of_month = transaction.date.day
            daily_spending[day_of_month] += abs(transaction.amount)
        
        if len(daily_spending) < 10:
            return insights
        
        # Identifier les pics de dépenses
        spending_values = list(daily_spending.values())
        avg_daily = sum(spending_values) / len(spending_values)
        max_daily = max(spending_values)
        
        # Trouver les jours avec pics de dépenses
        peak_days = [
            day for day, amount in daily_spending.items() 
            if amount > avg_daily * 2
        ]
        
        if len(peak_days) >= 3:  # Au moins 3 jours de pics
            # Analyser si les pics sont en début/fin de mois
            early_month_peaks = [d for d in peak_days if d <= 10]
            late_month_peaks = [d for d in peak_days if d >= 20]
            
            pattern_detected = False
            pattern_description = ""
            
            if len(early_month_peaks) >= 2:
                pattern_detected = True
                pattern_description = "début de mois"
            elif len(late_month_peaks) >= 2:
                pattern_detected = True
                pattern_description = "fin de mois"
            
            if pattern_detected:
                metrics = InsightMetrics(
                    numerical_value=max_daily,
                    comparative_value=avg_daily,
                    confidence_score=0.7
                )
                
                actions = [
                    InsightAction(
                        action_text=f"Planifier vos dépenses de {pattern_description}",
                        priority="medium"
                    ),
                    InsightAction(
                        action_text="Étaler vos achats sur tout le mois",
                        priority="medium"
                    ),
                    InsightAction(
                        action_text="Prévoir un budget spécifique pour ces périodes",
                        priority="low"
                    )
                ]
                
                insight = FinancialInsight(
                    insight_id=f"spending_cycle_{user_id}_{pattern_description.replace(' ', '_')}",
                    user_id=user_id,
                    insight_type=InsightType.SPENDING_PATTERN,
                    title=f"Pic de dépenses en {pattern_description}",
                    description=f"Vous dépensez plus en {pattern_description} ({max_daily:.0f}€ vs {avg_daily:.0f}€/jour)",
                    highlight=f"Pic: {max_daily:.0f}€/jour",
                    time_scope=TimeScope.SHORT_TERM,
                    financial_scope=FinancialScope.BUDGETING,
                    priority=Priority.LOW,
                    metrics=metrics,
                    suggested_actions=actions,
                    narrative=f"Vos dépenses se concentrent en {pattern_description}, ce qui peut créer des tensions budgétaires",
                    tags=["pattern", "cycle", "budget"]
                )
                
                insights.append(insight)
        
        return insights

class AlertAnalyzer(BaseInsightAnalyzer):
    """Analyseur spécialisé dans les alertes financières."""
    
    async def analyze_financial_alerts(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse et génère des alertes financières urgentes.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            
        Returns:
            List[FinancialInsight]: Alertes générées
        """
        insights = []
        
        # Analyser les dépenses excessives récentes
        insights.extend(await self._analyze_excessive_spending(
            user_id, recent_transactions
        ))
        
        # Analyser les doublons potentiels
        insights.extend(await self._analyze_duplicate_transactions(
            user_id, recent_transactions
        ))
        
        return insights
    
    async def _analyze_excessive_spending(
        self, 
        user_id: int, 
        transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse les dépenses excessives sur une courte période."""
        insights = []
        
        # Analyser les dépenses des 7 derniers jours
        recent_date = datetime.now() - timedelta(days=7)
        very_recent_expenses = [
            t for t in transactions 
            if t.amount < 0 and t.date >= recent_date
        ]
        
        if len(very_recent_expenses) < 3:
            return insights
        
        total_recent = sum(abs(t.amount) for t in very_recent_expenses)
        daily_average = total_recent / 7
        
        # Comparer avec la moyenne mensuelle
        monthly_expenses = [t for t in transactions if t.amount < 0]
        if monthly_expenses:
            monthly_total = sum(abs(t.amount) for t in monthly_expenses)
            monthly_daily_avg = monthly_total / 30
            
            # Si les dépenses récentes sont 50% plus élevées que la moyenne
            if daily_average > monthly_daily_avg * 1.5:
                excess_percentage = ((daily_average - monthly_daily_avg) / monthly_daily_avg) * 100
                
                metrics = InsightMetrics(
                    numerical_value=daily_average,
                    comparative_value=monthly_daily_avg,
                    change_percentage=excess_percentage,
                    confidence_score=0.9
                )
                
                actions = [
                    InsightAction(
                        action_text="Examiner vos dépenses des 7 derniers jours",
                        priority="high"
                    ),
                    InsightAction(
                        action_text="Ralentir les achats non essentiels",
                        priority="high"
                    ),
                    InsightAction(
                        action_text="Réviser votre budget pour le reste du mois",
                        priority="medium"
                    )
                ]
                
                insight = FinancialInsight(
                    insight_id=f"excessive_spending_{user_id}_{datetime.now().strftime('%Y%m%d')}",
                    user_id=user_id,
                    insight_type=InsightType.BUDGET_ALERT,
                    title="Dépenses élevées détectées",
                    description=f"Vos dépenses récentes sont {excess_percentage:.0f}% plus élevées que la moyenne",
                    highlight=f"7 derniers jours: {total_recent:.0f}€",
                    time_scope=TimeScope.SHORT_TERM,
                    financial_scope=FinancialScope.SPENDING,
                    priority=Priority.HIGH,
                    metrics=metrics,
                    suggested_actions=actions,
                    narrative=f"Dépenses récentes: {daily_average:.0f}€/jour vs moyenne {monthly_daily_avg:.0f}€/jour",
                    tags=["alerte", "dépenses", "budget"]
                )
                
                insights.append(insight)
        
        return insights
    
    async def _analyze_duplicate_transactions(
        self, 
        user_id: int, 
        transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """Analyse les transactions potentiellement dupliquées."""
        insights = []
        
        # Grouper les transactions par montant et date proche
        potential_duplicates = defaultdict(list)
        
        for transaction in transactions:
            # Clé basée sur le montant arrondi et la date
            amount_key = round(abs(transaction.amount), 2)
            date_key = transaction.date.strftime("%Y-%m-%d")
            
            key = (amount_key, date_key)
            potential_duplicates[key].append(transaction)
        
        # Identifier les vrais doublons
        confirmed_duplicates = []
        for (amount, date), tx_list in potential_duplicates.items():
            if len(tx_list) >= 2:
                # Vérifier si les descriptions sont similaires
                descriptions = [
                    (t.clean_description or t.provider_description or "").lower()
                    for t in tx_list
                ]
                
                # Si au moins 2 descriptions sont très similaires
                for i, desc1 in enumerate(descriptions):
                    for desc2 in descriptions[i+1:]:
                        if desc1 and desc2:
                            from difflib import SequenceMatcher
                            similarity = SequenceMatcher(None, desc1, desc2).ratio()
                            if similarity > 0.8:  # 80% de similarité
                                confirmed_duplicates.extend(tx_list)
                                break
        
        if confirmed_duplicates:
            total_duplicate_amount = sum(abs(t.amount) for t in confirmed_duplicates)
            
            metrics = InsightMetrics(
                numerical_value=total_duplicate_amount,
                confidence_score=0.8
            )
            
            context = InsightContext(
                related_transactions=[t.id for t in confirmed_duplicates[:10]]
            )
            
            actions = [
                InsightAction(
                    action_text="Vérifier les transactions en double dans vos comptes",
                    priority="high"
                ),
                InsightAction(
                    action_text="Contacter votre banque si nécessaire",
                    priority="medium"
                ),
                InsightAction(
                    action_text="Surveiller vos relevés pour éviter les doublons",
                    priority="low"
                )
            ]
            
            insight = FinancialInsight(
                insight_id=f"duplicate_transactions_{user_id}",
                user_id=user_id,
                insight_type=InsightType.ANOMALY_DETECTED,
                title="Transactions potentiellement dupliquées",
                description=f"{len(confirmed_duplicates)} transactions similaires détectées",
                highlight=f"{total_duplicate_amount:.2f}€ concernés",
                time_scope=TimeScope.SHORT_TERM,
                financial_scope=FinancialScope.GENERAL,
                priority=Priority.NORMAL,
                metrics=metrics,
                context=context,
                suggested_actions=actions,
                narrative="Des transactions similaires ont été détectées, vérifiez s'il n'y a pas de doublons",
                tags=["anomalie", "doublons", "vérification"]
            )
            
            insights.append(insight)
        
        return insights