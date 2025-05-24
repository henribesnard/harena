"""
Générateur d'insights financiers automatique.

Ce module analyse les données financières pour générer des insights
intelligents et des recommandations personnalisées.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from uuid import uuid4
import statistics

from enrichment_service.core.logging import get_contextual_logger, log_performance
from enrichment_service.core.exceptions import InsightGenerationError
from enrichment_service.core.config import enrichment_settings
from enrichment_service.db.models import RawTransaction, SyncAccount, BridgeCategory

logger = logging.getLogger(__name__)

@dataclass
class FinancialInsight:
    """Insight financier généré automatiquement."""
    insight_id: str
    user_id: int
    insight_type: str
    title: str
    description: str
    highlight: str
    
    # Portée temporelle
    time_scope: str  # short_term, medium_term, long_term
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    # Portée financière
    financial_scope: str = "general"  # spending, saving, budgeting, etc.
    categories_concerned: List[str] = field(default_factory=list)
    merchants_concerned: List[str] = field(default_factory=list)
    accounts_concerned: List[str] = field(default_factory=list)
    
    # Importance et impact
    priority: int = 3  # 1 (faible) à 5 (critique)
    numerical_value: Optional[float] = None
    comparative_value: Optional[float] = None
    change_percentage: Optional[float] = None
    
    # Données associées
    related_transactions: List[int] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    action_impact: Optional[float] = None
    
    # Métadonnées
    confidence_score: float = 0.0
    generation_method: str = "rule_based"  # rule_based, ml, hybrid
    narrative: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Dates
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

class InsightGenerator:
    """
    Générateur d'insights financiers intelligents.
    
    Cette classe analyse les données financières et génère automatiquement
    des insights personnalisés avec recommandations actionables.
    """
    
    def __init__(self, db: Session, embedding_service, qdrant_service):
        """
        Initialise le générateur d'insights.
        
        Args:
            db: Session de base de données
            embedding_service: Service d'embedding
            qdrant_service: Service Qdrant
        """
        self.db = db
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        
        # Configuration des insights
        self.confidence_threshold = enrichment_settings.insight_confidence_threshold
        self.max_insights_per_user = enrichment_settings.max_insights_per_user
        
        # Seuils pour la génération d'insights
        self.significant_change_threshold = 0.20  # 20%
        self.large_transaction_threshold = 200.0  # euros
        self.recurring_confidence_threshold = 0.7
        
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
    
    @log_performance
    async def generate_user_insights(self, user_id: int) -> List[FinancialInsight]:
        """
        Génère tous les insights pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[FinancialInsight]: Liste des insights générés
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="insight_generation"
        )
        
        ctx_logger.info(f"Génération d'insights pour l'utilisateur {user_id}")
        
        try:
            insights = []
            
            # Récupérer les données nécessaires
            recent_transactions = await self._get_recent_transactions(user_id, days=30)
            historical_transactions = await self._get_recent_transactions(user_id, days=90)
            
            if not recent_transactions:
                ctx_logger.info("Pas de transactions récentes, aucun insight généré")
                return []
            
            # Générer différents types d'insights
            insights.extend(await self._generate_spending_insights(user_id, recent_transactions, historical_transactions))
            insights.extend(await self._generate_saving_insights(user_id, recent_transactions, historical_transactions))
            insights.extend(await self._generate_trend_insights(user_id, recent_transactions, historical_transactions))
            insights.extend(await self._generate_anomaly_insights(user_id, recent_transactions))
            insights.extend(await self._generate_opportunity_insights(user_id, recent_transactions, historical_transactions))
            insights.extend(await self._generate_budget_insights(user_id, recent_transactions, historical_transactions))
            insights.extend(await self._generate_pattern_insights(user_id, historical_transactions))
            
            # Filtrer et classer les insights
            filtered_insights = await self._filter_and_rank_insights(insights)
            
            # Stocker dans Qdrant
            for insight in filtered_insights:
                await self._store_insight_in_qdrant(insight)
            
            ctx_logger.info(f"Génération terminée: {len(filtered_insights)} insights créés")
            
            return filtered_insights
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération d'insights: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise InsightGenerationError(error_msg, "general", {"user_id": user_id})
    
    async def _get_recent_transactions(self, user_id: int, days: int) -> List[RawTransaction]:
        """
        Récupère les transactions récentes d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days: Nombre de jours à récupérer
            
        Returns:
            List[RawTransaction]: Liste des transactions
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        transactions = self.db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.date >= cutoff_date,
            RawTransaction.deleted.is_(False),
            RawTransaction.amount != 0
        ).order_by(RawTransaction.date.desc()).all()
        
        return transactions
    
    async def _generate_spending_insights(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Génère des insights sur les dépenses.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions des 30 derniers jours
            historical_transactions: Transactions des 90 derniers jours
            
        Returns:
            List[FinancialInsight]: Insights sur les dépenses
        """
        insights = []
        
        # Analyser les dépenses par catégorie
        recent_expenses = [t for t in recent_transactions if t.amount < 0]
        historical_expenses = [t for t in historical_transactions if t.amount < 0]
        
        if not recent_expenses:
            return insights
        
        # Comparer les dépenses récentes vs historiques
        recent_total = sum(abs(t.amount) for t in recent_expenses)
        
        # Calculer la moyenne mensuelle historique
        if len(historical_expenses) > 30:
            historical_monthly_avg = sum(abs(t.amount) for t in historical_expenses) / 3  # 3 mois
            
            if recent_total > historical_monthly_avg * (1 + self.significant_change_threshold):
                change_pct = ((recent_total - historical_monthly_avg) / historical_monthly_avg) * 100
                
                insight = FinancialInsight(
                    insight_id=str(uuid4()),
                    user_id=user_id,
                    insight_type="spending_increase",
                    title="Augmentation significative des dépenses",
                    description=f"Vos dépenses ont augmenté de {change_pct:.1f}% ce mois-ci",
                    highlight=f"+{change_pct:.1f}% vs moyenne",
                    time_scope="short_term",
                    financial_scope="spending",
                    priority=4,
                    numerical_value=recent_total,
                    comparative_value=historical_monthly_avg,
                    change_percentage=change_pct,
                    confidence_score=0.9,
                    suggested_actions=[
                        "Examiner les dépenses récentes pour identifier les causes",
                        "Réviser votre budget pour le mois suivant",
                        "Considérer reporter les achats non essentiels"
                    ],
                    narrative=f"Vos dépenses de {recent_total:.2f}€ ce mois dépassent votre moyenne de {historical_monthly_avg:.2f}€",
                    tags=["dépenses", "augmentation", "budget"]
                )
                
                insights.append(insight)
        
        # Analyser les catégories de dépenses importantes
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
                insight = FinancialInsight(
                    insight_id=str(uuid4()),
                    user_id=user_id,
                    insight_type="category_concentration",
                    title=f"Concentration élevée des dépenses: {category_name}",
                    description=f"{category_percentage:.1f}% de vos dépenses concernent {category_name}",
                    highlight=f"{top_category_amount:.2f}€ en {category_name}",
                    time_scope="short_term",
                    financial_scope="spending",
                    categories_concerned=[category_name],
                    priority=3,
                    numerical_value=top_category_amount,
                    change_percentage=category_percentage,
                    confidence_score=0.8,
                    suggested_actions=[
                        f"Examiner si les dépenses en {category_name} sont justifiées",
                        "Chercher des moyens d'optimiser cette catégorie",
                        "Diversifier vos dépenses si possible"
                    ],
                    narrative=f"La catégorie {category_name} représente une part importante de votre budget mensuel",
                    tags=["dépenses", "concentration", category_name.lower()]
                )
                
                insights.append(insight)
        
        return insights
    
    async def _generate_saving_insights(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Génère des insights sur l'épargne.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights sur l'épargne
        """
        insights = []
        
        # Calculer le flux net récent
        recent_income = sum(t.amount for t in recent_transactions if t.amount > 0)
        recent_expenses = abs(sum(t.amount for t in recent_transactions if t.amount < 0))
        recent_net = recent_income - recent_expenses
        
        if recent_income == 0:
            return insights
        
        recent_savings_rate = (recent_net / recent_income) * 100
        
        # Comparer avec la performance historique
        if len(historical_transactions) > 30:
            historical_income = sum(t.amount for t in historical_transactions if t.amount > 0)
            historical_expenses = abs(sum(t.amount for t in historical_transactions if t.amount < 0))
            historical_net = historical_income - historical_expenses
            
            historical_savings_rate = (historical_net / historical_income * 3) * 100 if historical_income > 0 else 0  # Ajusté sur 3 mois
            
            # Insight sur l'amélioration du taux d'épargne
            if recent_savings_rate > historical_savings_rate + 5:  # Amélioration de 5%+
                insight = FinancialInsight(
                    insight_id=str(uuid4()),
                    user_id=user_id,
                    insight_type="savings_improvement",
                    title="Amélioration de votre taux d'épargne",
                    description=f"Votre taux d'épargne s'améliore: {recent_savings_rate:.1f}% ce mois",
                    highlight=f"Taux d'épargne: {recent_savings_rate:.1f}%",
                    time_scope="short_term",
                    financial_scope="saving",
                    priority=2,
                    numerical_value=recent_savings_rate,
                    comparative_value=historical_savings_rate,
                    change_percentage=recent_savings_rate - historical_savings_rate,
                    confidence_score=0.8,
                    suggested_actions=[
                        "Continuer sur cette lancée positive",
                        "Considérer augmenter vos objectifs d'épargne",
                        "Explorer des placements pour optimiser votre épargne"
                    ],
                    narrative=f"Félicitations ! Votre discipline financière porte ses fruits avec {recent_net:.2f}€ épargnés",
                    tags=["épargne", "amélioration", "félicitations"]
                )
                
                insights.append(insight)
        
        # Insights sur les opportunités d'épargne
        if recent_savings_rate < 10:  # Taux d'épargne faible
            priority = 5 if recent_savings_rate < 0 else 4
            
            insight = FinancialInsight(
                insight_id=str(uuid4()),
                user_id=user_id,
                insight_type="low_savings_rate",
                title="Taux d'épargne à améliorer",
                description=f"Votre taux d'épargne de {recent_savings_rate:.1f}% peut être optimisé",
                highlight=f"Seulement {recent_net:.2f}€ épargnés",
                time_scope="short_term",
                financial_scope="saving",
                priority=priority,
                numerical_value=recent_savings_rate,
                confidence_score=0.9,
                suggested_actions=[
                    "Examiner vos dépenses pour identifier des économies",
                    "Fixer un objectif d'épargne mensuel (recommandé: 10-20%)",
                    "Automatiser votre épargne via un virement programmé"
                ],
                narrative="Un taux d'épargne de 10-20% est recommandé pour une santé financière optimale",
                tags=["épargne", "amélioration", "objectif"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def _generate_trend_insights(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Génère des insights sur les tendances financières.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights sur les tendances
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
            insight = FinancialInsight(
                insight_id=str(uuid4()),
                user_id=user_id,
                insight_type="expense_trend_increase",
                title="Tendance à la hausse des dépenses",
                description=f"Vos dépenses augmentent de ~{expense_trend['slope']:.0f}€ par mois",
                highlight=f"Tendance: +{expense_trend['slope']:.0f}€/mois",
                time_scope="medium_term",
                financial_scope="spending",
                priority=4,
                numerical_value=expense_trend["slope"],
                confidence_score=expense_trend["confidence"],
                suggested_actions=[
                    "Identifier les causes de cette augmentation",
                    "Mettre en place un budget plus strict",
                    "Revoir vos abonnements et dépenses récurrentes"
                ],
                narrative=f"Vos dépenses mensuelles ont une tendance haussière de {expense_trend['slope']:.0f}€/mois",
                tags=["tendance", "dépenses", "augmentation"]
            )
            
            insights.append(insight)
        
        # Analyser la tendance des revenus
        income_values = [monthly_data[month]["income"] for month in months]
        income_trend = self._calculate_trend(income_values)
        
        if income_trend["slope"] > 100:  # Augmentation > 100€/mois
            insight = FinancialInsight(
                insight_id=str(uuid4()),
                user_id=user_id,
                insight_type="income_trend_increase",
                title="Progression positive de vos revenus",
                description=f"Vos revenus progressent de ~{income_trend['slope']:.0f}€ par mois",
                highlight=f"Croissance: +{income_trend['slope']:.0f}€/mois",
                time_scope="medium_term",
                financial_scope="income",
                priority=2,
                numerical_value=income_trend["slope"],
                confidence_score=income_trend["confidence"],
                suggested_actions=[
                    "Profiter de cette croissance pour augmenter votre épargne",
                    "Considérer investir le surplus",
                    "Revoir vos objectifs financiers à la hausse"
                ],
                narrative=f"Excellente nouvelle ! Vos revenus croissent de {income_trend['slope']:.0f}€ par mois",
                tags=["revenus", "croissance", "opportunité"]
            )
            
            insights.append(insight)
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """
        Calcule la tendance d'une série de valeurs.
        
        Args:
            values: Liste de valeurs
            
        Returns:
            Dict: Informations sur la tendance
        """
        if len(values) < 2:
            return {"slope": 0.0, "confidence": 0.0}
        
        # Régression linéaire simple
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
    
    async def _generate_anomaly_insights(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Génère des insights sur les anomalies détectées.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            
        Returns:
            List[FinancialInsight]: Insights sur les anomalies
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
                
                for transaction in large_transactions:
                    insight = FinancialInsight(
                        insight_id=str(uuid4()),
                        user_id=user_id,
                        insight_type="large_transaction",
                        title="Transaction importante détectée",
                        description=f"Transaction de {abs(transaction.amount):.2f}€ inhabituellement élevée",
                        highlight=f"{abs(transaction.amount):.2f}€",
                        time_scope="short_term",
                        financial_scope="spending" if transaction.amount < 0 else "income",
                        priority=3,
                        numerical_value=abs(transaction.amount),
                        confidence_score=0.9,
                        related_transactions=[transaction.id],
                        suggested_actions=[
                            "Vérifier que cette transaction est correcte",
                            "S'assurer qu'elle s'inscrit dans votre budget" if transaction.amount < 0 else "Considérer placer ce montant si c'est un revenu exceptionnel",
                            "Ajuster votre budget si nécessaire"
                        ],
                        narrative=f"Transaction de {abs(transaction.amount):.2f}€ le {transaction.date.strftime('%d/%m/%Y')}",
                        tags=["anomalie", "transaction_importante"]
                    )
                    
                    insights.append(insight)
        
        return insights
    
    async def _generate_opportunity_insights(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Génère des insights sur les opportunités d'optimisation.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights sur les opportunités
        """
        insights = []
        
        # Analyser les abonnements potentiels
        subscription_insights = await self._analyze_subscription_opportunities(user_id, historical_transactions)
        insights.extend(subscription_insights)
        
        # Analyser les opportunités d'économies
        savings_opportunities = await self._analyze_savings_opportunities(user_id, recent_transactions, historical_transactions)
        insights.extend(savings_opportunities)
        
        return insights
    
    async def _analyze_subscription_opportunities(
        self, 
        user_id: int, 
        transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse les abonnements pour identifier des opportunités d'optimisation.
        
        Args:
            user_id: ID de l'utilisateur
            transactions: Transactions à analyser
            
        Returns:
            List[FinancialInsight]: Insights sur les abonnements
        """
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
                        annual_cost = monthly_cost * 12
                        total_subscription_cost += monthly_cost
                        subscription_count += 1
        
        # Générer un insight si des abonnements significatifs sont détectés
        if total_subscription_cost > 50:  # Plus de 50€ d'abonnements
            insight = FinancialInsight(
                insight_id=str(uuid4()),
                user_id=user_id,
                insight_type="subscription_analysis",
                title="Analyse de vos abonnements",
                description=f"~{total_subscription_cost:.0f}€/mois en abonnements détectés ({subscription_count} services)",
                highlight=f"{total_subscription_cost:.0f}€/mois d'abonnements",
                time_scope="medium_term",
                financial_scope="spending",
                priority=3,
                numerical_value=total_subscription_cost,
                confidence_score=0.7,
                suggested_actions=[
                    "Faire le point sur vos abonnements actifs",
                    "Résilier les services non utilisés",
                    f"Économie potentielle: ~{total_subscription_cost * 0.2:.0f}€/mois"
                ],
                narrative=f"Vos abonnements représentent {total_subscription_cost:.0f}€ par mois, soit {total_subscription_cost * 12:.0f}€ par an",
                tags=["abonnements", "optimisation", "économies"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def _analyze_savings_opportunities(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Analyse les opportunités d'économies.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Opportunités d'économies
        """
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
                
                # Si l'écart max est significatif
                if max_amount > avg_amount * 2 and avg_amount > 20:
                    category_name = "Non catégorisé"
                    if category_id in self._categories_cache:
                        category_name = self._categories_cache[category_id].name
                    
                    potential_savings = (max_amount - avg_amount) * len(amounts) / 4  # Estimation conservative
                    
                    insight = FinancialInsight(
                        insight_id=str(uuid4()),
                        user_id=user_id,
                        insight_type="savings_opportunity",
                        title=f"Opportunité d'économies: {category_name}",
                        description=f"Forte variabilité dans {category_name} (moyenne: {avg_amount:.0f}€, max: {max_amount:.0f}€)",
                        highlight=f"Économie potentielle: {potential_savings:.0f}€",
                        time_scope="short_term",
                        financial_scope="spending",
                        categories_concerned=[category_name],
                        priority=3,
                        numerical_value=potential_savings,
                        confidence_score=0.6,
                        suggested_actions=[
                            f"Examiner vos dépenses en {category_name}",
                            "Comparer les prix et négocier si possible",
                            "Fixer un budget maximum pour cette catégorie"
                        ],
                        narrative=f"Optimiser vos dépenses en {category_name} pourrait vous faire économiser ~{potential_savings:.0f}€",
                        tags=["économies", "optimisation", category_name.lower()]
                    )
                    
                    insights.append(insight)
        
        return insights
    
    async def _generate_budget_insights(
        self, 
        user_id: int, 
        recent_transactions: List[RawTransaction], 
        historical_transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Génère des insights sur la gestion du budget.
        
        Args:
            user_id: ID de l'utilisateur
            recent_transactions: Transactions récentes
            historical_transactions: Transactions historiques
            
        Returns:
            List[FinancialInsight]: Insights sur le budget
        """
        insights = []
        
        # Analyser la régularité des revenus
        income_transactions = [t for t in historical_transactions if t.amount > 0]
        
        if len(income_transactions) >= 6:  # Au moins 6 revenus
            # Grouper par mois
            monthly_income = defaultdict(float)
            for transaction in income_transactions:
                month_key = transaction.date.strftime("%Y-%m")
                monthly_income[month_key] += transaction.amount
            
            if len(monthly_income) >= 2:
                income_values = list(monthly_income.values())
                avg_income = sum(income_values) / len(income_values)
                
                # Calculer la variation
                income_variation = max(income_values) - min(income_values)
                variation_percentage = (income_variation / avg_income) * 100 if avg_income > 0 else 0
                
                if variation_percentage > 25:  # Revenus variables
                    insight = FinancialInsight(
                        insight_id=str(uuid4()),
                        user_id=user_id,
                        insight_type="income_variability",
                        title="Revenus variables détectés",
                        description=f"Vos revenus varient de {variation_percentage:.0f}% (moyenne: {avg_income:.0f}€)",
                        highlight=f"Variation: {variation_percentage:.0f}%",
                        time_scope="medium_term",
                        financial_scope="budgeting",
                        priority=3,
                        numerical_value=variation_percentage,
                        confidence_score=0.8,
                        suggested_actions=[
                            "Budgeter sur la base du revenu minimum",
                            "Constituer un fonds d'urgence plus important",
                            "Lisser vos dépenses sur les mois de revenus élevés"
                        ],
                        narrative=f"Avec des revenus variables, une gestion budgétaire prudente est recommandée",
                        tags=["budget", "revenus_variables", "planification"]
                    )
                    
                    insights.append(insight)
        
        return insights
    
    async def _generate_pattern_insights(
        self, 
        user_id: int, 
        transactions: List[RawTransaction]
    ) -> List[FinancialInsight]:
        """
        Génère des insights basés sur les patterns de comportement.
        
        Args:
            user_id: ID de l'utilisateur
            transactions: Transactions à analyser
            
        Returns:
            List[FinancialInsight]: Insights sur les patterns
        """
        insights = []
        
        # Analyser les patterns journaliers
        daily_spending = defaultdict(float)
        for transaction in transactions:
            if transaction.amount < 0:
                day_name = transaction.date.strftime("%A")
                daily_spending[day_name] += abs(transaction.amount)
        
        if len(daily_spending) >= 5:  # Au moins 5 jours différents
            max_day = max(daily_spending.items(), key=lambda x: x[1])
            min_day = min(daily_spending.items(), key=lambda x: x[1])
            
            # Si l'écart est significatif
            if max_day[1] > min_day[1] * 2:
                insight = FinancialInsight(
                    insight_id=str(uuid4()),
                    user_id=user_id,
                    insight_type="spending_pattern",
                    title=f"Pattern de dépenses: forte activité le {max_day[0]}",
                    description=f"Vous dépensez plus le {max_day[0]} ({max_day[1]:.0f}€) vs {min_day[0]} ({min_day[1]:.0f}€)",
                    highlight=f"Pic le {max_day[0]}: {max_day[1]:.0f}€",
                    time_scope="medium_term",
                    financial_scope="spending",
                    priority=2,
                    numerical_value=max_day[1],
                    comparative_value=min_day[1],
                    confidence_score=0.7,
                    suggested_actions=[
                        f"Planifier vos dépenses du {max_day[0]}",
                        "Faire une liste avant les gros achats",
                        "Répartir vos achats sur la semaine"
                    ],
                    narrative=f"Vous avez tendance à dépenser plus le {max_day[0]}, ce qui peut déséquilibrer votre budget",
                    tags=["pattern", "comportement", "planification"]
                )
                
                insights.append(insight)
        
        return insights
    
    async def _filter_and_rank_insights(self, insights: List[FinancialInsight]) -> List[FinancialInsight]:
        """
        Filtre et classe les insights par importance.
        
        Args:
            insights: Liste des insights générés
            
        Returns:
            List[FinancialInsight]: Insights filtrés et classés
        """
        # Filtrer par score de confiance
        filtered_insights = [
            insight for insight in insights 
            if insight.confidence_score >= self.confidence_threshold
        ]
        
        # Éviter les doublons (même type d'insight)
        seen_types = set()
        unique_insights = []
        
        for insight in filtered_insights:
            if insight.insight_type not in seen_types:
                unique_insights.append(insight)
                seen_types.add(insight.insight_type)
        
        # Classer par priorité puis par score de confiance
        unique_insights.sort(
            key=lambda x: (x.priority, x.confidence_score), 
            reverse=True
        )
        
        # Limiter le nombre d'insights
        return unique_insights[:self.max_insights_per_user]
    
    async def _store_insight_in_qdrant(self, insight: FinancialInsight):
        """
        Stocke un insight dans Qdrant.
        
        Args:
            insight: Insight à stocker
        """
        # Générer l'embedding de l'insight
        embedding_text = f"{insight.title} {insight.description} {insight.narrative}"
        
        vector = await self.embedding_service.generate_embedding(embedding_text)
        
        # Construire le payload
        payload = {
            "id": insight.insight_id,
            "user_id": insight.user_id,
            "insight_type": insight.insight_type,
            "title": insight.title,
            "description": insight.description,
            "highlight": insight.highlight,
            
            # Portée temporelle
            "time_scope": insight.time_scope,
            "period_start": insight.period_start.isoformat() if insight.period_start else None,
            "period_end": insight.period_end.isoformat() if insight.period_end else None,
            
            # Portée financière
            "financial_scope": insight.financial_scope,
            "categories_concerned": insight.categories_concerned,
            "merchants_concerned": insight.merchants_concerned,
            "accounts_concerned": insight.accounts_concerned,
            
            # Importance
            "priority": insight.priority,
            "numerical_value": insight.numerical_value,
            "comparative_value": insight.comparative_value,
            "change_percentage": insight.change_percentage,
            
            # Données associées
            "related_transactions": insight.related_transactions,
            "related_patterns": insight.related_patterns,
            "suggested_actions": insight.suggested_actions,
            "action_impact": insight.action_impact,
            
            # Métadonnées
            "confidence_score": insight.confidence_score,
            "generation_method": insight.generation_method,
            "narrative": insight.narrative,
            "tags": insight.tags,
            
            # Dates
            "created_at": insight.created_at.isoformat(),
            "expires_at": insight.expires_at.isoformat() if insight.expires_at else None
        }
        
        # Stocker dans Qdrant
        await self.qdrant_service.upsert_point(
            collection_name="financial_insights",
            point_id=insight.insight_id,
            vector=vector,
            payload=payload
        )
    
    async def get_user_insights(self, user_id: int, insight_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les insights d'un utilisateur depuis Qdrant.
        
        Args:
            user_id: ID de l'utilisateur
            insight_type: Type d'insight à filtrer (optionnel)
            limit: Nombre maximum d'insights
            
        Returns:
            List[Dict]: Liste des insights
        """
        filter_conditions = {"user_id": user_id}
        if insight_type:
            filter_conditions["insight_type"] = insight_type
        
        try:
            results = await self.qdrant_service.search_points(
                collection_name="financial_insights",
                filter_conditions=filter_conditions,
                limit=limit
            )
            
            insights = []
            for result in results:
                insights.append(result["payload"])
            
            # Trier par priorité et date de création
            insights.sort(key=lambda x: (x.get("priority", 0), x.get("created_at", "")), reverse=True)
            
            return insights
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des insights: {e}")
            return []
    
    async def delete_user_insights(self, user_id: int) -> bool:
        """
        Supprime tous les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        try:
            return await self.qdrant_service.delete_points(
                collection_name="financial_insights",
                filter_conditions={"user_id": user_id}
            )
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des insights: {e}")
            return False
    
    async def refresh_user_insights(self, user_id: int) -> List[FinancialInsight]:
        """
        Rafraîchit tous les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[FinancialInsight]: Nouveaux insights générés
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="insight_refresh"
        )
        
        try:
            # Supprimer les anciens insights
            await self.delete_user_insights(user_id)
            ctx_logger.info("Anciens insights supprimés")
            
            # Générer de nouveaux insights
            new_insights = await self.generate_user_insights(user_id)
            ctx_logger.info(f"Génération de {len(new_insights)} nouveaux insights")
            
            return new_insights
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors du rafraîchissement des insights: {e}")
            raise
    
    async def generate_insight_summary(self, user_id: int) -> Dict[str, Any]:
        """
        Génère un résumé des insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Résumé des insights
        """
        insights = await self.get_user_insights(user_id, limit=50)
        
        if not insights:
            return {
                "total_insights": 0,
                "by_priority": {},
                "by_type": {},
                "by_scope": {},
                "key_recommendations": [],
                "summary_text": "Aucun insight disponible pour le moment."
            }
        
        # Analyser les insights
        by_priority = defaultdict(int)
        by_type = defaultdict(int)
        by_scope = defaultdict(int)
        
        high_priority_insights = []
        
        for insight in insights:
            priority = insight.get("priority", 3)
            insight_type = insight.get("insight_type", "unknown")
            financial_scope = insight.get("financial_scope", "general")
            
            by_priority[priority] += 1
            by_type[insight_type] += 1
            by_scope[financial_scope] += 1
            
            if priority >= 4:
                high_priority_insights.append(insight)
        
        # Extraire les recommandations clés
        key_recommendations = []
        for insight in high_priority_insights[:5]:  # Top 5 insights prioritaires
            actions = insight.get("suggested_actions", [])
            if actions:
                key_recommendations.extend(actions[:2])  # 2 premières actions
        
        # Générer le texte de résumé
        total_insights = len(insights)
        critical_count = by_priority.get(5, 0)
        high_count = by_priority.get(4, 0)
        
        summary_parts = [f"Vous avez {total_insights} insights actifs."]
        
        if critical_count > 0:
            summary_parts.append(f"{critical_count} insight(s) critique(s) nécessitent votre attention immédiate.")
        
        if high_count > 0:
            summary_parts.append(f"{high_count} insight(s) importante(s) à considérer.")
        
        # Domaines principaux
        top_scope = max(by_scope.items(), key=lambda x: x[1])[0] if by_scope else "général"
        summary_parts.append(f"Focus principal sur: {top_scope}.")
        
        summary_text = " ".join(summary_parts)
        
        return {
            "total_insights": total_insights,
            "by_priority": dict(by_priority),
            "by_type": dict(by_type),
            "by_scope": dict(by_scope),
            "key_recommendations": key_recommendations[:8],  # Limiter à 8 recommandations
            "high_priority_insights": len(high_priority_insights),
            "summary_text": summary_text
        }
    
    async def mark_insight_as_read(self, insight_id: str, user_id: int) -> bool:
        """
        Marque un insight comme lu par l'utilisateur.
        
        Args:
            insight_id: ID de l'insight
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si marqué avec succès
        """
        try:
            # Récupérer l'insight
            results = await self.qdrant_service.search_points(
                collection_name="financial_insights",
                filter_conditions={"id": insight_id, "user_id": user_id},
                limit=1
            )
            
            if not results:
                return False
            
            insight_payload = results[0]["payload"]
            
            # Ajouter le timestamp de lecture
            insight_payload["read_at"] = datetime.now().isoformat()
            insight_payload["status"] = "read"
            
            # Regénérer l'embedding (si nécessaire)
            embedding_text = f"{insight_payload['title']} {insight_payload['description']}"
            vector = await self.embedding_service.generate_embedding(embedding_text)
            
            # Mettre à jour dans Qdrant
            await self.qdrant_service.upsert_point(
                collection_name="financial_insights",
                point_id=insight_id,
                vector=vector,
                payload=insight_payload
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du marquage de l'insight comme lu: {e}")
            return False
    
    async def dismiss_insight(self, insight_id: str, user_id: int) -> bool:
        """
        Supprime/rejette un insight spécifique.
        
        Args:
            insight_id: ID de l'insight
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        try:
            return await self.qdrant_service.delete_points(
                collection_name="financial_insights",
                filter_conditions={"id": insight_id, "user_id": user_id}
            )
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'insight: {e}")
            return False
    
    async def get_insight_analytics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Génère des analytics sur les insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            days: Période d'analyse en jours
            
        Returns:
            Dict: Analytics des insights
        """
        insights = await self.get_user_insights(user_id, limit=100)
        
        if not insights:
            return {"message": "Aucune donnée d'analytics disponible"}
        
        # Filtrer par période
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_insights = [
            insight for insight in insights
            if datetime.fromisoformat(insight.get("created_at", "1970-01-01")) >= cutoff_date
        ]
        
        # Calculer les métriques
        total_generated = len(recent_insights)
        read_insights = len([i for i in recent_insights if i.get("read_at")])
        dismissed_insights = len([i for i in recent_insights if i.get("status") == "dismissed"])
        
        engagement_rate = (read_insights / total_generated * 100) if total_generated > 0 else 0
        
        # Analyser les types les plus générés
        type_frequency = defaultdict(int)
        for insight in recent_insights:
            type_frequency[insight.get("insight_type", "unknown")] += 1
        
        # Analyser les priorités
        priority_distribution = defaultdict(int)
        for insight in recent_insights:
            priority_distribution[insight.get("priority", 3)] += 1
        
        # Analyser les domaines financiers
        scope_distribution = defaultdict(int)
        for insight in recent_insights:
            scope_distribution[insight.get("financial_scope", "general")] += 1
        
        # Calculer l'impact potentiel
        total_potential_impact = 0
        for insight in recent_insights:
            if insight.get("action_impact"):
                total_potential_impact += insight["action_impact"]
        
        return {
            "period_days": days,
            "total_insights_generated": total_generated,
            "insights_read": read_insights,
            "insights_dismissed": dismissed_insights,
            "engagement_rate": round(engagement_rate, 1),
            "type_frequency": dict(type_frequency),
            "priority_distribution": dict(priority_distribution),
            "scope_distribution": dict(scope_distribution),
            "total_potential_impact": total_potential_impact,
            "average_confidence": round(
                sum(i.get("confidence_score", 0) for i in recent_insights) / len(recent_insights), 2
            ) if recent_insights else 0,
            "most_common_type": max(type_frequency.items(), key=lambda x: x[1])[0] if type_frequency else "none"
        }
    
    async def generate_personalized_recommendations(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Génère des recommandations personnalisées basées sur tous les insights.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            List[Dict]: Recommandations personnalisées
        """
        insights = await self.get_user_insights(user_id, limit=50)
        
        if not insights:
            return []
        
        recommendations = []
        
        # Analyser les patterns dans les insights
        spending_insights = [i for i in insights if i.get("financial_scope") == "spending"]
        saving_insights = [i for i in insights if i.get("financial_scope") == "saving"]
        
        # Recommandation globale sur les dépenses
        if len(spending_insights) >= 3:
            total_potential_savings = sum(
                i.get("numerical_value", 0) for i in spending_insights 
                if i.get("insight_type") in ["savings_opportunity", "subscription_analysis"]
            )
            
            if total_potential_savings > 50:
                recommendations.append({
                    "type": "spending_optimization",
                    "priority": "high",
                    "title": "Optimisation des dépenses",
                    "description": f"Économies potentielles identifiées: ~{total_potential_savings:.0f}€/mois",
                    "actions": [
                        "Réviser vos abonnements et services",
                        "Optimiser vos catégories de dépenses variables",
                        "Mettre en place un budget strict"
                    ],
                    "estimated_impact": total_potential_savings
                })
        
        # Recommandation sur l'épargne
        if len(saving_insights) >= 2:
            low_savings_insights = [i for i in saving_insights if "low_savings" in i.get("insight_type", "")]
            
            if low_savings_insights:
                recommendations.append({
                    "type": "savings_improvement",
                    "priority": "high",
                    "title": "Amélioration de l'épargne",
                    "description": "Votre taux d'épargne peut être optimisé",
                    "actions": [
                        "Automatiser un virement d'épargne mensuel",
                        "Fixer un objectif d'épargne de 10-20% des revenus",
                        "Explorer des placements pour faire fructifier votre épargne"
                    ],
                    "estimated_impact": "Amélioration de la sécurité financière"
                })
        
        # Recommandation sur la planification budgétaire
        budget_insights = [i for i in insights if "budget" in i.get("tags", [])]
        if len(budget_insights) >= 2:
            recommendations.append({
                "type": "budget_planning",
                "priority": "medium",
                "title": "Planification budgétaire",
                "description": "Votre gestion budgétaire pourrait être structurée",
                "actions": [
                    "Créer un budget mensuel détaillé",
                    "Suivre vos dépenses par catégorie",
                    "Planifier vos gros achats à l'avance"
                ],
                "estimated_impact": "Meilleure maîtrise des finances"
            })
        
        # Recommandation personnalisée basée sur les insights les plus fréquents
        insight_types = [i.get("insight_type", "") for i in insights]
        most_common_type = max(set(insight_types), key=insight_types.count) if insight_types else None
        
        if most_common_type and insight_types.count(most_common_type) >= 3:
            type_recommendations = {
                "spending_increase": {
                    "title": "Maîtrise des dépenses croissantes",
                    "description": "Vos dépenses ont tendance à augmenter",
                    "actions": ["Identifier les causes d'augmentation", "Mettre en place des alertes de budget"]
                },
                "category_concentration": {
                    "title": "Diversification des dépenses",
                    "description": "Vos dépenses se concentrent sur peu de catégories",
                    "actions": ["Analyser si cette concentration est justifiée", "Chercher des alternatives moins coûteuses"]
                },
                "large_transaction": {
                    "title": "Gestion des gros achats",
                    "description": "Vous effectuez régulièrement de gros achats",
                    "actions": ["Planifier les gros achats", "Constituer une réserve pour les achats importants"]
                }
            }
            
            if most_common_type in type_recommendations:
                rec = type_recommendations[most_common_type]
                recommendations.append({
                    "type": "personalized",
                    "priority": "medium",
                    "title": rec["title"],
                    "description": rec["description"],
                    "actions": rec["actions"],
                    "estimated_impact": "Amélioration ciblée"
                })
        
        return recommendations[:5]  # Limiter à 5 recommandations
    
    async def update_insights_for_transaction(self, transaction: RawTransaction):
        """
        Met à jour les insights affectés par une nouvelle transaction.
        
        Args:
            transaction: Nouvelle transaction
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=transaction.user_id,
            transaction_id=transaction.id,
            enrichment_type="insight_update"
        )
        
        try:
            # Pour l'instant, on régénère tous les insights de l'utilisateur
            # Une approche plus sophistiquée pourrait mettre à jour sélectivement
            await self.refresh_user_insights(transaction.user_id)
            ctx_logger.debug("Insights mis à jour suite à nouvelle transaction")
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors de la mise à jour des insights: {e}", exc_info=True)
    
    async def get_insight_by_id(self, insight_id: str, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupère un insight spécifique par son ID.
        
        Args:
            insight_id: ID de l'insight
            user_id: ID de l'utilisateur
            
        Returns:
            Optional[Dict]: Insight trouvé ou None
        """
        try:
            results = await self.qdrant_service.search_points(
                collection_name="financial_insights",
                filter_conditions={"id": insight_id, "user_id": user_id},
                limit=1
            )
            
            if results:
                return results[0]["payload"]
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'insight {insight_id}: {e}")
            return None
    
    async def search_insights(self, user_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recherche des insights par contenu textuel.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche
            limit: Nombre maximum de résultats
            
        Returns:
            List[Dict]: Insights correspondants
        """
        try:
            # Générer l'embedding de la requête
            query_vector = await self.embedding_service.generate_embedding(query)
            
            # Rechercher dans Qdrant
            results = await self.qdrant_service.search_similar(
                collection_name="financial_insights",
                query_vector=query_vector,
                filter_conditions={"user_id": user_id},
                limit=limit
            )
            
            insights = []
            for result in results:
                insight = result["payload"]
                insight["similarity_score"] = result.get("score", 0)
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche d'insights: {e}")
            return []
    
    async def get_insights_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Récupère les statistiques globales des insights d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Statistiques des insights
        """
        insights = await self.get_user_insights(user_id, limit=100)
        
        if not insights:
            return {
                "total_insights": 0,
                "insights_by_month": {},
                "average_priority": 0,
                "most_actionable_type": None
            }
        
        # Analyser par mois de création
        insights_by_month = defaultdict(int)
        total_priority = 0
        type_action_count = defaultdict(int)
        
        for insight in insights:
            # Par mois
            created_at = insight.get("created_at", "")
            if created_at:
                try:
                    month_key = datetime.fromisoformat(created_at).strftime("%Y-%m")
                    insights_by_month[month_key] += 1