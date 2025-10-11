"""
Recommendation Engine - Actionable Financial Insights
Architecture v3.0 - Phase 2

Responsabilité: Génération de recommandations personnalisées et actionnables
- Règles métier pour optimisations financières
- Détection opportunités d'économies
- Recommandations contextuelles basées sur comportement
- Scoring de pertinence avec ML (futur)
- Tracking efficacité et A/B testing
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types de recommandations"""
    OPTIMIZE_SUBSCRIPTIONS = "optimize_subscriptions"
    DETECT_DUPLICATE = "detect_duplicate"
    BUDGET_ALERT_SETUP = "budget_alert_setup"
    CASHBACK_OPPORTUNITY = "cashback_opportunity"
    UNUSUAL_PATTERN = "unusual_pattern"
    SAVINGS_GOAL = "savings_goal"
    CATEGORY_CONSOLIDATION = "category_consolidation"
    MERCHANT_LOYALTY = "merchant_loyalty"


class Priority(Enum):
    """Niveaux de priorité"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Recommendation:
    """Recommandation actionnable"""
    recommendation_id: str
    recommendation_type: RecommendationType
    title: str
    description: str
    estimated_savings: Optional[float] = None  # Économie estimée en €
    priority: Priority = Priority.MEDIUM
    confidence: float = 0.5  # 0.0 - 1.0
    actionable: bool = True
    cta_text: str = "Voir les détails"  # Call-to-Action
    cta_action: str = ""  # Action à exécuter
    data_support: Dict[str, Any] = None  # Données justificatives
    expires_at: Optional[datetime] = None


class RecommendationEngine:
    """
    Moteur de recommandations financières personnalisées

    Génère des recommandations actionnables basées sur:
    - Règles métier (ex: 2 abonnements streaming → consolidation)
    - Patterns de dépenses (ex: dépenses inhabituelles)
    - Profil utilisateur (préférences, historique)
    - Données transactionnelles
    """

    def __init__(
        self,
        user_profile_manager=None,
        analytics_agent=None
    ):
        self.user_profile_manager = user_profile_manager
        self.analytics_agent = analytics_agent

        # Règles de recommandation (config externe dans prod)
        self.rules = self._load_recommendation_rules()

        # Statistiques
        self.stats = {
            "recommendations_generated": 0,
            "recommendations_accepted": 0,
            "recommendations_dismissed": 0,
            "total_estimated_savings": 0.0,
            "by_type": defaultdict(int)
        }

        logger.info("RecommendationEngine initialized")

    async def generate_recommendations(
        self,
        transactions: List[Dict[str, Any]],
        user_id: int,
        user_context: Dict[str, Any] = None,
        max_recommendations: int = 3
    ) -> List[Recommendation]:
        """
        Génère top N recommandations personnalisées

        Args:
            transactions: Transactions récentes utilisateur
            user_id: ID utilisateur
            user_context: Contexte additionnel (profil, historique)
            max_recommendations: Nombre max de recommandations

        Returns:
            Liste de recommandations triées par pertinence
        """
        try:
            recommendations = []

            # Charger profil utilisateur
            user_profile = None
            if self.user_profile_manager:
                user_profile = await self.user_profile_manager.get_profile(user_id)

            # Évaluer toutes les règles
            for rule in self.rules:
                if rule["active"]:
                    rec = await self._evaluate_rule(
                        rule, transactions, user_profile, user_context
                    )
                    if rec:
                        recommendations.append(rec)

            # Filtrer recommandations déjà dismissées récemment
            if user_profile:
                dismissed_types = {
                    r.recommendation_type
                    for r in user_profile.dismissed_recommendations[-10:]
                }
                recommendations = [
                    r for r in recommendations
                    if r.recommendation_type.value not in dismissed_types
                ]

            # Scorer pertinence
            for rec in recommendations:
                rec.confidence = self._calculate_relevance_score(rec, user_profile)

            # Trier par priorité et score
            recommendations.sort(
                key=lambda r: (r.priority.value, -r.confidence)
            )

            # Limiter nombre
            recommendations = recommendations[:max_recommendations]

            # Mise à jour statistiques
            self.stats["recommendations_generated"] += len(recommendations)
            for rec in recommendations:
                self.stats["by_type"][rec.recommendation_type.value] += 1
                if rec.estimated_savings:
                    self.stats["total_estimated_savings"] += rec.estimated_savings

            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []

    async def _evaluate_rule(
        self,
        rule: Dict[str, Any],
        transactions: List[Dict[str, Any]],
        user_profile: Any,
        user_context: Dict[str, Any]
    ) -> Optional[Recommendation]:
        """Évalue une règle de recommandation"""

        rule_type = rule["type"]

        try:
            if rule_type == "optimize_subscriptions":
                return await self._check_optimize_subscriptions(
                    transactions, rule, user_profile
                )

            elif rule_type == "detect_duplicate":
                return await self._check_detect_duplicate(
                    transactions, rule, user_profile
                )

            elif rule_type == "budget_alert_setup":
                return await self._check_budget_alert_setup(
                    transactions, rule, user_profile
                )

            elif rule_type == "cashback_opportunity":
                return await self._check_cashback_opportunity(
                    transactions, rule, user_profile
                )

            elif rule_type == "unusual_pattern":
                return await self._check_unusual_pattern(
                    transactions, rule, user_profile
                )

        except Exception as e:
            logger.warning(f"Error evaluating rule {rule_type}: {str(e)}")

        return None

    async def _check_optimize_subscriptions(
        self,
        transactions: List[Dict[str, Any]],
        rule: Dict[str, Any],
        user_profile: Any
    ) -> Optional[Recommendation]:
        """Détecte opportunités de consolidation d'abonnements"""

        # Identifier abonnements (catégorie streaming, téléphonie, etc.)
        subscription_categories = rule["condition"]["categories"]
        min_count = rule["condition"]["min_count"]

        subscriptions = [
            tx for tx in transactions
            if tx.get("category_name") in subscription_categories
        ]

        # Grouper par marchand
        merchants = defaultdict(list)
        for sub in subscriptions:
            merchant = sub.get("merchant_name", "Unknown")
            merchants[merchant].append(sub)

        # Vérifier si plusieurs abonnements similaires
        if len(merchants) >= min_count:
            total_cost = sum(
                abs(float(tx.get("amount", 0)))
                for tx in subscriptions
            )

            estimated_savings = total_cost * 0.20  # Estimation 20% économie

            merchant_list = ", ".join(list(merchants.keys())[:3])

            return Recommendation(
                recommendation_id=f"opt_sub_{datetime.now().timestamp()}",
                recommendation_type=RecommendationType.OPTIMIZE_SUBSCRIPTIONS,
                title="Optimisez vos abonnements",
                description=f"Vous avez {len(merchants)} abonnements actifs ({merchant_list}...). Vous pourriez économiser en consolidant ou en choisissant une offre groupée.",
                estimated_savings=estimated_savings,
                priority=Priority.HIGH,
                confidence=0.8,
                cta_text="Voir mes abonnements",
                cta_action="show_subscriptions_detail",
                data_support={
                    "merchants": list(merchants.keys()),
                    "total_monthly_cost": total_cost,
                    "count": len(merchants)
                }
            )

        return None

    async def _check_detect_duplicate(
        self,
        transactions: List[Dict[str, Any]],
        rule: Dict[str, Any],
        user_profile: Any
    ) -> Optional[Recommendation]:
        """Détecte potentiels doublons de transactions"""

        # Grouper par date et montant
        date_amount_groups = defaultdict(list)

        for tx in transactions:
            date = tx.get("date", "")[:10]  # YYYY-MM-DD
            amount = abs(float(tx.get("amount", 0)))
            key = f"{date}_{amount}"
            date_amount_groups[key].append(tx)

        # Chercher doublons (même jour, même montant, même marchand)
        duplicates = []

        for key, group in date_amount_groups.items():
            if len(group) >= 2:
                # Vérifier si même marchand
                merchants = {tx.get("merchant_name") for tx in group}
                if len(merchants) == 1:
                    duplicates.extend(group)

        if duplicates:
            first_duplicate = duplicates[0]
            amount = abs(float(first_duplicate.get("amount", 0)))
            merchant = first_duplicate.get("merchant_name", "N/A")

            return Recommendation(
                recommendation_id=f"dup_{datetime.now().timestamp()}",
                recommendation_type=RecommendationType.DETECT_DUPLICATE,
                title="Transaction potentiellement dupliquée",
                description=f"Deux transactions identiques de {amount}€ chez {merchant} le même jour ont été détectées. Vérifiez qu'il ne s'agit pas d'un doublon.",
                estimated_savings=amount,  # Potentiel remboursement
                priority=Priority.HIGH,
                confidence=0.75,
                cta_text="Vérifier les transactions",
                cta_action="review_duplicates",
                data_support={
                    "duplicate_transactions": [tx.get("id") for tx in duplicates],
                    "amount": amount,
                    "merchant": merchant
                }
            )

        return None

    async def _check_budget_alert_setup(
        self,
        transactions: List[Dict[str, Any]],
        rule: Dict[str, Any],
        user_profile: Any
    ) -> Optional[Recommendation]:
        """Suggère création d'alertes budget"""

        # Vérifier si utilisateur n'a pas d'alerte configurée
        if user_profile and user_profile.created_alerts:
            return None  # Déjà configuré

        # Calculer dépenses par catégorie
        category_spending = defaultdict(float)

        for tx in transactions:
            if tx.get("transaction_type") == "debit":
                category = tx.get("category_name", "Autres")
                amount = abs(float(tx.get("amount", 0)))
                category_spending[category] += amount

        # Identifier catégorie principale
        if not category_spending:
            return None

        top_category, top_amount = max(
            category_spending.items(),
            key=lambda x: x[1]
        )

        threshold = rule["condition"]["percentage"]

        # Seuil suggéré (10% au-dessus de la dépense moyenne)
        suggested_threshold = top_amount * (1 + threshold / 100)

        return Recommendation(
            recommendation_id=f"alert_{datetime.now().timestamp()}",
            recommendation_type=RecommendationType.BUDGET_ALERT_SETUP,
            title=f"Créez une alerte budget pour {top_category}",
            description=f"Vous dépensez en moyenne {top_amount:.0f}€/mois en {top_category}. Configurez une alerte à {suggested_threshold:.0f}€ pour suivre votre budget.",
            estimated_savings=None,
            priority=Priority.MEDIUM,
            confidence=0.70,
            cta_text="Créer l'alerte",
            cta_action="create_budget_alert",
            data_support={
                "category": top_category,
                "current_spending": top_amount,
                "suggested_threshold": suggested_threshold
            }
        )

    async def _check_cashback_opportunity(
        self,
        transactions: List[Dict[str, Any]],
        rule: Dict[str, Any],
        user_profile: Any
    ) -> Optional[Recommendation]:
        """Détecte opportunités de cashback"""

        # Marchands avec programmes cashback connus
        cashback_merchants = rule["condition"].get("merchants", [])
        min_monthly_spending = rule["condition"].get("min_spending", 100)

        # Calculer dépenses par marchand
        merchant_spending = defaultdict(float)

        for tx in transactions:
            merchant = tx.get("merchant_name", "")
            if merchant in cashback_merchants:
                amount = abs(float(tx.get("amount", 0)))
                merchant_spending[merchant] += amount

        # Vérifier si éligible
        for merchant, amount in merchant_spending.items():
            if amount >= min_monthly_spending:
                cashback_rate = 0.03  # 3% estimation
                estimated_cashback = amount * cashback_rate

                return Recommendation(
                    recommendation_id=f"cashback_{datetime.now().timestamp()}",
                    recommendation_type=RecommendationType.CASHBACK_OPPORTUNITY,
                    title=f"Cashback disponible chez {merchant}",
                    description=f"Vous dépensez {amount:.0f}€/mois chez {merchant}. Activez une carte avec cashback pour économiser environ {estimated_cashback:.0f}€/mois.",
                    estimated_savings=estimated_cashback,
                    priority=Priority.MEDIUM,
                    confidence=0.65,
                    cta_text="Découvrir les offres",
                    cta_action="explore_cashback",
                    data_support={
                        "merchant": merchant,
                        "monthly_spending": amount,
                        "cashback_rate": cashback_rate
                    }
                )

        return None

    async def _check_unusual_pattern(
        self,
        transactions: List[Dict[str, Any]],
        rule: Dict[str, Any],
        user_profile: Any
    ) -> Optional[Recommendation]:
        """Détecte patterns inhabituels nécessitant attention"""

        if not self.analytics_agent or not transactions:
            return None

        # Utiliser Analytics Agent pour détection anomalies
        from .analytics_agent import AnomalyDetectionMethod

        anomalies = await self.analytics_agent.detect_anomalies(
            transactions,
            method=AnomalyDetectionMethod.Z_SCORE,
            threshold=2.0
        )

        if anomalies:
            # Prendre première anomalie (plus significative)
            anomaly = anomalies[0]

            return Recommendation(
                recommendation_id=f"unusual_{datetime.now().timestamp()}",
                recommendation_type=RecommendationType.UNUSUAL_PATTERN,
                title="Dépense inhabituelle détectée",
                description=anomaly.reason,
                estimated_savings=None,
                priority=Priority.HIGH,
                confidence=0.80,
                cta_text="Voir la transaction",
                cta_action="show_transaction_detail",
                data_support={
                    "transaction_id": anomaly.transaction_id,
                    "amount": anomaly.amount,
                    "merchant": anomaly.merchant,
                    "anomaly_score": anomaly.anomaly_score
                }
            )

        return None

    def _calculate_relevance_score(
        self,
        recommendation: Recommendation,
        user_profile: Any
    ) -> float:
        """
        Calcule score de pertinence (0.0 - 1.0)

        Facteurs:
        - Impact financier estimé (€)
        - Pertinence historique (a-t-il accepté ce type avant ?)
        - Urgence (risque)
        - Facilité d'implémentation
        """
        score = recommendation.confidence

        # Facteur impact financier
        if recommendation.estimated_savings:
            if recommendation.estimated_savings > 50:
                score += 0.2
            elif recommendation.estimated_savings > 20:
                score += 0.1

        # Facteur historique utilisateur
        if user_profile:
            rec_type = recommendation.recommendation_type.value

            # A-t-il accepté ce type avant ?
            accepted_types = {
                r.recommendation_type
                for r in user_profile.accepted_recommendations
            }

            if rec_type in accepted_types:
                score += 0.15  # Boost si déjà accepté similaire

            # A-t-il dismissé ce type récemment ?
            recent_dismissed = [
                r.recommendation_type
                for r in user_profile.dismissed_recommendations[-5:]
            ]

            if rec_type in recent_dismissed:
                score -= 0.2  # Pénalité si dismissé récemment

        # Normaliser [0.0 - 1.0]
        return min(max(score, 0.0), 1.0)

    def _load_recommendation_rules(self) -> List[Dict[str, Any]]:
        """Charge les règles de recommandation"""

        # Règles intégrées (en prod: charger depuis DB/config)
        return [
            {
                "type": "optimize_subscriptions",
                "active": True,
                "condition": {
                    "categories": ["streaming", "Téléphones/internet", "Abonnements"],
                    "min_count": 2
                }
            },
            {
                "type": "detect_duplicate",
                "active": True,
                "condition": {}
            },
            {
                "type": "budget_alert_setup",
                "active": True,
                "condition": {
                    "percentage": 10  # Seuil suggéré: +10% au-dessus moyenne
                }
            },
            {
                "type": "cashback_opportunity",
                "active": True,
                "condition": {
                    "merchants": ["Amazon", "Carrefour", "Leclerc"],
                    "min_spending": 100
                }
            },
            {
                "type": "unusual_pattern",
                "active": True,
                "condition": {}
            }
        ]

    async def record_recommendation_action(
        self,
        recommendation_id: str,
        user_id: int,
        action: str,  # "accepted", "dismissed", "deferred"
        user_profile: Any = None
    ):
        """Enregistre l'action utilisateur sur une recommandation"""

        if action == "accepted":
            self.stats["recommendations_accepted"] += 1
        elif action == "dismissed":
            self.stats["recommendations_dismissed"] += 1

        # Mettre à jour profil utilisateur
        if user_profile:
            user_profile.record_recommendation_feedback(
                recommendation_id=recommendation_id,
                recommendation_type="",  # À récupérer depuis la recommandation
                action=action
            )

            if self.user_profile_manager:
                await self.user_profile_manager.save_profile(user_profile)

        logger.info(f"Recommendation {recommendation_id} action: {action} by user {user_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du moteur"""

        acceptance_rate = 0.0
        total_feedback = (
            self.stats["recommendations_accepted"] +
            self.stats["recommendations_dismissed"]
        )

        if total_feedback > 0:
            acceptance_rate = (
                self.stats["recommendations_accepted"] / total_feedback
            )

        return {
            **self.stats,
            "acceptance_rate": acceptance_rate
        }
