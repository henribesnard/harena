"""
User Profile Model - Long-Term Memory
Architecture v3.0 - Phase 1

Responsabilité: Gestion profil utilisateur évolutif avec mémorisation long-terme
- Préférences explicites (catégories favorites, alertes)
- Habitudes implicites (patterns queries, spending behavior)
- Historique interactions et recommandations
- Apprentissage continu des préférences
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class NotificationPreference(Enum):
    """Préférences de notification"""
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    IN_APP = "in_app"
    NONE = "none"


@dataclass
class SpendingHabit:
    """Habitude de dépense détectée"""
    category: str
    average_monthly: float
    frequency: str  # "daily", "weekly", "monthly"
    peak_days: List[str]  # Jours de la semaine
    confidence: float  # 0.0 - 1.0


@dataclass
class QueryPattern:
    """Pattern de requête fréquent"""
    pattern_type: str  # "yoy_comparison", "category_filter", "merchant_tracking"
    frequency: int  # Nombre d'occurrences
    last_used: datetime
    parameters: Dict[str, Any]  # Paramètres typiques


@dataclass
class Alert:
    """Alerte configurée par l'utilisateur"""
    alert_id: str
    alert_type: str  # "budget_threshold", "unusual_transaction", "recurring_charge"
    parameters: Dict[str, Any]
    active: bool
    created_at: datetime
    last_triggered: Optional[datetime] = None


@dataclass
class RecommendationFeedback:
    """Feedback sur une recommandation"""
    recommendation_id: str
    recommendation_type: str
    action_taken: str  # "accepted", "dismissed", "deferred"
    timestamp: datetime
    impact_estimated: Optional[float] = None  # Impact financier estimé


class UserProfile:
    """
    Profil utilisateur évolutif avec mémoire long-terme

    Combine:
    - Préférences explicites (définies par l'utilisateur)
    - Habitudes implicites (apprises automatiquement)
    - Historique interactions
    - Métriques comportementales
    """

    def __init__(
        self,
        user_id: int,
        created_at: datetime = None,
        profile_data: Dict[str, Any] = None
    ):
        self.user_id = user_id
        self.created_at = created_at or datetime.now()
        self.last_updated = datetime.now()

        # Préférences explicites
        self.preferred_categories: List[str] = []
        self.preferred_merchants: List[str] = []
        self.notification_preferences: Dict[str, NotificationPreference] = {
            "budget_alerts": NotificationPreference.PUSH,
            "recommendations": NotificationPreference.IN_APP,
            "monthly_reports": NotificationPreference.EMAIL
        }

        # Habitudes implicites (ML/Analytics)
        self.spending_habits: List[SpendingHabit] = []
        self.frequent_query_patterns: List[QueryPattern] = []
        self.average_spending_by_category: Dict[str, float] = {}
        self.peak_spending_days: List[str] = []

        # Historique interactions
        self.accepted_recommendations: List[RecommendationFeedback] = []
        self.dismissed_recommendations: List[RecommendationFeedback] = []
        self.created_alerts: List[Alert] = []

        # Métriques comportementales
        self.query_count_total: int = 0
        self.last_active_date: Optional[datetime] = None
        self.preferred_query_complexity: str = "simple"  # "simple", "medium", "complex"

        # Scores ML (pour personnalisation)
        self.engagement_score: float = 0.5  # 0.0 - 1.0
        self.financial_literacy_score: float = 0.5  # 0.0 - 1.0
        self.recommendation_acceptance_rate: float = 0.0

        # Chargement données si fournies
        if profile_data:
            self._load_from_dict(profile_data)

        logger.info(f"UserProfile initialized for user {user_id}")

    def add_preferred_category(self, category: str):
        """Ajoute une catégorie favorite"""
        if category not in self.preferred_categories:
            self.preferred_categories.append(category)
            self.last_updated = datetime.now()
            logger.info(f"User {self.user_id} added preferred category: {category}")

    def add_preferred_merchant(self, merchant: str):
        """Ajoute un marchand favori"""
        if merchant not in self.preferred_merchants:
            self.preferred_merchants.append(merchant)
            self.last_updated = datetime.now()
            logger.info(f"User {self.user_id} added preferred merchant: {merchant}")

    def update_notification_preference(
        self,
        notification_type: str,
        preference: NotificationPreference
    ):
        """Met à jour préférence de notification"""
        self.notification_preferences[notification_type] = preference
        self.last_updated = datetime.now()

    def add_spending_habit(self, habit: SpendingHabit):
        """Ajoute ou met à jour une habitude de dépense"""
        # Remplacer si catégorie existe déjà
        self.spending_habits = [
            h for h in self.spending_habits if h.category != habit.category
        ]
        self.spending_habits.append(habit)
        self.last_updated = datetime.now()

    def track_query_pattern(self, pattern_type: str, parameters: Dict[str, Any]):
        """Enregistre un pattern de requête"""

        # Chercher pattern existant
        existing_pattern = next(
            (p for p in self.frequent_query_patterns if p.pattern_type == pattern_type),
            None
        )

        if existing_pattern:
            # Incrémenter fréquence
            existing_pattern.frequency += 1
            existing_pattern.last_used = datetime.now()
        else:
            # Nouveau pattern
            new_pattern = QueryPattern(
                pattern_type=pattern_type,
                frequency=1,
                last_used=datetime.now(),
                parameters=parameters
            )
            self.frequent_query_patterns.append(new_pattern)

        self.query_count_total += 1
        self.last_active_date = datetime.now()
        self.last_updated = datetime.now()

    def record_recommendation_feedback(
        self,
        recommendation_id: str,
        recommendation_type: str,
        action: str,
        impact: Optional[float] = None
    ):
        """Enregistre feedback sur une recommandation"""

        feedback = RecommendationFeedback(
            recommendation_id=recommendation_id,
            recommendation_type=recommendation_type,
            action_taken=action,
            timestamp=datetime.now(),
            impact_estimated=impact
        )

        if action == "accepted":
            self.accepted_recommendations.append(feedback)
        elif action == "dismissed":
            self.dismissed_recommendations.append(feedback)

        # Recalculer taux d'acceptation
        total_feedback = len(self.accepted_recommendations) + len(self.dismissed_recommendations)
        if total_feedback > 0:
            self.recommendation_acceptance_rate = (
                len(self.accepted_recommendations) / total_feedback
            )

        self.last_updated = datetime.now()

    def create_alert(
        self,
        alert_type: str,
        parameters: Dict[str, Any]
    ) -> Alert:
        """Crée une nouvelle alerte"""

        alert = Alert(
            alert_id=f"alert_{len(self.created_alerts) + 1}",
            alert_type=alert_type,
            parameters=parameters,
            active=True,
            created_at=datetime.now()
        )

        self.created_alerts.append(alert)
        self.last_updated = datetime.now()

        logger.info(f"User {self.user_id} created alert: {alert_type}")

        return alert

    def get_top_query_patterns(self, limit: int = 3) -> List[QueryPattern]:
        """Récupère les patterns de requête les plus fréquents"""

        sorted_patterns = sorted(
            self.frequent_query_patterns,
            key=lambda p: p.frequency,
            reverse=True
        )

        return sorted_patterns[:limit]

    def get_profile_completeness(self) -> float:
        """Calcule le niveau de complétude du profil (0.0 - 1.0)"""

        score = 0.0
        max_score = 10.0

        # Préférences explicites
        if self.preferred_categories:
            score += 2.0
        if self.preferred_merchants:
            score += 2.0
        if len(self.notification_preferences) >= 3:
            score += 1.0

        # Habitudes implicites
        if self.spending_habits:
            score += 2.0
        if self.frequent_query_patterns:
            score += 1.0

        # Historique
        if self.accepted_recommendations or self.dismissed_recommendations:
            score += 1.0
        if self.created_alerts:
            score += 1.0

        return min(score / max_score, 1.0)

    def suggest_shortcuts(self) -> List[Dict[str, Any]]:
        """Suggère des raccourcis basés sur les patterns fréquents"""

        suggestions = []

        top_patterns = self.get_top_query_patterns(limit=3)

        for pattern in top_patterns:
            if pattern.frequency >= 3:  # Au moins 3 utilisations
                suggestions.append({
                    "title": f"Raccourci: {pattern.pattern_type}",
                    "description": f"Utilisé {pattern.frequency} fois",
                    "parameters": pattern.parameters,
                    "action": "create_shortcut"
                })

        # Suggestion basée sur catégories favorites
        if len(self.preferred_categories) >= 2:
            suggestions.append({
                "title": "Rapport mensuel personnalisé",
                "description": f"Vos catégories: {', '.join(self.preferred_categories[:3])}",
                "action": "create_monthly_report"
            })

        return suggestions

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le profil en dictionnaire"""

        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "preferred_categories": self.preferred_categories,
            "preferred_merchants": self.preferred_merchants,
            "notification_preferences": {
                k: v.value for k, v in self.notification_preferences.items()
            },
            "spending_habits": [
                {
                    "category": h.category,
                    "average_monthly": h.average_monthly,
                    "frequency": h.frequency,
                    "peak_days": h.peak_days,
                    "confidence": h.confidence
                }
                for h in self.spending_habits
            ],
            "frequent_query_patterns": [
                {
                    "pattern_type": p.pattern_type,
                    "frequency": p.frequency,
                    "last_used": p.last_used.isoformat(),
                    "parameters": p.parameters
                }
                for p in self.frequent_query_patterns
            ],
            "average_spending_by_category": self.average_spending_by_category,
            "peak_spending_days": self.peak_spending_days,
            "accepted_recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "recommendation_type": r.recommendation_type,
                    "action_taken": r.action_taken,
                    "timestamp": r.timestamp.isoformat(),
                    "impact_estimated": r.impact_estimated
                }
                for r in self.accepted_recommendations
            ],
            "dismissed_recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "recommendation_type": r.recommendation_type,
                    "action_taken": r.action_taken,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.dismissed_recommendations
            ],
            "created_alerts": [
                {
                    "alert_id": a.alert_id,
                    "alert_type": a.alert_type,
                    "parameters": a.parameters,
                    "active": a.active,
                    "created_at": a.created_at.isoformat(),
                    "last_triggered": a.last_triggered.isoformat() if a.last_triggered else None
                }
                for a in self.created_alerts
            ],
            "query_count_total": self.query_count_total,
            "last_active_date": self.last_active_date.isoformat() if self.last_active_date else None,
            "preferred_query_complexity": self.preferred_query_complexity,
            "engagement_score": self.engagement_score,
            "financial_literacy_score": self.financial_literacy_score,
            "recommendation_acceptance_rate": self.recommendation_acceptance_rate,
            "profile_completeness": self.get_profile_completeness()
        }

    def _load_from_dict(self, data: Dict[str, Any]):
        """Charge le profil depuis un dictionnaire"""

        self.preferred_categories = data.get("preferred_categories", [])
        self.preferred_merchants = data.get("preferred_merchants", [])

        # Notification preferences
        notif_prefs = data.get("notification_preferences", {})
        self.notification_preferences = {
            k: NotificationPreference(v) for k, v in notif_prefs.items()
        }

        # Spending habits
        habits_data = data.get("spending_habits", [])
        self.spending_habits = [
            SpendingHabit(
                category=h["category"],
                average_monthly=h["average_monthly"],
                frequency=h["frequency"],
                peak_days=h["peak_days"],
                confidence=h["confidence"]
            )
            for h in habits_data
        ]

        # Query patterns
        patterns_data = data.get("frequent_query_patterns", [])
        self.frequent_query_patterns = [
            QueryPattern(
                pattern_type=p["pattern_type"],
                frequency=p["frequency"],
                last_used=datetime.fromisoformat(p["last_used"]),
                parameters=p["parameters"]
            )
            for p in patterns_data
        ]

        # Autres champs
        self.average_spending_by_category = data.get("average_spending_by_category", {})
        self.peak_spending_days = data.get("peak_spending_days", [])
        self.query_count_total = data.get("query_count_total", 0)

        last_active = data.get("last_active_date")
        if last_active:
            self.last_active_date = datetime.fromisoformat(last_active)

        self.preferred_query_complexity = data.get("preferred_query_complexity", "simple")
        self.engagement_score = data.get("engagement_score", 0.5)
        self.financial_literacy_score = data.get("financial_literacy_score", 0.5)
        self.recommendation_acceptance_rate = data.get("recommendation_acceptance_rate", 0.0)


class UserProfileManager:
    """
    Gestionnaire de profils utilisateurs

    Responsabilités:
    - Chargement/sauvegarde profils (Redis + DB)
    - Mise à jour automatique des habitudes
    - Suggestions personnalisées
    """

    def __init__(self, redis_client=None, db_client=None):
        self.redis_client = redis_client
        self.db_client = db_client

        # Cache mémoire (LRU)
        self._profile_cache: Dict[int, UserProfile] = {}
        self._cache_max_size = 100

        logger.info("UserProfileManager initialized")

    async def get_profile(self, user_id: int) -> UserProfile:
        """Récupère le profil utilisateur (avec cache)"""

        # Cache mémoire
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]

        # Cache Redis
        if self.redis_client:
            try:
                profile_data = await self._load_from_redis(user_id)
                if profile_data:
                    profile = UserProfile(user_id, profile_data=profile_data)
                    self._profile_cache[user_id] = profile
                    return profile
            except Exception as e:
                logger.warning(f"Redis load failed for user {user_id}: {str(e)}")

        # Database
        if self.db_client:
            try:
                profile_data = await self._load_from_db(user_id)
                if profile_data:
                    profile = UserProfile(user_id, profile_data=profile_data)
                    await self._save_to_redis(user_id, profile)  # Populate cache
                    self._profile_cache[user_id] = profile
                    return profile
            except Exception as e:
                logger.warning(f"DB load failed for user {user_id}: {str(e)}")

        # Nouveau profil
        profile = UserProfile(user_id)
        self._profile_cache[user_id] = profile
        await self.save_profile(profile)

        return profile

    async def save_profile(self, profile: UserProfile):
        """Sauvegarde le profil (Redis + DB)"""

        profile.last_updated = datetime.now()
        profile_data = profile.to_dict()

        # Redis (cache TTL 1h)
        if self.redis_client:
            try:
                await self._save_to_redis(profile.user_id, profile)
            except Exception as e:
                logger.error(f"Redis save failed: {str(e)}")

        # Database (persistance)
        if self.db_client:
            try:
                await self._save_to_db(profile.user_id, profile_data)
            except Exception as e:
                logger.error(f"DB save failed: {str(e)}")

        # Cache mémoire
        self._profile_cache[profile.user_id] = profile

    async def _load_from_redis(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Charge profil depuis Redis"""
        # TODO: Implémentation Redis
        return None

    async def _save_to_redis(self, user_id: int, profile: UserProfile):
        """Sauvegarde profil dans Redis"""
        # TODO: Implémentation Redis
        pass

    async def _load_from_db(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Charge profil depuis DB"""
        # TODO: Implémentation DB
        return None

    async def _save_to_db(self, user_id: int, profile_data: Dict[str, Any]):
        """Sauvegarde profil dans DB"""
        # TODO: Implémentation DB
        pass
