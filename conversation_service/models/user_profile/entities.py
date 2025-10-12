"""
User Profile Entities - Long-Term Memory and Personalization

This module defines the user profile structure for storing:
- Explicit preferences (categories, merchants, notification settings)
- Implicit habits (query patterns, spending behavior)
- Interaction history (recommendations accepted/dismissed, alerts created)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# ===== Pydantic Models (API/Business Logic) =====

class NotificationPreference(str, Enum):
    """Notification preference levels"""
    ALL = "all"
    IMPORTANT_ONLY = "important_only"
    NONE = "none"


class QueryPattern(str, Enum):
    """Common query patterns detected"""
    YOY_COMPARISONS = "yoy_comparisons"
    MOM_COMPARISONS = "mom_comparisons"
    MONTHLY_REPORTS = "monthly_reports"
    CATEGORY_ANALYSIS = "category_analysis"
    MERCHANT_TRACKING = "merchant_tracking"
    ANOMALY_DETECTION = "anomaly_detection"
    BUDGET_TRACKING = "budget_tracking"


class UserPreferences(BaseModel):
    """Explicit user preferences"""
    # Favorite categories and merchants
    preferred_categories: List[str] = Field(default_factory=list, description="User's favorite categories")
    preferred_merchants: List[str] = Field(default_factory=list, description="User's favorite merchants")

    # Notification settings
    notification_preference: NotificationPreference = Field(default=NotificationPreference.IMPORTANT_ONLY)
    email_notifications: bool = Field(default=True)
    push_notifications: bool = Field(default=False)

    # Display preferences
    currency: str = Field(default="EUR")
    date_format: str = Field(default="%Y-%m-%d")
    language: str = Field(default="fr")

    # Analysis preferences
    default_period: str = Field(default="month", description="Default analysis period")
    show_trends: bool = Field(default=True)
    show_insights: bool = Field(default=True)


class UserHabits(BaseModel):
    """Implicit user habits learned from behavior"""
    # Query patterns
    frequent_query_patterns: List[QueryPattern] = Field(default_factory=list)
    query_frequency: Dict[str, int] = Field(default_factory=dict, description="Pattern -> count")

    # Spending patterns
    average_spending_by_category: Dict[str, float] = Field(default_factory=dict)
    peak_spending_days: List[str] = Field(default_factory=list, description="Days of week with high spending")
    peak_spending_months: List[int] = Field(default_factory=list, description="Months with high spending")

    # Interaction patterns
    preferred_visualization_types: List[str] = Field(default_factory=list)
    average_session_duration_minutes: float = Field(default=0.0)
    queries_per_session: float = Field(default=0.0)


class InteractionHistory(BaseModel):
    """History of user interactions with recommendations and alerts"""
    # Recommendations
    accepted_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    dismissed_recommendations: List[Dict[str, Any]] = Field(default_factory=list)

    # Alerts and rules
    created_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    triggered_alerts_count: int = Field(default=0)

    # Feedback
    positive_feedback_count: int = Field(default=0)
    negative_feedback_count: int = Field(default=0)


class UserProfile(BaseModel):
    """Complete user profile with all preferences, habits, and history"""
    user_id: int = Field(..., description="User ID (foreign key)")

    # Sub-models
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    habits: UserHabits = Field(default_factory=UserHabits)
    interaction_history: InteractionHistory = Field(default_factory=InteractionHistory)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    profile_completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="0.0 to 1.0")

    # Analytics
    total_queries: int = Field(default=0)
    total_sessions: int = Field(default=0)
    last_active: Optional[datetime] = None

    def calculate_completeness(self) -> float:
        """Calculate profile completeness score (0.0 to 1.0)"""
        score = 0.0
        max_score = 7.0

        # Preferences (3 points max)
        if self.preferences.preferred_categories:
            score += 1.0
        if self.preferences.preferred_merchants:
            score += 1.0
        if self.preferences.notification_preference != NotificationPreference.ALL:
            score += 1.0

        # Habits (2 points max)
        if self.habits.frequent_query_patterns:
            score += 1.0
        if self.habits.average_spending_by_category:
            score += 1.0

        # Interaction history (2 points max)
        if self.interaction_history.accepted_recommendations or self.interaction_history.dismissed_recommendations:
            score += 1.0
        if self.interaction_history.created_alerts:
            score += 1.0

        return round(score / max_score, 2)

    def update_completeness(self):
        """Update the profile completeness score"""
        self.profile_completeness = self.calculate_completeness()

    def add_query_pattern(self, pattern: QueryPattern):
        """Add a detected query pattern"""
        if pattern not in self.habits.frequent_query_patterns:
            self.habits.frequent_query_patterns.append(pattern)

        # Update frequency
        pattern_str = pattern.value
        self.habits.query_frequency[pattern_str] = self.habits.query_frequency.get(pattern_str, 0) + 1

    def add_preferred_category(self, category: str):
        """Add a preferred category"""
        if category not in self.preferences.preferred_categories:
            self.preferences.preferred_categories.append(category)
            self.update_completeness()

    def add_preferred_merchant(self, merchant: str):
        """Add a preferred merchant"""
        if merchant not in self.preferences.preferred_merchants:
            self.preferences.preferred_merchants.append(merchant)
            self.update_completeness()

    def record_recommendation_feedback(self, recommendation_id: str, accepted: bool, recommendation_data: Dict[str, Any]):
        """Record user feedback on recommendation"""
        feedback_record = {
            "recommendation_id": recommendation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": recommendation_data
        }

        if accepted:
            self.interaction_history.accepted_recommendations.append(feedback_record)
            self.interaction_history.positive_feedback_count += 1
        else:
            self.interaction_history.dismissed_recommendations.append(feedback_record)
            self.interaction_history.negative_feedback_count += 1

        self.update_completeness()

    def record_alert_created(self, alert_data: Dict[str, Any]):
        """Record alert creation"""
        alert_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": alert_data
        }
        self.interaction_history.created_alerts.append(alert_record)
        self.update_completeness()


# ===== SQLAlchemy ORM Models (Database) =====

class UserProfileDB(Base):
    """SQLAlchemy model for user_profiles table"""
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, unique=True, nullable=False, index=True)

    # Preferences (JSON)
    preferred_categories = Column(JSON, default=list)
    preferred_merchants = Column(JSON, default=list)
    notification_preference = Column(String(50), default="important_only")
    email_notifications = Column(Boolean, default=True)
    push_notifications = Column(Boolean, default=False)
    currency = Column(String(10), default="EUR")
    date_format = Column(String(50), default="%Y-%m-%d")
    language = Column(String(10), default="fr")
    default_period = Column(String(20), default="month")
    show_trends = Column(Boolean, default=True)
    show_insights = Column(Boolean, default=True)

    # Habits (JSON)
    frequent_query_patterns = Column(JSON, default=list)
    query_frequency = Column(JSON, default=dict)
    average_spending_by_category = Column(JSON, default=dict)
    peak_spending_days = Column(JSON, default=list)
    peak_spending_months = Column(JSON, default=list)
    preferred_visualization_types = Column(JSON, default=list)
    average_session_duration_minutes = Column(Float, default=0.0)
    queries_per_session = Column(Float, default=0.0)

    # Interaction history (JSON)
    accepted_recommendations = Column(JSON, default=list)
    dismissed_recommendations = Column(JSON, default=list)
    created_alerts = Column(JSON, default=list)
    triggered_alerts_count = Column(Integer, default=0)
    positive_feedback_count = Column(Integer, default=0)
    negative_feedback_count = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    profile_completeness = Column(Float, default=0.0)
    total_queries = Column(Integer, default=0)
    total_sessions = Column(Integer, default=0)
    last_active = Column(DateTime, nullable=True)

    def to_pydantic(self) -> UserProfile:
        """Convert SQLAlchemy model to Pydantic model"""
        return UserProfile(
            user_id=self.user_id,
            preferences=UserPreferences(
                preferred_categories=self.preferred_categories or [],
                preferred_merchants=self.preferred_merchants or [],
                notification_preference=NotificationPreference(self.notification_preference),
                email_notifications=self.email_notifications,
                push_notifications=self.push_notifications,
                currency=self.currency,
                date_format=self.date_format,
                language=self.language,
                default_period=self.default_period,
                show_trends=self.show_trends,
                show_insights=self.show_insights
            ),
            habits=UserHabits(
                frequent_query_patterns=[QueryPattern(p) for p in (self.frequent_query_patterns or [])],
                query_frequency=self.query_frequency or {},
                average_spending_by_category=self.average_spending_by_category or {},
                peak_spending_days=self.peak_spending_days or [],
                peak_spending_months=self.peak_spending_months or [],
                preferred_visualization_types=self.preferred_visualization_types or [],
                average_session_duration_minutes=self.average_session_duration_minutes,
                queries_per_session=self.queries_per_session
            ),
            interaction_history=InteractionHistory(
                accepted_recommendations=self.accepted_recommendations or [],
                dismissed_recommendations=self.dismissed_recommendations or [],
                created_alerts=self.created_alerts or [],
                triggered_alerts_count=self.triggered_alerts_count,
                positive_feedback_count=self.positive_feedback_count,
                negative_feedback_count=self.negative_feedback_count
            ),
            created_at=self.created_at,
            last_updated=self.last_updated,
            profile_completeness=self.profile_completeness,
            total_queries=self.total_queries,
            total_sessions=self.total_sessions,
            last_active=self.last_active
        )

    @staticmethod
    def from_pydantic(profile: UserProfile) -> "UserProfileDB":
        """Create SQLAlchemy model from Pydantic model"""
        return UserProfileDB(
            user_id=profile.user_id,
            preferred_categories=profile.preferences.preferred_categories,
            preferred_merchants=profile.preferences.preferred_merchants,
            notification_preference=profile.preferences.notification_preference.value,
            email_notifications=profile.preferences.email_notifications,
            push_notifications=profile.preferences.push_notifications,
            currency=profile.preferences.currency,
            date_format=profile.preferences.date_format,
            language=profile.preferences.language,
            default_period=profile.preferences.default_period,
            show_trends=profile.preferences.show_trends,
            show_insights=profile.preferences.show_insights,
            frequent_query_patterns=[p.value for p in profile.habits.frequent_query_patterns],
            query_frequency=profile.habits.query_frequency,
            average_spending_by_category=profile.habits.average_spending_by_category,
            peak_spending_days=profile.habits.peak_spending_days,
            peak_spending_months=profile.habits.peak_spending_months,
            preferred_visualization_types=profile.habits.preferred_visualization_types,
            average_session_duration_minutes=profile.habits.average_session_duration_minutes,
            queries_per_session=profile.habits.queries_per_session,
            accepted_recommendations=profile.interaction_history.accepted_recommendations,
            dismissed_recommendations=profile.interaction_history.dismissed_recommendations,
            created_alerts=profile.interaction_history.created_alerts,
            triggered_alerts_count=profile.interaction_history.triggered_alerts_count,
            positive_feedback_count=profile.interaction_history.positive_feedback_count,
            negative_feedback_count=profile.interaction_history.negative_feedback_count,
            profile_completeness=profile.profile_completeness,
            total_queries=profile.total_queries,
            total_sessions=profile.total_sessions,
            last_active=profile.last_active
        )


class PreComputedMetric(Base):
    """SQLAlchemy model for pre_computed_metrics table

    Stores pre-calculated metrics for fast retrieval:
    - Monthly totals (spending, income)
    - Category breakdowns
    - MoM/YoY comparisons
    - 6-month trends

    Used by batch job (Celery) and cached in Redis for performance
    """
    __tablename__ = "pre_computed_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # 'monthly_total', 'category_breakdown', 'mom_comparison', 'trend_6m'
    period = Column(String(20), nullable=False, index=True)        # '2025-01', '2025-Q1', '2025'
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True, index=True)

    # Metric values (JSON for flexibility)
    metric_value = Column(JSON, nullable=False)

    # Metadata for monitoring
    computation_time_ms = Column(Integer, nullable=True)
    data_points_count = Column(Integer, nullable=True)
    cache_hit = Column(Boolean, default=False)

    def __repr__(self):
        return f"<PreComputedMetric user={self.user_id} type={self.metric_type} period={self.period}>"


__all__ = [
    "NotificationPreference",
    "QueryPattern",
    "UserPreferences",
    "UserHabits",
    "InteractionHistory",
    "UserProfile",
    "UserProfileDB",
    "PreComputedMetric"
]
