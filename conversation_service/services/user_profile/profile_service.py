"""
User Profile Service - CRUD Operations and Business Logic

This service handles all user profile operations:
- Create/Read/Update/Delete profiles
- Update habits and preferences
- Record interactions
- Analytics on user behavior
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from conversation_service.models.user_profile.entities import (
    UserProfile,
    UserProfileDB,
    QueryPattern,
    UserPreferences,
    UserHabits,
    InteractionHistory
)

logger = logging.getLogger(__name__)


class UserProfileService:
    """Service for managing user profiles"""

    def __init__(self, db_session: Session):
        """
        Initialize User Profile Service

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session

    async def get_profile(self, user_id: int) -> Optional[UserProfile]:
        """
        Get user profile by user_id

        Args:
            user_id: User ID

        Returns:
            UserProfile if exists, None otherwise
        """
        try:
            profile_db = self.db.query(UserProfileDB).filter_by(user_id=user_id).first()

            if not profile_db:
                logger.info(f"Profile not found for user_id={user_id}")
                return None

            return profile_db.to_pydantic()

        except Exception as e:
            logger.error(f"Error fetching profile for user_id={user_id}: {str(e)}")
            raise

    async def create_profile(self, user_id: int, profile: Optional[UserProfile] = None) -> UserProfile:
        """
        Create a new user profile

        Args:
            user_id: User ID
            profile: Optional pre-filled UserProfile (defaults to empty profile)

        Returns:
            Created UserProfile
        """
        try:
            # Check if profile already exists
            existing = await self.get_profile(user_id)
            if existing:
                logger.warning(f"Profile already exists for user_id={user_id}")
                return existing

            # Create profile (use provided or default)
            if profile is None:
                profile = UserProfile(user_id=user_id)
            else:
                profile.user_id = user_id  # Ensure user_id matches

            # Convert to DB model and save
            profile_db = UserProfileDB.from_pydantic(profile)
            self.db.add(profile_db)
            self.db.commit()
            self.db.refresh(profile_db)

            logger.info(f"Created profile for user_id={user_id}")
            return profile_db.to_pydantic()

        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity error creating profile for user_id={user_id}: {str(e)}")
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating profile for user_id={user_id}: {str(e)}")
            raise

    async def update_profile(self, user_id: int, profile: UserProfile) -> UserProfile:
        """
        Update existing user profile

        Args:
            user_id: User ID
            profile: Updated UserProfile

        Returns:
            Updated UserProfile
        """
        try:
            profile_db = self.db.query(UserProfileDB).filter_by(user_id=user_id).first()

            if not profile_db:
                logger.warning(f"Profile not found for user_id={user_id}, creating new one")
                return await self.create_profile(user_id, profile)

            # Update all fields from Pydantic model
            profile_db.preferred_categories = profile.preferences.preferred_categories
            profile_db.preferred_merchants = profile.preferences.preferred_merchants
            profile_db.notification_preference = profile.preferences.notification_preference.value
            profile_db.email_notifications = profile.preferences.email_notifications
            profile_db.push_notifications = profile.preferences.push_notifications
            profile_db.currency = profile.preferences.currency
            profile_db.date_format = profile.preferences.date_format
            profile_db.language = profile.preferences.language
            profile_db.default_period = profile.preferences.default_period
            profile_db.show_trends = profile.preferences.show_trends
            profile_db.show_insights = profile.preferences.show_insights

            profile_db.frequent_query_patterns = [p.value for p in profile.habits.frequent_query_patterns]
            profile_db.query_frequency = profile.habits.query_frequency
            profile_db.average_spending_by_category = profile.habits.average_spending_by_category
            profile_db.peak_spending_days = profile.habits.peak_spending_days
            profile_db.peak_spending_months = profile.habits.peak_spending_months
            profile_db.preferred_visualization_types = profile.habits.preferred_visualization_types
            profile_db.average_session_duration_minutes = profile.habits.average_session_duration_minutes
            profile_db.queries_per_session = profile.habits.queries_per_session

            profile_db.accepted_recommendations = profile.interaction_history.accepted_recommendations
            profile_db.dismissed_recommendations = profile.interaction_history.dismissed_recommendations
            profile_db.created_alerts = profile.interaction_history.created_alerts
            profile_db.triggered_alerts_count = profile.interaction_history.triggered_alerts_count
            profile_db.positive_feedback_count = profile.interaction_history.positive_feedback_count
            profile_db.negative_feedback_count = profile.interaction_history.negative_feedback_count

            profile_db.profile_completeness = profile.profile_completeness
            profile_db.total_queries = profile.total_queries
            profile_db.total_sessions = profile.total_sessions
            profile_db.last_active = profile.last_active
            profile_db.last_updated = datetime.utcnow()

            self.db.commit()
            self.db.refresh(profile_db)

            logger.info(f"Updated profile for user_id={user_id}")
            return profile_db.to_pydantic()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating profile for user_id={user_id}: {str(e)}")
            raise

    async def delete_profile(self, user_id: int) -> bool:
        """
        Delete user profile

        Args:
            user_id: User ID

        Returns:
            True if deleted, False if not found
        """
        try:
            profile_db = self.db.query(UserProfileDB).filter_by(user_id=user_id).first()

            if not profile_db:
                logger.warning(f"Profile not found for user_id={user_id}, cannot delete")
                return False

            self.db.delete(profile_db)
            self.db.commit()

            logger.info(f"Deleted profile for user_id={user_id}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting profile for user_id={user_id}: {str(e)}")
            raise

    async def add_query_pattern(self, user_id: int, pattern: QueryPattern) -> UserProfile:
        """
        Add detected query pattern to user profile

        Args:
            user_id: User ID
            pattern: Detected query pattern

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)
        if not profile:
            profile = await self.create_profile(user_id)

        profile.add_query_pattern(pattern)
        profile.update_completeness()

        return await self.update_profile(user_id, profile)

    async def add_preferred_category(self, user_id: int, category: str) -> UserProfile:
        """
        Add preferred category to user profile

        Args:
            user_id: User ID
            category: Category name

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)
        if not profile:
            profile = await self.create_profile(user_id)

        profile.add_preferred_category(category)

        return await self.update_profile(user_id, profile)

    async def add_preferred_merchant(self, user_id: int, merchant: str) -> UserProfile:
        """
        Add preferred merchant to user profile

        Args:
            user_id: User ID
            merchant: Merchant name

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)
        if not profile:
            profile = await self.create_profile(user_id)

        profile.add_preferred_merchant(merchant)

        return await self.update_profile(user_id, profile)

    async def record_recommendation_feedback(
        self,
        user_id: int,
        recommendation_id: str,
        accepted: bool,
        recommendation_data: Dict[str, Any]
    ) -> UserProfile:
        """
        Record user feedback on recommendation

        Args:
            user_id: User ID
            recommendation_id: Recommendation ID
            accepted: Whether recommendation was accepted
            recommendation_data: Recommendation metadata

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)
        if not profile:
            profile = await self.create_profile(user_id)

        profile.record_recommendation_feedback(recommendation_id, accepted, recommendation_data)

        return await self.update_profile(user_id, profile)

    async def record_alert_created(
        self,
        user_id: int,
        alert_data: Dict[str, Any]
    ) -> UserProfile:
        """
        Record alert creation

        Args:
            user_id: User ID
            alert_data: Alert metadata

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)
        if not profile:
            profile = await self.create_profile(user_id)

        profile.record_alert_created(alert_data)

        return await self.update_profile(user_id, profile)

    async def update_session_activity(self, user_id: int) -> UserProfile:
        """
        Update user session activity

        Args:
            user_id: User ID

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)
        if not profile:
            profile = await self.create_profile(user_id)

        profile.total_sessions += 1
        profile.last_active = datetime.utcnow()

        return await self.update_profile(user_id, profile)

    async def increment_query_count(self, user_id: int) -> UserProfile:
        """
        Increment user query count

        Args:
            user_id: User ID

        Returns:
            Updated UserProfile
        """
        profile = await self.get_profile(user_id)
        if not profile:
            profile = await self.create_profile(user_id)

        profile.total_queries += 1
        profile.last_active = datetime.utcnow()

        return await self.update_profile(user_id, profile)

    async def get_or_create_profile(self, user_id: int) -> UserProfile:
        """
        Get or create user profile with graceful degradation

        Convenience method that combines get_profile and create_profile

        Args:
            user_id: User ID

        Returns:
            UserProfile (existing or newly created)
        """
        profile = await self.get_profile(user_id)
        if not profile:
            profile = await self.create_profile(user_id)
        return profile

    async def update_query_patterns(
        self,
        user_id: int,
        intent_group: str,
        intent_subtype: str
    ) -> UserProfile:
        """
        Track query patterns for implicit learning

        Updates:
        - Intent frequency (which intents user uses most)
        - Total queries counter
        - Last active timestamp
        - Detects common patterns when threshold reached

        Args:
            user_id: User ID
            intent_group: Intent group (e.g., "transaction_search", "analytics")
            intent_subtype: Intent subtype (e.g., "simple", "comparison")

        Returns:
            Updated UserProfile
        """
        profile = await self.get_or_create_profile(user_id)

        # Increment total queries
        profile.total_queries += 1

        # Update query frequency
        intent_key = f"{intent_group}.{intent_subtype}"
        profile.habits.query_frequency[intent_key] = profile.habits.query_frequency.get(intent_key, 0) + 1

        # Detect common patterns based on frequency threshold (5+ occurrences)
        if profile.habits.query_frequency[intent_key] >= 5:
            pattern = self._detect_query_pattern(intent_group, intent_subtype)
            if pattern and pattern not in profile.habits.frequent_query_patterns:
                profile.add_query_pattern(pattern)

        # Update last_active
        profile.last_active = datetime.utcnow()

        # Update completeness
        profile.update_completeness()

        return await self.update_profile(user_id, profile)

    async def update_spending_patterns(
        self,
        user_id: int,
        average_spending_by_category: Dict[str, float],
        peak_spending_days: Optional[List[str]] = None,
        peak_spending_months: Optional[List[int]] = None
    ) -> UserProfile:
        """
        Update spending patterns for user

        Args:
            user_id: User ID
            average_spending_by_category: Dict of category -> average amount
            peak_spending_days: Days of week with high spending (e.g., ["Monday", "Friday"])
            peak_spending_months: Months with high spending (e.g., [1, 12])

        Returns:
            Updated UserProfile
        """
        profile = await self.get_or_create_profile(user_id)

        # Update spending patterns
        profile.habits.average_spending_by_category = average_spending_by_category or {}

        if peak_spending_days is not None:
            profile.habits.peak_spending_days = peak_spending_days

        if peak_spending_months is not None:
            profile.habits.peak_spending_months = peak_spending_months

        # Update completeness
        profile.update_completeness()

        return await self.update_profile(user_id, profile)

    async def update_session_stats(
        self,
        user_id: int,
        session_duration_minutes: float,
        queries_in_session: int
    ) -> UserProfile:
        """
        Update session statistics for analytics

        Calculates moving averages for:
        - Average session duration
        - Average queries per session

        Args:
            user_id: User ID
            session_duration_minutes: Duration of session in minutes
            queries_in_session: Number of queries in the session

        Returns:
            Updated UserProfile
        """
        profile = await self.get_or_create_profile(user_id)

        # Increment session count
        profile.total_sessions += 1
        total_sessions = profile.total_sessions

        # Calculate moving average for session duration
        prev_avg_duration = profile.habits.average_session_duration_minutes or 0
        new_avg_duration = ((prev_avg_duration * (total_sessions - 1)) + session_duration_minutes) / total_sessions
        profile.habits.average_session_duration_minutes = round(new_avg_duration, 2)

        # Calculate moving average for queries per session
        prev_avg_queries = profile.habits.queries_per_session or 0
        new_avg_queries = ((prev_avg_queries * (total_sessions - 1)) + queries_in_session) / total_sessions
        profile.habits.queries_per_session = round(new_avg_queries, 2)

        # Update last_active
        profile.last_active = datetime.utcnow()

        return await self.update_profile(user_id, profile)

    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """
        Get user preferences with fallback to defaults

        Args:
            user_id: User ID

        Returns:
            Dict containing user preferences and stats
        """
        profile = await self.get_or_create_profile(user_id)

        return {
            'preferred_categories': profile.preferences.preferred_categories,
            'preferred_merchants': profile.preferences.preferred_merchants,
            'notification_preference': profile.preferences.notification_preference.value,
            'email_notifications': profile.preferences.email_notifications,
            'currency': profile.preferences.currency,
            'language': profile.preferences.language,
            'query_patterns': profile.habits.query_frequency,
            'total_queries': profile.total_queries,
            'total_sessions': profile.total_sessions,
            'profile_completeness': profile.profile_completeness,
            'last_active': profile.last_active.isoformat() if profile.last_active else None
        }

    def _detect_query_pattern(self, intent_group: str, intent_subtype: str) -> Optional[QueryPattern]:
        """
        Detect common query pattern from intent

        Maps intents to QueryPattern enum values for tracking user behavior

        Args:
            intent_group: Intent group
            intent_subtype: Intent subtype

        Returns:
            QueryPattern enum value or None if no pattern detected
        """
        pattern_mapping = {
            ('analytics', 'comparison'): QueryPattern.YOY_COMPARISONS,
            ('analytics', 'yoy'): QueryPattern.YOY_COMPARISONS,
            ('analytics', 'mom'): QueryPattern.MOM_COMPARISONS,
            ('analytics', 'monthly_report'): QueryPattern.MONTHLY_REPORTS,
            ('transaction_search', 'by_category'): QueryPattern.CATEGORY_ANALYSIS,
            ('transaction_search', 'by_merchant'): QueryPattern.MERCHANT_TRACKING,
            ('analytics', 'anomaly'): QueryPattern.ANOMALY_DETECTION,
            ('budget', 'tracking'): QueryPattern.BUDGET_TRACKING,
        }

        return pattern_mapping.get((intent_group, intent_subtype))

    async def get_active_profiles(self, days: int = 30, limit: int = 100) -> List[UserProfile]:
        """
        Get recently active profiles

        Args:
            days: Number of days to look back
            limit: Maximum number of profiles to return

        Returns:
            List of active UserProfiles
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            profiles_db = (
                self.db.query(UserProfileDB)
                .filter(UserProfileDB.last_active >= cutoff_date)
                .order_by(UserProfileDB.last_active.desc())
                .limit(limit)
                .all()
            )

            return [p.to_pydantic() for p in profiles_db]

        except Exception as e:
            logger.error(f"Error fetching active profiles: {str(e)}")
            raise

    async def get_profiles_by_completeness(
        self,
        min_completeness: float = 0.0,
        max_completeness: float = 1.0,
        limit: int = 100
    ) -> List[UserProfile]:
        """
        Get profiles filtered by completeness score

        Args:
            min_completeness: Minimum completeness score
            max_completeness: Maximum completeness score
            limit: Maximum number of profiles to return

        Returns:
            List of UserProfiles
        """
        try:
            profiles_db = (
                self.db.query(UserProfileDB)
                .filter(
                    UserProfileDB.profile_completeness >= min_completeness,
                    UserProfileDB.profile_completeness <= max_completeness
                )
                .order_by(UserProfileDB.profile_completeness.desc())
                .limit(limit)
                .all()
            )

            return [p.to_pydantic() for p in profiles_db]

        except Exception as e:
            logger.error(f"Error fetching profiles by completeness: {str(e)}")
            raise


__all__ = ["UserProfileService"]
