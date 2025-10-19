"""
Modèles de données pour le module de profilage budgétaire
"""
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, Numeric, Date, JSON, Text, UniqueConstraint, and_
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from db_service.base import Base, TimestampMixin


class UserBudgetProfile(Base, TimestampMixin):
    """
    Profil budgétaire utilisateur
    Stocke les métriques financières et le comportement de l'utilisateur
    """
    __tablename__ = "user_budget_profile"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)

    # Segmentation
    user_segment = Column(String(50))  # 'budget_serré', 'équilibré', 'confortable'
    behavioral_pattern = Column(String(50))  # 'dépensier_hebdomadaire', 'acheteur_impulsif', 'planificateur'

    # Métriques moyennes (3 derniers mois)
    avg_monthly_income = Column(Numeric(10, 2))
    avg_monthly_expenses = Column(Numeric(10, 2))
    avg_monthly_savings = Column(Numeric(10, 2))
    savings_rate = Column(Numeric(5, 2))  # Pourcentage

    # Répartition charges
    fixed_charges_total = Column(Numeric(10, 2))
    semi_fixed_charges_total = Column(Numeric(10, 2))
    variable_charges_total = Column(Numeric(10, 2))

    # Reste à vivre après charges fixes
    remaining_to_live = Column(Numeric(10, 2))

    # Métadonnées
    profile_completeness = Column(Numeric(3, 2))  # 0.0 - 1.0
    last_analyzed_at = Column(DateTime(timezone=True))

    # Relations
    user = relationship("User", back_populates="budget_profile")
    fixed_charges = relationship(
        "FixedCharge",
        primaryjoin="UserBudgetProfile.user_id == FixedCharge.user_id",
        foreign_keys="[FixedCharge.user_id]"
    )
    savings_goals = relationship(
        "SavingsGoal",
        primaryjoin="UserBudgetProfile.user_id == SavingsGoal.user_id",
        foreign_keys="[SavingsGoal.user_id]"
    )
    recommendations = relationship(
        "BudgetRecommendation",
        primaryjoin="UserBudgetProfile.user_id == BudgetRecommendation.user_id",
        foreign_keys="[BudgetRecommendation.user_id]"
    )
    seasonal_patterns = relationship(
        "SeasonalPattern",
        primaryjoin="UserBudgetProfile.user_id == SeasonalPattern.user_id",
        foreign_keys="[SeasonalPattern.user_id]"
    )


class FixedCharge(Base, TimestampMixin):
    """
    Charge fixe détectée automatiquement
    Dépenses récurrentes mensuelles avec montant stable
    """
    __tablename__ = "fixed_charges"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Identification
    merchant_name = Column(String(255))
    category = Column(String(100))

    # Caractéristiques récurrence
    avg_amount = Column(Numeric(10, 2))
    amount_variance = Column(Numeric(5, 2))  # Pourcentage de variance
    recurrence_day = Column(Integer)  # Jour du mois (1-31)
    recurrence_confidence = Column(Numeric(3, 2))  # 0.0 - 1.0

    # Statut
    validated_by_user = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Historique
    first_detected_date = Column(Date)
    last_transaction_date = Column(Date)
    transaction_count = Column(Integer)

    # Relations
    user_profile = relationship(
        "UserBudgetProfile",
        primaryjoin="and_(FixedCharge.user_id == UserBudgetProfile.user_id)",
        foreign_keys="[FixedCharge.user_id]",
        uselist=False,
        viewonly=True
    )

    __table_args__ = (
        UniqueConstraint('user_id', 'merchant_name', name='uq_user_merchant'),
    )


class SavingsGoal(Base, TimestampMixin):
    """
    Objectif d'épargne défini par l'utilisateur
    """
    __tablename__ = "savings_goals"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Objectif
    goal_name = Column(String(255))  # "Vacances été", "Réserve urgence"
    target_amount = Column(Numeric(10, 2))
    target_date = Column(Date)

    # Progression
    current_amount = Column(Numeric(10, 2), default=0)
    monthly_contribution = Column(Numeric(10, 2))

    # Plan d'épargne suggéré
    suggested_categories = Column(JSON)  # [{"category": "loisirs", "reduction_pct": 20}]

    # Statut
    status = Column(String(50), default='active')  # 'active', 'completed', 'abandoned'

    # Relations
    user_profile = relationship(
        "UserBudgetProfile",
        primaryjoin="and_(SavingsGoal.user_id == UserBudgetProfile.user_id)",
        foreign_keys="[SavingsGoal.user_id]",
        uselist=False,
        viewonly=True
    )


class BudgetRecommendation(Base, TimestampMixin):
    """
    Recommandation budgétaire personnalisée
    """
    __tablename__ = "budget_recommendations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Type de recommandation
    recommendation_type = Column(String(100))  # 'monthly_savings', 'savings_goal', 'alert', 'optimization'
    priority = Column(String(20))  # 'high', 'medium', 'low'

    # Contenu
    title = Column(String(255))
    description = Column(Text)
    estimated_savings = Column(Numeric(10, 2))

    # Actions suggérées
    actions = Column(JSON)  # [{"type": "create_alert", "parameters": {...}}]

    # Feedback utilisateur
    status = Column(String(50), default='pending')  # 'pending', 'accepted', 'rejected', 'ignored'
    user_feedback = Column(Text)
    feedback_timestamp = Column(DateTime(timezone=True))

    # Efficacité (tracking)
    actual_impact = Column(Numeric(10, 2))

    # Expiration
    expires_at = Column(DateTime(timezone=True))

    # Relations
    user_profile = relationship(
        "UserBudgetProfile",
        primaryjoin="and_(BudgetRecommendation.user_id == UserBudgetProfile.user_id)",
        foreign_keys="[BudgetRecommendation.user_id]",
        uselist=False,
        viewonly=True
    )


class SeasonalPattern(Base, TimestampMixin):
    """
    Pattern saisonnier détecté dans les dépenses
    """
    __tablename__ = "seasonal_patterns"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Période
    month = Column(Integer)  # 1-12
    pattern_type = Column(String(50))  # 'high_spending', 'multiple_bills', 'vacation', 'holiday'

    # Métriques
    avg_amount = Column(Numeric(10, 2))
    variance_vs_avg = Column(Numeric(5, 2))  # Pourcentage de variance vs moyenne annuelle

    # Description
    description = Column(Text)
    key_expenses = Column(JSON)  # [{"merchant": "...", "amount": ..., "category": "..."}]

    # Confiance
    confidence = Column(Numeric(3, 2))  # 0.0 - 1.0
    years_data = Column(Integer)  # Nombre d'années de données utilisées

    # Relations
    user_profile = relationship(
        "UserBudgetProfile",
        primaryjoin="and_(SeasonalPattern.user_id == UserBudgetProfile.user_id)",
        foreign_keys="[SeasonalPattern.user_id]",
        uselist=False,
        viewonly=True
    )

    __table_args__ = (
        UniqueConstraint('user_id', 'month', 'pattern_type', name='uq_user_month_pattern'),
    )
