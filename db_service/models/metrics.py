"""
SQLAlchemy Models pour les Métriques - Harena Metrics Service
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class MetricsCache(Base):
    """Cache des métriques calculées"""
    __tablename__ = "metrics_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)
    metric_key = Column(String(255), nullable=False)  # Clé unique pour params
    data = Column(JSON, nullable=False)
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index('idx_cache_lookup', 'user_id', 'metric_type', 'metric_key'),
    )

class MetricsHistory(Base):
    """Historique des métriques pour analyse temporelle"""
    __tablename__ = "metrics_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    value = Column(Float, nullable=True)  # Valeur principale de la métrique
    data = Column(JSON, nullable=False)  # Données complètes
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_history_lookup', 'user_id', 'metric_type', 'period_start'),
    )

class UserMetricsConfig(Base):
    """Configuration des métriques par utilisateur"""
    __tablename__ = "user_metrics_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, unique=True)

    # Seuils personnalisés
    savings_rate_target = Column(Float, default=20.0)  # % cible
    expense_ratio_essentials = Column(Float, default=50.0)
    expense_ratio_lifestyle = Column(Float, default=30.0)
    expense_ratio_savings = Column(Float, default=20.0)

    # Alertes
    enable_low_balance_alert = Column(Boolean, default=True)
    low_balance_threshold = Column(Float, default=100.0)
    enable_burn_rate_alert = Column(Boolean, default=True)
    burn_rate_runway_threshold = Column(Integer, default=30)  # jours

    # Préférences de calcul
    forecast_default_periods = Column(Integer, default=90)
    recurring_detection_min_occurrences = Column(Integer, default=3)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
