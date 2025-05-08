from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
from user_service.models.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relations
    bridge_connections = relationship("BridgeConnection", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", uselist=False, back_populates="user", cascade="all, delete-orphan")
    
    # Utiliser une référence par chaîne pour les modèles sync_service afin d'éviter les imports circulaires
    # Supprimer la relation sync_items ici - elle sera définie dans sync_service
    raw_transactions = relationship("RawTransaction", back_populates="user")
    raw_stocks = relationship("RawStock", back_populates="user")


class BridgeConnection(Base, TimestampMixin):
    __tablename__ = "bridge_connections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    bridge_user_uuid = Column(String, nullable=False)
    external_user_id = Column(String, nullable=False, unique=True)
    last_token = Column(String, nullable=True)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relations
    user = relationship("User", back_populates="bridge_connections")


class UserPreference(Base, TimestampMixin):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    notification_settings = Column(JSON, default={})
    display_preferences = Column(JSON, default={})
    budget_settings = Column(JSON, default={})
    
    # Relations
    user = relationship("User", back_populates="preferences")