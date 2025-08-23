"""
Modèles SQLAlchemy SIMPLIFIÉS pour les conversations.

Version épurée - retour aux bases pour éviter la complexité.
On développera les fonctionnalités progressivement.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from db_service.base import Base, TimestampMixin


class Conversation(Base, TimestampMixin):
    """
    Table conversation SIMPLE - juste les essentiels.
    
    Principe: partir simple et enrichir progressivement au besoin.
    """
    
    __tablename__ = "conversations"
    
    # Clé primaire
    id = Column(Integer, primary_key=True, index=True)
    
    # Identifiant unique pour l'API
    conversation_id = Column(
        String(255), 
        unique=True, 
        index=True, 
        default=lambda: str(uuid.uuid4()),
        nullable=False
    )
    
    # SÉCURITÉ CRITIQUE: Cloisonnement par utilisateur
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    # Métadonnées essentielles
    title = Column(String(500), nullable=True)  # Titre généré ou donné par l'user
    status = Column(
        String(50), 
        default="active",  # active, archived, deleted
        nullable=False, 
        index=True
    )
    
    # Compteur simple
    total_turns = Column(Integer, default=0, nullable=False)
    
    # Activité
    last_activity_at = Column(
        DateTime(timezone=True), 
        default=func.now(), 
        nullable=False, 
        index=True
    )
    
    # Données flexibles - UN SEUL champ JSON pour commencer
    # Note: "metadata" est réservé par SQLAlchemy, on utilise "data"
    data = Column(JSON, default=dict, nullable=False)
    
    # Relations
    user = relationship("User", back_populates="conversations")
    turns = relationship(
        "ConversationTurn",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationTurn.turn_number",
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id}, turns={self.total_turns})>"


class ConversationTurn(Base, TimestampMixin):
    """
    Tour de conversation SIMPLE - un message user + une réponse IA.
    
    Principe: fonctionnel d'abord, optimisé ensuite.
    """
    
    __tablename__ = "conversation_turns"
    
    # Clé primaire
    id = Column(Integer, primary_key=True, index=True)
    
    # Référence à la conversation
    conversation_id = Column(
        Integer, 
        ForeignKey("conversations.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    # Numéro du tour
    turn_number = Column(Integer, nullable=False, index=True)
    
    # Contenu - SIMPLE
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    
    # Données flexibles - UN SEUL champ pour commencer
    # Note: "metadata" est réservé par SQLAlchemy, on utilise "data"
    data = Column(JSON, default=dict, nullable=False)
    
    # Relation
    conversation = relationship("Conversation", back_populates="turns")
    
    def __repr__(self):
        return f"<ConversationTurn(id={self.id}, conv_id={self.conversation_id}, turn={self.turn_number})>"