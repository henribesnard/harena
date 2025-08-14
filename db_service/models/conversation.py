# db_service/models/conversation.py
"""
Modèles SQLAlchemy pour les conversations IA.

Ce module définit les tables PostgreSQL pour stocker les conversations
entre utilisateurs et le service d'IA, avec cloisonnement par utilisateur.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone
import uuid

from db_service.base import Base, TimestampMixin


class Conversation(Base, TimestampMixin):
    """
    Table principale pour stocker les conversations complètes.
    
    Chaque conversation appartient à un utilisateur spécifique et contient
    des métadonnées globales sur la session conversationnelle.
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
    
    # Cloisonnement par utilisateur - SÉCURITÉ CRITIQUE
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    # Métadonnées conversation
    title = Column(String(500), nullable=True)  # Titre généré automatiquement
    status = Column(
        String(50), 
        default="active",  # active, archived, deleted
        nullable=False, 
        index=True
    )
    
    # Contexte et paramètres
    language = Column(String(10), default="fr", nullable=False)
    domain = Column(String(100), default="financial", nullable=False)
    
    # Compteurs
    total_turns = Column(Integer, default=0, nullable=False)
    max_turns = Column(Integer, default=50, nullable=False)
    
    # Activité
    last_activity_at = Column(
        DateTime(timezone=True), 
        default=func.now(), 
        nullable=False, 
        index=True
    )
    
    # Métadonnées JSON (metadata est réservé par SQLAlchemy)
    conversation_metadata = Column(JSON, default=dict, nullable=False)
    user_preferences = Column(JSON, default=dict, nullable=False)
    session_metadata = Column(JSON, default=dict, nullable=False)
    
    # Relations
    user = relationship("User", back_populates="conversations")
    turns = relationship(
        "ConversationTurn", 
        back_populates="conversation", 
        cascade="all, delete-orphan",
        order_by="ConversationTurn.turn_number"
    )
    summaries = relationship(
        "ConversationSummary", 
        back_populates="conversation", 
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id}, turns={self.total_turns})>"


class ConversationTurn(Base, TimestampMixin):
    """
    Table pour stocker chaque tour de conversation (user message + AI response).
    
    Chaque tour contient le message utilisateur, la réponse de l'IA,
    et toutes les métadonnées de traitement associées.
    """
    
    __tablename__ = "conversation_turns"
    
    # Clé primaire
    id = Column(Integer, primary_key=True, index=True)
    
    # Identifiant unique pour l'API
    turn_id = Column(
        String(255), 
        unique=True, 
        index=True, 
        default=lambda: str(uuid.uuid4()),
        nullable=False
    )
    
    # Référence à la conversation parent
    conversation_id = Column(
        Integer, 
        ForeignKey("conversations.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    # Numéro du tour dans la conversation
    turn_number = Column(Integer, nullable=False, index=True)
    
    # Contenu des messages
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    
    # Métriques de traitement
    processing_time_ms = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    error_occurred = Column(Boolean, default=False, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Intelligence artificielle - métadonnées
    intent_result = Column(JSON, nullable=True)
    agent_chain = Column(JSON, default=list, nullable=False)  # Séquence d'agents utilisés
    
    # Recherche et résultats
    search_query_used = Column(Text, nullable=True)
    search_results_count = Column(Integer, default=0, nullable=False)
    search_execution_time_ms = Column(Float, nullable=True)
    
    # Métadonnées flexibles (metadata est réservé par SQLAlchemy)
    turn_metadata = Column(JSON, default=dict, nullable=False)
    
    # Relations
    conversation = relationship("Conversation", back_populates="turns")
    
    def __repr__(self):
        return f"<ConversationTurn(id={self.id}, conv_id={self.conversation_id}, turn={self.turn_number})>"


class ConversationSummary(Base, TimestampMixin):
    """
    Table pour stocker des résumés de conversations longues.
    
    Permet de maintenir le contexte tout en optimisant les performances
    pour les conversations avec de nombreux tours.
    """
    
    __tablename__ = "conversation_summaries"
    
    # Clé primaire
    id = Column(Integer, primary_key=True, index=True)
    
    # Référence à la conversation
    conversation_id = Column(
        Integer, 
        ForeignKey("conversations.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    # Période résumée
    start_turn = Column(Integer, nullable=False)
    end_turn = Column(Integer, nullable=False)
    
    # Contenu du résumé
    summary_text = Column(Text, nullable=False)
    key_topics = Column(JSON, default=list, nullable=False)
    important_entities = Column(JSON, default=list, nullable=False)
    
    # Métadonnées du résumé
    summary_method = Column(String(50), default="auto", nullable=False)  # auto, manual
    
    # Relations
    conversation = relationship("Conversation", back_populates="summaries")
    
    def __repr__(self):
        return f"<ConversationSummary(conv_id={self.conversation_id}, turns={self.start_turn}-{self.end_turn})>"