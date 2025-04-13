"""
Modèles de base de données SQLAlchemy pour le service de conversation.

Ce module définit les modèles SQLAlchemy qui représentent les entités
persistantes dans la base de données: conversations et messages.
"""

import uuid
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from .session import Base


class ConversationState(enum.Enum):
    """État d'une conversation."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MessageRole(enum.Enum):
    """Rôle d'un message dans une conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    """Modèle représentant une conversation entre un utilisateur et l'assistant."""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, nullable=False, index=True)
    title = Column(String(255), nullable=True)
    state = Column(Enum(ConversationState), default=ConversationState.ACTIVE, nullable=False)
    meta_data = Column(JSON, default={}, nullable=False)
    last_activity = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relations
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id='{self.id}', user_id='{self.user_id}', title='{self.title}')>"


class Message(Base):
    """Modèle représentant un message dans une conversation."""
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    meta_data = Column(JSON, default={}, nullable=False)
    token_count = Column(Integer, default=0, nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relations
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id='{self.id}', role='{self.role}', processed='{self.processed}')>"


class ConversationContext(Base):
    """Modèle représentant le contexte d'une conversation."""
    __tablename__ = "conversation_contexts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, unique=True)
    context_data = Column(JSON, default={}, nullable=False)
    active_intent = Column(String(50), nullable=True)
    last_query_data = Column(JSON, nullable=True)
    last_response_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<ConversationContext(id='{self.id}', conversation_id='{self.conversation_id}', active_intent='{self.active_intent}')>"