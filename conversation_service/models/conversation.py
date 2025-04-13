"""
Modèles Pydantic pour représenter les conversations et sessions.

Ce module définit les modèles de données pour les conversations,
messages, et sessions utilisés dans l'API et la logique métier.
"""

import uuid
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ConversationState(str, Enum):
    """États possibles d'une conversation."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MessageRole(str, Enum):
    """Rôles possibles d'un message dans une conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageBase(BaseModel):
    """Modèle de base pour un message."""
    conversation_id: uuid.UUID
    role: MessageRole
    content: str
    meta_data: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None


class MessageCreate(MessageBase):
    """Modèle pour la création d'un message."""
    pass


class MessageRead(MessageBase):
    """Modèle pour la lecture d'un message."""
    id: uuid.UUID
    processed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    """Modèle de base pour une conversation."""
    user_id: int
    title: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None


class ConversationCreate(ConversationBase):
    """Modèle pour la création d'une conversation."""
    pass


class ConversationUpdate(BaseModel):
    """Modèle pour la mise à jour d'une conversation."""
    title: Optional[str] = None
    state: Optional[ConversationState] = None
    meta_data: Optional[Dict[str, Any]] = None


class ConversationRead(ConversationBase):
    """Modèle pour la lecture d'une conversation."""
    id: uuid.UUID
    state: ConversationState
    last_activity: datetime
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = None

    class Config:
        from_attributes = True


class ConversationDetail(ConversationRead):
    """Modèle détaillé pour une conversation avec ses messages."""
    messages: List[MessageRead]


class ConversationContext(BaseModel):
    """Contexte d'une conversation pour le traitement des messages."""
    conversation_id: uuid.UUID
    user_id: int
    messages: List[Dict[str, str]]
    active_intent: Optional[str] = None
    last_query_data: Optional[Dict[str, Any]] = None
    last_response_data: Optional[Dict[str, Any]] = None
    meta_data: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Réponse à une conversation contenant le message et des métadonnées."""
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    content: str
    token_count: int
    created_at: datetime
    intent: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None