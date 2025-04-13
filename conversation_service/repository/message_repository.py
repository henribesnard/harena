"""
Accès aux données des messages en base.

Ce module fournit une couche d'abstraction pour l'accès
aux messages dans la base de données.
"""

import uuid
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc
from datetime import datetime

from ..db.models import Message, MessageRole
from ..models.conversation import MessageCreate


class MessageRepository:
    """
    Repository pour les opérations CRUD sur les messages.
    
    Cette classe encapsule toutes les opérations de base de données
    liées aux messages.
    """
    
    def __init__(self, db: Session):
        """
        Initialise le repository avec une session de base de données.
        
        Args:
            db: Session de base de données SQLAlchemy
        """
        self.db = db
    
    async def get_message_by_id(self, message_id: uuid.UUID) -> Optional[Message]:
        """
        Récupère un message par son ID.
        
        Args:
            message_id: ID du message
            
        Returns:
            Message ou None si non trouvé
        """
        return self.db.query(Message).filter(Message.id == message_id).first()
    
    async def get_conversation_messages(
        self,
        conversation_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
        oldest_first: bool = True
    ) -> List[Message]:
        """
        Récupère les messages d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            skip: Nombre d'éléments à sauter (pour la pagination)
            limit: Nombre d'éléments à récupérer
            oldest_first: Ordonner du plus ancien au plus récent
            
        Returns:
            Liste des messages
        """
        query = self.db.query(Message).filter(Message.conversation_id == conversation_id)
        
        # Ordonner par date
        if oldest_first:
            query = query.order_by(asc(Message.created_at))
        else:
            query = query.order_by(desc(Message.created_at))
        
        # Appliquer la pagination
        return query.offset(skip).limit(limit).all()
    
    async def create_message(self, message_data: MessageCreate) -> Message:
        """
        Crée un nouveau message.
        
        Args:
            message_data: Données du message à créer
            
        Returns:
            Message créé
        """
        # Créer le message
        db_message = Message(
            conversation_id=message_data.conversation_id,
            role=getattr(MessageRole, message_data.role.upper()),
            content=message_data.content,
            meta_data=message_data.meta_data or {},
            token_count=message_data.token_count or 0
        )
        
        # Persister en base
        self.db.add(db_message)
        self.db.commit()
        self.db.refresh(db_message)
        
        return db_message
    
    async def create_user_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        token_count: int = 0,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Crée un message utilisateur.
        
        Args:
            conversation_id: ID de la conversation
            content: Contenu du message
            token_count: Nombre de tokens
            meta_data: Métadonnées du message
            
        Returns:
            Message créé
        """
        # Créer le message
        db_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
            meta_data=meta_data or {},
            token_count=token_count
        )
        
        # Persister en base
        self.db.add(db_message)
        self.db.commit()
        self.db.refresh(db_message)
        
        return db_message
    
    async def create_assistant_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        token_count: int = 0,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Crée un message assistant.
        
        Args:
            conversation_id: ID de la conversation
            content: Contenu du message
            token_count: Nombre de tokens
            meta_data: Métadonnées du message
            
        Returns:
            Message créé
        """
        # Créer le message
        db_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=content,
            meta_data=meta_data or {},
            token_count=token_count
        )
        
        # Persister en base
        self.db.add(db_message)
        self.db.commit()
        self.db.refresh(db_message)
        
        return db_message
    
    async def create_system_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        token_count: int = 0,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Crée un message système.
        
        Args:
            conversation_id: ID de la conversation
            content: Contenu du message
            token_count: Nombre de tokens
            meta_data: Métadonnées du message
            
        Returns:
            Message créé
        """
        # Créer le message
        db_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.SYSTEM,
            content=content,
            meta_data=meta_data or {},
            token_count=token_count
        )
        
        # Persister en base
        self.db.add(db_message)
        self.db.commit()
        self.db.refresh(db_message)
        
        return db_message
    
    async def mark_message_as_processed(self, message_id: uuid.UUID) -> Optional[Message]:
        """
        Marque un message comme traité.
        
        Args:
            message_id: ID du message
            
        Returns:
            Message mis à jour ou None si non trouvé
        """
        # Récupérer le message
        db_message = await self.get_message_by_id(message_id)
        if not db_message:
            return None
        
        # Marquer comme traité
        db_message.processed = True
        
        # Persister en base
        self.db.add(db_message)
        self.db.commit()
        self.db.refresh(db_message)
        
        return db_message
    
    async def update_token_count(self, message_id: uuid.UUID, token_count: int) -> Optional[Message]:
        """
        Met à jour le nombre de tokens d'un message.
        
        Args:
            message_id: ID du message
            token_count: Nouveau nombre de tokens
            
        Returns:
            Message mis à jour ou None si non trouvé
        """
        # Récupérer le message
        db_message = await self.get_message_by_id(message_id)
        if not db_message:
            return None
        
        # Mettre à jour le nombre de tokens
        db_message.token_count = token_count
        
        # Persister en base
        self.db.add(db_message)
        self.db.commit()
        self.db.refresh(db_message)
        
        return db_message
    
    async def get_conversation_context(
        self,
        conversation_id: uuid.UUID,
        max_messages: int = 20
    ) -> List[Dict[str, str]]:
        """
        Récupère le contexte de conversation au format LLM.
        
        Args:
            conversation_id: ID de la conversation
            max_messages: Nombre maximum de messages à inclure
            
        Returns:
            Liste de messages formatés pour le LLM
        """
        # Récupérer les messages récents
        messages = await self.get_conversation_messages(
            conversation_id=conversation_id,
            limit=max_messages,
            oldest_first=True
        )
        
        # Formater pour le LLM
        llm_messages = []
        for msg in messages:
            llm_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        return llm_messages