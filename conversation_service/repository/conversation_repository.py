"""
Accès aux données des conversations en base.

Ce module fournit une couche d'abstraction pour l'accès
aux conversations dans la base de données.
"""

import uuid
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from datetime import datetime

from ..db.models import Conversation, ConversationState
from ..models.conversation import ConversationCreate, ConversationUpdate


class ConversationRepository:
    """
    Repository pour les opérations CRUD sur les conversations.
    
    Cette classe encapsule toutes les opérations de base de données
    liées aux conversations.
    """
    
    def __init__(self, db: Session):
        """
        Initialise le repository avec une session de base de données.
        
        Args:
            db: Session de base de données SQLAlchemy
        """
        self.db = db
    
    async def get_conversation_by_id(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        """
        Récupère une conversation par son ID.
        
        Args:
            conversation_id: ID de la conversation
            
        Returns:
            Conversation ou None si non trouvée
        """
        return self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    async def get_user_conversations(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        include_archived: bool = False,
        include_deleted: bool = False
    ) -> List[Conversation]:
        """
        Récupère les conversations d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            skip: Nombre d'éléments à sauter (pour la pagination)
            limit: Nombre d'éléments à récupérer
            include_archived: Inclure les conversations archivées
            include_deleted: Inclure les conversations supprimées
            
        Returns:
            Liste des conversations
        """
        query = self.db.query(Conversation).filter(Conversation.user_id == user_id)
        
        # Filtrer par état
        if not include_deleted and not include_archived:
            query = query.filter(Conversation.state == ConversationState.ACTIVE)
        elif not include_deleted:
            query = query.filter(Conversation.state != ConversationState.DELETED)
        elif not include_archived:
            query = query.filter(Conversation.state != ConversationState.ARCHIVED)
        
        # Ordonner par activité récente
        query = query.order_by(desc(Conversation.last_activity))
        
        # Appliquer la pagination
        return query.offset(skip).limit(limit).all()
    
    async def create_conversation(self, conversation_data: ConversationCreate) -> Conversation:
        """
        Crée une nouvelle conversation.
        
        Args:
            conversation_data: Données de la conversation à créer
            
        Returns:
            Conversation créée
        """
        # Créer la conversation
        db_conversation = Conversation(
            user_id=conversation_data.user_id,
            title=conversation_data.title,
            state=ConversationState.ACTIVE,
            meta_data=conversation_data.meta_data or {}
        )
        
        # Persister en base
        self.db.add(db_conversation)
        self.db.commit()
        self.db.refresh(db_conversation)
        
        return db_conversation
    
    async def update_conversation(
        self,
        conversation_id: uuid.UUID,
        conversation_update: ConversationUpdate
    ) -> Optional[Conversation]:
        """
        Met à jour une conversation.
        
        Args:
            conversation_id: ID de la conversation
            conversation_update: Données de mise à jour
            
        Returns:
            Conversation mise à jour ou None si non trouvée
        """
        # Récupérer la conversation
        db_conversation = await self.get_conversation_by_id(conversation_id)
        if not db_conversation:
            return None
        
        # Mettre à jour les champs
        update_data = conversation_update.dict(exclude_unset=True)
        
        # Renommer meta_data si présent
        if "meta_data" in update_data:
            update_data["meta_data"] = update_data.pop("meta_data")
            
        # Mettre à jour chaque champ
        for key, value in update_data.items():
            setattr(db_conversation, key, value)
        
        # Mettre à jour la date d'activité
        db_conversation.last_activity = datetime.utcnow()
        
        # Persister en base
        self.db.add(db_conversation)
        self.db.commit()
        self.db.refresh(db_conversation)
        
        return db_conversation
    
    async def mark_conversation_as_archived(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        """
        Marque une conversation comme archivée.
        
        Args:
            conversation_id: ID de la conversation
            
        Returns:
            Conversation mise à jour ou None si non trouvée
        """
        # Récupérer la conversation
        db_conversation = await self.get_conversation_by_id(conversation_id)
        if not db_conversation:
            return None
        
        # Marquer comme archivée
        db_conversation.state = ConversationState.ARCHIVED
        db_conversation.last_activity = datetime.utcnow()
        
        # Persister en base
        self.db.add(db_conversation)
        self.db.commit()
        self.db.refresh(db_conversation)
        
        return db_conversation
    
    async def mark_conversation_as_deleted(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        """
        Marque une conversation comme supprimée.
        
        Args:
            conversation_id: ID de la conversation
            
        Returns:
            Conversation mise à jour ou None si non trouvée
        """
        # Récupérer la conversation
        db_conversation = await self.get_conversation_by_id(conversation_id)
        if not db_conversation:
            return None
        
        # Marquer comme supprimée
        db_conversation.state = ConversationState.DELETED
        db_conversation.last_activity = datetime.utcnow()
        
        # Persister en base
        self.db.add(db_conversation)
        self.db.commit()
        self.db.refresh(db_conversation)
        
        return db_conversation
    
    async def delete_conversation(self, conversation_id: uuid.UUID) -> bool:
        """
        Supprime définitivement une conversation.
        
        Args:
            conversation_id: ID de la conversation
            
        Returns:
            True si la suppression a réussi, False sinon
        """
        # Récupérer la conversation
        db_conversation = await self.get_conversation_by_id(conversation_id)
        if not db_conversation:
            return False
        
        # Supprimer de la base
        self.db.delete(db_conversation)
        self.db.commit()
        
        return True
    
    async def update_last_activity(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        """
        Met à jour la date de dernière activité d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            
        Returns:
            Conversation mise à jour ou None si non trouvée
        """
        # Récupérer la conversation
        db_conversation = await self.get_conversation_by_id(conversation_id)
        if not db_conversation:
            return None
        
        # Mettre à jour la date d'activité
        db_conversation.last_activity = datetime.utcnow()
        
        # Persister en base
        self.db.add(db_conversation)
        self.db.commit()
        self.db.refresh(db_conversation)
        
        return db_conversation