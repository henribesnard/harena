"""
Initialisation du package de base de données du conversation_service.

Ce module initialise l'accès à la base de données et exporte les
composants principaux pour une utilisation simplifiée.
"""

from .session import engine, SessionLocal, get_db
from .models import Base, Conversation, Message

__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "Base",
    "Conversation",
    "Message"
]