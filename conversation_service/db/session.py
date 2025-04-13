"""
Configuration des sessions SQLAlchemy et connexion à la base de données.

Ce module gère la connexion à la base de données et fournit les
objets et fonctions pour interagir avec elle via SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator

from ..config.settings import settings

# Création de l'engine SQLAlchemy
engine = create_engine(
    settings.DATABASE_URI,
    pool_pre_ping=True,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DATABASE_ECHO_SQL
)

# Création du sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Création de la classe de base déclarative
Base = declarative_base()


def get_db() -> Generator:
    """
    Fournit une session de base de données comme dépendance.
    
    Yields:
        Session SQLAlchemy
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()